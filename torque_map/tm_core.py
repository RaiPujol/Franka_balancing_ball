"""Torque-map ball balancer: the three blocks.

1. Init     put the arm in the balance pose (lab home, gripper down, joints 5 and 6 axes
            horizontal) automatically or by hand, and check it.
2. Map      the empty paddle, then the ball held still at known points: the torque change
            on joints 5 and 6 at each point. A plane through those points is the torque map.
            Plus an empty-paddle tilt sweep: what the paddle alone does to the torques when
            it tilts, which the balancer has to subtract.
3. Balance  drive the torque change on joints 5 and 6 to the value the map recorded for
            the centre. The map turns each N*m of error into metres of ball error; a Kalman
            filter and an LQR turn that into paddle tilt.

Everything here is independent of frankapy and MuJoCo: `io` is robot_io.FrankaIO on the
robot or sim_io.SimIO on a laptop. `ctx` is the UI (app.py) or a test harness.
"""

import json
import os
import time

import numpy as np

import kin
from kin import (D_GAINS, JOINT_MAX, JOINT_MIN, K_GAINS, LAGS, ROLL_ACCEL, TRACK_TIME, TRACK_TOL, AxisKF,
                 Excitation, SafetyStop, Tare, TiltServo, fk, lqr_gain, motion_features)

HERE = os.path.dirname(os.path.abspath(__file__))
CALIB_FILE = os.path.join(HERE, "tm_calib.json")
LOG_DIR = os.path.join(HERE, "logs")

# Default map points from the deck diagram, in mm. x: from the GRIP POINT (middle of the
# fingers on the tab) toward the far edge, away from the robot. y: to the left, seen from
# the robot. C is 125 mm out; the far row is another 101.29 mm.
DEFAULT_POINTS = [
    {"name": "C", "x": 125.0, "y": 0.0},
    {"name": "L", "x": 125.0, "y": 105.0},
    {"name": "R", "x": 125.0, "y": -101.0},
    {"name": "F", "x": 226.29, "y": 0.0},
    {"name": "FL", "x": 226.29, "y": 105.6},
    {"name": "FR", "x": 226.29, "y": -102.2},
]
GAIN_SCALES = {"1": 1.0, "2": 2.0}          # wrist stiffness: frankapy default, or twice as stiff


def to_paddle(x_mm, y_mm):
    """Point from the UI (mm from the grip point) to the paddle frame used inside (m from C)."""
    return np.array([x_mm / 1000.0 - kin.GRIP_TO_C, y_mm / 1000.0])
JOINT_NAMES = ["joint 1", "joint 2", "joint 3", "joint 4", "joint 5", "joint 6", "joint 7"]


# ----------------------------------------------------------------------------------- calibration
class Calib:
    """Everything the three blocks learn, saved to tm_calib.json after every step."""

    def __init__(self):
        self.q0 = None
        self.rate_hz = 50.0
        self.tilt_max_deg = 5.0
        self.channels = [4, 5]                  # joints 5 and 6
        self.ball_mass = kin.BALL_MASS
        self.points = [dict(p) for p in DEFAULT_POINTS]
        self.empty = None                       # {"mean", "std", "t"}
        self.empty_end = None
        self.readings = {}                      # name -> {"mean", "std", "x", "y", "t"}
        self.fit = None                         # {"a", "B", ...}
        self.tare_W = None
        self.tare_sigma = None
        self.frame = "grip"                     # point x measured from the grip point
        self.wrist_scale = 1.0                  # 1 = frankapy default gains on joints 5-7

    # -- persistence
    def to_json(self):
        d = {k: v for k, v in self.__dict__.items()}
        for k in ("q0", "tare_W", "tare_sigma"):
            if d[k] is not None:
                d[k] = np.asarray(d[k]).tolist()
        return d

    def save(self, path=CALIB_FILE):
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=1)

    @classmethod
    def load(cls, path=CALIB_FILE):
        c = cls()
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            for k, v in d.items():
                if hasattr(c, k):
                    setattr(c, k, v)
            for k in ("q0", "tare_W", "tare_sigma"):
                if getattr(c, k) is not None:
                    setattr(c, k, np.array(getattr(c, k), dtype=float))
            if "frame" not in d:
                # files from the first version stored x from C: add the 125 mm, keep the recordings
                off = kin.GRIP_TO_C * 1000.0
                for p in c.points:
                    p["x"] = round(p["x"] + off, 3)
                for r in c.readings.values():
                    r["x"] = round(r["x"] + off, 3)
                c.frame = "grip"
        return c

    @property
    def gains(self):
        """Joint impedance gains for streaming. Only the wrist (5-7) is scaled; damping
        scales with the square root so the damping ratio stays the same."""
        k, d = K_GAINS.copy(), D_GAINS.copy()
        k[4:] *= self.wrist_scale
        d[4:] *= np.sqrt(self.wrist_scale)
        return k, d

    @property
    def T_paddle(self):
        return kin.paddle_transform()

    @property
    def tilt_max(self):
        return np.radians(self.tilt_max_deg)


# ----------------------------------------------------------------------------------- block 1
def pose_report(q, calib=None):
    """How good is this pose for balancing: joints 5 and 6 axes should be horizontal (they
    see the ball at full sensitivity and tilt the paddle in two clean directions), the
    paddle should be level."""
    q = np.asarray(q, dtype=float)
    T, J = fk(q, kin.paddle_transform())
    axis_tilt = [float(np.degrees(np.arcsin(min(1.0, abs(J[5, j]))))) for j in (4, 5, 6)]
    level = float(np.degrees(np.arccos(np.clip(T[2, 2], -1, 1))))
    margin = float(np.min(np.minimum(q - JOINT_MIN, JOINT_MAX - q)))
    ok = axis_tilt[0] < 10 and axis_tilt[1] < 10 and level < 5 and margin > 0.25
    return {"j5_axis_from_horizontal_deg": axis_tilt[0], "j6_axis_from_horizontal_deg": axis_tilt[1],
            "j7_axis_from_vertical_deg": 90.0 - axis_tilt[2], "paddle_from_level_deg": level,
            "paddle_centre_m": T[:3, 3].round(3).tolist(), "joint_limit_margin_rad": margin, "ok": bool(ok)}


def job_go_home(io, calib, ctx):
    ctx.status("moving to the balance pose (lab home)")
    io.goto(np.array(kin.HOME_POSE), 5.0)
    io.wait(1.0)
    return _use_pose(io, calib, ctx)


def job_guide(io, calib, ctx, seconds=30.0):
    ctx.status(f"guide mode for {seconds:.0f} s: move the arm by hand, then let go")
    io.guide(seconds)
    ctx.log("guide mode ended. Press 'Use current pose' if the paddle is where you want it.")


def job_use_current(io, calib, ctx):
    return _use_pose(io, calib, ctx)


def _use_pose(io, calib, ctx):
    q = np.asarray(io.read()["q"], dtype=float)
    rep = pose_report(q)
    calib.q0 = q
    # a new pose invalidates everything learned at the old one
    calib.empty = calib.empty_end = calib.fit = calib.tare_W = calib.tare_sigma = None
    calib.readings = {}
    calib.save()
    ctx.log(f"balance pose saved. joint 5 axis {rep['j5_axis_from_horizontal_deg']:.1f} deg and joint 6 axis "
            f"{rep['j6_axis_from_horizontal_deg']:.1f} deg from horizontal, paddle {rep['paddle_from_level_deg']:.1f} "
            f"deg from level => {'PASS' if rep['ok'] else 'CHECK THE POSE'}")
    return rep


# ----------------------------------------------------------------------------------- block 2
def record_static(io, calib, ctx, seconds):
    """Average the joint torques with the arm still (no streaming)."""
    dt = 1.0 / calib.rate_hz
    io.wait(0.5)                                # let a placed ball settle
    taus, qs = [], []
    t0 = io.clock()
    while io.clock() - t0 < seconds:
        if ctx.cancelled():
            raise SafetyStop("cancelled")
        s = io.read()
        taus.append(np.asarray(s["tau"], dtype=float))
        qs.append(np.asarray(s["q"], dtype=float))
        ctx.live(tau=taus[-1], q=qs[-1])
        io.wait(dt)
    taus, qs = np.array(taus), np.array(qs)
    if np.degrees(np.ptp(qs, axis=0)).max() > 0.5:
        raise SafetyStop("the arm moved during the recording: hands off and record again")
    return taus.mean(axis=0), taus.std(axis=0)


def job_record_empty(io, calib, ctx, which="start", seconds=3.0):
    _need_pose(calib)
    io.place_ball(None)
    ctx.status(f"recording the EMPTY paddle for {seconds:.0f} s")
    mean, std = record_static(io, calib, ctx, seconds)
    entry = {"mean": mean.tolist(), "std": std.tolist(), "t": time.time()}
    if which == "end":
        calib.empty_end = entry
    else:
        calib.empty, calib.empty_end, calib.readings, calib.fit = entry, None, {}, None
    calib.save()
    ch = calib.channels
    ctx.log(f"empty paddle ({which}): " + ", ".join(
        f"{JOINT_NAMES[c]} {mean[c]:+.3f} Nm (noise {std[c] * 1000:.1f} mNm)" for c in ch))
    if which == "end" and calib.fit:
        job_fit_map(io, calib, ctx)


def job_record_point(io, calib, ctx, name, seconds=3.0):
    _need_pose(calib)
    if calib.empty is None:
        raise SafetyStop("record the empty paddle first")
    p = next((p for p in calib.points if p["name"] == name), None)
    if p is None:
        raise SafetyStop(f"no point called {name}")
    io.place_ball(tuple(to_paddle(p["x"], p["y"])), hold=True)
    ctx.status(f"recording point {name} ({p['x']:.0f}, {p['y']:.0f}) mm for {seconds:.0f} s")
    mean, std = record_static(io, calib, ctx, seconds)
    calib.readings[name] = {"mean": mean.tolist(), "std": std.tolist(), "x": p["x"], "y": p["y"],
                            "t": time.time()}
    calib.fit = None
    calib.save()
    d = mean - np.array(calib.empty["mean"])
    ctx.log(f"point {name}: change vs empty " + ", ".join(
        f"{JOINT_NAMES[c]} {d[c] * 1000:+.1f} mNm" for c in calib.channels))
    if len(calib.readings) >= 3:
        try:
            job_fit_map(io, calib, ctx)
        except SafetyStop:
            ctx.log("(map not fitted yet: needs points spread along both x and y)")


def job_fit_map(io, calib, ctx):
    """Plane through the recorded points: d_tau = a + B @ p, per channel (joints 5, 6).
    The physics says it is a plane (torque = weight x lever arm), so the residuals show the
    measurement quality, and the slopes should be about ball weight per metre."""
    names = [n for n in calib.readings if any(p["name"] == n for p in calib.points)]
    if len(names) < 3:
        raise SafetyStop("record at least 3 points")
    ch = calib.channels
    e0 = np.array(calib.empty["mean"])
    P = np.array([to_paddle(calib.readings[n]["x"], calib.readings[n]["y"]) for n in names])
    D = np.array([np.array(calib.readings[n]["mean"])[ch] - e0[ch] for n in names])
    X = np.c_[np.ones(len(P)), P]
    coef, *_ = np.linalg.lstsq(X, D, rcond=None)
    a, B = coef[0], coef[1:].T                       # B[channel, (x, y)] in Nm per m
    if abs(np.linalg.det(B)) < 1e-6:
        raise SafetyStop("the map cannot tell x from y: record points spread in both directions")
    est = np.linalg.solve(B, (D - a).T).T
    err_mm = np.linalg.norm(est - P, axis=1) * 1000
    fit = {"names": names, "a": a.tolist(), "B": B.tolist(), "err_mm": err_mm.tolist(),
           "rms_mm": float(np.sqrt(np.mean(err_mm ** 2))),
           "expected_slope_mNm_per_cm": calib.ball_mass * kin.G * 10.0,
           "slope_mNm_per_cm": (np.abs(B) * 10.0).tolist()}
    # noise: one sample of joint 5/6 noise, as ball position
    std = np.array(calib.empty["std"])[ch]
    Binv = np.linalg.inv(B)
    fit["noise_mm_per_sample"] = (np.sqrt(np.diag(Binv @ np.diag(std ** 2) @ Binv.T)) * 1000).tolist()
    if calib.empty_end is not None:
        drift = np.array(calib.empty_end["mean"])[ch] - e0[ch]
        fit["drift_mm"] = float(np.linalg.norm(Binv @ drift) * 1000)
    ok = fit["rms_mm"] < 10.0 and fit.get("drift_mm", 0.0) < 10.0
    fit["verdict"] = "PASS" if ok else ("MARGINAL" if fit["rms_mm"] < 20.0 else "FAIL")
    calib.fit = fit
    calib.save()
    ctx.log(f"map from {len(names)} points: slopes " + ", ".join(
        f"{JOINT_NAMES[c]} {np.abs(B[i]).max() * 10:.1f} mNm/cm" for i, c in enumerate(ch))
            + f" (expected about {fit['expected_slope_mNm_per_cm']:.1f}); position error rms "
            f"{fit['rms_mm']:.1f} mm" + (f", drift {fit['drift_mm']:.1f} mm" if "drift_mm" in fit else "")
            + f" => {fit['verdict']}")
    return fit


class Runner:
    """A streaming session: the paddle under the joint impedance controller, tilted by the
    wrist servo, with every safety check from the earlier package."""

    def __init__(self, io, calib, ctx, log_name):
        _need_pose(calib)
        self.io, self.calib, self.ctx = io, calib, ctx
        self.dt = 1.0 / calib.rate_hz
        self.k_gains, self.d_gains = calib.gains
        self.servo = TiltServo(calib.q0, calib.T_paddle, self.dt, calib.tilt_max, lead=self.d_gains / self.k_gains)
        self.P2 = self.servo.R0[:2, :2]           # paddle x, y in the base frame
        self.tare = Tare(calib.tare_W.copy(), calib.tare_sigma.copy()) if calib.tare_W is not None else None
        self.log, self.log_name = {}, log_name
        self.t = 0.0

    def start(self):
        rep = pose_report(self.calib.q0)
        if rep["paddle_from_level_deg"] > 10:
            raise SafetyStop("paddle is far from level at the saved pose: check the mount constants in kin.py")
        self.io.goto(self.calib.q0, 4.0)
        self.servo.reset()
        self._vel, self._vz, self._late, self.accel_z = [], None, np.zeros(3), 0.0
        self.io.start_stream(self.calib.q0, self.k_gains, self.d_gains)

    def stop(self):
        try:
            for _ in range(int(1.0 / self.dt)):
                self.io.command(self.servo.step([0.0, 0.0]))
        finally:
            self.io.stop_stream()
            if self.log_name and self.log:
                os.makedirs(LOG_DIR, exist_ok=True)
                np.savez(self.log_name, **{k: np.array(v) for k, v in self.log.items()})
                self.ctx.log(f"log saved: {os.path.relpath(self.log_name, HERE)}")

    def cycle(self, tilt_target, **extra):
        if self.ctx.cancelled():
            raise SafetyStop("stopped from the UI")
        state = self.io.read()
        q = np.asarray(state["q"], dtype=float)
        dq = np.asarray(state["dq"], dtype=float)
        tau = np.asarray(state["tau"], dtype=float)
        self._vel = ([dq] + self._vel)[:LAGS + 1]
        while len(self._vel) < LAGS + 1:
            self._vel.append(self._vel[-1])
        phi = motion_features(q[4:7] - self.calib.q0[4:7], self._vel)
        _, J = fk(q, self.calib.T_paddle)
        vz = J[2] @ dq
        self.accel_z = 0.0 if self._vz is None else (vz - self._vz) / self.dt
        self._vz = vz
        if np.max(np.abs(q - self.servo.q)) > 0.12:
            raise SafetyStop("arm is not following the command (blocked or pushed?)")
        err = np.abs(q[4:7] - self.servo.q[4:7])
        self._late = np.where(err > TRACK_TOL, self._late + self.dt, 0.0)
        if np.any(self._late > TRACK_TIME):
            j = int(np.argmax(self._late)) + 5
            i = j - 1
            self._rec(t=self.t, q=q, tau=tau, q_target=self.servo.q.copy())
            raise SafetyStop(
                f"joint {j} is not following its target at t={self.t:.1f} s: target "
                f"{np.degrees(self.servo.q[i] - self.calib.q0[i]):+.2f} deg, actual {np.degrees(q[i] - self.calib.q0[i]):+.2f} "
                f"deg (from the balance pose), torque {tau[i]:+.2f} Nm, velocity {np.degrees(dq[i]):+.1f} deg/s. "
                "Something blocks it (paddle, cable), the joint sticks, or a joint fault. Run the track test, "
                "check Desk for errors, and send the log.")
        tilt = self.servo.measured_tilt(q)
        q_des = self.servo.step(tilt_target, tilt)
        if np.max(np.abs(q_des - self.calib.q0)) > 0.35 or np.any(q_des < JOINT_MIN + 0.02) \
                or np.any(q_des > JOINT_MAX - 0.02):
            raise SafetyStop("joint target left the allowed window around the balance pose")
        self.io.command(q_des)
        self.t += self.dt
        self._rec(t=self.t, q=q, tau=tau, tilt=tilt, tilt_cmd=self.servo.cmd.copy(), q_target=self.servo.q.copy(),
                  dq=dq, **extra)
        return q, dq, tau, phi, tilt, J

    def hold(self, seconds, settle=1.0):
        res = []
        for i in range(int((settle + seconds) / self.dt)):
            q, _, tau, phi, _, _ = self.cycle([0.0, 0.0])
            if i * self.dt >= settle:
                res.append(tau - (self.tare.predict(phi) if self.tare else 0.0))
            self.ctx.live(tau=tau, q=q)
        return np.mean(res, axis=0)

    def prompt(self, message, ball=None, hold=False):
        """Ask the operator, keep the paddle level and streaming until they press Continue."""
        self.io.place_ball(ball, hold=hold)
        self.ctx.ask(message)
        while not self.ctx.answered():
            q, _, tau, _, _, _ = self.cycle([0.0, 0.0])
            self.ctx.live(tau=tau, q=q)

    def _rec(self, **kw):
        for k, v in kw.items():
            if v is not None:
                self.log.setdefault(k, []).append(np.array(v, dtype=float))


def job_tilt_sweep(io, calib, ctx, seconds=60.0):
    """Empty paddle rocks up to 0.8 x tilt limit: learns the paddle's own torques vs tilt."""
    if calib.fit is None:
        raise SafetyStop("build the map first (block 2)")
    run = Runner(io, calib, ctx, os.path.join(LOG_DIR, f"sweep_{time.strftime('%m%d_%H%M%S')}.npz"))
    run.start()
    Phi, Y = [], []
    try:
        run.prompt("Tilt sweep: REMOVE the ball, hands off the arm, then Continue. The paddle will rock "
                   f"for {seconds:.0f} s.", ball=None)
        exc = Excitation(0.8 * calib.tilt_max, seconds)
        n = int(seconds / run.dt)
        for i in range(n + int(2.0 / run.dt)):
            q, _, tau, phi, tilt, _ = run.cycle(exc(i, run.dt) if i < n else [0.0, 0.0])
            ctx.status(f"tilt sweep {i * run.dt:4.0f} / {seconds:.0f} s")
            ctx.live(tau=tau, q=q)
            if i > int(3.0 / run.dt):
                Phi.append(phi)
                Y.append(tau)
    finally:
        run.stop()
    tare = Tare.fit(np.array(Phi), np.array(Y))
    calib.tare_W, calib.tare_sigma = tare.W, tare.sigma
    B = np.array(calib.fit["B"])
    ch = calib.channels
    Binv = np.linalg.inv(B)
    noise = np.sqrt(np.diag(Binv @ np.diag(tare.sigma[ch] ** 2) @ Binv.T)) * 1000
    calib.fit["moving_noise_mm"] = noise.tolist()
    verdict = "PASS" if noise.max() < 30 else "MARGINAL" if noise.max() < 60 else "FAIL"
    calib.fit["moving_verdict"] = verdict
    calib.save()
    ctx.log("tilt sweep: leftover torque while moving " + ", ".join(
        f"{JOINT_NAMES[c]} {tare.sigma[c] * 1000:.0f} mNm" for c in ch)
            + f" = ball position noise {noise[0]:.0f} mm (x), {noise[1]:.0f} mm (y) per sample => {verdict}")


def job_track(io, calib, ctx, seconds=15.0, amp_deg=2.0):
    """No ball. Each tilt axis rocks alone; checks that the wrist joints follow their
    targets. This isolates the joint 6 problem from everything else."""
    run = Runner(io, calib, ctx, os.path.join(LOG_DIR, f"track_{time.strftime('%m%d_%H%M%S')}.npz"))
    amp = np.radians(amp_deg)
    out = {}
    run.start()
    try:
        run.prompt(f"Track test: no ball needed, hands off. Each tilt axis rocks alone for {seconds:.0f} s "
                   f"(max {amp_deg:.0f} deg). Continue.", ball=None)
        for ax, name in ((1, "B: forward/back (joint 6)"), (0, "A: left/right (joint 5)")):
            ctx.status(f"track test, axis {name}")
            cmd, meas, qt, qm = [], [], [], []
            n = int(seconds / run.dt)
            for i in range(n + int(1.5 / run.dt)):
                t = i * run.dt
                ramp = min(1.0, t / 2.0, max(0.0, (seconds - t) / 2.0))
                target = np.zeros(2)
                if i < n:
                    target[ax] = amp * ramp * (0.6 * np.sin(2 * np.pi * 0.2 * t) + 0.4 * np.sin(2 * np.pi * 0.6 * t + 1.0))
                q, _, tau, _, tilt, _ = run.cycle(target)
                ctx.live(tau=tau, q=q, tilt=tilt)
                if 1.0 < t < seconds - 1.0:
                    cmd.append(run.servo.cmd[ax]), meas.append(tilt[ax]), qt.append(run.servo.q.copy()), qm.append(q)
            out[name] = (np.array(cmd), np.array(meas), np.array(qt), np.array(qm))
    except SafetyStop as e:
        ctx.log(f"TRACK TEST FAILED: {e}")
    finally:
        run.stop()
    for name, (c, m, qt, qm) in out.items():
        best = max(range(0, 16), key=lambda k: np.corrcoef(c[:len(c) - k], m[k:])[0, 1])
        corr = float(np.corrcoef(c[:len(c) - best], m[best:])[0, 1])
        gain = float(np.polyfit(c[:len(c) - best], m[best:], 1)[0])
        lag_err = np.degrees(np.abs(qm[:, 4:7] - qt[:, 4:7]).max(axis=0))
        ok = 0.8 <= gain <= 1.2 and corr > 0.95 and best * run.dt <= 0.15
        ctx.log(f"track axis {name}: gain {gain:.2f} (0.8-1.2), correlation {corr:.3f} (>0.95), lag {best * run.dt:.2f} s; "
                f"worst joint error 5/6/7 = {lag_err[0]:.2f} / {lag_err[1]:.2f} / {lag_err[2]:.2f} deg => "
                + ("PASS" if ok else "FAIL"))
    if len(out) < 2:
        ctx.log("track test incomplete: see the STOP message above")


# ----------------------------------------------------------------------------------- block 3
def job_balance(io, calib, ctx, seconds=30.0, gain=0.6, axis="both", start=(0.03, -0.02)):
    """Drive the joint 5/6 torque change to the centre value from the map."""
    if calib.fit is None or calib.tare_W is None:
        raise SafetyStop("do block 2 first: map and tilt sweep")
    run = Runner(io, calib, ctx, os.path.join(LOG_DIR, f"balance_{time.strftime('%m%d_%H%M%S')}.npz"))
    ch = calib.channels
    a, B = np.array(calib.fit["a"]), np.array(calib.fit["B"])
    Binv = np.linalg.inv(B)
    target_p = np.zeros(2)                               # map point C
    target_tau = a + B @ target_p                        # the torque change recorded at the centre
    K = lqr_gain(run.dt, run.servo.wc, gain)
    kfx, kfy = AxisKF(run.dt), AxisKF(run.dt)
    m = calib.ball_mass
    run.start()
    lost = 0
    try:
        run.prompt("Balance: REMOVE the ball, hands off, then Continue (3 s re-zero).", ball=None)
        ctx.status("re-zero: empty paddle")
        run.tare.rezero(run.hold(3.0))
        where = "IN THE RAIL near the centre" if axis in ("x", "y") else "near the CENTRE"
        run.prompt(f"Put the ball {where}, let go, then Continue.", ball=tuple(start))
        sig_p = np.sqrt(np.diag(Binv @ np.diag(run.tare.sigma[ch] ** 2) @ Binv.T))

        def measure(q, tau, phi, J, ball_acc):
            dtau = (tau - run.tare.predict(phi))[ch]
            # the ball presses with m(g + a_z) when the paddle moves vertically,
            # and pushes back horizontally while it accelerates
            dtau = dtau - (J[:3, ch].T @ np.r_[-m * np.asarray(ball_acc), 0.0])
            dtau = dtau / (1.0 + run.accel_z / kin.G)
            p = Binv @ (dtau - a)                        # paddle frame, m
            return p, dtau

        q, _, tau, phi, tilt, J = run.cycle([0.0, 0.0])
        p, _ = measure(q, tau, phi, J, (0.0, 0.0))
        xy = run.P2 @ p
        kfx.reset(xy[0])
        kfy.reset(xy[1])
        target = np.zeros(2)
        tilt_prev = tilt
        ctx.status("balancing")
        for i in range(int(seconds / run.dt)):
            q, _, tau, phi, tilt, J = run.cycle(target)
            ball_acc = (ROLL_ACCEL * np.sin(tilt[1]) + kfx.x[2], -ROLL_ACCEL * np.sin(tilt[0]) + kfy.x[2])
            p, dtau = measure(q, tau, phi, J, ball_acc)
            z = run.P2 @ p
            r = (run.P2 @ sig_p) ** 2
            tilt_mid = 0.5 * (tilt + tilt_prev)
            tilt_prev = tilt
            x = kfx.step(ROLL_ACCEL * np.sin(tilt_mid[1]), z[0], abs(r[0]))
            yv = kfy.step(-ROLL_ACCEL * np.sin(tilt_mid[0]), z[1], abs(r[1]))
            goal = run.P2 @ target_p
            (a_cmd, b_cmd), (da_cmd, db_cmd) = run.servo.cmd, run.servo.dcmd
            bias_x, bias_y = x[2] / ROLL_ACCEL, yv[2] / ROLL_ACCEL
            b_t = -K @ np.array([x[0] - goal[0], x[1], b_cmd + bias_x, db_cmd]) - bias_x
            a_t = K @ np.array([yv[0] - goal[1], yv[1], -a_cmd + bias_y, -da_cmd]) + bias_y
            if axis == "x":
                a_t = 0.0
            elif axis == "y":
                b_t = 0.0
            target = np.array([a_t, b_t])
            est_p = np.linalg.solve(run.P2, np.array([x[0], yv[0]]))
            err = dtau - target_tau
            run._rec(p_meas=p, est=est_p, torque_error=err, truth=io.ball_truth())
            ctx.live(tau=tau, q=q, est=est_p, meas=p, err=err, tilt=tilt)
            watched = [est_p[0]] if axis == "x" else [est_p[1]] if axis == "y" else est_p
            lost = lost + 1 if np.max(np.abs(watched)) > kin.PLATE_HALF + 0.03 else 0
            if lost > int(0.5 / run.dt):
                raise SafetyStop("ball estimate left the paddle")
            if np.max(np.abs(dtau)) > 2.0:
                raise SafetyStop("large unexpected torque (contact?)")
    except SafetyStop as e:
        ctx.log(f"STOP: {e}. Levelling the paddle.")
    finally:
        run.stop()
    if "est" in run.log and len(run.log["est"]) > 10:
        e = np.array(run.log["est"]) * 1000
        err = np.array(run.log["torque_error"]) * 1000
        ctx.log(f"balance ran {len(e) * run.dt:.1f} s. ball from centre rms x {np.sqrt(np.mean(e[:, 0] ** 2)):.0f} mm, "
                f"y {np.sqrt(np.mean(e[:, 1] ** 2)):.0f} mm; torque error rms "
                + ", ".join(f"{JOINT_NAMES[c]} {np.sqrt(np.mean(err[:, i] ** 2)):.0f} mNm" for i, c in enumerate(ch)))
    return run.log


def _need_pose(calib):
    if calib.q0 is None:
        raise SafetyStop("set the balance pose first (block 1)")
