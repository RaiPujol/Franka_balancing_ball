"""Balancing along ONE paddle axis, with the ball in a rail (pencils or sticks).

Default: the rail runs along paddle y (left/right). Joint 5 alone tilts the paddle about
base x, and JOINT 3's torque tells where the ball is along the rail (13.4 mNm per cm).
Axis "x" (rail forward/back): joint 6 tilts, joints 2 and 4 sense.

Why not sense on the joint that tilts (Nov 20, 20:44-20:52 logs): joint 5's torque follows
its own motion (+-0.3 Nm, flipping when it reverses = 15 cm of "ball") and keeps a memory
of it (the empty paddle read 0.09 Nm = 90 mm different before and after the rail map).
Joint 3 holds still: its error is plain noise (19 mm per sample still, 45 mm rocking, no
drift), which the Kalman filter removes. The filter, an anchor at the start (ball placed at
a known mark) and a live "the ball is HERE" correction from the page handle the rest.

Why one axis at a time works better than the 2-D set-up:
* One wrist joint moves, by a commanded angle, no IK: at the lab home pose the joint 5 axis
  is the base x axis (checked: 1.000), so nothing else moves and nothing couples.
* The map is 1-D: 3-5 points along the rail and a straight line per joint, checked against
  the slope the kinematics predict (ball weight x lever: 18.9 mNm/cm on joint 5). A 2-D plane
  fit through 6 points let one bad point wreck both axes (the 19:58 map on Nov 20).
* The map is recorded WHILE STREAMING at the level pose, with the same controller and gains
  as balancing, so there is no offset between "idle" and "streaming" torques.
* Small steps at high rate: 100 Hz loop (the robot state comes at 100 Hz) and a hard cap on
  the tilt rate (4 deg/s = 0.04 deg per period by default).

Blocks (all need the balance pose from block 1):
  job_rail_track    no ball: the joint rocks alone; how well the tilt follows
  job_rail_map      streaming, level: empty, ball held at each rail point, empty again
  job_rail_sweep    no ball: the joint rocks; learns what the paddle alone does to the torques
  job_rail_balance  the balancer
"""

import os
import time

import numpy as np

import kin
from kin import ROLL_ACCEL, AxisKF, Excitation, JointTiltServo, SafetyStop, Tare, fk, lqr_gain, RailFeatures
from tm_core import JOINT_NAMES, LOG_DIR, Runner, _need_pose

# The tilting joint is never a sensor: its torque follows its own motion, so the balancer
# chases its own tilt (21:09 run: +-1 deg, 0.7 Hz self-oscillation, ball left on the right).
# idx: paddle/base coordinate along the rail. tilt: index in [a, b]. sgn: ball accel = sgn * 5/7 g sin(tilt).
AXES = {
    "y": {"idx": 1, "tilt": 0, "sgn": -1.0, "joint": 4, "channels": [2], "allowed": [2],
          "name": "y (left/right)", "plus": "left", "minus": "right"},
    "x": {"idx": 0, "tilt": 1, "sgn": 1.0, "joint": 5, "channels": [1, 3], "allowed": [1, 3],
          "name": "x (forward/back)", "plus": "far edge", "minus": "robot"},
}
DEFAULT_CFG = {
    "axis": "y",
    "rate_hz": 100.0,          # control loop; the robot state comes at 100 Hz
    "tilt_max_deg": 2.0,       # largest counter-angle the controller may ask for (preset "lqr")
    "rate_max_dps": 10.0,      # tilt rate cap: 0.1 deg per period at 100 Hz
    "wc": 15.0,                # tilt command filter bandwidth, rad/s (6 gave 330 ms of lag on Nov 20)
    "roll_scale": 1.0,         # real ball acceleration / (5/7 g sin tilt). A ball on two sticks rolls on a smaller radius
                               # and accelerates less; measure it with the roll test (Probe at a fixed tilt)
    "accel_max_dps2": 40.0,    # tilt acceleration cap: jerky tilts shake the arm and drown joint 3's signal
    "channels": None,          # sensing joints, 0-based (None = the axis default: joint 3 for y)
    "hand_yaw_deg": 0.0,       # 0: balance pose as saved. -90: joint 7 turned 90 deg, so a rail along paddle y
                               # points toward/away from the robot: joints 2 + 4 sense it, joint 6 tilts
    "smooth": "medium",        # Kalman filter: how much the estimate is smoothed (low / medium / high)
    # balance controller: cascaded PD with a ball-speed limit (see job_rail_balance)
    "ctrl": "lqr",             # "pd" (cascaded PD with a speed limit) or "lqr"
    "kp": 1.55,                # PD, 1/s: wanted ball speed per metre from the goal
    "kv": 2.0,                 # PD, 1/s: wanted ball acceleration per m/s of speed error
    "vmax_mm_s": 15.0,         # PD: never ask the ball to move faster than this
    "lqr_gain": 0.6,           # LQR: about the tilt asked per metre of error (rad/m)
    "lqr_qv": 0.1,             # LQR: weight on ball speed vs position (more = more damping)
    "lead_s": 0.15,            # predict the ball this far ahead (arm + filter delay)
    "ki": 0.0,                 # integral (both controllers), rad per (m s): removes a steady lean of the paddle (try 0.3)
    "ki_max_deg": 1.5,         # most tilt the integral may add
    # sticky rail (Nov 20: the ball needs ~1-2 deg to break free; it dropped off the ends)
    "brake_zone": False,       # brake at full counter-angle when the ball would otherwise stop past the margin
    "brake_margin_mm": 60.0,   # where the ball must be stopped by (the rail ends are at +-79 mm)
    "kick_deg": 1.0,           # ball still and off the goal: tilt at least this much toward the goal (0 = off)
    "deadband_mm": 8.0,        # no kicks inside this distance from the goal
    "hold_mm": 12.0,           # still-ball hold: inside this distance from the goal AND slower than hold_mm_s the paddle
    "hold_mm_s": 15.0,         # eases back to level instead of chasing sensor noise (0 = off). Nov 21 21:08: the ball sat
                               # still while the tilt dithered and wandered to -0.9 deg, then broke loose and ran off
    "stick_deg": 0.8,          # filter: a still ball stays put below this tilt (0 = no stiction in the model)
    "level_trim_deg": 0.0,     # where "level" is for the tilting joint at the rail pose (set with a phone level,
                               # live during a probe; + tips the rail like a + tilt command)
    "gate_accel_dps2": 0.0,    # reactive mode: ignore the torque reading while the tilt accelerates harder than this
                               # (the arm shakes and joint 3 hears it); the filter coasts on the model (0 = off)
    "goal_mm": 0.0,            # where to hold the ball, mm from C
    # Kalman filter
    "kf_accel": 0.05,          # m/s^2 of ball acceleration the model may not explain (more = faster, noisier)
    "kf_bias": 0.01,           # how fast the level offset may wander
    "kf_bias_max_deg": 1.0,    # largest level offset it may believe
    "kf_r_scale": 1.0,         # x measurement noise (more = trust the torque reading less)
    "kf_adapt": True,          # learn the measurement noise while running
    # rail ends
    "end_clamp": True,         # the estimate stops at the rail ends; the level offset is frozen there
    "unstick": True,           # ball held at an end: ramp the counter-angle up until it comes free
    "unstick_dps": 1.5,        # how fast that ramp grows, deg/s
    "balance_stops": True,     # False: no saturation stop and no ball-off-the-rail stop (aggressive testing).
                               # The tilt limit, joint window, blocked-arm, contact and frozen-robot stops stay.
    "cfg_version": 3,
}
# Starting points (Nov 20, fake robot with your rail's stickiness, arm delay and noise; the PD set is the one
# tuned by hand on the robot at 22:33). One click on the page loads either; then tune live.
_COMMON = {"goal_mm": 0.0, "wc": 15.0, "accel_max_dps2": 40.0, "ki": 0.0, "ki_max_deg": 1.5, "brake_zone": False,
           "brake_margin_mm": 60.0, "kick_deg": 1.0, "deadband_mm": 8.0, "stick_deg": 0.8, "kf_accel": 0.05, "kf_bias": 0.01, "kf_bias_max_deg": 2.0, "kf_r_scale": 1.0,
           "kf_adapt": True, "end_clamp": True, "unstick": True, "unstick_dps": 1.5}
PRESETS = {
    # sticky-rail sets (Nov 20 23:50, fake robot with 28 Hz state, 0.9 deg stiction, 0.8 deg lean): stiction in the
    # filter cut the time at a rail end 4.8 -> 1.5 %; PD with a 25 mm/s speed cap held the ball best
    "lqr": dict(_COMMON, ctrl="lqr", lqr_gain=0.6, lqr_qv=0.1, tilt_max_deg=2.5, lead_s=0.15, rate_max_dps=10.0,
                kf_accel=0.2, kf_bias_max_deg=2.0),
    # PD for the sticky rail (Nov 21 01:50, fake robot matched to run 6: 1.6 deg breakaway, joint-5 stiction,
    # vibration, 28 Hz): ramping kick + stronger catch. Back from the rail ends in ~3 s (run-6 settings: 13-17 s)
    # SMOOTH rail (Nov 21 16:33-16:47, real robot, rail fixed to roll freely): gentle PD, small tilts, no
    # stiction model. Best real result: holds of 21-46 s, ~3 drops per 90 s (with the start fix, 16:46).
    # smooth: the setup of the best run (iam-bruno Nov 21 19:22, 55 s at +-5-8 mm). Raising max tilt / ball speed and
    # lowering the rolling gain made the plate swing more every time (Nov 21 21:08, Nov 24 21:43)
    "smooth": dict(_COMMON, ctrl="pd", kp=1.0, kv=2.0, vmax_mm_s=20.0, tilt_max_deg=1.2, lead_s=0.35, rate_max_dps=10.0,
                   kf_accel=0.2, kf_bias=0.02, kf_bias_max_deg=2.0, kick_deg=0.0, stick_deg=0.0, unstick_dps=3.0),
    "pd": dict(_COMMON, ctrl="pd", kp=2.0, kv=3.0, vmax_mm_s=30.0, tilt_max_deg=2.5, lead_s=0.35, rate_max_dps=10.0,
               kf_accel=0.2, kf_bias_max_deg=2.0, kick_deg=1.2, unstick_dps=3.0),
}
TUNABLE = ("ctrl", "kp", "kv", "vmax_mm_s", "lqr_gain", "lqr_qv", "lead_s", "goal_mm", "tilt_max_deg", "rate_max_dps",
           "wc", "kf_accel", "kf_bias", "kf_bias_max_deg", "kf_r_scale", "kf_adapt", "end_clamp", "unstick", "unstick_dps", "balance_stops", "accel_max_dps2", "ki", "ki_max_deg",
           "brake_zone", "brake_margin_mm", "kick_deg", "deadband_mm", "stick_deg", "gate_accel_dps2",
           "level_trim_deg", "roll_scale", "hold_mm", "hold_mm_s")
BOOLS = ("kf_adapt", "end_clamp", "unstick", "balance_stops", "brake_zone")
# Kalman filter: how much ball acceleration the model may NOT explain (m/s^2). The model knows
# gravity along the tilt and the deck's sideways push; what is left is rolling friction and
# bumps on the rail, a few hundredths of g. 0.05 m/s^2 is about 0.4 deg of unknown tilt.
SMOOTH = {"low": 0.15, "medium": 0.05, "high": 0.025}
BIAS_MAX = np.radians(1.0)          # the paddle is level to better than this (tilt-equivalent of the bias state)
DEFAULT_POINTS = [-60.0, -30.0, 0.0, 30.0, 60.0]     # mm from C along the rail
LIMITS = {"rate_hz": (20.0, 100.0), "tilt_max_deg": (0.5, 5.0), "rate_max_dps": (0.5, 30.0), "wc": (2.0, 30.0),
          "kp": (0.05, 20.0), "kv": (0.2, 50.0), "vmax_mm_s": (5.0, 1000.0), "lead_s": (0.0, 1.0),
          "lqr_gain": (0.05, 20.0), "lqr_qv": (0.0, 20.0), "goal_mm": (-60.0, 60.0), "kf_accel": (0.005, 1.0),
          "kf_bias": (0.0, 0.2), "kf_bias_max_deg": (0.0, 3.0), "kf_r_scale": (0.1, 10.0), "unstick_dps": (0.1, 10.0),
          "accel_max_dps2": (5.0, 5000.0), "ki": (0.0, 20.0), "ki_max_deg": (0.0, 5.0),
          "brake_margin_mm": (10.0, 79.0), "kick_deg": (0.0, 5.0), "deadband_mm": (0.0, 50.0), "stick_deg": (0.0, 3.0),
          "gate_accel_dps2": (0.0, 5000.0), "level_trim_deg": (-3.0, 3.0), "roll_scale": (0.2, 1.5),
          "hold_mm": (0.0, 50.0), "hold_mm_s": (0.0, 100.0)}


def rail_q0(calib, c):
    """The pose the rail runs stream around: the saved balance pose, joint 7 turned by hand_yaw_deg."""
    q = np.array(calib.q0, dtype=float)
    q[6] += np.radians(c.get("hand_yaw_deg", 0.0))
    return q


def geometry(calib, axis, yaw_deg=0.0):
    """Everything that depends on where the rail points, computed from the kinematics at the actual pose:
      u        base-frame direction of +p (the rail, paddle x or y)
      joint    the wrist joint whose axis is horizontal and across the rail (it tilts the rail)
      tilt     which base tilt that is (0: about x, 1: about y); s: tilt per rad of that joint
      sgn      ball acceleration along +p = sgn * 5/7 g sin(tilt)
      slopes   d(joint torque)/dp for every joint (Nm per m), allowed: the joints that see the ball well
    Normal pose: y rail -> joint 5 tilts, joint 3 senses; x rail -> joint 6 tilts, joints 2 + 4 sense.
    Hand turned -90 deg: the y rail lies along base x -> joint 6 tilts, joints 2 + 4 sense."""
    base = AXES[axis]
    if calib.q0 is None:
        g = dict(base); g.update(u=None, s=1.0, slopes=None, q0=None)
        return g
    q0 = np.array(calib.q0, dtype=float); q0[6] += np.radians(yaw_deg)
    T0, J0 = fk(q0, calib.T_paddle)
    u = T0[:3, :3][:, base["idx"]].copy(); u[2] = 0.0; u /= max(np.linalg.norm(u), 1e-9)
    best = max((4, 5), key=lambda j: abs(np.cross(J0[3:, j], u)[2]))
    w = J0[3:, best]
    k = 0 if abs(w[0]) >= abs(w[1]) else 1
    e_k = np.eye(3)[k]
    sgn = -float(np.sign(np.cross(e_k, u)[2]))
    h = 0.02
    z = [fk(q0, _T_point(axis, sh))[1][2] for sh in (-h, h)]
    slopes = calib.ball_mass * kin.G * (z[1] - z[0]) / (2 * h)
    allowed = [j for j in range(7) if j != best and abs(slopes[j]) > 0.4 * calib.ball_mass * kin.G]
    lab = lambda v: ("the far edge (away from the robot)" if v[0] > 0.7 else "the robot" if v[0] < -0.7
                     else "the left" if v[1] > 0.7 else "the right")
    name = base["name"] + (", hand turned %+.0f deg" % yaw_deg if yaw_deg else "")
    return {"idx": base["idx"], "u": u, "joint": best, "tilt": k, "s": float(J0[3 + k, best]), "sgn": sgn,
            "slopes": slopes, "allowed": allowed, "channels": list(allowed), "name": name,
            "plus": lab(u), "minus": lab(-u), "q0": q0}


def kin_position(calib, c, d_all):
    """Ball position along the rail (m from C) from the torque CHANGES vs the empty paddle (all 7 joints),
    with no map: physics says d_j = s * m g (Jz_j(C) + slope_j/(m g) * p) for every sensing joint, s ~ 0.9
    (how much of the weight the joint shows). With two joints the scale s cancels: least squares on
    [s, s p]. With one joint s is taken as 0.88. Used to show the ball live while the map is recorded."""
    g = geometry(calib, c["axis"], c.get("hand_yaw_deg", 0.0))
    ch = list(g["allowed"])
    if not ch:
        return None
    A = calib.ball_mass * kin.G * fk(g["q0"], _T_point(c["axis"], 0.0))[1][2, ch]
    B = g["slopes"][ch]
    d = np.asarray(d_all, dtype=float)[ch]
    # each joint on its own with s = 0.88 (they agree within ~10 mm on Nov 21; the 2-joint ratio is
    # ill-conditioned at the turned pose). No ball: the change is far below a ball's weight.
    if np.max(np.abs(d)) < 0.3 * np.max(np.abs(A)):
        return None
    return float(np.mean((d / 0.88 - A) / B))


def cfg_of(calib):
    c = dict(DEFAULT_CFG)
    saved = dict(calib.rail_cfg or {})
    if saved.get("cfg_version", 1) < 2:                # older settings: slow command filter, no PD gains
        for k in ("wc", "rate_max_dps", "tilt_max_deg", "cfg_version"):
            saved.pop(k, None)
    if saved.get("cfg_version", 1) < 3 and "smooth" in saved:   # the smoothing preset becomes kf_accel
        saved["kf_accel"] = SMOOTH.get(saved["smooth"], 0.05)
    saved.pop("cfg_version", None)
    c.update(saved)
    if c["axis"] not in AXES:
        c["axis"] = "y"
    if c.get("smooth") not in SMOOTH:
        c["smooth"] = "medium"
    if c.get("hand_yaw_deg") not in (0.0, -90.0, 90.0):
        c["hand_yaw_deg"] = 0.0
    ax = geometry(calib, c["axis"], c["hand_yaw_deg"])
    ch = [j for j in (c["channels"] or []) if j in ax["allowed"]]
    c["channels"] = ch or list(ax["channels"])
    return c


def set_cfg(calib, new):
    c = cfg_of(calib)
    yaw = float(new["hand_yaw_deg"]) if new.get("hand_yaw_deg") not in (None, "") else c.get("hand_yaw_deg", 0.0)
    if (new.get("axis") in AXES and new["axis"] != c["axis"]) or yaw != c.get("hand_yaw_deg", 0.0):
        c["axis"] = new.get("axis") if new.get("axis") in AXES else c["axis"]
        c["hand_yaw_deg"], c["channels"] = yaw, None      # new geometry: its default sensing joints
        calib.railmap = calib.railmap_raw = calib.railtare_W = calib.railtare_sigma = calib.railtare_cfg = None
    elif "channels" in new:
        c["channels"] = sorted({int(j) for j in new["channels"]})
    for k, (lo, hi) in LIMITS.items():
        if new.get(k) not in (None, ""):
            c[k] = float(np.clip(float(new[k]), lo, hi))
    if new.get("smooth") in SMOOTH:
        c["smooth"] = new["smooth"]
        c["kf_accel"] = SMOOTH[new["smooth"]]
    if new.get("ctrl") in ("pd", "lqr"):
        c["ctrl"] = new["ctrl"]
    for k in BOOLS:
        if k in new and new[k] is not None:
            c[k] = bool(new[k])
    c["cfg_version"] = DEFAULT_CFG["cfg_version"]
    calib.rail_cfg = c
    return cfg_of(calib)


def points_of(calib):
    return list(calib.rail_points or DEFAULT_POINTS)


def _sweep_key(c, calib):
    """The settings the no-ball model depends on."""
    return {"axis": c["axis"], "hand_yaw_deg": c.get("hand_yaw_deg", 0.0), "rate_hz": c["rate_hz"],
            "wrist_scale": calib.wrist_scale, "channels": list(c["channels"]),
            "feat": RailFeatures.VERSION, "accel_max_dps2": c["accel_max_dps2"]}


def _ball_xy(axis, p):
    """Ball position on the paddle (m from C), for the simulator stand-in."""
    return (p, 0.0) if axis == "x" else (0.0, p)


def _T_point(axis, p):
    """Paddle frame moved to the point p (m from C) along the rail."""
    T = kin.paddle_transform()
    off = np.zeros(4)
    off[AXES[axis]["idx"]] = p
    T = T.copy()
    T[:3, 3] = T[:3, 3] + T[:3, :3] @ off[:3]
    return T


def expected_slopes(calib, axis, channels, yaw_deg=0.0):
    """d tau_j / d p (Nm per m along the rail) the kinematics predict at the rail pose: the measured
    torque is +m*g*J_z,j at the contact point (sign checked against the Nov 20 logs)."""
    return geometry(calib, axis, yaw_deg)["slopes"][list(channels)]


def combine(rmap, d, sigma):
    """Ball position along the rail (m from C) from the torque changes d on the map channels:
    inverse-variance weighted least squares. Returns position and its sigma."""
    a, b = np.array(rmap["a"]), np.array(rmap["b"])
    w = 1.0 / np.maximum(np.asarray(sigma, dtype=float), 1e-4) ** 2
    info = np.sum(w * b * b)
    return float(np.sum(w * b * (np.asarray(d) - a)) / info), float(1.0 / np.sqrt(info))


def make_runner(io, calib, ctx, kind, tare=None):
    _need_pose(calib)
    c = cfg_of(calib)
    ax = geometry(calib, c["axis"], c["hand_yaw_deg"])
    dt = 1.0 / c["rate_hz"]
    k, d = calib.gains
    j = ax["joint"]
    q0 = ax["q0"]
    q0s = q0.copy()
    q0s[j] += np.radians(c.get("level_trim_deg", 0.0)) / ax["s"]          # level, as set with the phone
    servo = JointTiltServo(q0s, calib.T_paddle, dt, axis=c["axis"], joint=j, tilt_index=ax["tilt"],
                           tilt_max=np.radians(c["tilt_max_deg"]),
                           wc=c["wc"], rate_max=np.radians(c["rate_max_dps"]), lead=d[j] / k[j],
                           accel_max=np.radians(c["accel_max_dps2"]))
    log = os.path.join(LOG_DIR, f"{kind}_{c['axis']}_{time.strftime('%m%d_%H%M%S')}.npz")
    run = Runner(io, calib, ctx, log, rate_hz=c["rate_hz"], servo=servo, tare=tare,
                 features=RailFeatures(q0, dt, j, c["channels"]), q0=q0)
    run.rcfg, run.ax = c, ax
    return run


def _tilt_target(run, t):
    out = np.zeros(2)
    out[run.ax["tilt"]] = t
    return out


def _record(run, seconds, settle=1.0):
    """Paddle level and streaming: mean and std of the joint torques."""
    taus = []
    for i in range(int((settle + seconds) / run.dt)):
        q, _, tau, _, tilt, _ = run.cycle([0.0, 0.0])
        if i * run.dt >= settle:
            taus.append(tau)
        run.ctx.live(tau=tau, q=q, tilt=tilt, tiltcmd=run.servo.cmd)
    taus = np.array(taus)
    return taus.mean(axis=0), taus.std(axis=0)


# ------------------------------------------------------------------------------- track test
def job_rail_track(io, calib, ctx, seconds=20.0, amp_deg=1.5):
    """No ball. The tilting joint follows a slow two-sine tilt alone. Reports gain and lag."""
    run = make_runner(io, calib, ctx, "railtrack")
    c, k = run.rcfg, run.ax["tilt"]
    amp = np.radians(min(amp_deg, 0.9 * c["tilt_max_deg"]))
    cmd, meas = [], []
    run.start()
    try:
        run.prompt(f"Rail track test ({run.ax['name']}): no ball needed, hands off. Joint {run.ax['joint'] + 1} rocks "
                   f"for {seconds:.0f} s (max {np.degrees(amp):.1f} deg). Continue.", ball=None)
        ctx.status("rail track test")
        n = int(seconds / run.dt)
        for i in range(n + int(1.5 / run.dt)):
            t = i * run.dt
            ramp = min(1.0, t / 2.0, max(0.0, (seconds - t) / 2.0))
            tt = amp * ramp * (0.6 * np.sin(2 * np.pi * 0.2 * t) + 0.4 * np.sin(2 * np.pi * 0.6 * t + 1.0)) if i < n else 0.0
            q, _, tau, _, tilt, _ = run.cycle(_tilt_target(run, tt))
            ctx.live(tau=tau, q=q, tilt=tilt, tiltcmd=run.servo.cmd)
            if 1.0 < t < seconds - 1.0:
                cmd.append(run.servo.cmd[k])
                meas.append(tilt[k])
    except SafetyStop as e:
        ctx.log(f"RAIL TRACK TEST STOPPED: {e}")
    finally:
        run.stop()
    if len(cmd) < 50:
        ctx.log("rail track test too short to judge")
        return
    c_, m_ = np.array(cmd), np.array(meas)
    best = max(range(0, 31), key=lambda s: np.corrcoef(c_[:len(c_) - s], m_[s:])[0, 1])
    corr = float(np.corrcoef(c_[:len(c_) - best], m_[best:])[0, 1])
    gain = float(np.polyfit(c_[:len(c_) - best], m_[best:], 1)[0])
    err_rms = float(np.degrees(np.sqrt(np.mean((m_ - c_) ** 2))))
    ok = 0.85 <= gain <= 1.15 and corr > 0.97 and best * run.dt <= 0.12
    ctx.log(f"rail track {run.ax['name']}, joint {run.ax['joint'] + 1}: gain {gain:.2f} (0.85-1.15), correlation "
            f"{corr:.3f} (>0.97), lag {best * run.dt * 1000:.0f} ms (<120), tilt error rms {err_rms:.2f} deg => "
            + ("PASS" if ok else "FAIL"))
    return {"gain": gain, "corr": corr, "lag_s": best * run.dt, "err_rms_deg": err_rms}


# ------------------------------------------------------------------------------- rail map
def job_rail_map(io, calib, ctx, seconds=3.0):
    """Streaming at the level pose (the same controller as balancing): empty paddle, the ball
    held still at each rail point, empty again. Then fits the 1-D map."""
    pts = points_of(calib)
    if len(pts) < 2:
        raise SafetyStop("the rail map needs at least 2 points (3 to 5 recommended)")
    run = make_runner(io, calib, ctx, "railmap")
    c, ax = run.rcfg, run.ax
    raw = {"empty": None, "empty_std": None, "empty_end": None, "points": [], "t": time.time(),
           "axis": c["axis"], "hand_yaw_deg": c["hand_yaw_deg"], "cfg": _sweep_key(c, calib)}
    ch = c["channels"]
    pts = sorted(pts, key=lambda v: (abs(v) > 1e-9, v))   # C first: the live display is relative to it
    show = {"e0": None, "c0": 0.0, "avg": None}

    def live_ball(tau, q, tilt):
        """Physics position of the ball (no map needed), relative to where C read: drawn on the page."""
        p_ = None
        if show["e0"] is not None:
            k_ = kin_position(calib, c, np.asarray(tau) - show["e0"])
            if k_ is not None:
                p_ = k_ - show["c0"]
                show["avg"] = p_ if show["avg"] is None else show["avg"] + 0.15 * (p_ - show["avg"])
            else:
                show["avg"] = None
        a_ = show["avg"]
        ctx.live(tau=tau, q=q, tilt=tilt, tiltcmd=run.servo.cmd,
                 est=None if a_ is None else (a_, 0.0), meas=None if p_ is None else (p_, 0.0))

    def prompt_live(msg, ball=None, hold=False):
        run.io.place_ball(ball, hold=hold)
        ctx.ask(msg)
        while not ctx.answered():
            q_, _, tau_, _, tilt_, _ = run.cycle([0.0, 0.0])
            live_ball(tau_, q_, tilt_)
            run._renew_stream()

    def record(seconds_):
        taus = []
        for i_ in range(int((1.0 + seconds_) / run.dt)):
            q_, _, tau_, _, tilt_, _ = run.cycle([0.0, 0.0])
            if i_ * run.dt >= 1.0:
                taus.append(tau_)
            live_ball(tau_, q_, tilt_)
        taus = np.array(taus)
        return taus.mean(axis=0), taus.std(axis=0)

    run.start()
    try:
        run.prompt("Rail map: REMOVE the ball, hands off, then Continue.", ball=None)
        ctx.status("rail map: empty paddle")
        m, s = _record(run, seconds)
        raw["empty"], raw["empty_std"] = m.tolist(), s.tolist()
        show["e0"] = m
        for n, p in enumerate(pts):
            side = ax["plus"] if p > 0 else ax["minus"] if p < 0 else "centre"
            for attempt in range(3):
                again = "" if attempt == 0 else "That reading was UNSTEADY (ball or hands moving). Again: "
                ctx.status(f"rail map: point {n + 1}/{len(pts)}, target {p:+.0f} mm")
                prompt_live(f"{again}Point {n + 1}/{len(pts)}: ball in the rail at {p:+.0f} mm from C ({side}). Watch the "
                            "ball on the deck drawing: it should sit on the mark. Hold it still (putty, a washer or a stick "
                            "against it), HANDS OFF the paddle and arm, then Continue.",
                            ball=_ball_xy(c["axis"], p / 1000.0), hold=True)
                pm, ps = record(seconds)
                if abs(p) < 1e-9:                                 # C recorded: the display is relative to it
                    k_ = kin_position(calib, c, pm - m)
                    show["c0"] = k_ if k_ is not None else 0.0
                k_ = kin_position(calib, c, pm - m)
                where = "" if k_ is None else f" (physics: {1000 * (k_ - show['c0']):+.0f} mm from C)"
                noisy = [JOINT_NAMES[j] for j in ch if ps[j] > 2.0 * max(s[j], 0.005)]
                dd = pm - m
                ctx.log(f"rail point {p:+.0f} mm{where}: change vs empty " + ", ".join(f"{JOINT_NAMES[j]} {dd[j] * 1000:+.0f} mNm" for j in ch)
                        + (f"  UNSTEADY on {', '.join(noisy)}" if noisy else ""))
                if not noisy:
                    break
            raw["points"].append({"p": float(p), "mean": pm.tolist(), "std": ps.tolist(), "steady": not noisy})
        run.prompt("Rail map: REMOVE the ball again (drift check), hands off, then Continue.", ball=None)
        ctx.status("rail map: empty again")
        m2, _ = _record(run, seconds)
        raw["empty_end"] = m2.tolist()
    finally:
        run.stop()
        if raw["points"]:
            calib.railmap_raw = raw
            calib.save()
    return rail_fit(calib, ctx)


def rail_fit(calib, ctx=None):
    raw = calib.railmap_raw
    if not raw or len(raw.get("points", [])) < 2:
        raise SafetyStop("record the rail map first (needs at least 2 points)")
    c = cfg_of(calib)
    if raw.get("axis", c["axis"]) != c["axis"] or raw.get("hand_yaw_deg", 0.0) != c["hand_yaw_deg"]:
        raise SafetyStop("the recorded rail map is for another axis or hand pose: record it again")
    ch = list(c["channels"])
    e0 = np.array(raw["empty"])
    P = np.array([pt["p"] / 1000.0 for pt in raw["points"]])                  # m from C
    D = np.array([np.array(pt["mean"])[ch] - e0[ch] for pt in raw["points"]])  # (points, channels)
    coef, *_ = np.linalg.lstsq(np.c_[np.ones(len(P)), P], D, rcond=None)
    a, b = coef[0], coef[1]
    sig = np.maximum(np.array(raw["empty_std"])[ch], 0.003)
    b_exp = expected_slopes(calib, c["axis"], ch, c["hand_yaw_deg"])
    ratio = b / b_exp
    notes = []
    good = [i for i in range(len(ch)) if 0.6 <= ratio[i] <= 1.3]
    for i, j in enumerate(ch):
        if i not in good:
            notes.append(f"{JOINT_NAMES[j]} slope is {ratio[i] * 100:.0f} % of the kinematic one"
                         + (" (wrong sign: check the point signs, + is toward " + geometry(calib, c['axis'], c['hand_yaw_deg'])['plus'] + ")" if ratio[i] < 0 else "")
                         + ("; left out of the map" if 0 < len(good) < len(ch) else ""))
    if 0 < len(good) < len(ch):                       # keep only the joints that see the ball properly
        ch, a, b, sig, D = [ch[i] for i in good], a[good], b[good], sig[good], D[:, good]
        b_exp, ratio = b_exp[good], ratio[good]
    fit = {"axis": c["axis"], "hand_yaw_deg": c["hand_yaw_deg"], "channels": ch, "a": a.tolist(), "b": b.tolist(), "b_expected": b_exp.tolist(),
           "ratio": ratio.tolist(), "sigma": sig.tolist(), "p_mm": (P * 1000).tolist()}
    est = np.array([combine(fit, d, sig)[0] for d in D])
    err = (est - P) * 1000
    fit["err_mm"] = err.tolist()
    fit["rms_mm"] = float(np.sqrt(np.mean(err ** 2)))
    fit["noise_mm"] = combine(fit, a, sig)[1] * 1000
    if raw.get("empty_end") is not None:
        drift = np.array(raw["empty_end"])[ch] - e0[ch]
        fit["drift_mm"] = abs(combine(fit, a + drift, sig)[0]) * 1000
    if len(P) == 2:
        notes.append("only 2 points: a bad point cannot show up in the error; record 3 or more")
    if fit["rms_mm"] < 6 and fit.get("drift_mm", 0.0) < 8 and good:
        fit["verdict"] = "PASS"
    elif fit["rms_mm"] < 12 and good:
        fit["verdict"] = "MARGINAL"
    else:
        fit["verdict"] = "FAIL"
    fit["notes"] = notes
    calib.railmap = fit
    calib.save()
    if ctx is not None:
        ctx.log(f"rail map {geometry(calib, c['axis'], c['hand_yaw_deg'])['name']}: slopes " + ", ".join(
            f"{JOINT_NAMES[j]} {b[i] * 10:+.1f} mNm/cm ({ratio[i] * 100:.0f} % of kinematic {b_exp[i] * 10:+.1f})"
            for i, j in enumerate(ch))
            + f"; point error rms {fit['rms_mm']:.1f} mm (" + ", ".join(f"{e:+.0f}" for e in err) + ")"
            + (f", drift {fit['drift_mm']:.1f} mm" if "drift_mm" in fit else "")
            + f", noise {fit['noise_mm']:.1f} mm per sample => {fit['verdict']}")
        for n in notes:
            ctx.log("  note: " + n)
    return fit


def _fit_sweep(calib, ctx, Phi, Y, c):
    tare = Tare.fit(np.asarray(Phi), np.asarray(Y))
    calib.railtare_W, calib.railtare_sigma = tare.W, tare.sigma
    calib.railtare_cfg = _sweep_key(c, calib)
    ch = calib.railmap["channels"]
    _, sp = combine(calib.railmap, np.zeros(len(ch)), tare.sigma[ch])
    verdict = "PASS" if sp < 0.02 else "MARGINAL" if sp < 0.04 else "FAIL"
    calib.railmap["moving_noise_mm"] = sp * 1000
    calib.railmap["moving_verdict"] = verdict
    calib.save()
    ctx.log("rail tilt sweep: leftover torque while rocking " + ", ".join(
        f"{JOINT_NAMES[j]} {tare.sigma[j] * 1000:.0f} mNm" for j in ch)
            + f" = ball noise {sp * 1000:.0f} mm per sample => {verdict}")
    return tare


def refit_sweep_from_log(calib, ctx):
    """Refit the no-ball model from the newest rail sweep log of this axis (it holds q, dq and
    all joint torques), e.g. after changing the sensing joints. No need to rock again."""
    import glob
    c = cfg_of(calib)
    if calib.railmap is None:
        raise SafetyStop("record the rail map first")
    files = sorted(glob.glob(os.path.join(LOG_DIR, f"railsweep_{c['axis']}_*.npz")), key=os.path.getmtime)
    files = [f for f in files if os.path.getmtime(f) >= (calib.railmap_raw or {}).get("t", 0) - 3600]
    if not files:
        raise SafetyStop("no rail sweep log for this axis: run the rail tilt sweep (C)")
    d = np.load(files[-1])
    gx = geometry(calib, c["axis"], c["hand_yaw_deg"])
    tc, t = d["tilt_cmd"][:, gx["tilt"]], d["t"]
    moving = np.nonzero(np.abs(tc) > np.radians(0.05))[0]
    if len(moving) < 500:
        raise SafetyStop("the last rail sweep log has too little motion: run the rail tilt sweep (C)")
    dt = float(np.median(np.diff(t)))
    if abs(dt - 1.0 / c["rate_hz"]) > 0.2 / c["rate_hz"]:
        raise SafetyStop("the last rail sweep ran at another rate: run the rail tilt sweep (C)")
    feat = RailFeatures(gx["q0"], dt, gx["joint"], c["channels"])
    vel, Phi = [], []
    for qq, vv in zip(d["q"], d["dq"]):
        vel = ([vv] + vel)[:3]
        while len(vel) < 3:
            vel.append(vel[-1])
        Phi.append(feat(qq, vel))
    i0, i1 = moving[0] + int(1.0 / dt), moving[-1]
    ctx.log(f"refitting the no-ball model from {os.path.basename(files[-1])} for the sensing joints "
            + ", ".join(JOINT_NAMES[j] for j in c["channels"]))
    return _fit_sweep(calib, ctx, np.array(Phi)[i0:i1], d["tau"][i0:i1], c)


def sync(calib, ctx):
    """Bring the map and the no-ball model in line with the current sensing joints, from the
    recordings already made (the map keeps all joint torques; the sweep log too)."""
    c = cfg_of(calib)
    if calib.railmap_raw and calib.railmap_raw.get("axis", c["axis"]) == c["axis"] \
            and calib.railmap_raw.get("hand_yaw_deg", 0.0) == c["hand_yaw_deg"] \
            and (calib.railmap is None or calib.railmap.get("channels") != list(c["channels"])):
        ctx.log("sensing joints changed: refitting the rail map from its recordings")
        rail_fit(calib, ctx)
    swept = calib.railtare_cfg or {}
    if calib.railmap is not None and (swept.get("channels") != list(c["channels"])
                                      or swept.get("feat") != RailFeatures.VERSION):
        try:
            refit_sweep_from_log(calib, ctx)
        except SafetyStop as e:
            ctx.log(f"could not refit the sweep: {e}")


# ------------------------------------------------------------------------------- tilt sweep
def job_rail_sweep(io, calib, ctx, seconds=40.0):
    """No ball. The tilting joint rocks like the balancer will; fits the no-ball model of all
    joint torques on that joint's angle, speed and acceleration."""
    sync(calib, ctx)
    if calib.railmap is None:
        raise SafetyStop("record the rail map first")
    run = make_runner(io, calib, ctx, "railsweep")
    c, k = run.rcfg, run.ax["tilt"]
    exc = Excitation(0.8 * np.radians(c["tilt_max_deg"]), seconds)
    Phi, Y = [], []
    run.start()
    try:
        run.prompt(f"Rail tilt sweep: REMOVE the ball, hands off, then Continue. Joint {run.ax['joint'] + 1} rocks "
                   f"for {seconds:.0f} s.", ball=None)
        n = int(seconds / run.dt)
        for i in range(n + int(2.0 / run.dt)):
            tt = exc(i, run.dt)[k] if i < n else 0.0
            q, _, tau, phi, tilt, _ = run.cycle(_tilt_target(run, tt))
            if i % 10 == 0:
                ctx.status(f"rail tilt sweep {i * run.dt:4.0f} / {seconds:.0f} s")
            ctx.live(tau=tau, q=q, tilt=tilt, tiltcmd=run.servo.cmd)
            if i > int(3.0 / run.dt):
                Phi.append(phi)
                Y.append(tau)
    finally:
        run.stop()
    if len(Y) < 200:
        raise SafetyStop("rail tilt sweep too short")
    _fit_sweep(calib, ctx, Phi, Y, c)


# ------------------------------------------------------------------------------- balance
def job_rail_probe(io, calib, ctx, seconds=40.0, anchor_mm=0.0, anchor=True):
    """No balancing: the paddle streams LEVEL with the ball in the rail, and the page shows the
    same estimate the balancer would use. Move the ball slowly with a stick (or let it sit at
    marks) and check the estimate follows. Separates the sensing from the control loop."""
    return job_rail_balance(io, calib, ctx, seconds=seconds, anchor_mm=anchor_mm, anchor=anchor, probe=True)


def job_rail_balance(io, calib, ctx, seconds=60.0, gain=None, goal_mm=None, anchor_mm=0.0, anchor=True, probe=False):
    """Keep the ball (in a rail) at the goal along the rail, tilting with one joint.

    anchor: the operator puts the ball at anchor_mm (a mark on the rail) before the start; two
    seconds of readings with the paddle level fix the map offset there. During the run the
    page can send "the ball is at p now" (ctx.anchor_request), and new tuning (ctx.tune_version
    changes: everything in TUNABLE is re-read from the settings, about every 0.1 s)."""
    sync(calib, ctx)
    if goal_mm is not None:
        set_cfg(calib, {"goal_mm": goal_mm})
    if calib.railmap is None or calib.railtare_W is None:
        raise SafetyStop("do the rail map and the rail tilt sweep first")
    c = cfg_of(calib)
    if calib.railmap.get("axis", c["axis"]) != c["axis"]:
        raise SafetyStop("the rail map is for the other axis: record it again")
    swept = calib.railtare_cfg or {}
    if swept.get("axis") != c["axis"] or swept.get("channels") != list(c["channels"]):
        raise SafetyStop("the rail tilt sweep was done for another axis or other sensing joints: run the sweep (C) again")
    if swept != _sweep_key(c, calib):
        ctx.log("warning: rate or stiffness changed since the rail tilt sweep; redo the sweep if it balances badly")
    tare = Tare(np.array(calib.railtare_W, dtype=float).copy(), np.array(calib.railtare_sigma, dtype=float).copy())
    run = make_runner(io, calib, ctx, "railprobe" if probe else "railbalance", tare=tare)
    ax, rm = run.ax, calib.railmap
    k, sgn, u = ax["tilt"], ax["sgn"], ax["u"]
    ch = rm["channels"]
    sig_ch = np.array(calib.railtare_sigma)[ch]
    _, sp = combine(rm, np.zeros(len(ch)), sig_ch)
    kf = AxisKF(run.dt, accel_noise=c["kf_accel"], dist_noise=c["kf_bias"])
    kf.reset()
    # the ball centre cannot go past the rail ends. A rail can run past "ball fully on the deck" (79 mm): on Nov 21 the
    # points were mapped at +-100 mm and the pinned ball read -110..-130 while the estimate sat clamped at -79
    lim = max(kin.PLATE_HALF - kin.BALL_RADIUS, max(abs(float(v)) for v in points_of(calib)) / 1000.0 + 0.010)
    lever = float(u @ fk(ax["q0"], calib.T_paddle)[1][:3, ax["joint"]]) / run.servo.s   # deck along the rail per rad of tilt
    m = calib.ball_mass
    T0, _ = fk(ax["q0"], calib.T_paddle)
    e_rail = np.eye(3)[ax["idx"]]
    far = lim + 0.05                                    # a raw reading this far out for 1 s: the ball is not where we think
    tune_version = getattr(ctx, "tune_version", lambda: 0)

    # ---- live tuning: everything in TUNABLE, re-read when the page changes it
    T = {}

    def load_tuning(first=False):
        new = {key: cfg_of(calib)[key] for key in TUNABLE}
        changed = {key: v for key, v in new.items() if T.get(key) != v}
        if not changed:
            return
        T.update(new)
        run.servo.tilt_max = np.radians(T["tilt_max_deg"])
        run.servo.rate_max = np.radians(T["rate_max_dps"])
        run.servo.wc = T["wc"]
        run.servo.q0[ax["joint"]] = ax["q0"][ax["joint"]] + np.radians(T["level_trim_deg"]) / ax["s"]
        run.servo.accel_max = np.radians(T["accel_max_dps2"])
        kf.set_noise(T["kf_accel"], T["kf_bias"])
        T["_ra"] = ROLL_ACCEL * float(T.get("roll_scale", 1.0))      # the rolling model, scaled by the roll test
        T["_bias_max"] = T["_ra"] * np.radians(T["kf_bias_max_deg"])
        if first:
            kf.P[2, 2] = max(T["_bias_max"], 1e-4) ** 2
        if T["ctrl"] == "lqr":
            T["_K"] = lqr_gain(run.dt, T["wc"], T["lqr_gain"], T["lqr_qv"])
        # how much tilt the estimate's own noise asks for (speed estimate noise is about 10 mm/s)
        noise_tilt = abs(T["_K"][1]) * 0.01 if T["ctrl"] == "lqr" else T["kv"] * 0.01 / T["_ra"]
        T["_noise_tilt"] = noise_tilt
        if noise_tilt > 0.5 * np.radians(T["tilt_max_deg"]):
            ctx.log(f"note: with these settings the speed-estimate noise alone asks for about "
                    f"{np.degrees(noise_tilt):.1f} deg of tilt (limit {T['tilt_max_deg']:.1f}): expect the paddle to slam "
                    "between the limits. Lower " + ("the LQR gain / speed weight" if T["ctrl"] == "lqr" else "Kv")
                    + " (recommended LQR: gain 0.6, speed weight 0.1)")
        if not first:
            ctx.log("live tuning: " + ", ".join(f"{key} {v}" for key, v in changed.items()))
            if "balance_stops" in changed:
                ctx.log("balance stops " + ("ON" if T["balance_stops"] else "OFF: no saturation or off-rail stop "
                        "(tilt limit, joint window, contact and frozen-robot stops stay). Partner on the e-stop"))
            run._rec(tune_t=run.t)

    load_tuning(first=True)
    if not T["balance_stops"]:
        ctx.log("balance stops OFF: no saturation or off-rail stop (tilt limit, joint window, contact and "
                "frozen-robot stops stay). Partner on the e-stop")
    sat = []                                            # was the tilt request at the limit, last 3 s
    run.start()
    lost = 0
    try:
        run.prompt("Rail balance: REMOVE the ball, hands off, then Continue (2 s re-zero).", ball=None)
        ctx.status("re-zero: empty paddle")
        run.tare.rezero(run.hold(2.0))
        anchor_p = float(anchor_mm) / 1000.0
        if anchor:
            run.prompt(f"Put the ball IN THE RAIL at the {anchor_mm:+.0f} mm mark (C = 0), let go, and press Continue once it "
                       "is STILL: it reads the ball there for 2 s to fix the map offset.", ball=_ball_xy(c["axis"], anchor_p))
        else:
            run.prompt("Put the ball IN THE RAIL near the centre (C), let go, then Continue.",
                       ball=_ball_xy(c["axis"], 0.01))

        def measure(q, tau, phi, J, ball_acc):
            d = (tau - run.tare.predict(phi))[ch]
            d = d - m * ball_acc * (u @ J[:3, ch])      # the ball pushes back along the rail while it accelerates
            pb, _ = combine(rm, d, sig_ch)              # base position of the contact point, from C at the level pose
            Tq, _ = fk(q, calib.T_paddle)
            p = (pb - u @ (Tq[:3, 3] - T0[:3, 3])) / max(float(u @ Tq[:3, :3] @ e_rail), 0.5)   # tilting moves the deck
            return p, d

        offset = 0.0
        if anchor:
            ctx.status("anchor: reading the ball at the mark")
            ps_ = []
            for k_ in range(int(2.5 / run.dt)):
                q, _, tau, phi, tilt, J = run.cycle([0.0, 0.0])
                if k_ * run.dt >= 0.5:                  # let the ball and the filters settle first
                    ps_.append(measure(q, tau, phi, J, 0.0)[0])
                ctx.live(tau=tau, q=q, tilt=tilt, tiltcmd=run.servo.cmd)
            nq = max(len(ps_) // 4, 1)
            moved = float(np.mean(ps_[-nq:]) - np.mean(ps_[:nq]))
            if abs(moved) > 0.015:                      # the ball rolled during the anchor: do not trust it
                anchor = False
                ctx.log(f"anchor NOT applied: the reading moved {moved * 1000:+.0f} mm during it (ball rolling?). "
                        "Click the rail drawing where the ball is to correct it.")
            else:
                offset = anchor_p - float(np.mean(ps_))
                ctx.log(f"anchor: the map read {np.mean(ps_) * 1000:+.0f} mm with the ball at {anchor_mm:+.0f} mm: "
                        f"offset {offset * 1000:+.0f} mm applied" + ("  (large: redo the rail map)" if abs(offset) > 0.04 else ""))
        q, _, tau, phi, tilt, J = run.cycle([0.0, 0.0])
        # ball presence: joint 4 carries the ball's weight (+1.0 Nm on Nov 20-21). Reference levels:
        # empty from the re-zero hold, ball on from now (the ball sits at the anchor mark).
        pres_j = 3
        tau_on = float(np.mean([run.cycle([0.0, 0.0])[2][pres_j] for _ in range(20)]))
        tau_empty = float(run.tare.predict(phi)[pres_j])
        pres_step = tau_on - tau_empty
        pres_ok = abs(pres_step) > 0.3
        # joint 4 is a SENSING joint at the turned-hand pose: its torque depends on where the ball is, so "joint 4
        # back to empty" fired when the ball rolled to one end (Nov 21 19:16). With 2+ sensing joints use the
        # ball's WEIGHT instead: d = w*(a + b*p) on every sensing joint -> solve for w (1 = ball on, 0 = off)
        AB = np.c_[np.array(rm["a"]), np.array(rm["b"])] if len(ch) >= 2 else None
        pres_by_weight = AB is not None and pres_j in ch
        if pres_by_weight:
            pres_ok = True
            ctx.log("ball presence: from the ball's weight on " + " + ".join(JOINT_NAMES[j] for j in ch)
                    + " (position-independent)")
        w_avg, d_avg = 1.0, None
        if not pres_ok:
            ctx.log(f"ball presence check off: joint {pres_j + 1} changes only {pres_step:+.2f} Nm with the ball")
        pres_avg, gone_t, drops = tau_on, 0.0, 0
        p, _ = measure(q, tau, phi, J, 0.0)
        kf.x[:] = [anchor_p if anchor else float(np.clip(p + offset, -lim, lim)), 0.0, 0.0]
        # confident start: the ball was placed STILL at the mark. A wide-open start let the first noisy
        # readings throw the speed to 187 mm/s and the offset to its 2 deg cap within 0.5 s (Nov 21 16:28),
        # and the controller flipped +-2.5 deg: the ball dropped 3-4 s after every start.
        START_P = np.diag([0.01 ** 2, 0.005 ** 2, (ROLL_ACCEL * np.radians(0.3)) ** 2])
        kf.P[:] = START_P
        kf.P[0, 0] = 0.01 ** 2 if anchor else 0.03 ** 2
        t_start = 0.0                                   # soft start: controller authority ramps over 2 s
        r_ema = sp ** 2                                 # measurement noise, learned while running
        r = sp ** 2
        contact_t = 0.0
        raw_avg, raw_hist = [], []
        dcmd_prev = run.servo.dcmd[k]
        show = []                                       # 0.25 s average of the raw reading, for the page only
        ask_anchor = getattr(ctx, "anchor_request", lambda: None)
        target = np.zeros(2)
        e_prev = sgn * tilt[k]
        seen_version = None                             # None: re-read the settings on the first step, so changes
                                                        # made while a start-up prompt was waiting are applied
        pinned_t, resync_t, resyncs = 0.0, 0.0, 0
        gate_t, dcmd_prev_g, gated = 0.0, run.servo.dcmd[k], False
        spread_avg, disagree_t, last_disagree_log, p_each_avg = 0.0, 0.0, -1e9, None
        outlier_t, outliers, gate_cool = 0.0, 0, 0.0
        gap_base, gap_t0 = None, None
        integ = 0.0                                     # integral of the position error (m s)
        still_t = 0.0
        ctx.status(f"probe: paddle level, move the ball slowly with a stick" if probe else f"balancing along {ax['name']}")
        for i in range(int(seconds / run.dt)):
            if i % 10 == 0 and tune_version() != seen_version:
                seen_version = None                             # None: re-read the settings on the first step, so changes
                                                        # made while a start-up prompt was waiting are applied
                load_tuning()
            goal, tmax, lead = T["goal_mm"] / 1000.0, np.radians(T["tilt_max_deg"]), T["lead_s"]
            q, _, tau, phi, tilt, J = run.cycle(target)
            e = sgn * tilt[k]                           # tilt that accelerates the ball toward +p
            # the joint swings the deck sideways (axis ~22 cm above it): a rolling ball gets 5/7 of
            # that push the other way, relative to the deck. Commanded tilt acceleration: smooth, known.
            a_deck = lever * (run.servo.dcmd[k] - dcmd_prev) / run.dt
            dcmd_prev = run.servo.dcmd[k]
            a_rel = T["_ra"] * np.sin(e) + kf.x[2] - 5.0 / 7.0 * a_deck
            p_meas, d = measure(q, tau, phi, J, a_rel + a_deck)
            p_meas += offset
            req = ask_anchor()
            if req is not None:                         # operator: "the ball is at req now"
                offset += req - kf.x[0]
                p_meas += req - kf.x[0]
                kf.x[0] = req
                kf.P[0, 0] = min(kf.P[0, 0], 0.005 ** 2)
                pinned_t = 0.0
                ctx.log(f"re-anchored: ball set to {req * 1000:+.0f} mm (offset now {offset * 1000:+.0f} mm)")
            bias_before = kf.x[2]
            a_in = T["_ra"] * np.sin(0.5 * (e + e_prev)) - 5.0 / 7.0 * a_deck
            if T["stick_deg"] > 0 and abs(kf.x[1]) < 0.008 and abs(a_in + kf.x[2]) < T["_ra"] * np.sin(np.radians(T["stick_deg"])):
                a_in = -kf.x[2]                         # a still ball on a sticky rail stays put below the breakaway tilt
                kf.x[1] *= 0.9
            kf.predict(a_in)
            cmd_acc = abs(run.servo.dcmd[k] - dcmd_prev_g) / run.dt
            dcmd_prev_g = run.servo.dcmd[k]
            if T["gate_accel_dps2"] > 0 and cmd_acc > np.radians(T["gate_accel_dps2"]):
                gate_t = 0.1                            # arm shaking: trust the model for now and 0.1 s after
            gate_t = max(0.0, gate_t - run.dt)
            gated = gate_t > 0
            if run.fresh and not gated:                 # new robot state: measurement update
                innov = p_meas - kf.x[0]
                r = T["kf_r_scale"] * (max(sp ** 2, r_ema - kf.P[0, 0]) if T["kf_adapt"] else sp ** 2)
                # one-off jumps (a load change springing the arm: -1700 mm on Nov 21 19:40 at 151.6 s) are not the ball:
                # skip them, for at most 0.3 s; a real jump that persists then goes through (and the resync below)
                if abs(innov) > max(0.06, 5.0 * np.sqrt(kf.P[0, 0] + r)) and outlier_t < 0.3 and gate_cool <= 0:
                    outlier_t += run.dt
                    outliers += 1
                else:
                    if outlier_t >= 0.3:
                        gate_cool = 1.0                 # skipped 0.3 s: maybe the ESTIMATE is off. Let readings in for 1 s
                    outlier_t = 0.0
                    r_ema += (0.035 / 1.0) * (innov ** 2 - r_ema)          # about 1 s memory at ~28 Hz
                    kf.update(p_meas, r)
            e_prev = e
            gate_cool = max(0.0, gate_cool - run.dt)
            # physics limits: the paddle is nearly level, the ball stays between the rail ends
            kf.x[2] = float(np.clip(kf.x[2], -T["_bias_max"], T["_bias_max"]))
            if T["end_clamp"] and abs(kf.x[0]) > lim:   # the ball centre cannot pass the rail ends
                kf.x[0] = float(np.sign(kf.x[0]) * lim)
                if kf.x[1] * kf.x[0] > 0:
                    kf.x[1] = 0.0
            # stalled at an end: near an end, tilted to roll inward, yet not moving inward. That is the
            # end stop (or the ball wedged), not a level offset: freeze the offset (it wound up to its
            # 1 deg cap on Nov 20 22:17 and cancelled the counter-angle) and, after 0.7 s, unstick.
            # judged on the RAW reading: the filter trusts the model and "sees" the ball roll inward
            # drift check: with 2+ sensing joints each gives its own position; they must agree
            if len(ch) >= 2:
                p_now = (np.asarray(d) - np.array(rm["a"])) / np.array(rm["b"])
                p_each = p_now if p_each_avg is None else p_each_avg + (run.dt / 0.5) * (p_now - p_each_avg)
                p_each_avg = p_each                       # smooth each joint first: noise averages out, drift stays
                gap = p_each - float(np.mean(p_each))
                # the joints never agree exactly (Nov 21 19:22: a steady -25/+25 mm from the start; the anchor only
                # fixes their average). Take their gap 1 s after each (re)anchor as normal; warn when it CHANGES
                if gap_t0 != t_start and i * run.dt - t_start >= 1.0:
                    gap_base, gap_t0 = gap.copy(), t_start
                spread_avg = float(np.ptp(gap - gap_base)) if gap_base is not None else 0.0
                disagree_t = disagree_t + run.dt if spread_avg > 0.025 else 0.0
                if disagree_t > 2.0 and run.t - last_disagree_log > 10.0:
                    last_disagree_log = run.t
                    ctx.log(f"drift: the gap between the sensing joints changed by {spread_avg * 1000:.0f} mm since the "
                            "anchor (" + ", ".join(f"{JOINT_NAMES[j]} {g * 1000:+.0f}" for j, g in zip(ch, gap - gap_base))
                            + " mm). One of them drifted: re-anchor (click the ball's real spot) or redo the sweep")
            raw_avg = (raw_avg + [p_meas])[-int(0.5 / run.dt):]
            ravg = float(np.median(raw_avg))           # median: a few spikes (ball placed, arm springing) cannot drag it
            raw_hist = (raw_hist + [ravg])[-int(1.0 / run.dt):]
            side = np.sign(ravg)
            moved_in = (raw_hist[0] - ravg) * side            # how far inward the reading went in the last 1 s
            pinned = bool(T["end_clamp"] and len(raw_hist) == int(1.0 / run.dt) and abs(ravg) > lim - 0.025
                          and e * side < -np.radians(0.3) and moved_in < 0.005)
            if pinned:
                kf.x[2] = bias_before
            # the moment the ball comes free (moving inward), end the unstick: decaying pinned_t slowly kept the full
            # unstick tilt for ~0.5 s after release on Nov 21 19:40, the ball reached 240-350 mm/s and shot off the far end
            freed = (kf.x[1] * side < -0.015) or (len(raw_hist) > 30 and (raw_hist[-30] - ravg) * side > 0.008)
            if pinned:
                pinned_t += run.dt
            elif freed:
                pinned_t = 0.0
            else:
                pinned_t = max(0.0, pinned_t - 2 * run.dt)
            # a stuck ball under tilt is the rail's stiction, not a level offset: do not learn from it
            if len(raw_hist) > int(0.5 / run.dt) and abs(ravg - raw_hist[-int(0.5 / run.dt)]) < 0.006:
                kf.x[2] = bias_before
            # resync: the model may say "tilted, so it rolls" while a sticky rail holds the ball. If
            # the estimate stays > 40 mm from the 0.5 s reading for 0.3 s, snap it back to the reading
            # (22:53 run: estimate +107 mm while the reading said -10 mm; LQR chased it to full tilt).
            far_off = (len(raw_hist) > int(0.5 / run.dt) and abs(kf.x[0] - ravg) > 0.04 and not gated
                       and i * run.dt - t_start > 1.0)  # not right after placing: the readings settle first (Nov 21 20:03)
            resync_t = resync_t + run.dt if far_off else 0.0
            if resync_t > 0.3:
                old_x = kf.x[0]
                kf.x[0] = float(np.clip(ravg, -lim, lim))
                kf.x[1] = float(np.clip((ravg - raw_hist[-int(0.5 / run.dt)]) / 0.5, -0.1, 0.1))
                # keep kf.x[2] (the learned lean): wiping it at every resync made the controller relearn the
                # level each time (Nov 21: 3-4 resyncs per run)
                kf.P[:] = np.diag([0.01 ** 2, 0.03 ** 2, max(T["_bias_max"], 1e-4) ** 2])
                resync_t, resyncs = 0.0, resyncs + 1
                if resyncs <= 3 or resyncs % 10 == 0:
                    ctx.log(f"estimate resynced to the reading ({old_x * 1000:+.0f} -> {kf.x[0] * 1000:+.0f} mm, "
                            f"{resyncs} so far): sticky rail or the filter trusting the model too much "
                            "(raise accel noise)")
            st = kf.x.copy()
            bias = st[2] / T["_ra"]
            a_now = T["_ra"] * np.sin(e) + st[2]        # (not the deck push: it follows the command itself)
            # SAFE look-ahead: count the model's acceleration only when it speeds the ball up, never as credit
            # for slowing it down (Nov 21 01:51: "+0.6 deg already brakes it" -> brake asked +0.37 deg; the deck
            # leans and the ball ran off the left end at 75 mm/s).
            if a_now * st[1] < 0:
                a_now = 0.0
            p_ahead = st[0] + st[1] * lead + 0.5 * a_now * lead ** 2
            v_ahead = st[1] + a_now * lead
            if T["ctrl"] == "lqr":
                # LQR on [position, speed, tilt, tilt rate] (tilt through the command filter), on the ball ahead
                v_star = 0.0
                e_t = float(-T["_K"] @ np.array([p_ahead - goal, v_ahead, e + bias, sgn * run.servo.dcmd[k]]) - bias)
            else:
                # cascaded PD with a speed limit:
                #   wanted speed  v* = -kp (p - goal), at most vmax       (P on position)
                #   wanted accel  a* = kv (v* - v) - bias                  (D: brake on speed error)
                v_star = float(np.clip(-T["kp"] * (p_ahead - goal), -T["vmax_mm_s"] / 1000.0, T["vmax_mm_s"] / 1000.0))
                a_star = T["kv"] * (v_star - v_ahead) - st[2]
                e_t = float(np.arcsin(np.clip(a_star / T["_ra"], -1.0, 1.0)))
            # end guard: in the outer zone, never tilt TOWARD that end. On Nov 21 21:03 the ball was coming back
            # from the open front end at 44 mm/s; the speed limit (20 mm/s) braked it by tilting toward the front, the
            # paddle lagged 0.3 s and the ball rolled back over the edge
            # still-ball hold: near the goal and slow, ease the paddle to level (smooth: full control again at twice
            # the hold distance or speed). Noise near the goal then cannot pile tilt up against a ball held by stiction
            if not probe and T["hold_mm"] > 0 and T["hold_mm_s"] > 0 and i * run.dt - t_start > 2.0:
                fp = (abs(st[0] - goal) - T["hold_mm"] / 1000.0) / (T["hold_mm"] / 1000.0)
                fv = (abs(st[1]) - T["hold_mm_s"] / 1000.0) / (T["hold_mm_s"] / 1000.0)
                hold_f = float(np.clip(max(fp, fv), 0.0, 1.0))
                # ease to the LEARNED level (the tilt that cancels the estimated lean), not the bare trim: holding the
                # bare trim let the ball roll out of the hold zone on a smooth rail (fake test, 1-2 drops)
                e_lean = float(np.arcsin(np.clip(-st[2] / T["_ra"], -1.0, 1.0)))
                e_t = e_lean + hold_f * (e_t - e_lean)
            else:
                hold_f = 1.0
            # ...but DO brake once it comes back faster than the max ball speed: with no braking at all the ball came
            # back from an end at 110-160 mm/s, overshot to the other end and the swing grew (Nov 21 21:08)
            end_zone = lim - 0.045
            inward = -st[1] * np.sign(st[0])
            if not probe and abs(st[0]) > end_zone and e_t * np.sign(st[0]) > 0 and inward < T["vmax_mm_s"] / 1000.0:
                e_t = 0.0
            # brake zone: where would the ball stop if we braked at full counter-angle from now on (after the
            # ~0.3 s it takes the tilt to get there)? If that is past the margin and it is heading out: brake.
            braking = kicking = False
            v_now = st[1]
            if not probe and T["brake_zone"] and abs(v_now) > 0.005:
                a_brk = 0.8 * T["_ra"] * np.sin(tmax)
                p_stop = st[0] + v_now * 0.3 + np.sign(v_now) * v_now ** 2 / (2 * a_brk)
                if abs(p_stop) > T["brake_margin_mm"] / 1000.0 and p_stop * v_now > 0:
                    e_t, braking = float(-np.sign(v_now) * tmax), True
            # kick: the ball sits still off the goal (sticky rail): tilt at least kick_deg toward the goal
            # kick, anywhere on the rail: the ball sits still off the goal (sticky rail: it needs ~1.6 deg to
            # break free, more than the controller asks for). Start at kick_deg toward the goal and RAMP UP
            # (unstick_dps) until it moves; then the controller catches it. "Still": the raw reading moved
            # < 8 mm in 0.5 s (the speed estimate is too noisy).
            raw_still = len(raw_hist) > int(0.5 / run.dt) and abs(ravg - raw_hist[-int(0.5 / run.dt)]) < 0.008
            if not probe and not braking and T["kick_deg"] > 0 and raw_still \
                    and abs(ravg - goal) > T["deadband_mm"] / 1000.0:
                still_t += run.dt
                if still_t > 0.3:
                    mag = min(tmax, np.radians(T["kick_deg"]) + np.radians(T["unstick_dps"]) * (still_t - 0.3))
                    want = np.sign(goal - ravg) * mag
                    if want * e_t <= 0 or abs(e_t) < abs(want):
                        e_t, kicking = float(want), True
            else:
                still_t = 0.0
            # I-term: a steady lean of the paddle keeps the ball on one side; the integral of the error adds
            # the tilt that cancels it (Nov 20 23:41: the ball sat left although the paddle averaged +0.45 deg).
            # It keeps integrating at the rail ends (that is where a lean shows most), clamped to ki_max_deg.
            if not probe and T["ki"] > 0:
                imax = np.radians(T["ki_max_deg"]) / T["ki"]
                integ = float(np.clip(integ + (st[0] - goal) * run.dt, -imax, imax))
                e_t -= T["ki"] * integ
            unsticking = False
            if T["unstick"] and pinned_t > 0.3:         # a ball sitting at an open end falls off within ~1.5 s
                # held at an end: ramp the counter-angle toward the centre until the ball comes free
                # up to the full max tilt: 0.4-1.0 deg could not pull the ball off the front edge (Nov 21 21:03). Safe now
                # that the unstick ends the moment the ball moves inward (19:40: it kept 2 deg for 0.5 s and launched it)
                umax = min(tmax, np.radians(c.get("unstick_max_deg", 90.0)))
                want = -side * min(umax, np.radians(0.5) + np.radians(T["unstick_dps"]) * (pinned_t - 0.3))
                if want * e_t <= 0 or abs(want) > abs(e_t):
                    e_t, unsticking = float(want), True
            ramp = min(1.0, max(0.0, (i * run.dt - t_start) / 2.0))   # soft start after (re)anchoring
            if not unsticking:
                e_t *= ramp
            e_t = float(np.clip(e_t, -tmax, tmax))
            target = np.zeros(2) if probe else _tilt_target(run, sgn * e_t)
            run._rec(p_meas=p_meas, p_est=st[0], v_est=st[1], bias=bias, tilt_target=sgn * e_t, d=d, offset=offset,
                     v_star=v_star, p_ahead=p_ahead, goal=goal, pinned=float(pinned), unstick=float(unsticking),
                     resyncs=float(resyncs), i_tilt=T["ki"] * integ, braking=float(braking), kicking=float(kicking),
                     gated=float(gated), cycle_t=run.t, spread=spread_avg, ball_w=w_avg, outliers=float(outliers),
                     hold=hold_f,       # cycle_t: matches the per-cycle "t" of this step
                     r=np.sqrt(r), a_deck=a_deck, truth=io.ball_truth())
            show = (show + [p_meas])[-max(int(0.25 / run.dt), 1):]
            if i % 2 == 0:
                ctx.live(tau=tau, q=q, est=(st[0], 0.0), meas=(float(np.mean(show)), 0.0), err=None, tilt=tilt,
                         tiltcmd=run.servo.cmd)
            sat = (sat + [abs(e_t) >= 0.98 * tmax and not probe and not unsticking])[-int(3.0 / run.dt):]
            if T["balance_stops"] and len(sat) == int(3.0 / run.dt) and np.mean(sat) > 0.8:
                raise SafetyStop("the controller has asked for the full tilt most of the last 3 s (oscillating or the "
                                 "estimate is off): lower the gains, re-anchor, or redo the rail map")
            lost = lost + 1 if abs(ravg) > far and not probe and T["balance_stops"] else 0
            if lost > int(1.0 / run.dt):
                raise SafetyStop(f"for 1 s the torque reading put the ball {np.mean(raw_avg) * 1000:+.0f} mm from C, "
                                 "past the rail end: ball off the rail, something touching, or re-do the map/anchor")
            event = None
            # ball dropped? joint 4 back near its empty level for 0.3 s: hold level, ask, re-anchor
            if pres_ok and not probe and pres_by_weight:
                w_now = float(np.linalg.lstsq(AB, np.asarray(d, dtype=float), rcond=None)[0][0])
                w_avg += (run.dt / 0.1) * (w_now - w_avg)
                d_avg = np.asarray(d, dtype=float) if d_avg is None else d_avg + (run.dt / 0.1) * (np.asarray(d) - d_avg)
                # the weight solve multiplies a one-joint error by ~4 (the two joints' columns nearly overlap): a spring
                # on joint 2 alone looked like a drop on the fake. A real drop empties EVERY sensing joint: require both
                each_empty = bool(np.all(np.abs(d_avg) < 0.5 * np.abs(np.array(rm["a"]))))
                gone_t = gone_t + run.dt if (w_avg < 0.4 and each_empty) else 0.0
            elif pres_ok and not probe:
                pres_avg += (run.dt / 0.1) * (tau[pres_j] - run.tare.predict(phi)[pres_j] + tau_empty - pres_avg)
                gone_t = gone_t + run.dt if (pres_avg - tau_empty) / pres_step < 0.4 else 0.0
            if pres_ok and not probe and gone_t > 0.3:
                event = "drop"
            # contact: 2.5 Nm SUSTAINED for 0.1 s (a hard tilt reversal gives 1-6 sample spikes). Not a stop any more
            # (Nov 21 19:02: a push ended a good run): level, re-zero with the ball off, ball back, keep balancing
            contact_t = contact_t + run.dt if np.max(np.abs(d)) > 2.5 else 0.0
            if contact_t >= 0.1 and event is None and not probe:
                event = "contact"
            if event is not None:
                drops += 1
                run._rec(drop_t=run.t)
                if event == "drop":
                    ctx.log(f"ball DROPPED at {i * run.dt:.1f} s (drop {drops}); last estimate {st[0] * 1000:+.0f} mm, "
                            f"speed {st[1] * 1000:+.0f} mm/s")
                    zero = run.hold(1.0, settle=0.3)       # the paddle should be empty now: re-zero on the spot
                    zfrac = float(np.max(np.abs(zero[ch]) / np.maximum(np.abs(np.array(rm["a"])), 0.05)))
                    if 0.25 < zfrac <= 0.5:
                        # neither empty nor the ball: a hand picking the ball up, the ball on the edge. Do not learn it
                        ctx.log(f"not re-zeroing: the paddle did not read clearly empty ("
                                + ", ".join(f"{JOINT_NAMES[j]} {zero[j] * 1000:+.0f} mNm" for j in ch)
                                + "); keeping the previous zero")
                        zero = np.zeros_like(zero)
                    if zfrac > 0.5:
                        # the ball's weight is still there: a false drop. Do NOT learn it as empty; keep balancing
                        ctx.log(f"false drop: the ball's weight is still on the paddle ("
                                + ", ".join(f"{JOINT_NAMES[j]} {zero[j] * 1000:+.0f} mNm" for j in ch)
                                + "); not re-zeroing, balancing on")
                        drops -= 1
                        gone_t, w_avg, event = 0.0, 1.0, None
                        pres_avg = tau_on
                        kf.x[1] = 0.0
                        t_start = i * run.dt
                        continue
                else:
                    ctx.log(f"CONTACT at {i * run.dt:.1f} s ({np.max(np.abs(d)):.1f} Nm on a sensing joint for 0.1 s: "
                            f"a hand, or the ball hitting/leaving hard); paddle levelled, asking for the ball (event {drops})")
                    msg = ("Something pressed hard on the paddle (or the ball fell). Take the ball OFF, hands off, "
                           "then Continue (1 s re-zero).")
                    for _try in range(5):
                        run.prompt(msg, ball=None)
                        zero = run.hold(1.0, settle=0.3)
                        # ball still on? its weight is the map intercept on each sensing joint: refuse to re-zero on it
                        if np.max(np.abs(zero[ch]) / np.maximum(np.abs(np.array(rm["a"])), 0.05)) < 0.5:
                            break
                        ctx.log(f"re-zero refused: the paddle is not empty ({', '.join(f'{JOINT_NAMES[j]} {zero[j] * 1000:+.0f} mNm' for j in ch)}"
                                ", about the ball's weight). Asking again")
                        msg = "The ball still seems to be ON the paddle (or a hand on it). Take it OFF, hands off, then Continue."
                    else:
                        zero = np.zeros_like(zero)          # never learn the ball's weight as "empty"
                run.tare.rezero(zero)
                z_mm = float(np.mean(zero[ch] / np.array(rm["b"]))) * 1000
                ctx.log(f"re-zeroed with the paddle empty: the sensing joints had drifted {z_mm:+.0f} mm-equivalent "
                        f"({', '.join(f'{JOINT_NAMES[j]} {zero[j] * 1000:+.0f} mNm' for j in ch)})")
                run._rec(rezero_mm=z_mm)
                run.prompt(f"Put the ball back IN THE RAIL at the {anchor_mm:+.0f} mm mark (event {drops}), "
                           "let it settle, then Continue.", ball=_ball_xy(c["axis"], anchor_p))
                ps_ = []
                for k_ in range(int(1.5 / run.dt)):
                    q, _, tau, phi, tilt, J = run.cycle([0.0, 0.0])
                    if k_ * run.dt >= 0.5:
                        ps_.append(measure(q, tau, phi, J, 0.0)[0])
                offset = anchor_p - float(np.mean(ps_))
                # RESET the learned lean when the ball is put back: the lean learned while the ball swung end to end is
                # junk, and carrying it over (-0.36 deg) pushed the re-placed ball off within 3 s (Nov 24 21:44)
                kf.x[:] = [anchor_p, 0.0, 0.0]
                kf.P[:] = START_P
                t_start = i * run.dt
                pres_avg, gone_t, integ, still_t, pinned_t, contact_t, w_avg, d_avg = tau_on, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, None
                # the empty-paddle readings (~+300 mm garbage) inflated the learned noise to ~200 mm (Nov 21 20:03): the
                # filter then ignored the ball after it was put back. Start the noise memory and the gate fresh
                r_ema, outlier_t, gate_cool, resync_t = sp ** 2, 0.0, 0.0, 0.0
                raw_avg, raw_hist = [], []
                target = np.zeros(2)
                event = None
                continue
        run.log["stop_reason"] = [f"finished: ran the full {seconds:.0f} s"]
    except SafetyStop as e:
        ctx.log(f"STOP: {e}. Levelling the paddle.")
        run.log["stop_reason"] = [f"safety stop: {e}"]
    except Exception as e:
        import traceback
        run.log["stop_reason"] = [f"python error: {e!r}\n{traceback.format_exc()}"]
        raise
    finally:
        run.stop()
    goal = T["goal_mm"] / 1000.0
    if "p_est" in run.log and len(run.log["p_est"]) > 10:
        err = (np.array(run.log["p_est"]) - goal) * 1000
        n = len(err)
        tl = np.degrees(np.array(run.log["tilt"])[-n:, k])
        if run.log.get("drop_t"):
            ctx.log(f"the ball dropped {len(run.log['drop_t'])} times in this run")
        ctx.log(f"rail {'probe' if probe else 'balance'} ran {n * run.dt:.1f} s: ball from the goal rms {np.sqrt(np.mean(err ** 2)):.0f} mm "
                f"(second half {np.sqrt(np.mean(err[n // 2:] ** 2)):.0f} mm, worst {np.abs(err).max():.0f} mm), "
                f"tilt rms {np.sqrt(np.mean(tl ** 2)):.2f} deg")
    return run.log


# ------------------------------------------------------------------------------- roll test
def job_rail_rolltest(io, calib, ctx, deg=1.0, rolls=3, roll_s=6.0):
    """Open loop: hold the paddle at +deg (ball rolls to the BACK, + end), then -deg (rolls to the FRONT), and let the
    operator release the ball from the high end `rolls` times each way. The ball's acceleration against the MEASURED
    tilt gives the real rolling gain (ball on two sticks: less than 5/7 g) and the leftover lean (where it would not
    roll). Nov 21: balancing failed toward the front every time; a closed-loop log cannot separate the two."""
    if calib.railmap is None or calib.railtare_W is None:
        raise SafetyStop("do the rail map and the rail tilt sweep first")
    tare = Tare(np.array(calib.railtare_W, dtype=float).copy(), np.array(calib.railtare_sigma, dtype=float).copy())
    run = make_runner(io, calib, ctx, "railroll", tare=tare)
    c, ax, rm = run.rcfg, run.ax, calib.railmap
    k, sgn = ax["tilt"], ax["sgn"]
    ch = rm["channels"]
    sig_ch = np.array(calib.railtare_sigma)[ch]
    amp = np.radians(min(float(deg), 0.9 * c["tilt_max_deg"]))
    lim = max(kin.PLATE_HALF - kin.BALL_RADIUS, max(abs(float(v)) for v in points_of(calib)) / 1000.0 + 0.010)
    results = []                                        # (direction, roll, accel m/s2, mean tilt e rad, ok)
    # release 5 cm IN from the end: both rail ends hold the ball in a pocket (Nov 21 20:51: at the back end it sat 6 s
    # with the back raised 1.9 deg), which measures the end, not the rolling
    near = "+50 mm mark (about 5 cm in from the BACK end, the end NEAR the robot), NOT in the end pocket"
    far = "-50 mm mark (about 5 cm in from the FRONT end, the end AWAY from the robot), NOT in the end pocket"

    def hold_tilt(e, seconds, rec=None):
        for _ in range(int(seconds / run.dt)):
            q, _, tau, phi, tilt, J = run.cycle(_tilt_target(run, sgn * e))
            d = (tau - run.tare.predict(phi))[ch]
            p, _ = combine(rm, d, sig_ch)
            ctx.live(tau=tau, q=q, est=(p, 0.0), meas=(p, 0.0), tilt=tilt, tiltcmd=run.servo.cmd)
            if rec is not None:
                rec.append((run.t, p, sgn * tilt[k]))
            run._rec(p_meas=p, e=sgn * tilt[k], cycle_t=run.t)

    run.start()
    try:
        run.prompt("Roll test: take the ball OFF the paddle, hands off, then Continue (1 s re-zero).", ball=None)
        run.tare.rezero(run.hold(1.0, settle=0.5))
        for direction, e_cmd, start_end, to_end in ((+1, +amp, far, "back"), (-1, -amp, near, "front")):
            ctx.status(f"roll test: tilting {np.degrees(e_cmd):+.1f} deg")
            hold_tilt(e_cmd, 1.5)                       # tilt and settle before asking
            for r in range(1, rolls + 1):
                run.prompt(f"Roll test {'A' if direction > 0 else 'B'}, roll {r} of {rolls}: hold the ball STILL at the "
                           f"{start_end}, hands off the paddle otherwise. Press Continue, then LET GO (no push) within "
                           f"1 s and let it roll to the {to_end}"
                           + ". CATCH it gently at that end; never let it drop.", ball=None)
                rec = []
                hold_tilt(e_cmd, roll_s, rec)
                t_, p_, e_ = (np.array(x) for x in zip(*rec))
                if abs(np.median(p_[:20])) > lim + 0.03:
                    ctx.log(f"roll {'A' if direction > 0 else 'B'}{r}: no ball on the paddle at the start (reading "
                            f"{1000 * float(np.median(p_[:20])):+.0f} mm): skipped")
                    results.append((direction, r, 0.0, float(np.mean(e_)), False))
                    continue
                # the rolling part: from when it starts moving (10 mm from its start) until 15 mm before the far end
                p_rel = (p_ - np.median(p_[:20])) * direction
                mv = np.flatnonzero(p_rel > 0.010)
                if len(mv) < 20:
                    ctx.log(f"roll {'A' if direction > 0 else 'B'}{r}: the ball did not roll (moved "
                            f"{1000 * float(np.max(p_rel)):.0f} mm in {roll_s:.0f} s at {np.degrees(np.mean(e_)):+.2f} deg)")
                    results.append((direction, r, 0.0, float(np.mean(e_)), False))
                    continue
                i0 = max(0, mv[0] - 15)
                # fit until 25 mm before the rail end it rolls to (or 2 s): after that it stops, or falls and reads junk
                j1 = i0
                while (j1 + 1 < len(p_) and t_[j1 + 1] - t_[i0] < 2.0
                       and direction * p_[j1 + 1] < lim - 0.025 and abs(p_[j1 + 1]) < lim + 0.05):
                    j1 += 1
                seg = slice(i0, j1 + 1)
                if j1 - i0 < 20:
                    ctx.log(f"roll {'A' if direction > 0 else 'B'}{r}: too short to fit")
                    results.append((direction, r, 0.0, float(np.mean(e_)), False))
                    continue
                tt = t_[seg] - t_[i0]
                a = 2.0 * float(np.polyfit(tt, p_[seg], 2)[0])
                results.append((direction, r, a, float(np.mean(e_[seg])), True))
                ctx.log(f"roll {'A' if direction > 0 else 'B'}{r}: accel {a * 1000:+.0f} mm/s2 at a measured tilt of "
                        f"{np.degrees(np.mean(e_[seg])):+.2f} deg ({tt[-1]:.1f} s, {1000 * abs(p_[j1] - p_[i0]):.0f} mm)")
        ctx.status("roll test: levelling")
        hold_tilt(0.0, 1.5)
    except SafetyStop as e:
        ctx.log(f"ROLL TEST STOPPED: {e}")
    finally:
        run.stop()
    good = [x for x in results if x[4]]
    ups = [x for x in good if x[0] > 0]
    dns = [x for x in good if x[0] < 0]
    if not ups or not dns:
        ctx.log("roll test: need good rolls in BOTH directions to separate the rolling gain from the lean")
        return {"results": results}
    A = np.array([[np.sin(x[3]), 1.0] for x in good]); y = np.array([x[2] for x in good])
    (gain, a0), *_ = np.linalg.lstsq(A, y, rcond=None)
    scale = gain / ROLL_ACCEL
    e0 = -a0 / gain if gain > 0 else 0.0               # the tilt where the ball would not roll (in "toward +" terms)
    trim_new = c.get("level_trim_deg", 0.0) + float(np.degrees(sgn * e0))
    ctx.log(f"ROLL TEST: real ball accel = {gain:.2f} m/s2 per rad of tilt = {scale:.2f} x the 5/7 g model "
            f"(set 'rolling gain' to {scale:.2f}); leftover lean {np.degrees(e0):+.2f} deg along the rail "
            f"({'the ball drifts to the FRONT' if a0 < 0 else 'the ball drifts to the BACK'} at level, "
            f"{abs(a0) * 1000:.0f} mm/s2) -> level trim {c.get('level_trim_deg', 0.0):+.2f} -> {trim_new:+.2f}")
    return {"results": results, "roll_scale": float(scale), "lean_deg": float(np.degrees(e0)), "trim_new": trim_new}
