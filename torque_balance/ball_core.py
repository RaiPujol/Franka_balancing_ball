"""Torque-only ball balancing on a Franka Panda: shared math and procedures.

Nothing in this file imports frankapy or mujoco. The real robot script
(franka_ball_balance.py) and the simulator (sim_ball_balance.py) both run the
procedures below through the same small "io" object, so the code tested in
simulation is the code that runs on the arm.

How the ball is sensed (no camera)
----------------------------------
A ball of mass m sitting on the paddle at horizontal offset (dX, dY) from the paddle
centre pushes down with k = m*g. In the base frame that changes the measured
link-side joint torques (libfranka tau_J) by

    d_tau = k * ( J_vz^T  -  J_wy^T * dX  +  J_wx^T * dY )

where J is the 6x7 Jacobian of the paddle centre. Everything else the torque sensors
see (arm + hand + paddle gravity, inertia while tilting) is removed with a "tare"
model fitted from a sweep with no ball on the paddle. (dX, dY) is then a weighted
least-squares solve, a small Kalman filter turns the noisy position into position +
velocity using the rolling-ball model, and a PD law tilts the paddle.

Frames
------
base  : robot base, z up.
EE    : libfranka end-effector frame with the Franka Hand (what frankapy get_pose gives).
paddle: origin at the centre of the paddle's top face, z out of the top face.
Tilt (a, b) is a small rotation about the BASE x and y axes away from the level pose.
"""

import json
import os

import numpy as np

G = 9.81
GOLF_BALL_MASS = 0.0459          # kg
ROLL_ACCEL = (5.0 / 7.0) * G     # m/s^2 per rad of tilt, solid ball rolling without slip
PLATE_HALF = 0.127               # m, half of the 10 inch paddle

# Gains of frankapy's joint impedance controller (its defaults). The real script passes
# these explicitly, the simulator uses the same numbers. torque = K (q_d - q) - D dq.
K_GAINS = np.array([600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0])
D_GAINS = np.array([50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0])


# --------------------------------------------------------------------------------------
# Paddle mounting. EDIT THESE TWO to match how the paddle sits in the gripper.
# --------------------------------------------------------------------------------------
# Centre of the paddle's top face, in the EE frame (metres).
# Default: handle held in the fingers, paddle sticking straight out along the hand's
# z axis, centre 16.7 cm past the fingertips.
PADDLE_CENTER_IN_EE = (0.0, 0.0, 0.167)
# Paddle x, y, z axes written in EE coordinates (rows). z is the top-face normal.
# Default: paddle x = hand z (pointing away from the wrist), paddle normal = hand y
# (the finger closing direction, i.e. the fingers pinch a flat handle).
PADDLE_AXES_IN_EE = ((0.0, 0.0, 1.0),
                     (1.0, 0.0, 0.0),
                     (0.0, 1.0, 0.0))


def paddle_transform(center=PADDLE_CENTER_IN_EE, axes=PADDLE_AXES_IN_EE):
    T = np.eye(4)
    T[:3, :3] = np.asarray(axes, dtype=float).T
    T[:3, 3] = center
    return T


# --------------------------------------------------------------------------------------
# Panda kinematics (Craig DH parameters from the Franka documentation)
# --------------------------------------------------------------------------------------
_DH = np.array([   # a_{i-1}, d_i, alpha_{i-1}
    [0.0,     0.333,  0.0],
    [0.0,     0.0,   -np.pi / 2],
    [0.0,     0.316,  np.pi / 2],
    [0.0825,  0.0,    np.pi / 2],
    [-0.0825, 0.384, -np.pi / 2],
    [0.0,     0.0,    np.pi / 2],
    [0.088,   0.0,    np.pi / 2],
])


def _link7_to_ee():
    c, s = np.cos(-np.pi / 4), np.sin(-np.pi / 4)
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    T[2, 3] = 0.107 + 0.1034     # flange offset + Franka Hand TCP offset
    return T


_T_LINK7_EE = _link7_to_ee()

JOINT_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
JOINT_MAX = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])


def fk(q, T_ee_tool=None):
    """Pose (4x4) and geometric Jacobian (6x7, rows vx vy vz wx wy wz) of a tool frame
    rigidly attached to the EE. T_ee_tool=None gives the EE frame itself."""
    T = np.eye(4)
    origins, axes = [], []
    for (a, d, alpha), th in zip(_DH, q):
        ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(th), np.sin(th)
        T = T @ np.array([[ct, -st, 0.0, a],
                          [st * ca, ct * ca, -sa, -d * sa],
                          [st * sa, ct * sa, ca, d * ca],
                          [0.0, 0.0, 0.0, 1.0]])
        origins.append(T[:3, 3].copy())
        axes.append(T[:3, 2].copy())
    T = T @ _T_LINK7_EE
    if T_ee_tool is not None:
        T = T @ T_ee_tool
    J = np.zeros((6, 7))
    for i in range(7):
        J[:3, i] = np.cross(axes[i], T[:3, 3] - origins[i])
        J[3:, i] = axes[i]
    return T, J


def rotvec_to_R(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = np.asarray(w) / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def R_to_rotvec(R):
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    th = np.arccos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < 1e-9:
        return v / 2
    return v * th / (2 * np.sin(th))


# --------------------------------------------------------------------------------------
# Tilt servo: tilt target -> smooth tilt command -> joint targets
# --------------------------------------------------------------------------------------
class TiltServo:
    """Keeps the paddle at tilt (a, b) about the base x, y axes relative to pose q0.

    Only the wrist (joints 5, 6, 7) moves; joints 1-4 get a constant target. That keeps
    the big joints' torque readings quiet and the motion small.

    The impedance controller follows a moving target with a lag of D/K seconds per joint
    (0.3 s on joint 7 with the default gains). `lead` adds that much velocity feed-forward
    to the joint target so the paddle tracks the tilt command.
    """

    def __init__(self, q0, T_ee_paddle, dt, tilt_max=np.radians(5.0), wc=5.0, lead=None):
        self.q0 = np.asarray(q0, dtype=float).copy()
        self.T_ee_paddle = T_ee_paddle
        self.dt, self.tilt_max, self.wc = dt, tilt_max, wc
        self.lead = np.zeros(7) if lead is None else np.asarray(lead, dtype=float)
        T0, _ = fk(self.q0, T_ee_paddle)
        self.R0, self.p0 = T0[:3, :3].copy(), T0[:3, 3].copy()
        self.reset()

    def reset(self):
        self.q = self.q0.copy()            # joint pose that realises the current tilt command
        self.cmd = np.zeros(2)
        self.dcmd = np.zeros(2)
        self.trim = np.zeros(2)

    def step(self, target, measured=None):
        """Advance the command filter one period toward `target`; return the joint target.
        measured: the tilt the encoders report right now. A slow integrator (`trim`) closes
        the gap between commanded and actual tilt, which soft joints leave open: the ball's
        weight sags joint 6, and friction holds joint 7 a fraction of a degree off."""
        target = np.clip(np.asarray(target, dtype=float), -self.tilt_max, self.tilt_max)
        # critically damped second-order filter: bounded tilt rate and acceleration
        ddcmd = self.wc ** 2 * (target - self.cmd) - 2 * self.wc * self.dcmd
        self.dcmd = self.dcmd + ddcmd * self.dt
        self.cmd = self.cmd + self.dcmd * self.dt
        if measured is not None:
            self.trim = np.clip(self.trim + 2.0 * (self.cmd - measured) * self.dt,
                                -np.radians(2.0), np.radians(2.0))
        q_new = self._ik(self.cmd + self.trim)
        q_send = q_new + self.lead * (q_new - self.q) / self.dt
        self.q = q_new
        return q_send

    def measured_tilt(self, q):
        T, _ = fk(q, self.T_ee_paddle)
        return R_to_rotvec(T[:3, :3] @ self.R0.T)[:2]

    def _ik(self, tilt, iters=3):
        R_des = rotvec_to_R([tilt[0], tilt[1], 0.0]) @ self.R0
        q = self.q.copy()
        weight = np.array([1.0, 1.0, 0.2])          # yaw matters little, tilt matters
        for _ in range(iters):
            T, J = fk(q, self.T_ee_paddle)
            e_w = weight * R_to_rotvec(R_des @ T[:3, :3].T)
            Jw = weight[:, None] * J[3:, 4:7]
            q[4:7] += np.linalg.solve(Jw.T @ Jw + 1e-4 * np.eye(3), Jw.T @ e_w)
        return q


# --------------------------------------------------------------------------------------
# Ball sensing from torques
# --------------------------------------------------------------------------------------
class BallSensor:
    """Maps a robot state to a measurement vector y and its ball model

        y_ball = k * ( c + A @ [dX, dY] )

    source "tau"   : y = the 7 measured joint torques (tau_J).
    source "wrench": y = [Mx, My] of libfranka's external wrench estimate O_F_ext_hat_K
                     (what frankapy returns from get_ee_force_torque).
    """

    def __init__(self, source, T_ee_paddle):
        if source not in ("tau", "wrench"):
            raise ValueError("source must be 'tau' or 'wrench'")
        self.source, self.T_ee_paddle = source, T_ee_paddle
        self.n = 7 if source == "tau" else 2

    def raw(self, state):
        if self.source == "tau":
            return np.asarray(state["tau"], dtype=float)
        if state.get("wrench") is None:
            raise RuntimeError("this robot interface does not provide a wrench estimate")
        return np.asarray(state["wrench"], dtype=float)[3:5]

    def model(self, q):
        """Returns c, A, H with  y_ball = k * ( (1 + az/g) * (c + A @ [dX, dY]) + H @ [ax, ay] ).
        H is the effect of the ball's own horizontal acceleration (it pushes back on the
        paddle while it speeds up), az is the paddle's vertical acceleration."""
        T, J = fk(q, self.T_ee_paddle)
        if self.source == "tau":
            return J[2].copy(), np.c_[-J[4], J[3]], np.c_[J[0], J[1]] / G
        T_ee, _ = fk(q)
        r0 = T[:3, 3] - T_ee[:3, 3]          # EE origin -> paddle centre, base frame
        return (np.array([-r0[1], r0[0]]), np.array([[0.0, -1.0], [1.0, 0.0]]),
                np.array([[0.0, r0[2]], [-r0[2], 0.0]]) / G)

    def locate(self, q, dy, k, sigma, accel_z=0.0, accel_h=(0.0, 0.0)):
        """Weighted least squares ball position from the tared measurement dy.
        accel_z: upward acceleration of the paddle centre (a ball on an accelerating paddle
                 presses with m*(g + az); with the paddle 0.4 m from joint 6 that matters
                 as soon as the paddle pitches).
        accel_h: the ball's expected horizontal acceleration (zero for a ball that is held
                 in place).
        Returns (xy in base axes relative to the paddle centre, 2x2 covariance)."""
        c, A, H = self.model(q)
        w = 1.0 / np.square(sigma)
        Hinv = np.linalg.inv(A.T @ (A * w[:, None]))
        rhs = (dy / k - H @ np.asarray(accel_h, dtype=float)) / (1.0 + accel_z / G) - c
        xy = Hinv @ (A.T @ (w * rhs))
        return xy, Hinv / (k * k)


WRIST = [4, 5, 6]        # joints 5, 6, 7 do the tilting
ARM = [1, 3]             # joints 2 and 4 carry the reaction of the wrist motion
LAGS = 2                 # how many past velocity samples the model sees


def motion_features(q_wrist_dev, vel_hist):
    """Regressors for the no-ball torque model.
    q_wrist_dev: measured wrist joint angles minus the balance pose.
    vel_hist   : the last LAGS+1 measured joint velocity vectors, newest first."""
    q5, q6, q7 = q_wrist_dev
    quad = [q5 * q5, q5 * q6, q5 * q7, q6 * q6, q6 * q7, q7 * q7]
    return np.r_[1.0, q_wrist_dev, quad,
                 np.concatenate([v[WRIST] for v in vel_hist]),
                 np.concatenate([v[ARM] for v in vel_hist])]


class Tare:
    """No-ball model of the measurement: y0 = features @ W, one column per channel.

    Features are the MEASURED motion (see motion_features): rigid-body physics says the
    no-ball joint torques are gravity(q) + inertia * acceleration, so a regression on the
    measured wrist angles and a short history of joint velocities captures both without
    a robot model. They deliberately leave out the joint targets and the angles of joints
    1-4: torque is K*(target - q) by construction, so features that reveal the deflection
    would explain the ball away as well."""

    def __init__(self, W, sigma):
        self.W, self.sigma = np.asarray(W, dtype=float), np.asarray(sigma, dtype=float)

    @classmethod
    def fit(cls, Phi, Y):
        scale = np.maximum(np.abs(Phi).max(axis=0), 1e-9)
        Ws, *_ = np.linalg.lstsq(Phi / scale, Y, rcond=None)
        W = Ws / scale[:, None]
        sigma = np.maximum((Y - Phi @ W).std(axis=0), 1e-4)
        return cls(W, sigma)

    def predict(self, phi):
        return phi @ self.W

    def rezero(self, mean_residual):
        self.W[0] += mean_residual


class AxisKF:
    """Kalman filter for one axis. State [position, velocity, disturbance accel].
    The disturbance state soaks up a paddle that is not quite level."""

    def __init__(self, dt, accel_noise=0.4, dist_noise=0.05):
        self.dt = dt
        self.F = np.array([[1, dt, 0.5 * dt * dt], [0, 1, dt], [0, 0, 1]], dtype=float)
        self.B = np.array([0.5 * dt * dt, dt, 0.0])
        g = self.B
        self.Q = accel_noise ** 2 * np.outer(g, g) + np.diag([0, 0, dist_noise ** 2 * dt])
        self.reset()

    def reset(self, pos=0.0, pos_sigma=0.03):
        self.x = np.array([pos, 0.0, 0.0])
        self.P = np.diag([pos_sigma ** 2, 0.05 ** 2, 0.1 ** 2])

    def step(self, accel, z, r):
        """Predict with the known tilt acceleration, then update with measured position z
        (variance r)."""
        self.x = self.F @ self.x + self.B * accel
        self.P = self.F @ self.P @ self.F.T + self.Q
        r = max(r, 0.005 ** 2)                # never trust a single sample to better than 5 mm
        innov = z - self.x[0]
        s = self.P[0, 0] + r
        if innov * innov > 9.0 * s:           # wild sample: de-weight it instead of obeying it
            s = innov * innov / 9.0
        K = self.P[:, 0] / s
        self.x = self.x + K * innov
        self.P = self.P - np.outer(K, self.P[0])
        return self.x


# --------------------------------------------------------------------------------------
# Calibration file
# --------------------------------------------------------------------------------------
class Calib:
    FIELDS = ("q0", "source", "rate_hz", "tilt_max", "paddle_center", "paddle_axes",
              "tare_W", "tare_sigma", "k", "center_offset")

    def __init__(self, q0, source="tau", rate_hz=50.0, tilt_max=np.radians(5.0),
                 paddle_center=PADDLE_CENTER_IN_EE, paddle_axes=PADDLE_AXES_IN_EE,
                 tare_W=None, tare_sigma=None, k=GOLF_BALL_MASS * G, center_offset=(0.0, 0.0)):
        self.q0 = np.asarray(q0, dtype=float)
        self.source, self.rate_hz, self.tilt_max = source, float(rate_hz), float(tilt_max)
        self.paddle_center, self.paddle_axes = paddle_center, paddle_axes
        self.tare_W = None if tare_W is None else np.asarray(tare_W, dtype=float)
        self.tare_sigma = None if tare_sigma is None else np.asarray(tare_sigma, dtype=float)
        self.k = float(k)
        self.center_offset = np.asarray(center_offset, dtype=float)

    @property
    def T_ee_paddle(self):
        return paddle_transform(self.paddle_center, self.paddle_axes)

    def save(self, path):
        d = {}
        for f in self.FIELDS:
            v = getattr(self, f)
            d[f] = v.tolist() if isinstance(v, np.ndarray) else v
        with open(path, "w") as fh:
            json.dump(d, fh, indent=1)

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            return cls(**json.load(fh))


class SafetyStop(RuntimeError):
    pass


class Session:
    """One connected run: servo + sensor + tare, with the per-cycle bookkeeping."""

    def __init__(self, io, calib, log_name=None):
        self.io, self.calib = io, calib
        self.dt = 1.0 / calib.rate_hz
        self.servo = TiltServo(calib.q0, calib.T_ee_paddle, self.dt, calib.tilt_max, lead=D_GAINS / K_GAINS)
        self._vel = []
        self._vz = None
        self.accel_z = 0.0
        self.sensor = BallSensor(calib.source, calib.T_ee_paddle)
        self.tare = None
        if calib.tare_W is not None:
            self.tare = Tare(calib.tare_W.copy(), calib.tare_sigma.copy())
        self.log = {}
        self.log_name = log_name
        self.t = 0.0
        # paddle x, y axes in the base horizontal plane (to report positions in paddle coordinates)
        self.P2 = self.servo.R0[:2, :2]

    # -- unit conversion between base-frame offsets and marks drawn on the paddle
    def to_paddle(self, xy_base):
        return np.linalg.solve(self.P2, xy_base)

    def to_base(self, xy_paddle):
        return self.P2 @ np.asarray(xy_paddle, dtype=float)

    def start(self):
        normal = self.servo.R0[:, 2]
        off = np.degrees(np.arccos(np.clip(normal[2], -1, 1)))
        if off > 10.0:
            raise SafetyStop(
                f"paddle normal is {off:.0f} deg away from vertical at the balance pose. "
                "Re-teach the pose, or fix PADDLE_AXES_IN_EE at the top of ball_core.py.")
        print(f"balance pose: paddle centre at {np.round(self.servo.p0, 3)} m, "
              f"normal {off:.1f} deg from vertical (by the kinematic model)")
        self.io.goto(self.calib.q0, 4.0)
        self.servo.reset()
        self._vel, self._vz = [], None
        self.io.start_stream(self.calib.q0)

    def stop(self):
        try:
            for _ in range(int(1.0 / self.dt)):
                self.io.command(self.servo.step([0.0, 0.0]))
        finally:
            self.io.stop_stream()
            self.save_log()

    def cycle(self, tilt_target, **extra):
        """Read the robot, send the next joint target.
        Returns (state, features of the measured motion, measurement y)."""
        state = self.io.read()
        q = np.asarray(state["q"], dtype=float)
        dq = np.asarray(state["dq"], dtype=float)
        self._vel = ([dq] + self._vel)[:LAGS + 1]
        while len(self._vel) < LAGS + 1:
            self._vel.append(self._vel[-1])
        phi = motion_features(q[4:7] - self.calib.q0[4:7], self._vel)
        _, J = fk(q, self.calib.T_ee_paddle)
        vz = J[2] @ dq                         # vertical velocity of the paddle centre
        self.accel_z = 0.0 if self._vz is None else (vz - self._vz) / self.dt
        self._vz = vz
        y = self.sensor.raw(state)
        if np.max(np.abs(q - self.servo.q)) > 0.12:
            raise SafetyStop("arm is not following the command (blocked or pushed?)")
        tilt = self.servo.measured_tilt(q)
        q_des = self.servo.step(tilt_target, tilt)
        if np.max(np.abs(q_des - self.calib.q0)) > 0.35 or np.any(q_des < JOINT_MIN + 0.02) \
                or np.any(q_des > JOINT_MAX - 0.02):
            raise SafetyStop("joint target left the allowed window around the balance pose")
        self.io.command(q_des)
        self.t += self.dt
        self._rec(t=self.t, q=q, tau=state["tau"], y=y, tilt_cmd=self.servo.cmd.copy(),
                  tilt=tilt, **extra)
        return state, phi, y

    def hold(self, seconds, tilt=(0.0, 0.0), settle=1.0):
        """Hold a tilt; return the mean and std of the tared measurement and the mean q."""
        res, qs = [], []
        for i in range(int((settle + seconds) / self.dt)):
            state, phi, y = self.cycle(tilt)
            if i * self.dt >= settle:
                res.append(y - (self.tare.predict(phi) if self.tare else 0.0))
                qs.append(state["q"])
        res = np.array(res)
        return res.mean(axis=0), res.std(axis=0), np.mean(qs, axis=0)

    def rezero(self, seconds=3.0):
        self.io.ask("REMOVE the ball from the paddle, hands off, then press Enter: ", ball=None)
        mean, _, _ = self.hold(seconds)
        self.tare.rezero(mean)

    def _rec(self, **kw):
        for key, v in kw.items():
            if v is not None:
                self.log.setdefault(key, []).append(np.array(v, dtype=float))

    def save_log(self):
        if not self.log_name or not self.log:
            return
        os.makedirs(os.path.dirname(self.log_name) or ".", exist_ok=True)
        np.savez(self.log_name, **{k: np.array(v) for k, v in self.log.items()})
        print(f"log saved to {self.log_name}")


# --------------------------------------------------------------------------------------
# Procedures
# --------------------------------------------------------------------------------------
def run_log(io, seconds, rate_hz, out_csv, calib=None):
    """Step 1: just measure. Records time, joint positions, velocities and torques (and the
    external wrench if available) while the arm holds still, then prints the noise floor."""
    dt = 1.0 / rate_hz
    rows = []
    t0 = io.clock()
    n = int(seconds * rate_hz)
    for _ in range(n):
        s = io.read()
        w = s.get("wrench")
        rows.append(np.r_[io.clock() - t0, s["q"], s["dq"], s["tau"], np.zeros(6) if w is None else w])
        io.wait(dt)
    rows = np.array(rows)
    names = ["t"] + [f"q{i}" for i in range(1, 8)] + [f"dq{i}" for i in range(1, 8)] \
        + [f"tau{i}" for i in range(1, 8)] + ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
    np.savetxt(out_csv, rows, delimiter=",", header=",".join(names), comments="", fmt="%.6f")
    tau, q, wr = rows[:, 15:22], rows[:, 1:8], rows[:, 22:28]
    real_rate = (n - 1) / max(rows[-1, 0], 1e-9)
    print(f"\n{n} samples at {real_rate:.1f} Hz saved to {out_csv}")
    print("joint        " + "".join(f"{i:>9d}" for i in range(1, 8)))
    print("q mean  [rad]" + "".join(f"{v:9.4f}" for v in q.mean(axis=0)))
    print("tau mean [Nm]" + "".join(f"{v:9.3f}" for v in tau.mean(axis=0)))
    print("tau std  [Nm]" + "".join(f"{v:9.4f}" for v in tau.std(axis=0)))
    half = n // 2
    drift = tau[half:].mean(axis=0) - tau[:half].mean(axis=0)
    print("tau drift[Nm]" + "".join(f"{v:9.4f}" for v in drift) + "   (2nd half minus 1st half)")
    if np.any(wr):
        print(f"wrench std: Mx {wr[:, 3].std():.4f}  My {wr[:, 4].std():.4f} Nm   Fz {wr[:, 2].std():.3f} N")
    T_ee_paddle = calib.T_ee_paddle if calib else paddle_transform()
    k = GOLF_BALL_MASS * G
    sensor = BallSensor("tau", T_ee_paddle)
    _, cov = sensor.locate(q.mean(axis=0), np.zeros(7), k, np.maximum(tau.std(axis=0), 1e-4))
    sig = np.sqrt(np.diag(cov)) * 100
    c, A, _ = sensor.model(q.mean(axis=0))
    print(f"\nA golf ball moved 1 cm changes the joint torques by at most "
          f"{np.max(np.abs(A)) * k * 0.01 * 1000:.1f} mNm.")
    print(f"With this noise one sample locates a golf ball to about +/-{sig[0]:.1f} cm (x), "
          f"+/-{sig[1]:.1f} cm (y); the filter averages roughly 10-20 samples.")
    return rows


def run_tare(io, calib, seconds=60.0, log_name=None):
    """Step 2: rock the paddle with NO ball and fit the no-ball torque model.
    The motion mixes slow sines with small random steps so the model sees the same kind
    of movement the balance controller produces."""
    ses = Session(io, calib, log_name)
    ses.start()
    try:
        io.ask("TARE: REMOVE the ball, hands off the arm. The paddle will rock for "
               f"{seconds:.0f} s. Press Enter: ", ball=None)
        amp = 0.8 * calib.tilt_max
        rng = np.random.default_rng(0)
        step = np.zeros(2)
        Phi, Y = [], []
        n = int(seconds / ses.dt)
        for i in range(n + int(2.0 / ses.dt)):
            t = i * ses.dt
            ramp = min(1.0, t / 3.0, max(0.0, (seconds - t) / 3.0))
            sine = np.array([
                0.5 * np.sin(2 * np.pi * 0.13 * t) + 0.3 * np.sin(2 * np.pi * 0.41 * t + 1.0)
                + 0.2 * np.sin(2 * np.pi * 0.97 * t + 2.0),
                0.5 * np.sin(2 * np.pi * 0.17 * t + 0.5) + 0.3 * np.sin(2 * np.pi * 0.53 * t + 2.0)
                + 0.2 * np.sin(2 * np.pi * 1.13 * t)])
            if i % int(0.3 / ses.dt) == 0:
                step = rng.uniform(-1.0, 1.0, 2)
            target = amp * ramp * (0.5 * sine + 0.5 * step)
            _, phi, y = ses.cycle(target if i < n else [0.0, 0.0])
            if t > 1.0:
                Phi.append(phi)
                Y.append(y)
        tare = Tare.fit(np.array(Phi), np.array(Y))
    finally:
        ses.stop()
    calib.tare_W, calib.tare_sigma = tare.W, tare.sigma
    _, cov = ses.sensor.locate(calib.q0, np.zeros(ses.sensor.n), calib.k, tare.sigma)
    sig = np.sqrt(np.diag(cov)) * 100
    print("\nno-ball model fitted. Residual per channel "
          f"({'Nm, joints 1-7' if calib.source == 'tau' else 'Nm, Mx My'}):")
    print("  " + "  ".join(f"{v:.4f}" for v in tare.sigma))
    print(f"=> single-sample ball position noise while moving: +/-{sig[0]:.1f} cm, +/-{sig[1]:.1f} cm "
          f"(paddle half-width is {PLATE_HALF * 100:.1f} cm)")
    if max(sig) > 6.0:
        print("   That is too coarse to balance this ball. Options: heavier ball (see README), "
              "slower loop gains, or try --source wrench / --source tau.")
    return calib


def run_calibrate(io, calib, mark=0.08, seconds=4.0, log_name=None):
    """Step 3: the experiment itself. Measure the torque change a ball causes at two known
    spots, and from it get the sign and scale (k = m*g) and the centre offset."""
    ses = Session(io, calib, log_name)
    ses.start()
    try:
        ses.rezero(seconds)
        io.ask("Put the ball on the CENTRE mark (a washer or a dab of putty keeps it from rolling), "
               "let go, press Enter: ", ball=(0.0, 0.0), hold=True)
        d1, s1, q1 = ses.hold(seconds)
        io.ask(f"Move the ball {mark * 100:.0f} cm from the centre along paddle +x "
               "(away from the handle), let go, press Enter: ", ball=(mark, 0.0), hold=True)
        d2, s2, q2 = ses.hold(seconds)
    finally:
        ses.stop()

    c, A, _ = ses.sensor.model(q1)
    w = 1.0 / np.square(ses.tare.sigma)
    step = A @ ses.to_base([mark, 0.0])                      # expected change per unit k
    k = float((w * step) @ (d2 - d1) / ((w * step) @ step))
    nominal = GOLF_BALL_MASS * G
    if calib.source == "tau":
        print("\ntorque change caused by the ball [mNm]      joint:" + "".join(f"{i:>8d}" for i in range(1, 8)))
        print("  measured, ball at centre                       " + "".join(f"{v * 1000:8.1f}" for v in d1))
        print("  model (golf ball) at centre                    " + "".join(f"{v * 1000:8.1f}" for v in nominal * c))
        print(f"  measured, centre -> +{mark * 100:.0f} cm                       "
              + "".join(f"{v * 1000:8.1f}" for v in d2 - d1))
        print(f"  model (golf ball), centre -> +{mark * 100:.0f} cm              "
              + "".join(f"{v * 1000:8.1f}" for v in nominal * step))
        print("  noise (1 sigma, single sample)                 " + "".join(f"{v * 1000:8.1f}" for v in s1))
    else:
        print(f"\nmoment change [mNm]: centre {np.round(d1 * 1000, 1)}, "
              f"centre -> +{mark * 100:.0f} cm {np.round((d2 - d1) * 1000, 1)}")
    print(f"\nfitted k = {k:+.3f} N  (a golf ball is {nominal:.3f} N; mass by torque = {abs(k) / G * 1000:.0f} g)")
    if abs(k) < 0.3 * nominal:
        raise SafetyStop("the ball's torque signature was not found (k is tiny). Check the marks, "
                         "the paddle axes in ball_core.py, and the noise numbers from the tare step.")
    xy1, _ = ses.sensor.locate(q1, d1, k, ses.tare.sigma)
    calib.k, calib.center_offset = k, xy1
    print(f"apparent position of the centre mark: {np.round(ses.to_paddle(xy1) * 100, 1)} cm "
          "(paddle x, y). Stored as the zero.")
    return calib


def run_locate(io, calib, spots=None, seconds=30.0, log_name=None):
    """Step 4: live ball position readout on a level, still paddle. Move the ball around by
    hand and compare with a ruler. `spots` (paddle coordinates) is used by the simulator."""
    ses = Session(io, calib, log_name)
    ses.start()
    out = []
    try:
        ses.rezero()
        if spots is None:
            io.ask("Put the ball anywhere on the paddle and press Enter. Readout every 0.5 s: ", ball="keep")
            for _ in range(int(seconds / 0.5)):
                d, _, q = ses.hold(0.5, settle=0.0)
                xy, _ = ses.sensor.locate(q, d, calib.k, ses.tare.sigma)
                p = ses.to_paddle(xy - calib.center_offset) * 100
                print(f"  ball at paddle x {p[0]:+6.1f} cm   y {p[1]:+6.1f} cm")
        else:
            for spot in spots:
                io.ask(f"ball -> {np.round(np.array(spot) * 100, 1)} cm ", ball=tuple(spot), hold=True)
                d, _, q = ses.hold(2.0)
                xy, _ = ses.sensor.locate(q, d, calib.k, ses.tare.sigma)
                out.append(ses.to_paddle(xy - calib.center_offset))
    finally:
        ses.stop()
    return np.array(out)


def lqr_gain(dt, wc, gain):
    """Discrete LQR for one axis. State [ball position, ball velocity, tilt, tilt rate],
    input = tilt target. The tilt follows its target through the servo's second-order
    filter (bandwidth wc), so the controller knows about that lag instead of fighting it.
    `gain` is roughly the tilt asked per metre of ball error [rad/m]."""
    Ac = np.array([[0.0, 1.0, 0.0, 0.0],
                   [0.0, 0.0, ROLL_ACCEL, 0.0],
                   [0.0, 0.0, 0.0, 1.0],
                   [0.0, 0.0, -wc * wc, -2.0 * wc]])
    Bc = np.array([0.0, 0.0, 0.0, wc * wc])
    A, B = np.eye(4), np.zeros(4)
    for _ in range(50):                       # fine Euler steps over one control period
        B = B + A @ Bc * (dt / 50)
        A = A + Ac @ A * (dt / 50)
    Q = np.diag([1.0, 0.02, 0.0, 0.0])
    R = 1.0 / (gain * gain)
    P = Q.copy()
    for _ in range(5000):
        K = (B @ P @ A) / (R + B @ P @ B)
        P_new = Q + A.T @ P @ A - np.outer(A.T @ P @ B, K)
        if np.max(np.abs(P_new - P)) < 1e-12:
            break
        P = P_new
    return K


def run_balance(io, calib, seconds=30.0, gain=0.6, start=(0.03, 0.0), rezero=True,
                log_name=None, quiet=False):
    """Step 5: closed loop. Torques -> ball position -> Kalman filter -> LQR -> paddle tilt."""
    ses = Session(io, calib, log_name)
    K = lqr_gain(ses.dt, ses.servo.wc, gain)
    kfx, kfy = AxisKF(ses.dt), AxisKF(ses.dt)
    ses.start()
    lost = 0
    try:
        if rezero:
            ses.rezero()
        io.ask("Put the ball near the CENTRE of the paddle, let go, press Enter: ", ball=tuple(start))
        d, _, q = ses.hold(0.4, settle=0.2)
        xy, _ = ses.sensor.locate(q, d, calib.k, ses.tare.sigma)
        xy = xy - calib.center_offset
        kfx.reset(xy[0])
        kfy.reset(xy[1])
        target = np.zeros(2)
        tilt_prev = ses.servo.measured_tilt(q)
        for i in range(int(seconds / ses.dt)):
            state, phi, y = ses.cycle(target, est=np.r_[kfx.x, kfy.x], truth=io.ball_truth())
            q = np.asarray(state["q"], dtype=float)
            dy = y - ses.tare.predict(phi)
            if calib.source == "tau" and np.max(np.abs(dy)) > 3.0:
                raise SafetyStop("large unexpected torque (contact?)")
            tilt = ses.servo.measured_tilt(q)
            # the ball's own acceleration right now, as the filter believes it
            ball_acc = (ROLL_ACCEL * np.sin(tilt[1]) + kfx.x[2], -ROLL_ACCEL * np.sin(tilt[0]) + kfy.x[2])
            z, cov = ses.sensor.locate(q, dy, calib.k, ses.tare.sigma, ses.accel_z, ball_acc)
            z = z - calib.center_offset
            tilt_mid = 0.5 * (tilt + tilt_prev)       # tilt over the interval just elapsed
            tilt_prev = tilt
            # ball accelerates toward +x when the paddle rotates +b about y, toward -y for +a about x
            x = kfx.step(ROLL_ACCEL * np.sin(tilt_mid[1]), z[0], cov[0, 0])
            yv = kfy.step(-ROLL_ACCEL * np.sin(tilt_mid[0]), z[1], cov[1, 1])
            (a_cmd, b_cmd), (da_cmd, db_cmd) = ses.servo.cmd, ses.servo.dcmd
            bias_x, bias_y = x[2] / ROLL_ACCEL, yv[2] / ROLL_ACCEL      # tilt that cancels the disturbance
            b_target = -K @ np.array([x[0], x[1], b_cmd + bias_x, db_cmd]) - bias_x
            a_target = K @ np.array([yv[0], yv[1], -a_cmd + bias_y, -da_cmd]) + bias_y
            target = np.array([a_target, b_target])
            ses._rec(z=z)
            lost = lost + 1 if max(abs(x[0]), abs(yv[0])) > PLATE_HALF + 0.03 else 0
            if lost > int(0.5 / ses.dt):
                raise SafetyStop("ball estimate left the paddle")
            if not quiet and i % int(1.0 / ses.dt) == 0:
                p = ses.to_paddle(np.array([x[0], yv[0]])) * 100
                print(f"  t={i * ses.dt:5.1f}s  ball x {p[0]:+6.1f} cm  y {p[1]:+6.1f} cm   "
                      f"tilt {np.degrees(ses.servo.cmd[0]):+5.2f} {np.degrees(ses.servo.cmd[1]):+5.2f} deg")
    except SafetyStop as e:
        print(f"STOP: {e}. Levelling the paddle.")
    finally:
        ses.stop()
    return {k: np.array(v) for k, v in ses.log.items()}
