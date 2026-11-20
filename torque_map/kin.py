"""Kinematics, wrist servo, empty-paddle model, Kalman filter and LQR for the torque-map balancer.

Taken from the earlier torque_balance package, where these parts were tested in simulation
and (servo, streaming, watchdog) on the lab Franka. No frankapy or MuJoCo imports here.
"""

import numpy as np

G = 9.81
ROLL_ACCEL = 5.0 / 7.0 * G            # solid ball rolling without slipping: a = 5/7 g sin(tilt)
BALL_MASS = 0.193                     # kg, the solid 74.5 mm ball
BALL_RADIUS = 0.03725                 # m
PLATE_MASS = 0.1625                   # kg, measured print
PLATE_HALF = 0.1165                   # m, half of the 233 mm deck

# Franka joint impedance gains used while streaming (frankapy defaults).
K_GAINS = np.array([600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0])
D_GAINS = np.array([50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0])

# Paddle mount: gripper straight down, fingers pinch the tab, deck flat 2 cm below the
# fingertips, deck centre 125 mm out from the grip point along hand x (your diagram).
PADDLE_CENTER_IN_EE = (0.125, 0.0, 0.020)
PADDLE_AXES_IN_EE = ((1.0, 0.0, 0.0),        # paddle x: away from the tab (away from the robot)
                     (0.0, -1.0, 0.0),       # paddle y: to the left, seen from behind the robot
                     (0.0, 0.0, -1.0))       # paddle normal (up): -hand z

# Balance pose: lab home pose. Joints 5 and 6 axes both horizontal: joint 6 tilts the
# paddle forward/back, joint 5 left/right, joint 7 only spins it.
HOME_POSE = (0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398)

# Stop if a wrist joint stays this far from its target for this long.
TRACK_TOL = np.radians(1.5)
TRACK_TIME = 0.5


def paddle_transform(center=PADDLE_CENTER_IN_EE, axes=PADDLE_AXES_IN_EE):
    T = np.eye(4)
    T[:3, :3] = np.asarray(axes, dtype=float).T
    T[:3, 3] = center
    return T

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

class SafetyStop(RuntimeError):
    pass

class Excitation:
    """The rocking motion used by tare and fixedball: slow sines mixed with small random
    steps, like what the balance controller produces. Ramps in and out over 3 s."""

    def __init__(self, amp, seconds, seed=0):
        self.amp, self.seconds = amp, seconds
        self.rng = np.random.default_rng(seed)
        self.step = np.zeros(2)

    def __call__(self, i, dt):
        t = i * dt
        ramp = min(1.0, t / 3.0, max(0.0, (self.seconds - t) / 3.0))
        sine = np.array([
            0.5 * np.sin(2 * np.pi * 0.13 * t) + 0.3 * np.sin(2 * np.pi * 0.41 * t + 1.0)
            + 0.2 * np.sin(2 * np.pi * 0.97 * t + 2.0),
            0.5 * np.sin(2 * np.pi * 0.17 * t + 0.5) + 0.3 * np.sin(2 * np.pi * 0.53 * t + 2.0)
            + 0.2 * np.sin(2 * np.pi * 1.13 * t)])
        if i % max(int(0.3 / dt), 1) == 0:
            self.step = self.rng.uniform(-1.0, 1.0, 2)
        return self.amp * ramp * (0.5 * sine + 0.5 * self.step)

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
