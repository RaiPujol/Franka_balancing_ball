"""MuJoCo test bench for the torque-only ball balancer.

Uses the Lab 2 Panda model (torque actuators) and adds a 10 x 10 inch paddle on the
hand plus a free golf ball. The arm is driven the way frankapy's dynamic joint skill
drives the real one (joint impedance + gravity compensation, new joint targets at
50 Hz), and the "torque sensor" is the link-side joint torque with noise added.
It then runs the SAME procedures as the real robot script: tare, calibrate, locate,
balance.

    python sim_ball_balance.py                # full pipeline, no window
    mjpython sim_ball_balance.py --view       # watch it (plain python on Linux)
    python sim_ball_balance.py --noise 2      # double the sensor noise
    python sim_ball_balance.py --ball-mass 0.16 --source wrench
"""

import argparse
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import mujoco as mj

HERE = Path(__file__).resolve().parent
LAB2_DIR = HERE.parent / "Lab2"
sys.path.insert(0, str(LAB2_DIR))
import RobotUtil as rt          # noqa: E402  (Lab 2 helpers: interp_min_jerk)

import ball_core as bc          # noqa: E402

MODEL_DIR = LAB2_DIR / "franka_emika_panda"
ROOT_MODEL_XML = MODEL_DIR / "panda_torque_table.xml"
SCENE_XML = HERE / "panda_paddle_ball.xml"

# Level balance pose used in simulation: hand pointing forward, paddle flat.
Q_BALANCE = np.array([0.0, -0.5, 0.0, -1.9, 0.0, 2.971, -0.785])

# Assumed torque sensor noise, Nm (1 sigma per 1 kHz sample). Scaled by --noise.
# Measure the real numbers with `franka_ball_balance.py log` and put them here.
TAU_NOISE = np.array([0.03, 0.03, 0.03, 0.03, 0.01, 0.01, 0.01])
TAU_DRIFT = 0.3 * TAU_NOISE        # slow wander (thermal, friction), time constant 20 s
DQ_NOISE = 1e-3                    # rad/s, joint velocity noise

BALL_RADIUS = 0.02135
PLATE_THICKNESS = 0.006
PLATE_MASS = 0.25


def build_scene(ball_mass, out_xml=SCENE_XML):
    """Lab 2 Panda + paddle welded to the hand + one free ball. Same ElementTree approach
    as Lab 2's build_scene."""
    tree = ET.parse(ROOT_MODEL_XML)
    root = tree.getroot()
    root.find("compiler").set("meshdir", str(MODEL_DIR / "assets"))
    ET.SubElement(root, "option", {"timestep": "0.001"})

    hand = next(b for b in root.iter("body") if b.get("name") == "hand")
    T = bc.paddle_transform()
    quat = np.zeros(4)
    mj.mju_mat2Quat(quat, T[:3, :3].flatten())
    pos = T[:3, 3] + np.array([0.0, 0.0, 0.1034])           # hand body origin -> EE (TCP)
    paddle = ET.SubElement(hand, "body", {
        "name": "paddle", "pos": " ".join(f"{v:.6f}" for v in pos),
        "quat": " ".join(f"{v:.8f}" for v in quat)})
    h = bc.PLATE_HALF
    # top face of the plate is the paddle frame's z = 0 plane
    ET.SubElement(paddle, "geom", {
        "name": "plate", "type": "box", "size": f"{h} {h} {PLATE_THICKNESS / 2}",
        "pos": f"0 0 {-PLATE_THICKNESS / 2}", "mass": str(PLATE_MASS), "rgba": "0.85 0.65 0.35 1",
        "contype": "0", "conaffinity": "2", "condim": "6", "friction": "0.6 0.005 0.0002"})
    ET.SubElement(paddle, "geom", {
        "name": "handle", "type": "box", "size": f"0.05 0.012 {PLATE_THICKNESS / 2}",
        "pos": f"{-h - 0.05} 0 {-PLATE_THICKNESS / 2}", "mass": "0.03", "rgba": "0.3 0.3 0.3 1",
        "contype": "0", "conaffinity": "0"})

    world = root.find("worldbody")
    ball = ET.SubElement(world, "body", {"name": "ball", "pos": "2 0 0.5"})
    ET.SubElement(ball, "freejoint", {"name": "ball_free"})
    ET.SubElement(ball, "geom", {
        "name": "ball_geom", "type": "sphere", "size": str(BALL_RADIUS), "mass": str(ball_mass),
        "rgba": "1 1 1 1", "contype": "2", "conaffinity": "0", "condim": "6",
        "friction": "0.6 0.005 0.0002"})
    # "putty": pins the ball to a spot on the paddle for the calibration placements
    ET.SubElement(root.find("equality"), "connect", {
        "name": "pin", "body1": "ball", "body2": "paddle", "anchor": "0 0 0", "active": "false"})
    tree.write(out_xml, encoding="utf-8", xml_declaration=True)
    return out_xml


class SimIO:
    """Same interface as FrankaIO in franka_ball_balance.py, backed by MuJoCo."""

    def __init__(self, rate_hz=50.0, noise=1.0, ball_mass=bc.GOLF_BALL_MASS, view=False,
                 realtime=False, seed=0, verbose=True):
        self.model = mj.MjModel.from_xml_path(str(build_scene(ball_mass)))
        self.data = mj.MjData(self.model)
        self.dt_ctrl = 1.0 / rate_hz
        self.substeps = int(round(self.dt_ctrl / self.model.opt.timestep))
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose
        self.realtime = realtime
        self.paddle_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "paddle")
        self.ball_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "ball")
        jid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "ball_free")
        self.ball_qadr = self.model.jnt_qposadr[jid]
        self.ball_dadr = self.model.jnt_dofadr[jid]
        self.pin_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_EQUALITY, "pin")
        self.ball_on = False
        self.drift = np.zeros(7)
        self.q_des = Q_BALANCE.copy()
        self.data.qpos[:7] = Q_BALANCE
        self.data.qpos[7:9] = 0.01
        self._park_ball()
        mj.mj_forward(self.model, self.data)
        self.viewer = None
        if view:
            from mujoco import viewer as mj_viewer
            self.viewer = mj_viewer.launch_passive(self.model, self.data)
            self.viewer.cam.distance = 2.0
            self.viewer.cam.azimuth += 90
        self._wall0 = time.perf_counter()
        self._advance(int(2.0 / self.model.opt.timestep))       # let the arm settle

    # ---- interface used by ball_core -------------------------------------------------
    def read(self):
        m, d = self.model, self.data
        # link-side joint torque: what the actuator delivers to the link after rotor
        # inertia (armature) and joint damping, like Franka's joint torque sensors
        tau_link = d.qfrc_actuator[:7] + d.qfrc_passive[:7] - m.dof_armature[:7] * d.qacc[:7]
        tau = tau_link + self.noise * (TAU_NOISE * self.rng.standard_normal(7) + self.drift)
        # libfranka-style external wrench estimate at the EE: measured torque minus the
        # robot model (which, like the real one, does not know about the ball)
        acc = np.zeros(m.nv)
        acc[:7] = d.qacc[:7]
        m_acc = np.zeros(m.nv)
        mj.mj_mulM(m, d, m_acc, acc)                      # mass matrix times arm acceleration
        tau_model = m_acc[:7] - m.dof_armature[:7] * d.qacc[:7] + d.qfrc_bias[:7]
        _, J = bc.fk(d.qpos[:7])
        wrench = -np.linalg.pinv(J.T) @ (tau - tau_model)
        dq = d.qvel[:7] + self.noise * DQ_NOISE * self.rng.standard_normal(7)
        return {"q": d.qpos[:7].copy(), "dq": dq, "tau": tau, "wrench": wrench}

    def clock(self):
        return self.data.time

    def start_stream(self, q):
        self.q_des = np.asarray(q, dtype=float).copy()

    def stop_stream(self):
        pass

    def command(self, q_des):
        self.q_des = np.asarray(q_des, dtype=float).copy()
        self._advance(self.substeps)

    def wait(self, seconds):
        self._advance(int(round(seconds / self.model.opt.timestep)))

    def goto(self, q, duration):
        q_start, q_goal = self.data.qpos[:7].copy(), np.asarray(q, dtype=float)
        if np.allclose(q_start, q_goal, atol=1e-4):
            duration = 0.5
        steps = int(duration / self.dt_ctrl)
        for i in range(steps):
            self.q_des, _ = rt.interp_min_jerk(q_start, q_goal, (i + 1) * self.dt_ctrl, duration)
            self._advance(self.substeps)

    def ask(self, message, ball="keep", hold=False):
        """On the robot this waits for Enter. Here it moves the ball instead.
        ball: None = take it off, (x, y) = put it there in paddle coordinates, "keep".
        hold: pin the ball to that spot (the washer / putty used on the real paddle)."""
        if self.verbose:
            print(f"[sim] {message}")
        if ball is None:
            self._park_ball()
        elif ball != "keep":
            spot = np.array([ball[0], ball[1], BALL_RADIUS])
            R = self.data.xmat[self.paddle_id].reshape(3, 3)
            p = self.data.xpos[self.paddle_id] + R @ spot
            self.data.qpos[self.ball_qadr:self.ball_qadr + 7] = np.r_[p, 1, 0, 0, 0]
            self.data.qvel[self.ball_dadr:self.ball_dadr + 6] = 0
            self.model.body_gravcomp[self.ball_id] = 0.0
            self.model.eq_data[self.pin_id, 0:3] = 0.0
            self.model.eq_data[self.pin_id, 3:6] = spot
            self._set_pin(hold)
            self.ball_on = True
        self._advance(self.substeps)

    def ball_truth(self):
        """True ball offset from the paddle centre along base x, y (None if off the paddle)."""
        if not self.ball_on:
            return None
        return (self.data.xpos[self.ball_id] - self.data.xpos[self.paddle_id])[:2].copy()

    def ball_fell(self):
        return self.ball_on and self.data.xpos[self.ball_id][2] < self.data.xpos[self.paddle_id][2] - 0.05

    # ---- internals -------------------------------------------------------------------
    def _set_pin(self, on):
        active = self.data.eq_active if hasattr(self.data, "eq_active") else self.model.eq_active
        active[self.pin_id] = 1 if on else 0

    def _park_ball(self):
        self._set_pin(False)
        self.data.qpos[self.ball_qadr:self.ball_qadr + 7] = [2, 0, 0.5, 1, 0, 0, 0]
        self.data.qvel[self.ball_dadr:self.ball_dadr + 6] = 0
        self.model.body_gravcomp[self.ball_id] = 1.0
        self.ball_on = False

    def _advance(self, n):
        m, d = self.model, self.data
        for _ in range(n):
            # 1 kHz joint impedance + gravity compensation, as in Lab 2's controller
            tau = bc.K_GAINS * (self.q_des - d.qpos[:7]) - bc.D_GAINS * d.qvel[:7]
            d.ctrl[:7] = tau + d.qfrc_bias[:7]
            d.ctrl[7] = 0.005
            mj.mj_step(m, d)
            self.drift += (-self.drift / 20.0) * m.opt.timestep \
                + TAU_DRIFT * np.sqrt(2 * m.opt.timestep / 20.0) * self.rng.standard_normal(7)
        if self.viewer is not None:
            self.viewer.sync()
            if self.realtime:
                delay = d.time - (time.perf_counter() - self._wall0)
                if delay > 0:
                    time.sleep(delay)


def main():
    ap = argparse.ArgumentParser(description="Simulated torque-only ball balancing.")
    ap.add_argument("--view", action="store_true", help="Open the MuJoCo viewer (mjpython on macOS).")
    ap.add_argument("--noise", type=float, default=1.0, help="Scale on the assumed torque sensor noise.")
    ap.add_argument("--ball-mass", type=float, default=bc.GOLF_BALL_MASS, help="kg. Golf ball = 0.0459.")
    ap.add_argument("--source", choices=["tau", "wrench"], default="tau")
    ap.add_argument("--rate", type=float, default=50.0)
    ap.add_argument("--seconds", type=float, default=30.0, help="Length of the balance run.")
    ap.add_argument("--start", type=float, nargs=2, default=[0.05, -0.04], help="Ball start, paddle x y in m.")
    ap.add_argument("--gain", type=float, default=0.6, help="Tilt per metre of ball error [rad/m].")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    io = SimIO(rate_hz=args.rate, noise=args.noise, ball_mass=args.ball_mass, view=args.view,
               realtime=args.view, seed=args.seed)
    calib = bc.Calib(Q_BALANCE, source=args.source, rate_hz=args.rate,
                     k=args.ball_mass * bc.G)

    print("\n=== 1. tare (no ball) ===")
    bc.run_tare(io, calib)
    print("\n=== 2. calibrate (ball at centre, then +8 cm) ===")
    bc.run_calibrate(io, calib)
    print("\n=== 3. locate (still paddle, ball placed at known spots) ===")
    spots = [(0.0, 0.0), (0.08, 0.0), (-0.08, 0.0), (0.0, 0.08), (0.0, -0.08), (0.06, 0.06), (-0.10, -0.05)]
    est = bc.run_locate(io, calib, spots=spots)
    for s, e in zip(spots, est):
        print(f"  true ({s[0] * 100:+5.1f}, {s[1] * 100:+5.1f}) cm   estimated ({e[0] * 100:+5.1f}, {e[1] * 100:+5.1f}) cm")
    print(f"  rms error {np.sqrt(np.mean(np.sum((est - np.array(spots)) ** 2, axis=1))) * 100:.2f} cm")

    print("\n=== 4. balance ===")
    log = bc.run_balance(io, calib, seconds=args.seconds, start=args.start, gain=args.gain,
                         log_name=str(HERE / "logs" / "sim_balance.npz"))
    if "truth" in log:
        r = np.linalg.norm(log["truth"], axis=1) * 100
        fell = r.max() > bc.PLATE_HALF * 100 * np.sqrt(2)
        tail = r[len(r) // 2:]
        print(f"\nTRUE ball position: {'FELL OFF' if fell else 'stayed on'} the paddle. "
              f"distance from centre: start {r[0]:.1f} cm, max {r.max():.1f} cm, "
              f"rms over the second half {np.sqrt(np.mean(tail ** 2)):.1f} cm")
    if io.viewer is not None:
        io.viewer.close()


if __name__ == "__main__":
    main()
