"""Balance a ball on a paddle with the Franka, using only its joint torque sensors.

Runs on the robot PC inside the lab docker container (same setup as Lab 3's
real_robot.py): control PC server up, a partner with a hand on the e-stop.
Needs only numpy + frankapy, and ball_core.py next to this file.

Run the stages in this order. Each prints PASS / MARGINAL / FAIL; do not go on past a FAIL.
Full instructions: GUIDE.pdf.

    python3 franka_ball_balance.py home                 # MOVES: arm to the balance pose, saves calib.json
    python3 franka_ball_balance.py watch                # live torque change per joint (no motion)
    python3 franka_ball_balance.py grid                 # ball held at 5 marks, paddle still: k + accuracy
    python3 franka_ball_balance.py track                # MOVES: does each tilt axis follow? no ball
    python3 franka_ball_balance.py tare                 # MOVES: no ball, learns the paddle's own torques
    python3 franka_ball_balance.py fixedball            # MOVES: ball stuck with putty, does the estimate stay put?
    python3 franka_ball_balance.py balance --axis x     # MOVES: one axis, ball in a rail along x
    python3 franka_ball_balance.py balance              # MOVES: both axes

Ctrl-C at any time cancels the running skill and the arm holds position.
"""

import argparse
import os
import threading
import time

import numpy as np

import ball_core as bc

HERE = os.path.dirname(os.path.abspath(__file__))
STREAM_SECONDS = 600.0     # longest a single run may last before the skill times out


class FrankaIO:
    """The few robot calls ball_core needs, on top of frankapy."""

    def __init__(self, rate_hz):
        # imported here so the file can be read and syntax-checked off the robot
        import rospy
        from frankapy import FrankaArm, SensorDataMessageType
        from frankapy import FrankaConstants as FC
        from frankapy.proto import JointPositionSensorMessage, ShouldTerminateSensorMessage
        from frankapy.proto_utils import make_sensor_group_msg, sensor_proto2ros_msg
        from franka_interface_msgs.msg import SensorDataGroup

        self._rospy = rospy
        self._JointMsg, self._StopMsg = JointPositionSensorMessage, ShouldTerminateSensorMessage
        self._group, self._to_ros, self._types = make_sensor_group_msg, sensor_proto2ros_msg, SensorDataMessageType
        self.fa = FrankaArm()
        self.pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)
        self.rate = rospy.Rate(rate_hz)
        self.dt = 1.0 / rate_hz
        self.streaming = False
        self._id, self._t0, self._last_q = 0, 0.0, None

    # ---- measuring -------------------------------------------------------------------
    def read(self):
        """One robot state: joint angles, velocities, link-side joint torques (tau_J) and
        libfranka's external wrench estimate at the EE (O_F_ext_hat_K, base frame)."""
        s = self.fa.get_robot_state()
        return {"q": np.array(s["joints"]), "dq": np.array(s["joint_velocities"]),
                "tau": np.array(s["joint_torques"]), "wrench": np.array(s["ee_force_torque"])}

    def clock(self):
        return time.time()

    def ball_truth(self):
        return None            # no camera: nobody knows (the simulator overrides this)

    # ---- moving ----------------------------------------------------------------------
    def goto(self, q, duration):
        q = np.asarray(q, dtype=float)
        dist = float(np.max(np.abs(q - self.fa.get_joints())))
        if dist < 0.002:
            return
        self.fa.goto_joints(q.tolist(), duration=max(duration, dist / 0.3))

    def start_stream(self, q):
        """Start frankapy's dynamic joint skill (joint impedance, targets streamed by us)."""
        q = np.asarray(q, dtype=float)
        self.fa.goto_joints(q.tolist(), duration=STREAM_SECONDS, dynamic=True, buffer_time=10,
                            k_gains=bc.K_GAINS.tolist(), d_gains=bc.D_GAINS.tolist())
        self._t0 = self._rospy.Time.now().to_time()
        self._id, self._last_q, self.streaming = 0, q, True

    def command(self, q_des):
        self._publish(q_des)
        self.rate.sleep()

    def stop_stream(self):
        if not self.streaming:
            return
        self.streaming = False
        msg = self._StopMsg(timestamp=self._rospy.Time.now().to_time() - self._t0, should_terminate=True)
        self.pub.publish(self._group(
            termination_handler_sensor_msg=self._to_ros(msg, self._types.SHOULD_TERMINATE)))
        self.fa.stop_skill()

    def wait(self, seconds):
        self._rospy.sleep(seconds)

    def ask(self, message, ball="keep", hold=False):
        """Wait for Enter. Keeps re-sending the last joint target meanwhile so the
        impedance skill keeps holding the paddle. (ball / hold are for the simulator.)"""
        done = threading.Event()

        def keep_alive():
            while not done.is_set() and self.streaming:
                self._publish(self._last_q)
                time.sleep(self.dt)

        worker = threading.Thread(target=keep_alive, daemon=True)
        worker.start()
        try:
            input("\n" + message)
        finally:
            done.set()
            worker.join()

    def _publish(self, q):
        self._id += 1
        self._last_q = np.asarray(q, dtype=float)
        msg = self._JointMsg(id=self._id, timestamp=self._rospy.Time.now().to_time() - self._t0,
                             joints=self._last_q.tolist())
        self.pub.publish(self._group(
            trajectory_generator_sensor_msg=self._to_ros(msg, self._types.JOINT_POSITION)))


def check_kinematics(io):
    """Compare our Panda kinematics with what the robot reports for the EE."""
    pose = io.fa.get_pose(include_tool_offset=False)
    T, _ = bc.fk(io.fa.get_joints())
    err = float(np.linalg.norm(T[:3, 3] - np.asarray(pose.translation)))
    if err > 0.01:
        print(f"WARNING: the robot's EE frame is {err * 100:.1f} cm from the Franka Hand default this "
              "code assumes. Check the end-effector set in Desk; PADDLE_CENTER_IN_EE in ball_core.py "
              "is measured from the default hand TCP.")


def teach(io, path, seconds, args):
    print("Guide mode: hold the arm, move the paddle to where you want to balance, in free space")
    print("away from the table and shelves. Make the paddle LEVEL (use a spirit level or a phone),")
    print("then let go before the time runs out.")
    input(f"Enter to start {seconds:.0f} s of guide mode: ")
    io.fa.run_guide_mode(seconds)
    q0 = io.fa.get_joints()
    calib = bc.Calib(q0, source=args.source, rate_hz=args.rate, tilt_max=np.radians(args.tilt_max),
                     k=args.ball_mass * bc.G)
    T, _ = bc.fk(q0, calib.T_ee_paddle)
    off = np.degrees(np.arccos(np.clip(T[2, 2], -1, 1)))
    margin = float(np.min(np.minimum(q0 - bc.JOINT_MIN, bc.JOINT_MAX - q0)))
    print("balance pose: " + " ".join(f"{v:.4f}" for v in q0))
    print(f"paddle normal is {off:.1f} deg from vertical by the kinematic model, "
          f"closest joint limit {margin:.2f} rad away")
    if off > 10:
        print("That is far from level. Either the pose is off or PADDLE_AXES_IN_EE in ball_core.py "
              "does not match how the paddle sits in the gripper.")
    if margin < 0.25:
        print("A joint is close to its limit; the wrist needs about 0.15 rad of room. Try another pose.")
    calib.save(path)
    print(f"saved to {path}. Next: log, then tare.")


def watch(io, seconds, rate_hz):
    """Live readout of the joint torque CHANGE, for pressing on the paddle by hand.
    The arm is not commanded: it just holds wherever it is. No calibration needed."""
    def average(n):
        tau = np.array([_sample(io) for _ in range(n)])
        return tau.mean(axis=0), tau.std(axis=0)

    def take_zero():
        print("\nzeroing for 1 s, hands off the paddle ...")
        zero, noise = average(max(int(rate_hz), 2))
        print("change in joint torque [mNm]. Enter = zero again, Ctrl-C = quit.")
        print("noise  " + "".join(f"{v * 1000:8.1f}" for v in noise) + "   (1 sigma per raw sample)")
        print("joint  " + "".join(f"{i:>8d}" for i in range(1, 8)))
        return zero

    rezero = threading.Event()

    def keys():
        while True:
            try:
                input()
            except EOFError:
                return
            rezero.set()

    threading.Thread(target=keys, daemon=True).start()
    block = max(int(round(rate_hz / 10.0)), 1)          # one printed line = 0.1 s average
    zero = take_zero()
    peak = np.zeros(7)
    try:
        for _ in range(int(seconds * 10)):
            if rezero.is_set():
                rezero.clear()
                zero, peak = take_zero(), np.zeros(7)
            d = (average(block)[0] - zero) * 1000.0
            peak = np.maximum(peak, np.abs(d))
            print("\r       " + "".join(f"{v:+8.0f}" for v in d), end="", flush=True)
    except KeyboardInterrupt:
        pass
    print("\npeak   " + "".join(f"{v:8.0f}" for v in peak) + "   (largest change seen since the last zero)")


def _sample(io):
    tau = io.read()["tau"]
    io.rate.sleep()
    return tau


def home(io, path, args):
    """Move to the recommended balance pose (lab home) and save it as the balance pose."""
    print("Moving to the balance pose (the lab home pose). Partner on the e-stop.")
    input("Enter to move: ")
    io.goto(np.array(bc.HOME_POSE), 5.0)
    io.wait(1.0)
    q0 = io.fa.get_joints()
    calib = bc.Calib(q0, source=args.source, rate_hz=args.rate, tilt_max=np.radians(args.tilt_max),
                     k=args.ball_mass * bc.G)
    T, _ = bc.fk(q0, calib.T_ee_paddle)
    off = np.degrees(np.arccos(np.clip(T[2, 2], -1, 1)))
    print(f"paddle normal is {off:.1f} deg from vertical by the kinematic model "
          f"(PASS under 5; check with a phone level too). Paddle centre at {np.round(T[:3, 3], 3)} m.")
    if off > 5:
        print("FAIL: the mounting constants do not match. Run set_mounting.py, then home again.")
    calib.save(path)
    print(f"saved to {path}. Next: watch, then grid.")


def main():
    ap = argparse.ArgumentParser(description="Torque-only ball balancing on the Franka.")
    ap.add_argument("step", choices=["grip", "release", "home", "teach", "watch", "log", "grid", "track",
                                     "tare", "fixedball", "calibrate", "locate", "balance"])
    ap.add_argument("--calib", default=os.path.join(HERE, "calib.json"))
    ap.add_argument("--seconds", type=float, default=None, help="Duration of the step.")
    ap.add_argument("--rate", type=float, default=50.0, help="Loop rate in Hz (teach). Robot state updates at 100 Hz.")
    ap.add_argument("--source", choices=["tau", "wrench"], default="tau",
                    help="tau: raw joint torques. wrench: libfranka's external wrench estimate. (teach)")
    ap.add_argument("--tilt-max", type=float, default=5.0, help="Largest paddle tilt in degrees (teach).")
    ap.add_argument("--ball-mass", type=float, default=bc.GOLF_BALL_MASS, help="kg (teach). Golf ball 0.0459.")
    ap.add_argument("--gain", type=float, default=0.6, help="balance: tilt per metre of ball error [rad/m].")
    ap.add_argument("--no-rezero", action="store_true", help="balance: skip the no-ball zeroing.")
    ap.add_argument("--out", default=None, help="log: csv file name.")
    ap.add_argument("--axis", choices=["both", "x", "y"], default="both",
                    help="balance: x = only along x (rail along x, joint 6), y = only along y (joint 5).")
    ap.add_argument("--spot", type=float, nargs=2, default=[0.0, 0.0],
                    help="fixedball: where the ball is stuck, paddle x y in cm.")
    args = ap.parse_args()

    calib = bc.Calib.load(args.calib) if os.path.exists(args.calib) else None
    rate = calib.rate_hz if calib and args.step not in ("teach", "home", "log", "watch") else args.rate
    io = FrankaIO(rate)
    stamp = time.strftime("%m%d_%H%M%S")
    log_name = os.path.join(HERE, "logs", f"{args.step}_{stamp}.npz")

    if args.step == "grip":
        io.fa.close_gripper()
        return
    if args.step == "release":
        io.fa.open_gripper()
        return
    if args.step == "teach":
        teach(io, args.calib, args.seconds or 30.0, args)
        return
    if args.step == "home":
        try:
            home(io, args.calib, args)
        except ValueError as e:
            print(f"frankapy refused the motion: {e}")
        return
    if args.step == "watch":
        watch(io, args.seconds or 300.0, args.rate)
        return
    check_kinematics(io)
    if args.step == "log":
        os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
        out = args.out or os.path.join(HERE, "logs", f"log_{stamp}.csv")
        bc.run_log(io, args.seconds or 10.0, args.rate, out, calib)
        return
    if calib is None:
        ap.error("no calib.json yet: run the teach step first")
    if args.step not in ("tare", "grid", "track") and calib.tare_W is None:
        ap.error("run the tare step first")

    try:
        if args.step == "grid":
            bc.run_grid(io, calib, log_name=log_name)
            calib.save(args.calib)
        elif args.step == "track":
            bc.run_track(io, calib, log_name=log_name)
        elif args.step == "fixedball":
            bc.run_fixedball(io, calib, seconds=args.seconds or 30.0,
                             spot=(args.spot[0] / 100.0, args.spot[1] / 100.0), log_name=log_name)
        elif args.step == "tare":
            bc.run_tare(io, calib, args.seconds or 60.0, log_name)
            calib.save(args.calib)
        elif args.step == "calibrate":
            bc.run_calibrate(io, calib, log_name=log_name)
            calib.save(args.calib)
        elif args.step == "locate":
            bc.run_locate(io, calib, seconds=args.seconds or 30.0, log_name=log_name)
        elif args.step == "balance":
            bc.run_balance(io, calib, seconds=args.seconds or 30.0, gain=args.gain,
                           rezero=not args.no_rezero, log_name=log_name,
                           axis=None if args.axis == "both" else args.axis)
    except bc.SafetyStop as e:
        print(f"STOPPED: {e}")
    except ValueError as e:
        # frankapy raises this when a target is unreachable or inside its virtual walls
        print(f"frankapy refused the motion: {e}. Teach a pose further inside the workspace.")


if __name__ == "__main__":
    main()
