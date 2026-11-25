"""frankapy front end for tm_core. Runs in the lab docker container (numpy + frankapy)."""

import time

import numpy as np

import kin

STREAM_SECONDS = 3600.0    # longest a single streaming skill may last before frankapy ends it (Runner renews it)


class FrankaIO:
    """The few robot calls tm_core needs, on top of frankapy. Same calls as the earlier
    torque_balance package, which streamed and read torques on the lab Franka."""

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
        self._wait_for_control_pc()
        # the gripper is only used by the Close/Open buttons. If its server is not running on the control
        # PC, FrankaArm() would wait forever for it (Nov 21 16:00): start without it instead.
        self.has_gripper = self._gripper_running()
        if not self.has_gripper:
            print("robot_io: no gripper server running on the control PC: starting WITHOUT the gripper "
                  "(Close/Open gripper buttons off; balancing does not need it)", flush=True)
        self.fa = FrankaArm(with_gripper=self.has_gripper)
        # Robot state from the 100 Hz TOPIC, not frankapy's get_robot_state() SERVICE: the service is a round
        # trip per call and delivered only ~28 different states per second (72 % of the 100 Hz reads repeated,
        # Nov 20-21, both arms), while the topic carries every 10 ms state.
        import threading
        from franka_interface_msgs.msg import RobotState
        self._state_lock = threading.Lock()
        self._state_msg, self._state_n, self._state_t = None, 0, 0.0
        self._state_sub = rospy.Subscriber("/robot_state_publisher_node_1/robot_state", RobotState,
                                           self._on_state, queue_size=1, tcp_nodelay=True)
        self.pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)
        self.rate = rospy.Rate(rate_hz)
        self.dt = 1.0 / rate_hz
        self.streaming = False
        self._id, self._t0, self._last_q = 0, 0.0, None

    # ---- measuring -------------------------------------------------------------------
    def _on_state(self, msg):
        with self._state_lock:
            self._state_msg, self._state_n, self._state_t = msg, self._state_n + 1, time.time()

    def read(self):
        """The newest robot state (100 Hz topic): joint angles, velocities, link-side joint torques (tau_J),
        libfranka's model-based external joint torques (tau_ext_hat_filtered: arm gravity and dynamics
        removed by Franka's own model) and external wrench at the EE (O_F_ext_hat_K, base frame).
        Falls back to the frankapy service if the topic has been silent for 0.3 s."""
        with self._state_lock:
            m, n, t = self._state_msg, self._state_n, self._state_t
        if m is None or time.time() - t > 0.3:
            s = self.fa.get_robot_state()
            return {"q": np.array(s["joints"]), "dq": np.array(s["joint_velocities"]),
                    "tau": np.array(s["joint_torques"]), "wrench": np.array(s["ee_force_torque"]),
                    "source": "service"}
        return {"q": np.array(m.q), "dq": np.array(m.dq), "tau": np.array(m.tau_J),
                "wrench": np.array(m.O_F_ext_hat_K), "tau_ext": np.array(m.tau_ext_hat_filtered),
                "mode": int(m.robot_mode), "seq": n, "source": "topic"}

    def state_rate(self, seconds=1.0):
        """Messages per second on the state topic (for the start-up log)."""
        n0 = self._state_n
        time.sleep(seconds)
        return (self._state_n - n0) / seconds

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

    def start_stream(self, q, k_gains=None, d_gains=None):
        """Start frankapy's dynamic joint skill (its JointImpedanceFeedbackController with
        these gains, targets streamed by us)."""
        q = np.asarray(q, dtype=float)
        k = kin.K_GAINS if k_gains is None else np.asarray(k_gains, dtype=float)
        d = kin.D_GAINS if d_gains is None else np.asarray(d_gains, dtype=float)
        self.fa.goto_joints(q.tolist(), duration=STREAM_SECONDS, dynamic=True, buffer_time=10,
                            k_gains=k.tolist(), d_gains=d.tolist())
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

    def set_rate(self, rate_hz):
        """Control loop rate for streaming. franka-interface publishes the robot state at
        100 Hz, so faster than 100 Hz only re-reads the same state."""
        self.rate = self._rospy.Rate(rate_hz)
        self.dt = 1.0 / rate_hz

    def stream_status(self):
        """None if the streaming skill is still running; otherwise why not (Franka error text,
        franka-interface not ready, or the skill ended). Cheap enough to call now and then."""
        with self._state_lock:
            m = self._state_msg
        if m is not None and int(m.robot_mode) in (4, 5):
            return "robot in " + ("REFLEX (safety stop)" if int(m.robot_mode) == 4 else "USER STOP (button pressed)")
        try:
            st = self.fa._get_current_franka_interface_status().franka_interface_status
            if st.error_description:
                return f"Franka error: {st.error_description}"
            if not st.is_ready:
                return "franka-interface is not ready (robot stopped: user stop / reflex / error)"
        except Exception:                            # status service missing: do not guess
            return None
        try:
            from actionlib_msgs.msg import GoalStatus
            if self.fa._client.get_state() not in (GoalStatus.ACTIVE, GoalStatus.PENDING):
                return "the streaming skill ended (frankapy goal no longer active)"
        except Exception:
            pass
        return None

    def wait(self, seconds):
        self._rospy.sleep(seconds)

    def guide(self, seconds):
        """Hand-guiding: the arm can be pushed freely for `seconds`."""
        self.fa.run_guide_mode(seconds)

    @staticmethod
    def _wait_for_control_pc(service="/get_current_robot_state_server_node_1/get_current_robot_state_server"):
        """FrankaArm() waits silently and forever for the robot state service. Say what is missing instead
        (Nov 21: twice the control PC server was not running and the page never came up)."""
        import time as _t
        try:
            import rosgraph
            master = rosgraph.Master("/torque_map_check")
        except Exception:
            return
        t0 = _t.time()
        while True:
            try:
                master.lookupService(service)
                if _t.time() - t0 > 1.0:
                    print("robot_io: control PC server found, connecting", flush=True)
                return
            except Exception:
                pass
            if int(_t.time() - t0) % 5 == 0:
                print(f"robot_io: waiting for the control PC server ({_t.time() - t0:.0f} s): no robot state service. "
                      "Start it on the workstation: cd ~/Documents/frankapy && bash ./bash_scripts/start_control_pc.sh "
                      "-u student -i <control PC> (iam-camilo: iam-sneezy, iam-bruno: iam-snowwhite); Desk unlocked, "
                      "user stop released", flush=True)
            _t.sleep(1.0)

    @staticmethod
    def _gripper_running(name="/franka_gripper_1/move/status"):
        """True if some node PUBLISHES the gripper's action status (only the gripper server does)."""
        try:
            import rosgraph
            pubs = rosgraph.Master("/torque_map_check").getPublishedTopics("")
            return any(t == name for t, _ in pubs)
        except Exception:
            return True                                  # cannot tell: behave as before

    def grip(self, close=True):
        if not self.has_gripper:
            raise RuntimeError("no gripper server on the control PC (restart the control PC server with the gripper on)")
        if close:
            self.fa.close_gripper()
        else:
            self.fa.open_gripper()

    def place_ball(self, ball, hold=False):
        pass                   # a person does this on the real robot


    def _publish(self, q):
        self._id += 1
        self._last_q = np.asarray(q, dtype=float)
        msg = self._JointMsg(id=self._id, timestamp=self._rospy.Time.now().to_time() - self._t0,
                             joints=self._last_q.tolist())
        self.pub.publish(self._group(
            trajectory_generator_sensor_msg=self._to_ros(msg, self._types.JOINT_POSITION)))
