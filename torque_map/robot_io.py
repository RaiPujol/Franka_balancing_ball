"""frankapy front end for tm_core. Runs in the lab docker container (numpy + frankapy)."""

import time

import numpy as np

import kin

STREAM_SECONDS = 600.0     # longest a single streaming run may last before the skill times out


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

    def wait(self, seconds):
        self._rospy.sleep(seconds)

    def guide(self, seconds):
        """Hand-guiding: the arm can be pushed freely for `seconds`."""
        self.fa.run_guide_mode(seconds)

    def grip(self, close=True):
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
