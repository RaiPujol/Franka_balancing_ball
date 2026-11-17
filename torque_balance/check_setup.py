"""Check the lab setup step by step, without moving the arm.

Run inside the lab docker container:

    python3 check_setup.py

It stops at the first thing that is missing and says which step of the guide to redo.
"""

import sys


def fail(message):
    print("\nFAILED: " + message)
    sys.exit(1)


def ok(message):
    print("  ok   " + message)


print("1. Python and numpy")
ok("python " + sys.version.split()[0])
try:
    import numpy as np
except ImportError:
    fail("numpy is not installed. You are probably outside the docker container: "
         "open it with  bash terminal_docker.sh  (guide step 9).")
ok("numpy " + np.__version__)

print("2. frankapy (only installed inside the lab docker container)")
try:
    import frankapy  # noqa: F401
    from frankapy import FrankaArm
    from frankapy.proto import JointPositionSensorMessage  # noqa: F401
    from frankapy.proto_utils import make_sensor_group_msg  # noqa: F401
    from franka_interface_msgs.msg import SensorDataGroup  # noqa: F401
except ImportError as e:
    fail(f"{e}. You are outside the docker container. Open it with  bash terminal_docker.sh  (guide step 9).")
ok("frankapy and its message types import")

print("3. roscore")
try:
    import rosgraph
except ImportError:
    fail("ROS python packages missing. You are outside the docker container (guide step 9).")
if not rosgraph.is_master_online():
    fail("cannot reach roscore. Start  roscore  in its own terminal on the user PC (guide step 6). "
         "If roscore is running, close this container terminal and open it again after roscore.")
ok("roscore is reachable")

print("4. control PC server (start_control_pc.sh)")
topics = [t for t, _ in rosgraph.Master("/check_setup").getPublishedTopics("")]
franka = [t for t in topics if "franka" in t]
if not franka:
    fail("roscore is up but no franka topics exist: the control PC server is not running. "
         "Check the 3 windows from start_control_pc.sh for an error, and that you used  -i iam-sneezy  "
         "(the CONTROL PC name, not the user PC) (guide step 7).")
for t in franka[:6]:
    ok(t)

print("5. robot state (waits up to about 10 s)")
try:
    fa = FrankaArm()
    state = fa.get_robot_state()
except (Exception, SystemExit) as e:  # frankapy raises or exits, depending on version
    fail(f"{e}. The server is running but the robot does not answer: unlock the joints in Desk "
         "and get a blue light (guide step 5), then restart the server (guide step 7).")
print("  joints  [rad] " + " ".join(f"{v:7.3f}" for v in state["joints"]))
print("  torques [Nm]  " + " ".join(f"{v:7.3f}" for v in state["joint_torques"]))
print("\nALL GOOD. Next:  python3 torque_watch.py")
