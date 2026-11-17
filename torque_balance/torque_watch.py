import time
import numpy as np
from frankapy import FrankaArm

fa = FrankaArm()
tau = lambda: np.array(fa.get_robot_state()["joint_torques"])

def avg(seconds):
    t0, s = time.time(), []
    while time.time() - t0 < seconds:
        s.append(tau()); time.sleep(0.01)
    return np.array(s)

print("Hands off the paddle, zeroing for 2 s ...")
a = avg(2.0)
zero = a.mean(axis=0)
print("noise [mNm] " + "".join(f"{v*1000:8.1f}" for v in a.std(axis=0)))
print("joint       " + "".join(f"{i:>8d}" for i in range(1, 8)))
print("torque change [mNm] per joint, Ctrl-C to stop")
try:
    while True:
        print("\r            " + "".join(f"{v*1000:+8.0f}" for v in (avg(0.1).mean(axis=0) - zero)), end="", flush=True)
except KeyboardInterrupt:
    print()
