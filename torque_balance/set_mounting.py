"""Tell the code how the paddle sits in the gripper. Moves nothing.

Setup this is for: gripper pointing DOWN, fingers pinching a small vertical tab at one edge
of the paddle, paddle flat below the fingertips and sticking out sideways from them.

    python3 set_mounting.py --out 0.13 --drop 0.02

--out   horizontal distance from the middle of the fingertips to the paddle centre [m]
--drop  height of the fingertips above the paddle's top face [m]
--flip  use if the printed check says the paddle points the wrong way

Updates calib.json (keeps the taught pose) and the defaults in ball_core.py, then prints
where the code now thinks the paddle is, so you can check it against the real arm.
"""

import argparse
import json
import os
import re

import numpy as np

import ball_core as bc

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--out", type=float, required=True)
ap.add_argument("--drop", type=float, required=True)
ap.add_argument("--flip", action="store_true")
ap.add_argument("--calib", default=os.path.join(HERE, "calib.json"))
a = ap.parse_args()

with open(a.calib) as f:
    d = json.load(f)
q0 = np.array(d["q0"])
T, _ = bc.fk(q0)
R, p = T[:3, :3], T[:3, 3]

# The paddle sticks out along +/- hand x. Default: the sign that points away from the robot base.
s = 1.0 if R[:2, 0] @ p[:2] >= 0 else -1.0
if a.flip:
    s = -s
axes = ((s, 0.0, 0.0), (0.0, -s, 0.0), (0.0, 0.0, -1.0))   # paddle x, y, z (normal) in hand coords
center = (s * a.out, 0.0, a.drop)                          # hand z points down, so +drop is below

d["paddle_axes"], d["paddle_center"] = [list(r) for r in axes], list(center)
with open(a.calib, "w") as f:
    json.dump(d, f, indent=1)

src_path = os.path.join(HERE, "ball_core.py")
src = open(src_path).read()
src = re.sub(r"PADDLE_CENTER_IN_EE = \([^)]*\)", "PADDLE_CENTER_IN_EE = (%.4f, 0.0, %.4f)" % (center[0], center[2]), src, count=1)
src = re.sub(r"PADDLE_AXES_IN_EE = \(\([^)]*\),\s*\([^)]*\),\s*\([^)]*\)\)",
             "PADDLE_AXES_IN_EE = ((%.1f, 0.0, 0.0),\n                     (0.0, %.1f, 0.0),\n"
             "                     (0.0, 0.0, -1.0))" % (s, -s), src, count=1)
open(src_path, "w").write(src)

Tp, _ = bc.fk(q0, bc.paddle_transform(center, axes))
off = np.degrees(np.arccos(np.clip(Tp[2, 2], -1, 1)))
away = Tp[:2, 3] - p[:2]
print(f"paddle normal is {off:.1f} deg from vertical (must be under 10)")
print(f"paddle centre [m, robot base frame]: x {Tp[0, 3]:.3f}  y {Tp[1, 3]:.3f}  z {Tp[2, 3]:.3f}")
print(f"fingertips    [m, robot base frame]: x {p[0]:.3f}  y {p[1]:.3f}  z {p[2]:.3f}")
print("so the code thinks the paddle sticks out from the fingers "
      f"{'AWAY FROM' if away @ p[:2] > 0 else 'TOWARD'} the robot base"
      f" (x {away[0]*100:+.0f} cm, y {away[1]*100:+.0f} cm).")
print("If that is the wrong way, run this again with --flip.")
print("calib.json and ball_core.py updated. Next: python3 franka_ball_balance.py home")
