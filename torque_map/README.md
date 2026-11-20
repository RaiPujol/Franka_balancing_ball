# Torque map balancer

Balance a ball on the paddle using only the Franka's joint 5 and joint 6 torque sensors,
from a web page. Three blocks:

1. **Init**: put the arm in the balance pose (lab home: gripper down, paddle level, joints 5
   and 6 axes horizontal), automatically or by hand, and check it live.
2. **Torque map**: record the empty paddle, then the ball held still at known points. The
   torque change at each point is the map. Then an empty-paddle tilt sweep: what the
   paddle alone does to the torques when it tilts.
3. **Balance**: drive the joint 5/6 torque change to the value recorded with the ball at
   the centre. The map turns N·m of error into mm of ball error; a Kalman filter and an LQR
   turn that into paddle tilt.

Nothing to install: the robot side uses only the Python standard library, numpy and
frankapy (all in the lab container). The page is plain HTML served by `app.py`.

| File | What it is |
|---|---|
| `app.py` | Starts the robot connection and the web page |
| `ui.html` | The page (3 tabs, editable point table, live plots) |
| `tm_core.py` | The three blocks: pose check, map, tilt sweep, balance loop, safety |
| `kin.py` | Panda kinematics, wrist servo, Kalman filter, LQR, mount and ball constants |
| `robot_io.py` | frankapy calls (same ones the earlier `torque_balance` package used on the lab arm) |
| `sim_io.py` | Simulator stand-in, laptop only (needs mujoco and `../torque_balance` + `../Lab2`) |

## Setup in `kin.py`

- Ball: solid, 74.5 mm, 193 g. Deck: 233 mm square, 162.5 g.
- Mount: gripper straight down, fingers on the tab, deck 2 cm below the fingertips, deck
  centre 125 mm out from the grip point (`PADDLE_CENTER_IN_EE`).
- Default map points (editable in the page), in mm. x is measured from the grip point (middle
  of the fingers on the tab) toward the far edge, y to the left seen from the robot:
  C (125, 0), L (125, 105), R (125, -101), F (226.29, 0), FL (226.29, 105.6), FR (226.29, -102.2).
  A `tm_calib.json` from the first version (x from C) is converted automatically.

## At the lab

Bring up the station exactly as before (Desk unlocked with a blue light, `roscore`, the
control PC server with `-i iam-sneezy`, docker). Copy this folder into
`/home/student/16662_RobotAutonomy/src/devel_packages/` (drop.lol or USB). Then in the
container terminal:

```bash
cd /home/ros_ws/src/devel_packages/torque_map
python3 app.py
```

Open **http://localhost:8000** in Firefox on the user PC. If the page does not load, the
container cannot serve to the host: tell me and I will switch to a terminal UI.

Partner on the e-stop for every button marked **MOVES**. The red STOP button cancels the
running step and levels the paddle; Ctrl-C in the terminal stops everything.

### 1 · Init
1. *Close gripper* on the tab (open it first if needed).
2. *Go to balance pose* (MOVES), or *Guide mode 30 s* then *Use current pose*.
3. The live check must show joint 5 and joint 6 axes under 10° from horizontal: **PASS**.
4. **Level the paddle**: phone level on the deck, then *Far edge up/down* and *Left edge
   up/down* (0.1 to 0.5° steps, MOVES) until it reads 0, then *Use current pose*. The home
   pose has the flange level; a deck that still tilts there is the tab sitting at an angle in
   the fingers or the deck flexing under its own weight. Level first, then map.
5. Optional *Run track test* (MOVES, no ball): each tilt axis rocks alone for 15 s.

Joint errors: by default (Settings: *warn, keep going*) a wrist joint that lags its target
is logged and the run continues. Choose *stop* to stop on it instead. The hard limits always
stay on: max tilt, a 0.35 rad joint window, a 7° gross following error, unexpected torque,
ball estimate off the deck. If joint 6 really sticks, try *wrist stiffness 2x*, and test
balancing on the left/right axis alone (axis y, joint 5) first.

### 2 · Torque map
1. Ball off, hands off: **Record EMPTY**.
2. For each point: ball on the mark, held with putty or a washer, hands off, **Record**.
   Each point shows the change on joints 5 and 6 (a 193 g ball gives about 19 mNm per cm).
   After 3+ points spread in x and y the map fits itself and shows the error per point.
3. Ball off: **Record EMPTY again** (drift check).
4. Gate: static map **PASS** (point error under 10 mm, drift under 10 mm).
5. Ball off: **Run tilt sweep** (MOVES, 60 s). Gate: noise while tilting under 30 mm. This
   is the number that failed on Nov 17 with the golf ball.
6. *Download map CSV* and *Download calibration* for the team.

Edit, add or delete points in the table, then *Save point list* before recording. Changing
a point's coordinates drops its old recording.

### 3 · Balance
Set seconds, gain (start at 0.6; lower if twitchy) and axis (both, or x / y with a rail),
then **Start balancing** (MOVES). It asks for the ball off (3 s re-zero), then for the ball
near the centre: answer with **Continue** in the yellow bar. Watch the torque error go to
zero and the ball estimate on the deck drawing.

Every run saves a log in `logs/`, and `tm_calib.json` holds everything learned.

## Safety built in

Tilt limit (5° default, editable), a joint window of 0.35 rad around the pose, a watchdog
that stops if a wrist joint stays 1.5° off its target for 0.5 s (the Nov 17 joint 6 fault),
a stop on unexpected torque, a stop if the ball estimate leaves the deck. Every stop
levels the paddle first.

## Simulator (laptop)

```bash
python3 app.py --sim          # same page, simulated arm: http://localhost:8000
python3 app.py --sim --view   # plus the MuJoCo window (mjpython on macOS)
```

In the simulator the 193 g ball gives map slopes of 18.5 to 18.8 mNm/cm (expected 18.9),
point errors under 1 mm, and the ball stays within about 1 cm of the centre. The simulator
does not reproduce the extra wrist torque the real arm showed while the paddle moved, so
the tilt sweep on the robot is the real test.
