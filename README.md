# Ball on a paddle, sensed with joint torques only

No camera. The ball's weight changes the Franka's joint torque readings; where the
torque shows up tells you where the ball is.

| File | Runs on | Purpose |
|---|---|---|
| `ball_core.py` | both | Kinematics, torque -> ball position, Kalman filter, controller, the 5 procedures |
| `sim_ball_balance.py` | laptop | MuJoCo test bench built on the Lab 2 model (`../Lab2/franka_emika_panda`) |
| `franka_ball_balance.py` | robot PC | frankapy version (numpy + frankapy only, like Lab 3's `real_robot.py`) |

## 1. Try it in simulation (laptop)

```bash
cd ~/Robot_Autonomy/Labs/BallBalance
pip install numpy mujoco
python sim_ball_balance.py                 # tare -> calibrate -> locate -> balance, prints results
mjpython sim_ball_balance.py --view        # same, with the viewer (plain python on Linux)
python sim_ball_balance.py --noise 2       # double the assumed torque noise
python sim_ball_balance.py --ball-mass 0.16
```

The sim drives the arm the way frankapy does (joint impedance, K/D = frankapy defaults,
new joint targets at 50 Hz) and adds noise to the torque readings. The noise level is an
assumption (`TAU_NOISE` at the top of the file). Step `log` on the robot gives you the real one.

## 2. On the robot

Copy `ball_core.py` and `franka_ball_balance.py` to the robot PC. Partner on the e-stop.

```bash
python3 franka_ball_balance.py grip                 # fingers close on the paddle handle
python3 franka_ball_balance.py teach --seconds 30   # guide the paddle level, in free space
python3 franka_ball_balance.py log --seconds 10     # joint angles + torques -> logs/*.csv, noise floor
python3 franka_ball_balance.py tare                 # NO ball. Paddle rocks 60 s (max 4 deg)
python3 franka_ball_balance.py calibrate            # ball at centre, then at +8 cm
python3 franka_ball_balance.py locate               # live position readout, paddle still
python3 franka_ball_balance.py balance --seconds 30
```

What each step tells you:

- **teach**: prints how far from vertical the paddle normal is. More than a few degrees means
  the pose is off or the mounting constants are wrong (see below).
- **log**: `tau std` for joints 5, 6, 7 is the number that decides everything. A golf ball
  moved 1 cm changes the wrist torques by 4.5 mNm. In simulation a golf ball balances
  reliably with 0.01 Nm of wrist noise per sample (1 to 2 cm rms), is marginal at 0.02 Nm
  (fell off in 1 run of 3) and a 160 g ball still holds at 0.04 Nm.
- **tare**: prints the single-sample position noise while the paddle moves. Under about
  3 cm is workable, over 6 cm is not.
- **calibrate**: prints the measured torque change per joint next to the model prediction.
  This is the "does the idea work" check. `fitted k` should be near +0.45 N for a golf ball.
  Mark the paddle centre and a point 8 cm toward the far edge (away from the handle), and
  park the ball on a small washer or a dab of putty so it does not roll during the 4 s read.
- **locate**: move the ball by hand, compare with a ruler.
- **balance**: it first asks you to take the ball off for 3 s (re-zero), then to put it on.

Every run saves a log in `logs/` (`.npz`: time, joints, torques, tilt, estimates).

## Paddle mounting (edit if yours differs)

Top of `ball_core.py`:

- `PADDLE_CENTER_IN_EE`: centre of the paddle's top face in the hand frame. Default
  `(0, 0, 0.167)` = 16.7 cm straight out past the fingertips.
- `PADDLE_AXES_IN_EE`: paddle x, y, z written in hand coordinates. Default = flat handle
  pinched by the fingers, paddle pointing along the hand's z axis.

The centre does not have to be exact: `calibrate` measures the leftover offset.
Best pose: hand pointing forward and horizontal. Then joint 7 rolls the paddle and joint 6
pitches it, and both axes read the ball cleanly.

## Knobs

- `balance --gain 0.6`: tilt per metre of ball error (rad/m). Lower = calmer, slower.
  In simulation 1.0 and above amplified the sensor noise enough to lose the ball.
- `teach --tilt-max 5`: tilt limit in degrees.
- `teach --source wrench`: use libfranka's external wrench estimate instead of raw joint
  torques. Re-run tare and calibrate after changing it. Try it if `tau` is too noisy.
- `teach --rate 100`: loop rate. frankapy publishes robot state at 100 Hz, so that is the ceiling.
- `teach --ball-mass 0.16`: heavier ball. Signal scales with mass; this is the first thing
  to change if the golf ball is too light for your arm's noise.

## What is inside the measurement (found in simulation, all compensated)

1. Arm, hand and paddle weight and inertia: removed by the tare model, a regression of the
   torques on the measured wrist angles and a short history of joint velocities.
2. The ball's own inertia when the paddle pitches: it presses with m(g + a), not mg.
3. A rolling ball pushes back horizontally on the paddle while it accelerates.
4. The impedance controller lags its target by D/K (0.3 s on joint 7 with frankapy's
   default gains). The joint targets carry velocity feed-forward and the controller is an
   LQR that includes the tilt lag.
5. The soft wrist sags under the ball. A slow integrator on the encoder-measured tilt
   closes that gap (it also takes out joint friction offsets on the real arm).

## Safety built in

Tilt limit, joint window of 0.35 rad around the taught pose, stop if the arm does not
follow, stop on an unexpected torque, stop if the ball estimate leaves the paddle. Every
stop levels the paddle first. Ctrl-C cancels the skill.
