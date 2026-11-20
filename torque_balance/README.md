# Ball on a paddle, sensed with joint torques only

No camera. The ball's weight changes the Franka's joint torque readings; where the
torque shows up tells you where the ball is. Full step-by-step lab instructions: `GUIDE.pdf`.

| File | Runs on | Purpose |
|---|---|---|
| `ball_core.py` | both | Kinematics, torque -> ball position, Kalman filter, controller, all stages |
| `franka_ball_balance.py` | robot (lab docker) | frankapy front end: one command per stage |
| `check_setup.py` | robot | Checks container, roscore, control PC server, robot state. No motion |
| `torque_watch.py` | robot | Live torque change per joint (mNm). No motion, frankapy only |
| `set_mounting.py` | robot | Tells the code how the paddle sits in the gripper |
| `sim_ball_balance.py` | laptop | MuJoCo test bench, needs the Lab 2 model in `../Lab2/franka_emika_panda` |

## Stages (each prints PASS / MARGINAL / FAIL; stop at the first FAIL)

```bash
python3 franka_ball_balance.py home                    # MOVES: lab home pose, gripper down, saves calib.json
python3 torque_watch.py                                # noise: joints 5, 6 under 10 mNm
python3 franka_ball_balance.py grid                    # ball held at 5 marks, still: rms < 2 cm, drift < 1 cm
python3 franka_ball_balance.py track                   # MOVES, no ball: each tilt axis follows its command
python3 franka_ball_balance.py tare                    # MOVES, no ball: learns the paddle's own torques
python3 franka_ball_balance.py fixedball --spot 0 0    # MOVES: ball stuck with putty, estimate must stay put
python3 franka_ball_balance.py balance --axis x        # MOVES: one axis, ball in a rail along x (joint 6)
python3 franka_ball_balance.py balance --axis y        # MOVES: rail along y (joint 5)
python3 franka_ball_balance.py balance                 # MOVES: both axes
```

Re-run `tare` whenever the paddle changes (rail on/off, re-gripped, new pose).

## Mounting and pose

Default (top of `ball_core.py`): gripper pointing straight down, the fingers pinch a small
vertical tab at the paddle edge, paddle flat 2 cm below the fingertips, centre 13 cm out
along the hand's x axis. If yours differs: `python3 set_mounting.py --out <m> --drop <m> [--flip]`.

Balance pose: the lab home pose (`HOME_POSE`). There joint 6 tilts the paddle forward/back
(ball along base x), joint 5 tilts it left/right (ball along base y), joint 7 only spins it.
Directions on the paddle: +x away from the robot, +y to the left seen from behind the robot.

## Safety built in

Tilt limit (5 deg, `--tilt-max`), joint window of 0.35 rad around the balance pose, a
watchdog that stops if a wrist joint stays 1.5 deg off its target for 0.5 s (the Nov 17
joint 6 fault), stop on an unexpected torque, stop if the ball estimate leaves the paddle.
Every stop levels the paddle first. Ctrl-C cancels the skill.

## Simulation

```bash
python sim_ball_balance.py                           # track, grid, tare, fixedball, balance
python sim_ball_balance.py --rail x --axis x         # one axis with a rail
python sim_ball_balance.py --stick-joint 6 --stick-at 20   # replay the Nov 17 joint 6 fault
```

The sim uses the torque noise measured on the lab arm (Nov 17). It does NOT reproduce the
extra 0.14-0.19 Nm of wrist torque the real arm showed while the paddle moved, so it is
optimistic for everything under motion. The fixedball stage measures that on the robot.
