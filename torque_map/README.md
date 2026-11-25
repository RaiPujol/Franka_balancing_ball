# torque_map: balancing a ball with joint torques only

A Franka arm holds a paddle with a ball in a rail. **There is no camera.** The ball's position is read from the
arm's own joint torque sensors: the ball's weight, times its distance along the rail, changes the torque on every
joint whose axis is horizontal and across the rail. One wrist joint tilts the paddle to keep the ball at the goal.
Everything runs from a web page.

| | |
|---|---|
| **Best result so far** | ball held **55 s** without a drop, within about ±5–8 mm of the centre (iam-bruno, Nov 21 2026) |
| **Sensing noise** | about 4 mm per sample while the paddle moves, 100 samples per second |
| **Ball** | solid, 74.5 mm, 193 g, in a rail of two sticks on a 233 mm printed deck |
| **Robot station** | lab PC iam-bruno (172.26.18.213), control PC iam-snowwhite |

The measured findings behind every design choice, run by run, are in [LAB_NOTES.md](LAB_NOTES.md).

---

## 1. Quick start

**On the robot station (iam-bruno)**, with Desk unlocked (blue light), FCI on and the user stop released:

```bash
# terminal 1
roscore
# terminal 2: the control PC server (the name must be iam-snowwhite, not snowwhite)
cd ~/Documents/frankapy && bash ./bash_scripts/start_control_pc.sh -u student -i iam-snowwhite
# terminal 3: the lab container, then the app inside it
cd ~/16662_RobotAutonomy && bash run_docker.sh
cd /home/ros_ws/src/devel_packages/torque_map && python3 app.py
```

Open **http://localhost:8000** when the terminal says `UI ready`. Keep a partner on the e-stop for every button
marked **MOVES**.

- If the app prints `waiting for the control PC server`, terminal 2 is not running (or used the wrong `-i` name).
- Python loads the code once: **after any code change, Ctrl-C and restart `app.py`**. The page shows a red banner
  when the files are newer than the running code.
- No gripper server? The app starts without it (balancing does not need the gripper).

**Without a robot** (laptop, simulator): `python3 app.py --sim` (needs MuJoCo and `../torque_balance`).

---

## 2. Set up and calibrate (tab "Rail · one axis")

Do this once per mounting of the paddle or rail, and again after a reboot if the readings look off.

1. **Pose** (tab 1): *Go to SAVED balance pose*. It must say PASS (joint axes horizontal, paddle level).
2. **Settings**: rail along **y**, hand **turned 90°**. The page then senses with **joints 2 + 4** and tilts with
   **joint 6** (the geometry is computed from the pose). Rail points: `-100 -50 0 50 100` mm. *Save*.
   + is **toward the robot** (the back end), − is the front end.
3. **A · Track test** (ball off, 20 s): joint 6 must follow the tilt command (gain ≥ 0.85, lag ≤ 120 ms).
4. **B · Rail map**: ball off, then the ball **held still** on each mark, hands off, *Continue*. The deck drawing
   shows where the torques put the ball, so you can see it sit on its mark before you press Continue.
   Gate: point error under ~10 mm. **Redo it whenever the rail or the ball moved on the paddle.**
5. **C · Tilt sweep**: **ball off**, 40 s. This learns what the empty paddle does to the torques while it tilts.
   Gate: under ~20 mm of ball-equivalent noise. A sweep with the ball on ruins the model.
6. **Level**: start the **Probe** (holds the paddle still), lay a phone level **along the rail** on the deck and
   change **level trim** (±0.1 buttons, live) until it reads 0. A ball placed at C should then stay put.
7. **Roll test** (optional, measures the real rolling gain and the leftover lean): see section 5.

## 3. Balance

In section D press **Load recommended: smooth rail**, then **Start rail balancing**:

1. *Ball OFF, Continue*: 2 s re-zero of the empty paddle.
2. *Ball at C, Continue*: 2 s of readings fix the map offset (the anchor). Balancing starts with a 2 s soft start.

During the run:

- **Every field in section D applies live** (about 0.1 s): gains, limits, filter settings, level trim, goal.
- **Click the rail drawing** where the ball really is to correct the estimate on the spot.
- **If the ball drops** (its weight disappears from both sensing joints) the paddle levels, re-zeroes and asks for
  the ball back at C. **Leave the ball until the yellow bar asks**, so the re-zero sees an empty paddle.
- **If something presses on the paddle** (more than 2.5 Nm for 0.1 s) the run pauses the same way and asks you to
  take the ball off, then to put it back.
- The yellow bar with **Continue** floats at the bottom of the screen; STOP levels the paddle and ends the run.

Every run saves `logs/railbalance_y_<date>_<time>.npz` and appends to `logs/session_<date>.log`.

---

## 4. How it works

Each 10 ms cycle:

```
 joint torques (100 Hz state topic)
   -> minus the no-ball model (tilt sweep)          = the ball's torque on joints 2 and 4
   -> rail map (intercept + slope per joint)         = measured ball position
   -> Kalman filter [position, speed, lean]          = estimated ball position and speed
   -> cascaded PD controller (or LQR)                = wanted paddle tilt
   -> tilt servo (rate and acceleration limits)      = joint 6 target, streamed to the joint impedance controller
```

**Measuring.** The robot state is read from the 100 Hz topic (frankapy's state service only delivers ~28 new states
per second). The no-ball model predicts the empty paddle's torques from the tilt, its speed and acceleration and a
"vibration cancel" term (the fast part of the sensing joints' own angles, capped at ±30 µrad so a load change cannot
fake a 3 Nm reading). The rail map turns the remaining torque into a position; both joints are combined by their noise.

**Filtering.** The Kalman filter predicts the ball with physics, `a = rolling gain · 5/7 g · sin(tilt)`, plus a
slowly learned **lean** (whatever pushes the ball when the paddle is "level"). One-off jumps of more than 60 mm are
skipped for up to 0.3 s. If the estimate and the raw reading disagree by more than 40 mm for 0.3 s, the estimate is
resynced to the (median) reading.

**Controlling.** The ball is predicted `look-ahead` seconds ahead. Wanted speed `v* = −Kp·(p − goal)`, at most
`max ball speed`. Wanted acceleration `a* = Kv·(v* − v) − lean`. Tilt `= asin(a* / (rolling gain · 5/7 g))`, at most
`max counter-angle`. Three rules on top:

- **Still-ball hold**: within `hold mm` of the goal and slower than `hold mm/s`, the paddle eases to the learned level
  instead of reacting to noise (smooth blend; full control again at twice those values). This stops tilt piling up
  against a ball held by stiction.
- **End guard**: within 45 mm of an end, the paddle never tilts toward that end unless the ball comes back faster
  than `max ball speed`.
- **Unstick**: a ball held at an end for 0.3 s gets a tilt ramp toward the centre (up to the max counter-angle) that
  ends **the moment the ball moves inward**.

**Safety and recovery.** Hard limits stay on: max tilt, a joint window around the pose, a frozen-joint check, Franka
reflexes. With `balance stops` off (default for experiments) saturation and off-rail readings do not end the run; a
drop or a press pauses it and asks for the ball.

---

## 5. Settings that matter (section D)

| Setting | Recommended | What it does / when to change |
|---|---|---|
| Kp, Kv | 1.0, 2.0 /s | position and speed gains. Overshoots: raise Kv or lower max ball speed. Sluggish: raise Kp |
| max ball speed | 20 mm/s | the speed the controller steers the ball back with |
| max counter-angle | 1.2° | tilt limit. Higher catches harder pushes but swings the plate more |
| look-ahead | 0.35 s | makes up for the paddle's lag (~0.3 s at each reversal) |
| max tilt accel / rate | 40 °/s², 10 °/s | jerky tilts shake the arm and swamp the torque signal |
| **rolling gain** | 1.0 | real ball acceleration ÷ the 5/7 g model. A ball on two sticks rolls on a smaller radius and speeds up less. Too low: the plate over-swings. Too high: corrections are too weak |
| **level trim** | from the phone level | where "level" is along the rail |
| **hold mm, hold mm/s** | 12, 15 | still-ball hold (0 = off) |
| accel noise, offset drift | 0.2, 0.02 | Kalman filter: how fast it follows the ball / learns the lean |

**Roll test** (measures the rolling gain and the lean directly): start it from a terminal with
`python3 tools/rb.py start rail_rolltest '{"deg": 2.0, "rolls": 3}'` (raise *max counter-angle* to 2.5° first, set
it back after). The page asks you to release the ball 5 cm in from each end; the log line `ROLL TEST: ...` gives the
rolling gain and the level trim to set. Both rail ends hold the ball in a pocket, so never release it from an end.

---

## 6. Troubleshooting

| Symptom | Likely cause | What to do |
|---|---|---|
| The page never appears | control PC server not running | terminal 2; check `rosnode list` shows `robot_state_publisher_node_1` |
| Ball estimate jumps far while the ball is still | load change sprang the arm, or the map is stale | the outlier gate handles single jumps; if it persists, redo map + sweep |
| "drift: the gap between the sensing joints changed" | one joint's zero moved | re-anchor (click the drawing), or redo the sweep |
| Ball always runs to one end | level trim off, or a real slope in the rail | phone level in Probe; roll test |
| Plate swings larger and larger | rolling gain too low or max ball speed too high | raise rolling gain toward 1.0, lower max ball speed |
| Ball falls off an open end | no end stop on the rail | add a low stop (tape, putty) at both ends at ±100 mm |
| Endless "put the ball back" prompts | re-zero learned the ball as empty (fixed Nov 21) | restart the app; leave the ball until the bar asks |

---

## 7. Files

| File | What it is |
|---|---|
| `app.py`, `ui.html` | web server (jobs, live tuning, presets) and the page |
| `rail.py` | the rail balancer: geometry, rail map, tilt sweep, probe, roll test, balance loop, recovery |
| `kin.py` | Panda kinematics, tilt servos, Kalman filter, LQR, no-ball model features |
| `tm_core.py` | calibration storage, the streaming `Runner` (safety checks), the original 2-axis tabs |
| `robot_io.py` | frankapy I/O: 100 Hz state topic, joint streaming, status. The only robot-specific file |
| `sim_io.py` | MuJoCo stand-in for `--sim` |
| `sim_tests/` | **fake-robot tests**: `python3 sim_tests/run_all.py` (~1 min, numpy only) |
| `tools/rb.py` | drive the page from a terminal: state, tune, start jobs |
| `tools/index_logs.py` | one line per log file (kind, length, why it ended, drops) |
| `calibration/` | snapshot of a working calibration (iam-bruno, Nov 21). The live `tm_calib.json` is arm-specific and not in git |
| `LAB_NOTES.md` | findings with numbers, results timeline, open problems |
| `docs/` | earlier notes (the Nov 21 handoff for the iam-sneezy arm) |

`logs/` and `tm_calib.json` stay on the robot PC (they are machine data, ignored by git).

## 8. Changing the code

1. Edit, then run `python3 sim_tests/run_all.py`: every test must PASS.
2. Copy the changed files to the robot PC and restart `app.py`.
3. Commit and push, so every machine and GitHub run the same code.
