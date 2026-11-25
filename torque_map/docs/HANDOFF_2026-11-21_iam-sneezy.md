# Torque-sensed ball balancer: handoff (state on Nov 21, 2026)

A ball rolls in a rail on a paddle held by a Franka arm (iam-sneezy). There is **no camera**: the ball's
position comes from the arm's **joint torque sensors**, because the ball's weight times its lever arm shows
up as a torque change on every joint whose axis is horizontal and across the rail. One wrist joint tilts
the paddle to keep the ball at a goal. Everything runs from a web page (`app.py`, http://localhost:8000).

This file is the starting point for moving to a **newer arm and a better rail**. Read sections 5 and 7 first.

## 1. What is in the package

| Folder / file | What it is |
|---|---|
| `torque_map/` | **The current system.** Web UI, rail balancer, calibration, all robot logs of Nov 20-21 |
| `torque_map/rail.py` | One-axis rail balancer: geometry from the pose, rail map, tilt sweep, probe, balance loop, drop handling |
| `torque_map/kin.py` | Panda kinematics, tilt servos, Kalman filter, LQR, no-ball model features |
| `torque_map/tm_core.py` | Calibration storage, the streaming `Runner` (safety checks), the original 2-axis blocks (tabs 1-3) |
| `torque_map/app.py`, `ui.html` | Web server (jobs, live tuning, presets) and the page |
| `torque_map/robot_io.py` | frankapy calls (state, goto, joint streaming, status). The only robot-specific I/O |
| `torque_map/sim_io.py` | MuJoCo stand-in. **Needs `sim_ball_balance.py` + `panda_tm_scene.xml` from the laptop copy of torque_balance; they are not in this copy** |
| `torque_map/tm_calib.json` | Saved pose, rail map, no-ball model, settings. **Arm-specific: do not reuse on a new arm** |
| `torque_map/logs/` | Every run (`.npz`) + `session_*.log` (all page log lines). See `LOGS_INDEX.md` |
| `torque_balance/` | The earlier command-line package (stages home/grid/track/tare/fixedball/balance, `GUIDE.pdf`). Kept for reference; torque_map's simulator imports from it |
| `original_torque_map_before_nov20/` | torque_map as it was before this session (code only) |
| `tools/` | Offline fake robots and the scripts that drove the robot from a terminal (section 9) |
| `claude_conversation/` | The full Claude Code session: readable `transcript.md` + raw `.jsonl` (e-mail redacted) |

## 2. Station bring-up (lab PC iam-camilo, control PC iam-sneezy)

1. Franka Desk: joints unlocked (blue light), user stop released.
2. `roscore` running on the workstation.
3. Control PC server: `cd ~/Documents/frankapy && bash ./bash_scripts/start_control_pc.sh -u student -i iam-sneezy`
   - **`-i iam-sneezy`, not `sneezy`**: `sneezy` does not resolve, the script still prints "Done", nothing starts.
   - Check in the container: `rosnode list` must show `robot_state_publisher_node_1`, `get_current_robot_state_server_node_1`, ...
   - On Nov 21 the **gripper server did not start**. The app now detects that and starts without the gripper
     (one line in the terminal; the Close/Open gripper buttons then refuse). Balancing does not need it.
4. In the container: `cd /home/ros_ws/src/devel_packages/torque_map && python3 app.py` -> "UI ready" -> open http://localhost:8000.
5. Python loads the code once: **after any code change, Ctrl-C and restart `app.py`** (the page shows a red banner when the code is newer).

## 3. Workflow (tab "Rail · one axis")

1. **Pose** (tab 1): *Go to SAVED balance pose* (keeps calibration) or set a new pose and *Use current pose* (saves it, **clears all calibration**).
2. **Settings**: rail axis (y = left/right, x = toward/away), hand pose (normal / turned 90°), sensing joints (computed from the
   geometry), rail points (mm from C). *Save*.
3. **A · Track test** (optional, no ball): does the tilting joint follow its command (gain, lag)?
4. **B · Rail map**: ball off, then **C first**, then the other points. While you place the ball, the deck drawing shows where
   the torques say it is (physics, no map needed, relative to C, ~5 mm). Hold it still, hands off, Continue. Then ball off.
   Gate: slopes 60-130 % of the kinematic value, point error < 6 mm.
5. **C · Tilt sweep**: **BALL OFF** (a sweep with the ball on corrupts the no-ball model), 40 s. Gate: < 20 mm per sample.
6. **D · Balance (tune live)**: *Probe* (paddle held level, shows the estimate; use it to set the **level trim** with a phone
   level along the rail) or *Start rail balancing*. Presets: **smooth rail** (recommended), PD, LQR. Every field in D applies
   live (~0.1 s). The ball is anchored at C at the start; if it drops, the run pauses and asks you to put it back.

Redo map + sweep after changing: pose, axis, hand pose, sensing joints, rate, wrist stiffness, paddle or rail.

## 4. Current settings (end of Nov 21)

- Pose: saved balance pose (lab home, levelled), hand normal, **y rail, joint 5 tilts, joint 3 senses** (the only option at
  this pose on this arm, see 5.2).
- Preset **smooth rail**: PD Kp 1.0, Kv 2.0, max ball speed 20 mm/s, max counter-angle 1.2°, look-ahead 0.35 s, tilt filter
  15 rad/s, rate cap 10°/s, **acceleration cap 40°/s²**, Kalman accel noise 0.2, offset drift 0.02, no stiction model, no kick,
  unstick on, drop detection on, balance stops off (runs only stop on tilt limit / joint window / contact / frozen robot).
- Last trial (16:50): same + I-term Ki 0.3 (max 1°) and max counter-angle 1.6°. Result (incomplete): end of this file.

## 5. What we learned (with numbers): carry this to the new setup

### 5.1 Sensing the ball with joint torques
- Ball weight 0.193 kg -> **18.9 mNm per cm** of ball travel on a joint whose axis is horizontal and perpendicular to the rail
  (`kin.py` computes it; the map checks the measured slope against it). Measured slopes were 88-99 % of that.
- **The joint that tilts cannot sense.** Joint 5's torque followed its own motion (±0.3 Nm, flipping with direction) and kept
  a memory (90 mm offset before/after a map). Same for joint 6 on Nov 20. The balancer never uses the tilting joint as a sensor.
- At the lab home pose with a **y rail**, only **joint 3** (axis tilted 45°) sees the ball: 13.4 mNm/cm, readings quantized in
  24 mNm steps, ~20 mm per sample. With an **x rail** (or the hand turned 90° so a y rail points at the robot), **joints 2 + 4**
  both see it at the full 18.9 mNm/cm: **7 mm per sample** (sweep at 02:21 on Nov 21). That is the geometry to aim for.
- **Arm vibration vs tilt acceleration** (joint 3, Nov 20 logs): ~25 mNm below 5°/s², ~50 at 30°/s², ~100 at 100°/s², ~180 at
  300°/s². Jerky tilts blind the sensor -> the tilt servo has an acceleration cap (40°/s²).
- The **robot state only refreshes at ~28 Hz** here (72 % of the 100 Hz reads repeat the last state; `fresh` in the logs).
  The filter and the no-ball model use fresh samples only. Check this on the new arm (section 7).
- A **high-pass of the sensing joint's own angle** cancels its controller vibration (60 -> 21 mNm on joint 3).
- **Drops are invisible to the position estimate** (an empty paddle reads like a ball near C). Ball presence comes from
  **joint 4** (+1.0 Nm with the ball on). That is how the run now detects drops.

### 5.2 Actuation on iam-sneezy (arm-specific: check the new arm!)
- **Joint 6 does not respond to commands** (README "Nov 17 joint 6 fault"; Nov 21 02:26 and 02:33: target moved 4.3°, joint
  still, torque flat at 3.1 Nm). Logs: `railbalance_y_1121_022555.npz`, `railprobe_y_1121_022947.npz`. That ruled out the turned-hand pose.
- **Joint 5 sticks** now and then (needs ~5 Nm to break away; its torque follows 56-62 Nm/rad of error, a quarter of its 250
  Nm/rad stiffness). The tilt servo has a stiction push (up to 3° extra while the joint is stuck).
- Tilt reached the command ~0.08-0.12 s (arm) + filter delay later; total loop delay ~0.5 s with sensing. Look-ahead 0.35 s.
- frankapy's streaming skill ends after its duration -> 3600 s now, renewed during long prompts (a 45 min wait once killed it).

### 5.3 Rail and deck
- The first rail was **sticky** (ball needed 1.6-2° to start rolling) -> drift, then overshoot, then drops. It was rebuilt on
  Nov 21 and now rolls freely. On a smooth rail, small tilts (≤1.2-1.6°) are enough and large ones only overshoot.
- The deck **leans ~0.5° to the left** at the saved pose. On a smooth rail that alone rolls the ball off in a few seconds.
- A filter that starts "wide open" invented a 2° lean and 190 mm/s in the first 0.5 s -> drop 3-4 s after every start.
  Fixed: confident start (speed 0 ± 5 mm/s, offset 0 ± 0.3°) + 2 s soft start of the tilt.

### 5.4 Results timeline (real robot)

| When | Setup | Result |
|---|---|---|
| Nov 20 19:40-20:01 | original 2-axis code | map broken (joint 6 at 108 mNm/cm), balance 1-20 s |
| 20:44-21:09 | y rail, joint 5 sensing | map FAIL, 145 mm noise, 0.7 Hz self-oscillation |
| 21:54 | joint 3 sensing, PD | first 56 s run; 410 ms tilt lag (330 ms from my filter) |
| 23:22 | aggressive LQR | 11 mm rms for 19 s, contact stops (1-sample spikes from the 28 Hz state) |
| 23:31-23:35 | sweep jerky vs accel-capped | 154 vs 24 mNm -> acceleration cap |
| Nov 21 01:19-01:57 | sticky rail, ramp kick, drop detection | recovery from the ends 15 s -> 2 s; holds 3-32 s |
| 02:10-02:33 | hand turned 90°, joints 2+4 | sensing 7 mm/sample; **joint 6 does not follow** -> abandoned |
| 16:08 | smooth rail, new map | slope 99 % of kinematic, point error 5 mm |
| 16:28 | smooth, PD vs LQR | PD held 33 s; LQR dropped every 3-5 s |
| 16:33 | smooth, gentle PD (1.2° max) | holds 46 s and 32 s |
| 16:46 | + start fix | holds 8.5, 22, 21, 38+ s; 3 drops in 90 s |

## 6. Known limitations / open items
- y rail + joint-3-only sensing limits holds to ~20-45 s here. The x rail (joints 2+4) should be clearly better on a healthy arm.
- The 2-axis path (tabs 2-3, `tm_core.job_balance`) did not get any of these fixes.
- Log arrays: per-cycle (`t, q, tau, tilt, tilt_cmd, q_target, dq, fresh`) vs per-balancing-step (`p_est, v_est, tilt_target, ...`).
  Line them up with `cycle_t` (added Nov 21 17:00). Older logs: the drop prompts sit between them.
- `tools/rb.py analyze` lines per-step data up from the end: wrong after a drop in old logs.
- The simulator stand-in needs files that are not in this copy (section 1).

## 7. Migrating to the new arm and rail: checklist

**Code to check for a different arm**
- [ ] `kin.py`: DH table `_DH`, flange/TCP offset `_link7_to_ee` (0.107 + 0.1034 m, -45°), joint limits `JOINT_MIN/MAX`,
      impedance gains `K_GAINS/D_GAINS`, `HOME_POSE`. A Franka FR3 has different joint limits than a Panda: check its datasheet.
- [ ] `robot_io.py`: frankapy calls (`get_robot_state` keys `joints / joint_velocities / joint_torques`, `goto_joints`,
      dynamic joint streaming, `stream_status`). Another robot or SDK means porting only this file.
- [ ] Paddle/ball constants in `kin.py`: `PADDLE_CENTER_IN_EE` (deck centre 125 mm out, 20 mm below the fingertips),
      `PADDLE_AXES_IN_EE`, ball mass/radius, deck size. Measure them for the new mount.
- [ ] Start with a **fresh `tm_calib.json`** (or *Use current pose*, which clears everything).

**First hour on the new arm**
- [ ] State rate: run a probe, look at `fresh` in the log (25 % = 28 Hz here). Faster is better; the code handles both.
- [ ] Track test on the tilting joint: gain 0.85-1.15, lag < 120 ms, and that it **moves on small commands** (joint 6 did not here).
- [ ] Torque noise per joint: tilt sweep, ball off. Leftover < 20 mm per sample.
- [ ] Choose the geometry: **two non-moving joints across the rail at full slope** (x rail at the home pose, or hand turned
      90° for a y rail), and a healthy tilting joint. Settings shows which joints will sense.

**Rail**
- [ ] Ball rolls freely at ≤0.3° of tilt, straight, ball resting on the deck (guides do not pinch). Marks at C, ±30, ±60 mm.
- [ ] **End stops** at both rail ends (turns drops into roll-backs; the code rolls the ball back from an end).

**Then**
- [ ] Level: Probe + phone level along the rail + level trim (now applies even if changed during a prompt).
- [ ] Rail map (live display) -> tilt sweep (**ball off**) -> Probe (estimate follows a ball moved with a stick?) ->
      **Load recommended: smooth rail** -> Start rail balancing -> tune live (section 8).

## 8. Tuning hints (live, section D)
- Ball overshoots / paddle slams: lower max counter-angle or Kv; keep the acceleration cap.
- Ball drifts to one side and stays: I-term (Ki 0.2-0.4) or re-level (level trim).
- Sluggish: raise Kp a little. Estimate lags the ball: raise Kalman accel noise or look-ahead.
- Every log line is in `logs/session_<date>.log`; every run records why it ended (`stop_reason`), drop times (`drop_t`).

## 9. Tools (`tools/`)
- `rb.py`: drive the page from a terminal: `state`, `tune '{json}'`, `start <job> '{params}'`, `watch`, `analyze`.
- `schedule.py '<settings json>' <seconds> '[[t, {changes}], ...]'`: start a balance run and apply live changes at given times.
- Fake robots for offline tests (no ROS needed, run from the torque_map folder's parent with `tools/` on the path):
  `fake_rail.FakeIO` (ideal), `fake_real.RealishIO`, `fake_hard.HardIO` (delay, under-reach, noise),
  `real_fake.RealFake` (sticky rail, joint 5 stiction, vibration, 28 Hz), `yaw_test.GenFake` (any pose, lean, ball falls off).
- `*_test.py`, `pick*.py`: the comparisons quoted in the conversation. `export_transcript.py`, `index_logs.py`.

## 10. The Claude Code conversation
`claude_conversation/transcript.md` is a readable export (system context and hidden reasoning removed, tool output shortened).
The raw session `.jsonl` is included with e-mail addresses redacted. Another Claude session on this account
(`devel_packages/mike/franka_balancing_lqr_preflight`) belongs to a different project and is **not** included.

## Result of the last trial (I-term, 16:50)
Smooth-rail PD + Ki 0.3 (max 1°), max counter-angle 1.6°, offset drift 0.02 (`railbalance_y_1121_165054.npz`):
one hold of **19.2 s** (previous run's first hold: 8.5 s), then a drop; the run was stopped while waiting at the drop prompt.
During the hold the integral settled at **-0.44° on average (touching its 1° cap)**, i.e. it found and cancelled the deck
lean, and the ball's average position was **-7 mm** (nearly centred; without the I-term it sat on the left).
Conclusion: keep the I-term (Ki ~0.3) on a deck that is not perfectly level; level the new deck first anyway.
One hold is not a result: repeat it on the new setup against the plain smooth-rail preset (Ki 0).
