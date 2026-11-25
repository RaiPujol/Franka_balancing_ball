# Lab notes: torque-sensed rail balancer

> **Best run: Nov 24 2026 22:15–22:18 on iam-bruno: 175.9 s without a drop, pushed 54 times, back to the centre after
> 48 of them (up to ~220 mm/s, and from the rail end; median recovery 1.4 s).** Code `9a991aa`; settings: smooth
> preset (Kp 1.0, Kv 2.0, max ball speed 20 mm/s, max tilt 1.2°, look-ahead 0.35 s), rolling gain 1.0, level trim
> −0.85°, still-ball hold 12 mm / 15 mm/s, emergency brake 2.5° (used 7 s in total). Calibration:
> `calibration/iam-bruno_2026-11-24.json`; log `railbalance_y_1124_221351.npz` on the robot PC.

What we measured, what failed, and why each setting is what it is. Newest first. All numbers come from the logs in
`torque_map/logs/` on the robot PC (`railbalance_*`, `railmap_*`, `railsweep_*`, `railroll_*`, `session_*.log`).

## Open problems (Nov 24 2026)

1. **The plate swing builds up after ~40 s.** In the best runs the ball holds for 30–55 s, then an end-to-end swing
   grows until the ball drops. Fix in place since Nov 24: the still-ball hold, braking allowed for fast returns, the
   learned lean kept through resyncs (but reset when the ball is put back). On Nov 24 the more aggressive setup
   (rolling gain 0.8, 40 mm/s, 1.5°) swung the plate ±1°: back to the 19:22 values.
2. **The rail map goes stale.** On Nov 21 at 21:08, with the ball at C, joint 2 read about 0 mm and joint 4 about
   +40 mm (94 mm apart at the start of the run). The rail or ball had moved since the 18:57 map. Redo the map after any
   change, and after a reboot.
3. **The rail ends.** Neither end has a stop. The ball wedges in a pocket at both ends (it sat 6 s at the back end
   with the back raised 1.9°) and falls off the front edge. Low stops at ±100 mm would make end events recoverable.
4. **The true rolling gain is uncertain:** 0.53–0.86 × the 5/7 g model from long windows of the balancing logs, but
   the runs swing more whenever it is set below 1.0. The roll test (release 5 cm in from the ends) is the clean way
   to measure it.
5. **Joint 4 drifts** (+86 mNm over one map on Nov 24; the joint gap changed 80–98 mm in a run); joint 2 stays put.

## Nov 24 2026, iam-bruno (after a reboot)

Re-calibrated: track test PASS (gain 0.86, lag 80 ms); rail map **point errors 1.2 mm** (best so far) but joint 4's
empty reading drifted +86 mNm over the map and its still-noise was ~3x Nov 21 (26-30 mNm); sweep PASS, **6 mm per
sample**; **level trim −0.85°** by phone level (the balancing-log estimate of −0.2° was wrong: stiction).

| Time | Result | What we learned / changed |
|---|---|---|
| 21:43 | 41.7 s, then 2 fast drops (4.5 s, 7.3 s) | settings rolling gain 0.8, max ball speed 40, max tilt 1.5: the plate swung ±1.0° rms and the ball went ±60–100 mm every ~4 s (the 19:22 best run: ±0.2–0.3°, ±5–8 mm). The learned lean (−0.36°, learned during that swing) was carried into the re-placements and pushed the ball off the front within 3 s → lean reset on every re-placement again; preset back to the 19:22 values (rolling gain 1.0, 20 mm/s, 1.2°) keeping the end guard, hold and recovery fixes |

| **21:52** | **94.6 s, no drop until a hard push**; 20 mm rms, plate 0.20° rms | the best run so far: the ball sits still when not pushed, and came back from 17 pushes up to ~70 mm/s (out to ±60 mm, back within 15 mm in 0.3–1.3 s). The last push (~250 mm/s) was lost: the outlier gate treated 0.5 s of consistent readings as spikes, so the estimate stayed at −16 mm while the ball reached the end → consistent readings that both joints agree on are now accepted at once; push detection in the filter; emergency brake up to 2.5° (twice the tilt rate). On the fake, on a ±115 mm rail: catches up to ~120 mm/s (110 without the brake); a 150 mm/s push cannot be caught (the readings lag ~0.15 s, the paddle needs ~0.2 s to tilt) |

Settings of the 21:52 run: rolling gain 1.0, Kp 1.0, Kv 2.0, max ball speed 20 mm/s, max tilt 1.2°, look-ahead
0.35 s, level −0.85°, still-ball hold 12 mm / 15 mm/s.

| **22:15** | **175.9 s, best run: 54 pushes, 48 returned to the centre** | the consistent-reading acceptance, push detection and emergency brake (7 s of use in total) caught pushes up to ~220 mm/s and balls that reached the rail end. One drop at the end |

Lesson: every time the controller was made more aggressive (lower rolling gain, higher ball speed or tilt) the plate
swung more. The real rolling gain is probably near 1.0.

## Nov 21 2026, iam-bruno (control PC iam-snowwhite)

### Setup that works

- **Pose: hand turned 90°** (joint 7 −90°). The rail then runs toward/away from the robot; **joint 6 tilts** it and
  **joints 2 and 4 sense** the ball at full strength (18.9 mNm/cm by physics; mapped at 82 % and 89 %).
  On the old arm this pose was impossible because its joint 6 did not follow (stick-slip, 6 s behind). Here joint 6
  passed the track test: gain 1.03, correlation 0.997, never more than 0.5° behind.
- **100 Hz state topic** instead of frankapy's state service: 100 new states per second instead of ~28.
- **Results of the calibration:** rail map point errors **6.6 mm** (−2, +8, −7, −7, +8), empty-paddle drift 3 mm,
  noise 10.7 mm per sample. Tilt sweep: **4 mm per sample** while the paddle rocks (joint 3 at the normal pose: 13–45 mm).

### Runs

| Time | Result | What we learned / changed |
|---|---|---|
| 19:02 | 11 s at **8 mm rms**, then "contact" | a contact check ended the run on a push → contacts now pause and ask for the ball |
| 19:16 | endless "put the ball back" | the drop check used joint 4, a *sensing* joint at this pose: the ball rolling to an end looked like a drop, and the re-zero learned the ball's weight → presence now from the ball's weight on both joints; re-zero refused if the ball's weight is still there |
| **19:22** | **120 s run, one hold of 55 s at about ±4–8 mm** | best result. Paddle tilt rms only 0.2–0.3° |
| 19:40 | 3 failures with a fake "something pressed hard" | the vibration-cancel term (−13,700 Nm/rad × the joint angle's fast part) turned a 140–270 µrad spring of the arm (ball hitting an end, breaking free) into a fake 2–4 Nm. Franka's own force estimate showed the load had *dropped*. → term capped at ±30 µrad; filter skips one-off jumps > 60 mm; unstick ends the moment the ball moves (it kept full tilt 0.5 s after release and launched the ball at 240–350 mm/s); rail end limit from the map points (±110 mm, was ±79) |
| 20:02 | 6 drops | the filter's noise memory kept the empty-paddle garbage (~200 mm) after the ball was put back; a resync then followed 0.5 s of placement spikes to the end → noise memory reset on every re-placement, median raw average, no resync in the first second, the gate can no longer lock the filter out, re-zero only on a clearly empty paddle |
| 20:15 | 4 drops off the **front** | the ball ran to the front at ~100 mm/s; the paddle lags ~0.3 s at each reversal |
| 20:43–20:52 | roll test | both rail ends hold the ball in a pocket; free rolls too short to fit → roll test now releases 5 cm in from the ends |
| 21:01 | **29 s hold**, then 2 drops off the front | the speed limit (20 mm/s) *braked* a ball coming back from the front edge by tilting toward the front, and the paddle lag rolled it back over the edge; edge recovery too weak (0.4–1.0°) → end guard (never tilt toward a nearby end), unstick after 0.3 s up to the full max tilt, max ball speed 50 mm/s |
| **21:08** | **43 s hold**, then a growing swing, 2 drops | while the ball sat still the controller dithered the plate (±0.2–0.5°) around a tilt that wandered from −0.05° to −0.89° across holds. Stiction held the ball until it broke loose with the stored tilt, then the end recovery threw it back faster each time → still-ball hold, braking allowed above max ball speed, learned lean kept, rolling gain 0.8, max ball speed 40 mm/s |

### Numbers worth remembering

- The paddle reaches about **80 % of a commanded tilt** at the turned pose (joint 6 droop and elbow sag): commanded
  1.1° gave 0.87° at the joint and ~0.6° at the deck. The controller works from the *measured* tilt.
- The average tilt needed to hold the ball still was **−0.32° ± 0.29°** relative to trim −0.5° over 19 ten-second holds
  (so the true level is around trim −0.2°, but it wanders by a few tenths of a degree).
- Re-zeros after drops: the empty-paddle zero moved by 2–24 mm-equivalent per minute.
- The rail stiction: a still ball does not move below about 0.3–0.6° of tilt.

## Nov 20–21 2026, iam-sneezy (earlier arm, control PC iam-sneezy)

- Normal hand pose, rail left/right: **joint 5 tilts, joint 3 senses** (13.4 mNm/cm, quantised at 24 mNm, ~20 mm per
  sample). Joint 5 itself cannot sense: its own motion and friction swamp the ball (87 mm vs 45 mm of noise while
  rocking; 49 mm shift of the empty reading over one map).
- Joint 6 on that arm did not follow commands (stick-slip), so the better turned-hand pose was not usable there.
- The original sticky rail needed ~1.6° to break the ball free; after the rail was rebuilt smooth, the gentle PD
  preset held the ball for 46 s and 32 s.
- The original Nov 21 handoff with all details is in [docs/HANDOFF_2026-11-21_iam-sneezy.md](docs/HANDOFF_2026-11-21_iam-sneezy.md).
