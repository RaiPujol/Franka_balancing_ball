# Offline static shadow filter

`shadow_static_filter.py` is a **read-only offline analysis** of the wrench
recorded by `record_wrench_readonly.py`. It contains no ROS subscription,
connection, publisher, service, arm command or gripper command. It does not
authorize physical movement. `dispatch_to_robot()` is permanently locked and
always raises, regardless of inputs or settings.

The program loads an empty and loaded fixed-pose recording, transforms each
actual recorded base-frame wrench about stiffness frame K into the independently
measured tray frame, subtracts the empty-tray baseline, and filters the resulting
static contact-point estimates. No assumed or simulated ball position is used.
It requires the same payload setup and fixed robot/tray pose as the earlier static
replay; it rejects source records marked invalid or stale, and its filter rejects
invalid measurements, nonadvancing source times and large sample gaps.

Example from the project folder, **after recording real data** and preparing a
measured transform as described in `hardware/README.md`:

```sh
python3 -m hardware.shadow_static_filter --empty logs/empty.jsonl --loaded logs/ball_point_a.jsonl --ee-to-tray measured_EE_to_tray.json --output logs/shadow_point_a.json
```

This is an offline *static* filter. The 0.2 N normal force, 0.15 s maximum
sample gap, and Kalman noise assumptions are unvalidated analysis parameters,
not robot safety or usability thresholds. A sample marked valid is not evidence
the ball remains within the tray or that motion is safe. True sensor capture
time, latency, wrench accuracy, friction and dynamic coupling are not measured.
The program uses `robot_time` only for ordering and relative sample intervals;
the zero prediction horizon does **not** mean the physical sensor delay is zero.

Before using any result, independently compare the static raw estimates with
known measured ball positions, characterize drift and pose effects, and verify
the installed ROS topic and stiffness-frame convention. The output is for review
and cannot produce motion or gripper setpoints. Dynamic sensing, calibrated
timing, a real robot control interface and its physical safety checks still need
separate work.
