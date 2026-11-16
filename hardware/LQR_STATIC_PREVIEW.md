# Offline static LQR preview

`lqr_static_shadow_preview.py` analyzes a previously recorded **stationary**
Franka wrench stream. It is for examining the candidate in a fixed-pose test;
it cannot connect to the robot, publish commands, move the arm, operate the
gripper, or enable physical execution. Its `dispatch_to_robot` function always
raises with no override.

The input is the actual `O_F_ext_hat_K` data captured by the read-only recorder,
an empty-tray recording, and an **independently measured** end-effector-to-tray
transform. The existing static replay changes the wrench from the base frame
at stiffness point K to the measured tray frame, checks the recorded tool
configuration and fixed pose, and removes the empty-tray baseline. A delay-aware
Kalman filter uses the same measurement/process settings as the tested candidate
(0.002 m and 0.3 m/s²) to filter the resulting contact-position estimates.

For the stationary target at tray center, the preview applies the candidate's
same position/velocity gains, `kp=0.6`, `kd=0.8`, and its 7° norm limit. In the
simulator, the full moving controller also includes trajectory and acceleration
feedforward. Those terms are **zero in this static preview**, which does not
evaluate tracking or the arm's physical behavior. The output slope is a local
tray-frame model quantity labelled `STATIC PREVIEW ONLY - NOT A ROBOT COMMAND`;
it is not a robot pose, command, or verified safe tilt.

Example from the project root, after safely recording data and independently
measuring the transform as described in `hardware/README.md`:

```sh
python3 -m hardware.lqr_static_shadow_preview --empty logs/empty.jsonl --loaded logs/ball_point_a.jsonl --ee-to-tray measured_EE_to_tray.json --output logs/lqr_static_preview.json
```

The normal-load gate (default 0.2 N), 0.15 s sample-gap limit, filter parameters
and 7° numerical limit are **unvalidated simulation/analysis assumptions**,
not physical safety limits. Invalid or nonadvancing observations and gaps do not
produce preview slopes. Source robot time is used only to order and separate
samples. This offline preview does not know the true age of the wrench estimate,
so it performs no latency prediction and does not claim physical delay is zero.

A contact point inferred from static wrench is not necessarily a rolling ball's
center under acceleration, friction or other contact. Passing this offline
preview does not establish sensor accuracy, dynamic observability, collision
clearance, safe command cancellation, or permission to move. Independent
position measurements and real-robot timing/noise characterization remain
necessary before a separate dynamic integration can be considered.
