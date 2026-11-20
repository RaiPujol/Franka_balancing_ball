# Franka measurement and controller preparation

See [real robot setup](REAL_ROBOT_SETUP.md) for the new transport-independent Kalman + LQR core, read-only preflight, and measurements needed for the 162.5 g plate.

These files collect and analyze stationary robot-state data. They contain no motion, gripper, controller-start, payload-setting, or collision-setting commands. They have only been tested with offline fake messages, not connected to the lab robot. They are **not** a ready-to-run balancing controller.

## What the wrench means

The lab's previously installed FrankaPy source maps its end-effector force/torque reading to `O_F_ext_hat_K`. This is the **estimated external wrench at stiffness-frame origin K, expressed in the robot base O**, not a dedicated six-axis load cell at the tray. Sign convention is robot-on-environment positive; forces applied by the environment to the robot are negative. `EE_T_K` and `O_T_EE` are needed to transform it to the tray. Franka warns that its estimated external wrench becomes zero near singularities. These details must be verified for the installed robot/software version and the specific ROS state message before relying on readings.

Sources: https://frankarobotics.github.io/libfranka/latest/structfranka_1_1RobotState.html and the lab's archived `franka_arm_state_client.py` from the October 2026 session.

A conforming 45.93 g golf ball has about 0.45 N of weight. Moving it 10 mm produces only about 0.0045 N m change in static moment. The actual noise, drift, bias, latency and load compensation must be measured; a 5 g sensitivity claim alone is not evidence that a ball can be localized.

## Prepare at the lab

1. Follow the lab's safety procedure and keep the arm stationary. Your partner is required by the course safety procedure. The robot and tray design, mass, center of mass, and mounting are still to be checked.
2. Make the script folder available on the lab PC under your group work area (previously `~/devel_packages/mike`). Use the existing, correctly sourced ROS1 / Franka interface environment. Do not modify the shared controller or install or change robot parameters just to use the recorder.
3. Discover rather than assume the live state topic:

   ```sh
   rostopic list
   rostopic type /robot_state_publisher_node_1/robot_state
   rosmsg show franka_interface_msgs/RobotState
   ```

   `/robot_state_publisher_node_1/robot_state` was present on the October 2026 robot, but the current name and type may differ. Confirm that the topic really is the state topic and the type is `franka_interface_msgs/RobotState`; the recorder deliberately requires the topic argument. If the required fields or message type differ, stop and adapt the reader instead of substituting zeros.
4. Measure a rigid transform from the **EE frame to the tray-top frame**: origin at tray center on the top surface, +z normal to the surface, +x/+y chosen and recorded on the tray. Save a JSON file with key `T_EE_tray_row_major` containing a measured 4-by-4 homogeneous transform. All translations are metres. Do not substitute an identity transform or an assumed hand-to-tray offset. Reconfirm it whenever the attachment changes.
5. Review the configured payload and stiffness frames from the captured state with the course staff. The recorder logs `m_ee`, `m_load`, `m_total`, center-of-mass and inertia settings, `F_T_EE`, and `EE_T_K`; it does not change them. Configured values are not measurements of the attachment's actual inertia.

## Collect a static calibration set

Use a stationary arm and follow the lab's procedure for safely placing/removing a ball. Do not put hands in the workspace during commanded movement. The arm must remain at the same tray pose while collecting empty and loaded recordings. Independently mark and measure the ball's x and y location in the defined tray frame, in mm. The recording label is just a label; it is not fed to the robot.

Example commands after confirming the live topic and adjusting file paths (the directory must already exist):

```sh
python3 hardware/record_wrench_readonly.py --topic /robot_state_publisher_node_1/robot_state --output logs/empty.jsonl --label empty --duration 20
python3 hardware/record_wrench_readonly.py --topic /robot_state_publisher_node_1/robot_state --output logs/ball_point_a.jsonl --label point_a --duration 20
```

Repeat with several independently measured ball locations spread across the tray and repeat recordings to assess drift. Obtain an empty-tray baseline at the same pose; record again after loaded trials to evaluate change over time. Use different new filenames; the recorder refuses to overwrite files. It subscribes only: no `FrankaArm` object, publisher, action client, service call, or command is created.

The recorder preserves ROS header sequence/stamp, source `robot_time`, `is_fresh`, local receipt times, joint state and wrench. Its candidate flag checks only that source values advance; it cannot prove the signal is recent, detect every replay, or measure true sensor latency. It stops on robot-time reversal or stalled stream. Do not interpret the ROS header as the physical sample timestamp without verification. Require a complete end record with no reported errors.

## Offline analysis (no connection to robot)

Use independently measured coordinates as validation labels, not as a control signal. Example: if point A was measured at x=20 mm, y=-10 mm, and B at x=-15 mm, y=15 mm:

```sh
python3 hardware/analyze_static_wrench.py --empty logs/empty.jsonl --point 20 -10 logs/ball_point_a.jsonl --point -15 15 logs/ball_point_b.jsonl --ee-to-tray measured_EE_to_tray.json --output logs/static_evaluation.json
python3 hardware/replay_static_wrench.py --empty logs/empty.jsonl --loaded logs/ball_point_a.jsonl --ee-to-tray measured_EE_to_tray.json --output logs/static_replay.json
```

The analyzer transforms the wrench from the base at K to the tray center, subtracts the empty baseline, and reports position error against measured points, pointwise variation, empty-tray drift and insufficient-force counts. The replay writes per-sample static contact-position estimates and gate results without using the known position labels. Both reject incomplete recordings, changing configured load/stiffness frames, moving joints, and tray-pose changes outside their preliminary tolerance. The 2 mm/2 degree pose gate and 0.2 N normal-force gate are analysis assumptions, not validated robot safety limits. A passing sample is **not** a guarantee of freshness, absence of singularity, sensor accuracy, or safe motion. The model assumes a static ball in contact with the tray and no other touching loads.

These static data cannot establish dynamic tracking accuracy or end-to-end latency. Only after sensor noise/bias and actual attachment geometry are measured should the team revisit the simulation and design staged robot tests with the instructor/TA. Neither script enables movement.
