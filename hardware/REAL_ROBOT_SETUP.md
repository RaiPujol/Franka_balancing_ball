# Preparing the measured plate for robot tests

The 233 mm plate weighs 162.5 g without modeled fasteners. The matching simulation used a golf ball of nominal diameter 42.67 mm and mass 45.93 g. The candidate uses a Kalman filter and LQR position and velocity gains of 0.6 and 0.8. `hardware/real_lqr_core.py` carries the tested slope calculation into a transport-independent module. The ball radius must be supplied explicitly as `ball_radius_m=0.03725` for the measured 74.5 mm ball. Its input must be a ball position and normal force from a separately validated dynamic estimator, with acquisition and current time in the same characterized clock domain. Its output is a slope in tray coordinates.

There is no validated connection from this core to a robot motion command. The empty-tray subtraction used by the existing recorder analysis is valid at the stationary pose at which it was recorded. Tilting changes the tray's wrench. The teammate's `torque_balance/` package includes a separate experimental controller and learned dynamic tare, but its mount, timing and behavior have not been verified for this physical setup. Several of its stages start motion before the first interactive prompt. Its stop behavior also needs checking on the installed arm.

## Lab computer

The prepared checkout is under `/home/student/16662_RobotAutonomy/src/devel_packages/mike/`. In the course Docker environment the corresponding path starts with `/home/ros_ws/src/devel_packages/mike/`. Source the installed ROS and workspace setup and enter the checkout root:

```sh
source /opt/ros/noetic/setup.bash
source /home/ros_ws/devel/setup.bash
cd /home/ros_ws/src/devel_packages/mike/franka_balancing_lqr_preflight
python3 -m hardware.real_robot_preflight
```

This checks imports without making a robot connection. Once the lab's normal robot-state publisher is running, discover the topic and check the stream:

```sh
rostopic list
rostopic type /robot_state_publisher_node_1/robot_state
python3 -m hardware.real_robot_preflight --topic /robot_state_publisher_node_1/robot_state --seconds 10
```

Use the actual topic name if it differs and confirm its type is `franka_interface_msgs/RobotState`. Advancing timestamps show stream behavior but cannot establish sensor acquisition latency or end-to-end command latency.

## Measurements for the plate

1. Weigh the bolts and other parts added to the 162.5 g print. Check the actual golf ball mass and diameter. Record the robot's configured payload and stiffness frame.
2. With the robot stationary, measure the end-effector-to-tray rigid transform. Use the tray's top center as the origin, +z upward and mark +x and +y on the surface. Save a JSON file containing `T_EE_tray_row_major`, a 4×4 rigid transform in metres. Neither the simulated mount nor the teammate's mount estimate substitutes for this measurement.
3. Check the mechanical fit and clearances with the real gripper. The nominal CAD scene reported a deck-to-finger gap of about 0.16 mm at the sampled poses, and treats the attachment as rigid.
4. Collect an empty-tray recording and at least five ball locations spanning both tray axes. One possible set is the center and (±30 mm, ±30 mm). Measure the actual locations. Take each recording at the same stationary robot pose, mounting and load settings, using the lab's procedure for safely placing the ball. Repeat the empty recording afterwards to assess drift.

For the first empty-tray baseline, the passive initialization tool captures the state and summarizes the torque reference for all joints. It writes new files with unique names. Run it only after the plate is empty and the arm is stationary:

```sh
python3 -m hardware.initialize_torque_baseline --topic /robot_state_publisher_node_1/robot_state --output-dir logs/session_name
```

Create or substitute an existing output directory and verify the topic. The result is a software comparison reference at this pose. It does not alter the robot or physically zero the torque sensors. For other recordings, change the output filename and label. The script creates a new file and will not overwrite an existing one:

```sh
mkdir -p logs
python3 hardware/record_wrench_readonly.py --topic /robot_state_publisher_node_1/robot_state --output logs/empty.jsonl --label empty --duration 20
```

Run the static analysis with the measured transform and each measured location. Coordinates in the `--point` arguments are millimetres. Replace the example coordinates and file names with the actual measurements.

```sh
python3 hardware/analyze_static_wrench.py --empty logs/empty.jsonl --ee-to-tray measured_EE_to_tray.json --output logs/static_evaluation.json --point -30 -30 logs/point_1.jsonl --point -30 30 logs/point_2.jsonl --point 30 -30 logs/point_3.jsonl --point 30 30 logs/point_4.jsonl --point 0 0 logs/point_5.jsonl
python3 -m hardware.calib_observability --analysis logs/static_evaluation.json --output logs/observability.json
python3 -m hardware.lqr_static_shadow_preview --empty logs/empty.jsonl --loaded logs/point_5.jsonl --ee-to-tray measured_EE_to_tray.json --output logs/lqr_preview.json
```

The observability report scores independently held-out locations. The stationary LQR preview computes a slope for inspection; it does not command motion.

## Remaining integration

Dynamic ball position and normal-force estimates must be validated over actual tray motion, including wrist motion compensation, source freshness, timing, noise, drift and frame conventions. The real robot interface still needs a tested conversion from tray-frame slope to robot commands, plus a verified watchdog and stop behavior. Calibration, geometry, collision clearance and the resulting control loop must be checked on the installed arm before autonomous ball balancing.
