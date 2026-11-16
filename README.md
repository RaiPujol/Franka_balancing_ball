# Golf ball balancing: STL simulation and measurement tools

This folder contains an offline MuJoCo model of the supplied platform on a Franka hand, a force/torque-based estimator and controller for simulation, and separate read-only tools for collecting stationary robot measurements. It does not contain a hardware balancing controller. Do not use the simulated attachment or its simulated clearances to command the real robot.

## Install and test

Create a Python environment and install the pinned packages in `requirements.txt`. From this folder run:

```sh
python -m pip install -r requirements.txt
python -m unittest discover -q
python -m unittest discover -s tests -q
python stl_evaluate.py --only-task circle --only-case nominal_20ms --only-seed 401 --output-dir /tmp/actual-stl-smoke
```

The last command runs one offline trial and writes results to the specified output directory. A completed command is not by itself evidence that a trial passed; inspect the results. Run `python stl_evaluate.py --help` for the available tasks, disturbances and seeds. `stl_evaluate.py` and `stl_render_selected.py` explicitly install the STL scene for both the plant and the unloaded observer. Other retained scripts such as `wrench_shapes.py` and `robot_scene.py` use the earlier placeholder tray if run directly.

## Model and its limits

- `inputs/ball balancing platform.STL` is the user's original STL, in confirmed millimetres. `assets/platform/platform_tray_frame.stl` is a translated copy for simulation. The tray-top frame has its origin at the top surface's centre; the flat top is 150 by 150 mm. The user supplied a 92.77 g printed-part mass. Optional bolt mass is zero by default, and the model's uniform-density inertia has not been measured.
- The STL is used for rendering and estimating an idealized inertia. The contact model uses a flat box for the tray and a conservative box for the tab: it omits the small central dimple and tab undercuts. It does not simulate bolt fastening or the detailed fit of the fingers, and it is not an exact contact representation of the nonconvex STL.
- The orientation and hand-to-tray transform are provisional. The real part's grip, bolt position and transform have not been measured. The nominal model has approximately 0.16 mm clearance between the tray and the hand at its initial pose; that is not a verified safe physical clearance. Recheck the actual mount, all clearances, payload and arm configuration with the lab before any physical motion.
- The ball is a dynamic free body. Controller estimates are derived from simulated wrench/joint-torque signals, after transforming the wrench to the tray frame; simulation ground truth is used for evaluation and visualizations, not as feedback. Simulated noise, delay, friction and rigid contacts are assumptions, not measurements of the hardware.

The user's standard Franka hand and a nominal golf ball are modeled. Golf ball and tray properties should be measured for the real setup. This simulation alone cannot establish real-robot reliability.

## Read-only robot data

See `hardware/README.md` for static data collection, frame conventions and calibration requirements. The ROS1 recorder reads a robot state topic and writes local data; the analysis and shadow scripts are offline. They have not been validated against the currently installed lab robot. They do not start a robot controller or command arm or gripper motion. The robot's estimated external wrench, actual latency, noise and ball-position observability must be independently measured before designing hardware motion tests.

See `RESULTS.md` and `results/` for the tested cases and limitations, and `THIRD_PARTY.md` for asset attribution.
