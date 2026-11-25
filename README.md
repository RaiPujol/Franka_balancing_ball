# Franka ball balancing

A Franka Panda arm balances a ball on a hand-held paddle **without a camera**: the ball is sensed through the arm's
own joint torque sensors. The repository has two parts.

> **Headline result (Nov 24 2026, iam-bruno):** the 193 g ball balanced in a rail for **175.9 s (~3 min) without a
> drop**, sensed by joint torques only, while being pushed **54 times**. The ball returned to the centre after 48 of
> them, including pushes up to ~220 mm/s and runs all the way to the rail end (median recovery 1.4 s).

| Part | Where | Status |
|---|---|---|
| **Real-robot balancer** (ball in a rail, one axis) | [`torque_map/`](torque_map/README.md) | runs on the lab arm from a web page; best run **175.9 s** with 54 pushes |
| **Simulation and controller studies** (MuJoCo, two axes) | repository root, `studies/`, `hardware/` | PD, Kalman + PD, LQR and MPC compared on the printed tray; [results](RESULTS.md) |

**Start here:**

- To run the robot: [torque_map/README.md](torque_map/README.md) (quick start, calibration, settings, troubleshooting).
- To see what was learned on the robot, run by run: [torque_map/LAB_NOTES.md](torque_map/LAB_NOTES.md).
- To reproduce the simulation results: [Simulation studies](#simulation-studies) below.

## Repository map

| Path | What it is |
|---|---|
| `torque_map/` | the real-robot rail balancer: web UI (`app.py`), balancer (`rail.py`), estimator and kinematics (`kin.py`), fake-robot tests (`sim_tests/`) |
| `torque_balance/` | the earlier command-line robot package (`GUIDE.md`); `sim_ball_balance.py` is also used by `torque_map --sim` |
| `hardware/` | transport-independent Kalman + LQR core, read-only robot preflight and calibration analysis ([setup](hardware/REAL_ROBOT_SETUP.md)) |
| `*.py` at the root | the MuJoCo simulation, controllers (`lqr_control.py`, `mpc_control.py`, ...) and study scripts |
| `studies/actual_ball_193g/` | simulation study with the measured 74.5 mm, 193 g ball |
| `inputs/`, `assets/` | tray CAD (STL) and the Panda model ([provenance](THIRD_PARTY.md)) |
| `results/` | simulation result tables |

---

## Real-robot balancer (torque_map)

The ball rolls in a rail of two sticks on the 233 mm printed deck. With the hand turned 90°, joint 6 tilts the rail
and joints 2 and 4 feel the ball's weight shift (about 4 mm of position noise per sample at 100 Hz). A Kalman filter
and a cascaded PD controller turn that into a paddle tilt. Calibration (rail map, empty-paddle tilt sweep, level) and
balancing are buttons on a web page, with live tuning. Quick start:

```bash
roscore                                                                                  # terminal 1
cd ~/Documents/frankapy && bash ./bash_scripts/start_control_pc.sh -u student -i iam-snowwhite   # terminal 2
cd ~/16662_RobotAutonomy && bash run_docker.sh                                           # terminal 3, then in the container:
cd /home/ros_ws/src/devel_packages/torque_map && python3 app.py                          # open http://localhost:8000
```

Fake-robot tests (no robot, numpy only): `python3 torque_map/sim_tests/run_all.py`.
Everything else is in [torque_map/README.md](torque_map/README.md).

---

## Simulation studies

This MuJoCo simulation puts a standard golf ball on the revised 233 × 233 mm printed tray. The Franka hand holds the
tab with the deck pointing away from the base. A delayed joint-torque estimate supplies the ball feedback for
carrying, centering, circle, rounded-square and figure-eight trials.

The [verification results](RESULTS.md) include failed trials, controller comparisons and the limits of the modeled
mount. Those results use the earlier nominal golf ball. For the team's measured 74.5 mm, 193 g ball, see the separate
[measured-ball study](studies/actual_ball_193g/README.md).

### Reproduce a trial

Install the versions in `requirements.txt` in a Python environment. Run the following from the repository root.

```sh
python -m pip install -r requirements.txt
python -m unittest discover -q
python -m unittest discover -s tests -q
python stl_evaluate.py --only-task circle --only-case nominal_20ms --only-seed 401 --output-dir output/smoke
```

The trial writes its pass flag, stop reason and tracking error to `output/smoke/results.csv`. The full studies and
video entry points are below.

```sh
python stl_evaluate.py --output-dir output/main
python stress_stl_material.py --output-dir output/material
python stress_stl_mass.py --previous-observer-g 209.36256678704676 --output-dir output/mass
python stress_stl_reach.py --output-dir output/reach
python audit_trajectory_distances.py --output-dir output/clearance
for method in pd kalman_pd lqr mpc; do
  python stl_compare_methods.py --method "$method" --output-dir output/methods
done
python stl_render_selected.py --method lqr --output-dir output/task_videos
python stl_render_methods.py --output-dir output/comparison_videos
```

The `stl_*` entry points load the revised CAD for the plant and empty-tray observer. Running `run_sim.py` or
`wrench_shapes.py` directly loads an earlier box model. In the shape videos blue is the requested path, green is the
ball estimate and orange is the simulated ball position. The simulator uses the orange trace to score tracking.

### Tray and ball

The current source is `inputs/ball_ballancing_platform.STL`, in millimetres. The earlier 150 mm design remains in
`inputs/ball balancing platform.STL`. To regenerate the translated simulation mesh and check it against the supplied
Bambu Studio project, run the following with the local path to that project.

```sh
python prepare_platform_mesh.py --verify-3mf /path/to/ball_balancing_platform.3mf
```

The new deck is 233 mm square. Its 35 mm wide tab extends 21 mm past the deck. The model places the deck center
127 mm outward from the tab center. The STL supplies the rendered geometry and mass distribution. For contact, the
model uses a flat box enclosing the underside ribs and a solid tab. It omits the central dimple, tab hole, mounting
bolts and finger compliance. The team weighed the revised print at 162.5 g. The nominal model uses that mass for the
plate and zero added bolt mass. The material sweep also tests 10 g of fasteners. The nominal ball is 42.67 mm across
and weighs 45.93 g.

### Simulation hardware files

The `hardware/` directory contains a transport-independent Kalman + LQR core, read-only robot-state preflight and
calibration analysis. See [hardware setup](hardware/REAL_ROBOT_SETUP.md). The `torque_balance/` package contains an
earlier experimental controller and FrankaPy motion scripts. The validated simulation results here apply to the
outward-facing CAD model; the robot results are those of `torque_map/` above.

See [asset provenance](THIRD_PARTY.md) for the original models and dependencies.
