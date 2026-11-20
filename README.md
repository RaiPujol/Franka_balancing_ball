# Franka golf-ball balancing

This MuJoCo simulation puts a standard golf ball on the revised 233 × 233 mm printed tray. The Franka hand holds the tab with the deck pointing away from the base. A delayed joint-torque estimate supplies the ball feedback for carrying, centering, circle, rounded-square and figure-eight trials.

The [verification results](RESULTS.md) include failed trials, controller comparisons and the limits of the modeled mount.

## Reproduce a trial

Install the versions in `requirements.txt` in a Python environment. Run the following from the repository root.

```sh
python -m pip install -r requirements.txt
python -m unittest discover -q
python -m unittest discover -s tests -q
python stl_evaluate.py --only-task circle --only-case nominal_20ms --only-seed 401 --output-dir output/smoke
```

The trial writes its pass flag, stop reason and tracking error to `output/smoke/results.csv`. The full studies and video entry points are below.

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

The `stl_*` entry points load the revised CAD for the plant and empty-tray observer. Running `run_sim.py` or `wrench_shapes.py` directly loads an earlier box model. In the shape videos blue is the requested path, green is the ball estimate and orange is the simulated ball position. The simulator uses the orange trace to score tracking.

## Tray and ball

The current source is `inputs/ball_ballancing_platform.STL`, in millimetres. The earlier 150 mm design remains in `inputs/ball balancing platform.STL`. To regenerate the translated simulation mesh and check it against the supplied Bambu Studio project, run the following with the local path to that project.

```sh
python prepare_platform_mesh.py --verify-3mf /path/to/ball_balancing_platform.3mf
```

The new deck is 233 mm square. Its 35 mm wide tab extends 21 mm past the deck. The model places the deck center 127 mm outward from the tab center. The STL supplies the rendered geometry and mass distribution. For contact, the model uses a flat box enclosing the underside ribs and a solid tab. It omits the central dimple, tab hole, mounting bolts and finger compliance. The team weighed the revised print at 162.5 g. The nominal model uses that mass for the plate and zero added bolt mass. The material sweep also tests 10 g of fasteners. The nominal ball is 42.67 mm across and weighs 45.93 g.

## Hardware files

The `hardware/` directory contains stationary recording and offline analysis tools. The separate `torque_balance/` package has a Franka deployment script and its own simulation with a different mount and a 254 mm box. Its default simulation pose puts the revised CAD almost vertical. The results here apply to the outward-facing CAD model. We have not tested the revised print on hardware.

See [asset provenance](THIRD_PARTY.md) for the original models and dependencies.
