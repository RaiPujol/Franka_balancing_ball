"""Compare LQR performance with a matched or mismatched printed-part mass."""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from math import isfinite
from pathlib import Path

from stl_bridge import use_actual_stl_scene
from stl_evaluate import factory_for
from stl_scene import DEFAULTS
from wrench_center_demos import run_one
from wrench_shapes import run


def trial(job):
    task, seed, offset, plant_scale, observer_mass, calibration, output = job
    plant_mass = DEFAULTS['tray_mass_kg'] * plant_scale
    scene = {'tray_mass_kg': plant_mass}
    settings = dict(sensor_model='joint_torque', joint_torque_std=.0005,
                    joint_velocity_std=.0001, delay_s=.05, seed=seed,
                    controller_factory=factory_for(0., observer_mass/plant_mass))
    with use_actual_stl_scene():
        if task == 'circle':
            scene['ball_initial_offset_m'] = offset
            detail = output / f'{task}_{seed}_{plant_scale}_{calibration}'
            result = run(task, config=scene, output_dir=detail, **settings)
            error = result['plateau_tracking_rms_m']
            passed = (result['numerical_pass'] and error is not None and error < .005
                      and result['max_tray_position_error_m'] < .02)
        else:
            result = run_one(offset, task, scene=scene, **settings)
            error = result['last_second_max_true_error_m']
            passed = result['success']
    return dict(task=task, seed=seed, offset_x_m=offset[0], offset_y_m=offset[1],
                mass_fraction=plant_scale, plant_mass_kg=plant_mass, observer_mass_kg=observer_mass,
                calibration=calibration, delay_s=.05, joint_torque_noise_nm=.0005,
                numerical_pass=bool(passed), task_error_m=error, stop_reason=result['stop_reason'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('output/mass_study'))
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--previous-observer-g', type=float,
                        help='also test the measured plate with this previous observer mass in grams')
    args = parser.parse_args()
    if args.previous_observer_g is not None and (
            not isfinite(args.previous_observer_g) or args.previous_observer_g <= 0):
        parser.error('--previous-observer-g must be finite and positive')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [(1103, (.018, -.012)), (1201, (-.016, .018))]
    nominal = DEFAULTS['tray_mass_kg']
    jobs = []
    for task, (seed, offset) in product(('circle', 'carrying'), seeds):
        for scale in (.75, 1., 1.25):
            jobs.append((task, seed, offset, scale, nominal * scale, 'matched', args.output_dir))
            if scale != 1.:
                jobs.append((task, seed, offset, scale, nominal, 'fixed_nominal', args.output_dir))
        if args.previous_observer_g is not None:
            jobs.append((task, seed, offset, 1., args.previous_observer_g / 1000,
                         'previous_estimate', args.output_dir))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(trial, jobs))
    with (args.output_dir/'results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(results)
    print(f'{sum(row["numerical_pass"] for row in results)}/{len(results)} trials met the criteria')


if __name__ == '__main__':
    main()
