"""Test trajectories and starting positions that use the larger deck."""
import argparse
import csv
from pathlib import Path

from stl_bridge import use_actual_stl_scene
from stl_evaluate import SEEDS, factory_for
from wrench_center_demos import run_one
from wrench_shapes import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('output/deck_reach'))
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    records = []
    with use_actual_stl_scene():
        for seed, offset in SEEDS:
            settings = dict(sensor_model='joint_torque', joint_torque_std=.0005,
                            joint_velocity_std=.0001, delay_s=.05,
                            controller_factory=factory_for(0., 1.), seed=seed)
            for radius in (.060, .080):
                # Keep the steady tangential speed equal to the 20 mm, 8 s reference.
                period = 8. * radius / .020
                r = run('circle', config={'ball_initial_offset_m': offset}, radius_m=radius,
                        period_s=period, output_dir=out/f'circle_{radius}_{seed}', **settings)
                err = r['plateau_tracking_rms_m']
                passed = bool(r['numerical_pass'] and err is not None and err < .005
                              and r['max_tray_position_error_m'] < .02)
                records.append([seed, 'circle', *offset, radius, period, passed, r['stop_reason'],
                                err, r['minimum_full_footprint_margin_m']])
            for start in ((.070, .0), (-.080, .060)):
                r = run_one(start, 'centering', **settings)
                records.append([seed, 'centering', *start, None, None, r['success'], r['stop_reason'],
                                r['last_second_max_true_error_m'], r['minimum_full_footprint_edge_margin_m']])
    with (out/'results.csv').open('w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['seed','task','start_x_m','start_y_m','circle_radius_m','period_s',
                         'numerical_pass','stop_reason','task_error_m','min_footprint_margin_m'])
        writer.writerows(records)
    print(f'{sum(bool(row[6]) for row in records)}/{len(records)} trials passed')


if __name__ == '__main__':
    main()
