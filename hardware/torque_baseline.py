#!/usr/bin/env python3
"""Summarize an empty-tray stationary recording of measured joint torques.

The output describes a digital baseline for comparison at the same pose. It
does not change the robot, set its payload, or zero a physical sensor. It
does not establish the validity of a moving-tray torque estimator.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from hardware.analyze_static_wrench import load_recording


def summarize(path, *, min_samples=20, max_joint_speed=0.01,
              max_joint_range=0.01):
    if (min_samples < 2 or not np.isfinite([max_joint_speed, max_joint_range]).all()
            or max_joint_speed <= 0 or max_joint_range <= 0):
        raise ValueError('sample count and stationary thresholds must be positive')
    metadata, samples, rejected = load_recording(
        path, min_samples=min_samples, max_joint_speed=max_joint_speed)
    q = np.asarray([s.get('q', []) for s in samples], dtype=float)
    dq = np.asarray([s.get('dq', []) for s in samples], dtype=float)
    tau = np.asarray([s.get('tau_J', []) for s in samples], dtype=float)
    times = np.asarray([s['source_robot_time_s'] for s in samples], dtype=float)
    expected = (len(samples), 7)
    if any(a.shape != expected or not np.isfinite(a).all() for a in (q, dq, tau)):
        raise ValueError('q, dq and tau_J must contain seven finite values in every sample')
    joint_range = np.ptp(q, axis=0)
    if np.max(joint_range) > max_joint_range:
        raise ValueError('joint position range exceeded the stationary threshold')
    moving = rejected.get('robot_moving_or_invalid_velocity', 0)
    if moving:
        raise ValueError('recording included %d moving or invalid-velocity samples' % moving)

    setup_fields = ('F_T_EE', 'EE_T_K', 'm_ee', 'm_load', 'm_total',
                    'F_x_Cee', 'F_x_Cload', 'F_x_Ctotal',
                    'I_ee', 'I_load', 'I_total')
    try:
        reference_setup = {name: np.asarray(samples[0][name], dtype=float)
                           for name in setup_fields}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('recording is missing valid configured tool/load fields') from exc
    expected_shapes = {'F_T_EE': (16,), 'EE_T_K': (16,),
                       'm_ee': (), 'm_load': (), 'm_total': (),
                       'F_x_Cee': (3,), 'F_x_Cload': (3,), 'F_x_Ctotal': (3,),
                       'I_ee': (9,), 'I_load': (9,), 'I_total': (9,)}
    if any(value.shape != expected_shapes[name] or not np.isfinite(value).all()
           for name, value in reference_setup.items()):
        raise ValueError('invalid configured tool/load field shape or value')
    for sample in samples:
        for name, ref in reference_setup.items():
            try:
                value = np.asarray(sample[name], dtype=float)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError('missing or invalid configured tool/load field: ' + name) from exc
            if (value.shape != ref.shape or not np.isfinite(value).all()
                    or not np.allclose(value, ref, atol=1e-8, rtol=1e-5)):
                raise ValueError('configured tool/load or stiffness frame changed: ' + name)

    tail = max(1, len(tau) // 10)
    start_to_end = np.mean(tau[-tail:], axis=0) - np.mean(tau[:tail], axis=0)
    return {
        'status': 'offline_stationary_torque_baseline',
        'input_file': str(path),
        'label': metadata.get('label'),
        'accepted_samples': len(samples),
        'rejected_counts': rejected,
        'robot_time_span_s': float(times[-1] - times[0]),
        'max_robot_time_gap_s': float(np.max(np.diff(times))),
        'max_abs_joint_speed_rad_s': float(np.max(np.abs(dq))),
        'joint_reference_median_rad': np.median(q, axis=0).tolist(),
        'joint_range_rad': joint_range.tolist(),
        'configured_tool_and_load': {name: value.tolist()
                                     for name, value in reference_setup.items()},
        'frame_matrix_layout': 'F_T_EE and EE_T_K are 16-element column-major arrays',
        'torque_mean_nm': np.mean(tau, axis=0).tolist(),
        'torque_median_nm': np.median(tau, axis=0).tolist(),
        'torque_std_nm': np.std(tau, axis=0, ddof=1).tolist(),
        'torque_start_to_end_drift_nm': start_to_end.tolist(),
        'joint_5_6_median_nm': np.median(tau, axis=0)[4:6].tolist(),
        'joint_5_6_std_nm': np.std(tau, axis=0, ddof=1)[4:6].tolist(),
        'preliminary_checks': {
            'max_joint_speed_rad_s': max_joint_speed,
            'max_joint_range_rad': max_joint_range,
        },
        'limitations': [
            'This is a digital comparison baseline, not a physical torque-sensor zero.',
            'The recording must be genuinely empty, with unchanged plate and mount.',
            'Advancing robot time cannot prove acquisition latency or physical freshness.',
            'The reported tool and payload settings are configured values, not physical measurements.',
            'Stationary noise and drift do not validate a moving-tray estimator.',
            'Stationary thresholds are preliminary analysis settings, not safety limits.',
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', required=True, help='completed empty-tray JSONL recording')
    parser.add_argument('--output', required=True, help='new JSON result; existing files are preserved')
    parser.add_argument('--min-samples', type=int, default=20)
    parser.add_argument('--max-joint-speed', type=float, default=0.01,
                        help='preliminary stationary analysis threshold in rad/s')
    parser.add_argument('--max-joint-range', type=float, default=0.01,
                        help='preliminary within-recording joint position range in rad')
    args = parser.parse_args()
    result = summarize(args.recording, min_samples=args.min_samples,
                       max_joint_speed=args.max_joint_speed,
                       max_joint_range=args.max_joint_range)
    with Path(args.output).open('x', encoding='utf-8') as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write('\n')
    print('Accepted %d samples. Joint 5/6 median [N m]: %.6f, %.6f' % (
        result['accepted_samples'], *result['joint_5_6_median_nm']))


if __name__ == '__main__':
    main()
