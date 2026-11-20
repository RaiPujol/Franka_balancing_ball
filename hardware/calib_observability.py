#!/usr/bin/env python3
"""Evaluate 2D stationary ball-position observability from a static analysis.

The input is the JSON output of analyze_static_wrench.py, collected with the
robot stationary at one verified pose. Regression is a diagnostic only; it
is not loaded by a robot controller and does not model moving-tray dynamics.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def summarize(data):
    if data.get('status') != 'offline_static_evaluation_only':
        raise ValueError('input must be an offline static evaluation')
    positions = data.get('positions', [])
    if len(positions) < 4:
        raise ValueError('at least four non-collinear locations are required')

    known = np.asarray([p['known_xy_m'] for p in positions], dtype=float)
    estimated = np.asarray([p['estimated_mean_xy_m'] for p in positions], dtype=float)
    std = np.asarray([p['estimated_std_xy_m'] for p in positions], dtype=float)
    if any(a.shape != (len(positions), 2) or not np.isfinite(a).all()
           for a in (known, estimated, std)) or (std < 0).any():
        raise ValueError('point coordinates and standard deviations must be finite two-vectors')

    # Millimetre scaling avoids mixing tiny metre-valued coordinates and an
    # intercept when checking rank. Each location receives equal weight.
    known_mm = 1000 * known
    estimated_mm = 1000 * estimated
    if len(np.unique(known_mm, axis=0)) != len(known_mm):
        raise ValueError('each file must describe a distinct measured ball location')
    X = np.column_stack((estimated_mm, np.ones(len(estimated_mm))))
    K = np.column_stack((known_mm, np.ones(len(known_mm))))
    if np.linalg.matrix_rank(K) < 3:
        raise ValueError('known locations are collinear; two-axis fit is unobservable')
    if np.linalg.matrix_rank(X) < 3:
        raise ValueError('estimated locations are collinear; two-axis fit is unobservable')
    coefficients = np.linalg.lstsq(X, known_mm, rcond=None)[0]
    fitted_error = X @ coefficients - known_mm
    holdout = []
    for i in range(len(X)):
        train = np.arange(len(X)) != i
        if np.linalg.matrix_rank(X[train]) < 3 or np.linalg.matrix_rank(K[train]) < 3:
            raise ValueError('each held-out fit needs at least three non-collinear locations')
        beta = np.linalg.lstsq(X[train], known_mm[train], rcond=None)[0]
        error = X[i] @ beta - known_mm[i]
        holdout.append({'file': positions[i].get('file'),
                        'known_xy_mm': known_mm[i].tolist(),
                        'estimated_mean_xy_mm': estimated_mm[i].tolist(),
                        'raw_error_xy_mm': (estimated_mm[i] - known_mm[i]).tolist(),
                        'estimated_std_xy_mm': (1000 * std[i]).tolist(),
                        'holdout_error_xy_mm': error.tolist(),
                        'holdout_error_mm': float(np.linalg.norm(error))})
    errors = np.asarray([p['holdout_error_mm'] for p in holdout])
    return {
        'status': 'offline_static_observability_diagnostic',
        'input_scope': 'stationary same-pose analysis; equal weighting per independently measured location',
        'point_count': len(X),
        'affine_map_note': 'known_xy_mm = [estimated_x_mm, estimated_y_mm, 1] @ coefficients',
        'coefficients': coefficients.tolist(),
        'in_sample_rms_mm': float(np.sqrt(np.mean(np.sum(fitted_error ** 2, axis=1)))),
        'leave_one_location_out_rms_mm': float(np.sqrt(np.mean(errors ** 2))),
        'leave_one_location_out_max_mm': float(np.max(errors)),
        'locations': holdout,
        'limitations': [
            'Static records do not measure rolling contact, moving tray compensation or sensor latency.',
            'Held-out locations must be independently measured and span both axes.',
            'This fit is a diagnostic and is not loaded into any robot controller.'
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', required=True, help='output JSON of analyze_static_wrench.py')
    parser.add_argument('--output', required=True, help='new output JSON, never overwritten')
    args = parser.parse_args()
    data = json.loads(Path(args.analysis).read_text())
    result = summarize(data)
    with Path(args.output).open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print('Held-out position RMS %.2f mm, maximum %.2f mm' %
          (result['leave_one_location_out_rms_mm'], result['leave_one_location_out_max_mm']))


if __name__ == '__main__':
    main()
