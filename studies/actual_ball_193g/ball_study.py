"""Compare controllers on the 74.5 mm ball using the outward-facing STL model."""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from hashlib import sha256
from itertools import product
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import stl_scene
from render_methods import controller_factory
from stl_bridge import use_actual_stl_scene
from stl_evaluate import CASES, SEEDS, TASKS
from wrench_center_demos import run_one
from wrench_shapes import run as run_shape

DEFAULT_OUT = REPO / 'output' / 'actual_ball_193g'
MEASURED = dict(ball_diameter_m=0.0745, ball_mass_kg=0.193, tray_mass_kg=0.1625, bolt_mass_kg=0.0)
CASES_BY_NAME = {name: (delay, noise) for name, delay, noise, _, _ in CASES[:3]}
METHODS = ('lqr', 'kalman_pd', 'pd', 'mpc')
SHAPES = ('circle', 'rounded_square', 'figure_eight')
NEW_STARTS = ((1103, (0.024, 0.016)), (1201, (-0.027, -0.020)))
CONTACT_CASES = tuple(product((0.1, 0.3, 0.6), (0.2, 0.4, 2/3)))
SOURCE_FILES = (
    'stl_scene.py', 'stl_bridge.py', 'stl_evaluate.py', 'render_methods.py',
    'wrench_shapes.py', 'wrench_center_demos.py', 'wrench_control.py', 'joint_wrench_proxy.py',
    'controller.py', 'lqr_control.py', 'mpc_control.py', 'robust_control.py',
    'assets/platform/platform_tray_frame.stl',
)


def gravity_level_reference(controller, gravity):
    """Retain tray heading and make its nominal normal vertical in the base frame."""
    initial = controller.initial_rotation
    vertical = -gravity / np.linalg.norm(gravity)
    heading = initial[:, 0] - vertical * np.dot(initial[:, 0], vertical)
    if np.linalg.norm(heading) <= 0.7:
        raise ValueError('Cannot define this study heading from a near-vertical tray axis')
    heading /= np.linalg.norm(heading)
    tilt_deg = np.rad2deg(np.arccos(np.clip(np.dot(initial[:, 2], vertical), -1.0, 1.0)))
    controller.initial_rotation = np.column_stack((heading, np.cross(vertical, heading), vertical))
    return float(tilt_deg)


def jobs_for(suite):
    regular = list(product(TASKS, CASES_BY_NAME, SEEDS))
    if suite == 'challenge':
        return [
            (method, task, case, seed, start, 0.3, 0.4)
            for method, task, case, (seed, start) in product(('mpc', 'lqr'), TASKS, CASES_BY_NAME, NEW_STARTS)
        ] + [
            ('mpc', task, 'noisy_50ms', seed, start, mu, inertia_ratio)
            for (mu, inertia_ratio), task, (seed, start) in product(CONTACT_CASES, TASKS, SEEDS)
            if (mu, inertia_ratio) != (0.3, 0.4)
        ]
    return [
        (method, task, case, seed, start, 0.3, 0.4)
        for method in METHODS for task, case, (seed, start) in regular
    ] + [
        ('lqr', task, 'noisy_50ms', seed, start, mu, inertia_ratio)
        for (mu, inertia_ratio), task, (seed, start) in product(CONTACT_CASES, TASKS, SEEDS)
        if (mu, inertia_ratio) != (0.3, 0.4)
    ]


def trial(job, *, suite, out, capture_video=False):
    method, task, case, seed, start, friction, inertia_ratio = job
    delay_s, torque_noise = CASES_BY_NAME[case]
    trial_dir = out / suite / f'{method}_{task}_{case}_{seed}_mu{friction:g}_k{inertia_ratio:.6f}'
    trial_dir.mkdir(exist_ok=True, parents=True)
    scene = dict(MEASURED, sliding_friction=friction)
    radius = MEASURED['ball_diameter_m'] / 2
    inertia = inertia_ratio * MEASURED['ball_mass_kg'] * radius**2
    original_builder = stl_scene.build_model
    build_count = 0
    reference_tilt = None

    def checked_scene(config):
        nonlocal build_count
        assert all(config[key] == val for key, val in scene.items())
        model, data, meta = original_builder(config)
        ball = meta['ball_body_id']
        geom = meta['ball_geom_id']
        if inertia_ratio != 0.4:
            # MuJoCo gives a sphere 2/5*m*r² by default. Update the free body's
            # moments and dependent constants; keep its mass and surface unchanged.
            model.body_inertia[ball] = inertia
            mujoco.mj_setConst(model, mujoco.MjData(model))
            mujoco.mj_forward(model, data)
        np.testing.assert_allclose(model.body_inertia[ball], [inertia]*3, rtol=1.e-9)
        assert np.isclose(model.body_mass[ball], MEASURED['ball_mass_kg'])
        assert np.isclose(model.geom_size[geom, 0], radius)
        assert np.isclose(model.body_mass[meta['tray_body_id']], MEASURED['tray_mass_kg'])
        assert all(np.isclose(model.geom_friction[g, 0], friction) for g in (geom, meta['tray_geom_id']))
        build_count += 1
        return model, data, meta

    factory = controller_factory(method)

    def initialize(model, data, config=None, **kwargs):
        nonlocal reference_tilt
        assert 'ball_initial_offset_m' not in config
        assert all(config[key] == val for key, val in scene.items())
        control = factory(model, data, config=config, **kwargs)
        if suite != 'original':
            reference_tilt = gravity_level_reference(control, model.opt.gravity)
        return control

    options = dict(sensor_model='joint_torque', joint_torque_std=torque_noise,
                   joint_velocity_std=0.0001, delay_s=delay_s, seed=seed,
                   controller_factory=initialize)
    video_path = trial_dir / f'{task}.mp4'
    label = '193 g / 74.5 mm | gravity-level ' + ('hybrid MPC/PD' if method == 'mpc' else method)
    with patch.object(stl_scene, 'build_model', checked_scene), use_actual_stl_scene():
        if task in SHAPES:
            raw = run_shape(task, config=dict(scene, ball_initial_offset_m=list(start)),
                            period_s=16.0 if task == 'rounded_square' else 8.0, output_dir=trial_dir,
                            video=str(video_path) if capture_video else None, controller_label=label, **options)
            rms = raw['plateau_tracking_rms_m']
            path_error = raw['max_tray_position_error_m']
            passed = bool(raw['numerical_pass'] and rms is not None and rms < 0.005 and path_error < 0.020)
            completed = raw['controller_reached_end']
            misses = raw['no_contact_samples_after_half_s']
            edge = raw['minimum_full_footprint_margin_m']
            estimator_error = raw['estimation_error_rms_m']
            center_error = None
        else:
            video = None
            if capture_video:
                from wrench_center_demos import WrenchVideo
                video = WrenchVideo(video_path, 0., 0., delay_s*1000, .0001, 'joint_torque', torque_noise)
            try:
                raw = run_one(start, task, scene=scene, video=video, label=f'{label} | {task}',
                              output_csv=trial_dir/f'{task}.csv' if capture_video else None, **options)
            finally:
                if video is not None:
                    video.close()
            passed = raw['success']
            completed = raw['completed']
            path_error = raw['max_tray_path_error_m']
            center_error = raw['last_second_max_true_error_m']
            misses = raw['ball_tray_no_contact_steps_after_half_second']
            edge = raw['minimum_full_footprint_edge_margin_m']
            estimator_error = raw['estimator_rms_error_m']
            rms = None
    assert build_count >= 2
    row = dict(method=method, task=task, case=case, seed=seed, start_xy_m=start, **scene,
               ball_inertia_factor=inertia_ratio, principal_inertia_kg_m2=inertia,
               delay_s=delay_s, joint_torque_noise_std_nm=torque_noise, performance_pass=passed,
               completed=completed, stop_reason=raw['stop_reason'], no_contact_samples_after_half_s=misses,
               min_ball_footprint_margin_m=edge, max_tray_path_error_m=path_error, estimator_rms_m=estimator_error,
               shape_tracking_rms_m=rms, final_second_max_center_error_m=center_error, model_builds_checked=build_count)
    if reference_tilt is not None:
        row['initial_reference_tilt_deg'] = reference_tilt
    if capture_video:
        (trial_dir/'video_result.json').write_text(json.dumps(raw, indent=2)+'\n')
        if task in SHAPES and raw['stop_reason']:
            from render_methods import show_exact_stop_reason
            show_exact_stop_reason(video_path, raw['stop_reason'])
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=('original','gravity','challenge','all'), default='all')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = '1'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suite in (('original','gravity','challenge') if args.suite == 'all' else (args.suite,)):
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(partial(trial, suite=suite, out=args.output_dir), jobs_for(suite)))
        with (args.output_dir / f'{suite}.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
        print(f'{suite}: {len(rows)} trials recorded', flush=True)
    hashes = {name: sha256((REPO/name).read_bytes()).hexdigest() for name in SOURCE_FILES}
    (args.output_dir/'source_hashes.json').write_text(json.dumps(hashes, indent=2)+'\n')


if __name__ == '__main__':
    main()
