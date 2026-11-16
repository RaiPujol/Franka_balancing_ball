#!/usr/bin/env python3
"""Offline MuJoCo trials. No physical robot interface is present."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from controller import SmoothPath, TrayController
from robot_scene import build_model


def run_trial(scene=None, *, mode='level', segment_s=3., video=None):
    model, data, meta = build_model(scene)
    cfg = meta['config']
    path = SmoothPath(segment_s=segment_s)
    controller = TrayController(model, data, path=path, mode=mode,
        arm_qpos=meta['arm_qpos_ids'], arm_dofs=meta['arm_dof_ids'],
        arm_actuators=meta['arm_actuator_ids'])
    renderer = writer = None
    next_frame = 0.
    fps = 24
    if video:
        import imageio.v2 as imageio
        from PIL import Image, ImageDraw
        renderer = mujoco.Renderer(model, height=480, width=640)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [.32, 0., .46]
        camera.distance = 1.6
        camera.azimuth = 135
        camera.elevation = -25
        Path(video).parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(video, fps=fps, codec='libx264', macro_block_size=16)
    dt = model.opt.timestep
    nsteps = int(np.ceil((path.end_time + 1.) / dt))
    max_error, min_margin, max_path_error = 0., np.inf, 0.
    max_height_error = 0.
    ball_contact_steps = 0
    no_contact_s = max_no_contact_s = 0.
    max_tilt_deg = max_orientation_error_deg = 0.
    contacts = {}
    failure = None
    radius = cfg['ball_diameter_m'] / 2.
    try:
        for _ in range(nsteps):
            state = controller.step(data.time)
            mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                failure = 'nonfinite physics state'
                break
            # Compute local coordinates after the actual physics step, not the
            # pre-step controller estimate, so no visualization drives metrics.
            mujoco.mj_forward(model, data)
            R = data.site_xmat[meta['tray_site_id']].reshape(3, 3)
            bp = R.T @ (data.xpos[meta['ball_body_id']] - data.site_xpos[meta['tray_site_id']])
            max_error = max(max_error, float(np.linalg.norm(bp[:2])))
            min_margin = min(min_margin, float(cfg['tray_length_m']/2 - radius - abs(bp[0])),
                             float(cfg['tray_width_m']/2 - radius - abs(bp[1])))
            max_height_error = max(max_height_error, float(abs(bp[2] - radius)))
            max_path_error = max(max_path_error, float(np.linalg.norm(
                data.site_xpos[meta['tray_site_id']] - state['target_pos'])))
            max_tilt_deg = max(max_tilt_deg, float(np.degrees(np.arccos(np.clip(R[2, 2], -1., 1.)))))
            rot_error = R.T @ state['target_rotation']
            max_orientation_error_deg = max(max_orientation_error_deg, float(np.degrees(
                np.arccos(np.clip((np.trace(rot_error)-1.)/2., -1., 1.)))))
            on_tray = False
            for c in data.contact[:data.ncon]:
                if {int(c.geom1), int(c.geom2)} == {meta['ball_geom_id'], meta['tray_geom_id']}:
                    on_tray = True
                # Record penetrating collisions between different robot/scene
                # bodies; the idealized tray attachment is excluded explicitly.
                if c.dist < -1e-4:
                    a = int(model.geom_bodyid[c.geom1]); b = int(model.geom_bodyid[c.geom2])
                    names = tuple(sorted((mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, a) or 'world',
                                          mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or 'world')))
                    if 'ball' not in names:
                        key = ' / '.join(names)
                        contacts[key] = min(contacts.get(key, 0.), float(c.dist))
                if (meta['ball_geom_id'] in (int(c.geom1), int(c.geom2))
                        and meta['tray_geom_id'] not in (int(c.geom1), int(c.geom2))):
                    other_geom = int(c.geom2) if int(c.geom1) == meta['ball_geom_id'] else int(c.geom1)
                    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                            int(model.geom_bodyid[other_geom])) or 'world'
                    key = 'ball / ' + name
                    contacts[key] = min(contacts.get(key, 0.), float(c.dist))
            ball_contact_steps += int(on_tray)
            # Give the initially 0.2 mm elevated ball time to settle. Brief
            # solver separations (<20 ms) are recorded but not treated as a fall.
            if data.time > .05:
                no_contact_s = 0. if on_tray else no_contact_s + dt
                max_no_contact_s = max(max_no_contact_s, no_contact_s)
            if writer and data.time + 1e-10 >= next_frame:
                renderer.update_scene(data, camera=camera)
                image = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, 640, 38), fill='white')
                draw.text((8, 7), f'SIMULATION ONLY | {mode} | {data.time:.1f} s', fill='black')
                writer.append_data(np.asarray(image))
                next_frame += 1. / fps
    finally:
        if writer: writer.close()
        if renderer: renderer.close()
    warnings = {str(i): int(w.number) for i, w in enumerate(data.warning) if w.number}
    if not np.isfinite(min_margin):
        failure = failure or 'trial ended without valid state'
    result = {
        'mode': mode, 'scene': cfg, 'segment_s': segment_s,
        'duration_s': float(data.time), 'intended_duration_s': path.end_time + 1.,
        'max_ball_error_m': max_error, 'min_edge_margin_m': float(min_margin),
        'max_height_error_m': max_height_error, 'max_tray_path_error_m': max_path_error,
        'max_tray_tilt_deg': max_tilt_deg, 'max_orientation_tracking_error_deg': max_orientation_error_deg,
        'ball_tray_contact_fraction': ball_contact_steps / max(1, nsteps),
        'longest_ball_tray_separation_s_after_settling': max_no_contact_s,
        'other_penetrating_contacts': contacts, 'mujoco_warnings': warnings,
        'error': failure, 'scope': 'simulation only; rigid ideal bracket; no shelves or physical sensing',
    }
    # A conservative numerical criterion, not proof of physical success.
    result['success'] = bool(failure is None and not warnings and not contacts and min_margin >= 0.
                             and data.time >= path.end_time + 1. - dt/2
                             and max_no_contact_s <= .02 and max_height_error < .008 and max_path_error < .02)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', help='JSON scene overrides')
    p.add_argument('--mode', choices=['level','feedforward','oracle_feedback'], default='level')
    p.add_argument('--segment-s', type=float, default=3.)
    p.add_argument('--video')
    p.add_argument('--output', default='output/trial.json')
    args = p.parse_args()
    scene = json.loads(Path(args.config).read_text()) if args.config else None
    result = run_trial(scene, mode=args.mode, segment_s=args.segment_s, video=args.video)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return 0 if result['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
