#!/usr/bin/env python3
"""Offline STATIC estimate and LQR slope preview; never controls hardware.

The input is a previously recorded Franka wrench stream, not a simulation
proxy. The center target and zero feedforward isolate the static slice of the
held-out LQR candidate. No trajectory or dynamic hardware claim is made.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from hardware.shadow_static_filter import shadow_replay, dispatch_to_robot, HardwareExecutionLocked

# Fixed to the held-out candidate. Deliberately not exposed as a CLI setting.
KP=0.6
KD=0.8
MAX_TILT_RAD=float(np.deg2rad(7.0))
KALMAN_MEASUREMENT_STD_M=.002
KALMAN_ACCEL_STD=.3
MAX_GAP_S=.15


def static_center_preview(filtered_samples):
    """Return analysis-only slopes; never return a pose or command message.

    Invalid estimator samples yield no preview. Sources with nonadvancing
    timestamps are also rejected even if a caller incorrectly marks them valid.
    Slopes are relative to the measured tray +x/+y axes, not robot base axes.
    """
    results=[]
    last_time=None
    for sample in filtered_samples:
        try:t=float(sample['source_robot_time_s'])
        except (KeyError,TypeError,ValueError):t=float('nan')
        invalid=None
        if not np.isfinite(t):invalid='invalid_robot_time'
        elif last_time is not None and t<=last_time:invalid='nonadvancing_robot_time'
        elif last_time is not None and t-last_time>MAX_GAP_S:invalid='sample_gap_exceeded'
        elif not sample.get('valid',False):invalid='invalid_observation'
        last_time=t if np.isfinite(t) else None
        try:
            xy=np.asarray(sample.get('estimate_xy_m'),float)
            velocity=np.asarray(sample.get('estimated_velocity_m_s'),float)
        except (TypeError,ValueError):
            xy=velocity=np.asarray([])
        if invalid is None and (xy.shape!=(2,) or velocity.shape!=(2,) or
                                not np.isfinite(xy).all() or not np.isfinite(velocity).all()):
            invalid='nonfinite_or_malformed_state'
        if invalid is not None:
            results.append({'source_robot_time_s':t if np.isfinite(t) else None,
                            'preview_valid':False,'reason':invalid,
                            'preview_slope_xy_rad':None})
            continue
        # Matches the existing simulator controller for a fixed target at the
        # tray center, zero tray acceleration and zero target acceleration.
        # The bounded slope vector parameterizes the rotation [slope_y,
        # -slope_x,0]. It is a model output, never an arm pose or command.
        slope=KP*xy+KD*velocity
        magnitude=float(np.linalg.norm(slope))
        if magnitude>MAX_TILT_RAD:
            slope=slope*(MAX_TILT_RAD/magnitude)
        results.append({'source_robot_time_s':t,'preview_valid':True,
                        'reason':'STATIC PREVIEW ONLY - NOT A ROBOT COMMAND',
                        'preview_slope_xy_rad':slope.tolist()})
    return results


def run_preview(empty_path,loaded_path,transform,*,min_normal=.2):
    """Use recorded wrench and independently measured transform only."""
    shadow=shadow_replay(empty_path,loaded_path,transform,min_normal=min_normal,
        measurement_std_m=KALMAN_MEASUREMENT_STD_M,accel_std=KALMAN_ACCEL_STD,max_gap_s=MAX_GAP_S)
    previews=static_center_preview(shadow['samples'])
    return {'kind':'static_lqr_preview_only', 'hardware_execution_allowed':False,
            'motion_commands_issued':0, 'target':'tray center, stationary only',
            'gains':{'kp':KP,'kd':KD,'max_tilt_rad':MAX_TILT_RAD},
            'kalman':{'measurement_std_m':KALMAN_MEASUREMENT_STD_M,
                      'accel_std':KALMAN_ACCEL_STD,'max_gap_s':MAX_GAP_S},
            'time_limitations':shadow['time_limitations'],
            'limitations':'STATIC PREVIEW ONLY. Not dynamic, no verified latency, physical robot command or motion authorization.',
            'samples':previews,'valid_previews':sum(int(p['preview_valid']) for p in previews)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--empty',required=True,help='recorded static empty tray JSONL')
    p.add_argument('--loaded',required=True,help='recorded static loaded tray JSONL')
    p.add_argument('--ee-to-tray',required=True,help='JSON with measured T_EE_tray_row_major')
    p.add_argument('--output',required=True,help='new JSON path; no overwrite')
    p.add_argument('--min-normal',type=float,default=.2)
    a=p.parse_args()
    transform=json.loads(Path(a.ee_to_tray).read_text())['T_EE_tray_row_major']
    result=run_preview(a.empty,a.loaded,transform,min_normal=a.min_normal)
    with Path(a.output).open('x',encoding='utf-8') as f:
        json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print('STATIC PREVIEW ONLY: %d/%d estimates; physical motion remains unavailable.'
          % (result['valid_previews'],len(result['samples'])))
    return 0 if result['valid_previews'] else 2


if __name__=='__main__':
    raise SystemExit(main())
