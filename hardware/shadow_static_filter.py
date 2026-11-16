#!/usr/bin/env python3
"""Offline static shadow filter for recorded Franka wrench data.

No ROS, robot connection, command creation or live control is present. The
source robot timestamp orders measurements; its absolute latency is unknown.
The measurement model and empty baseline are valid only for a stationary,
fixed-pose, independently measured tray. Outputs are evaluation data, never
robot targets or an authorization to move.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from hardware.replay_static_wrench import replay
from position_filter import DelayedPositionKalman


class HardwareExecutionLocked(RuntimeError):
    """An attempt to use this analysis module as a robot controller."""


def dispatch_to_robot(*_args, **_kwargs):
    """There is deliberately no switch, argument, or configuration to unlock."""
    raise HardwareExecutionLocked('Shadow analysis has no physical execution capability')


def filter_samples(samples, *, measurement_std_m=.002, accel_std=.20, max_gap_s=.15):
    """Filter static measurements in source-robot-time order, with no commands.

    Unknown sample latency is *not* set to an assumed value. `now=stamp` is
    used solely to perform an offline sample-time update; it does not measure or
    compensate delay. Any invalid reading, duplicate/backwards stamp, nonfinite
    position or long gap invalidates the current estimate and resets the filter.
    A subsequent valid sample starts a new independent segment.
    """
    vals = [measurement_std_m, accel_std, max_gap_s]
    if not np.isfinite(vals).all() or min(vals) <= 0:
        raise ValueError('filter settings must be positive and finite')
    result = []
    observer = None
    previous_stamp = None
    segment = 0
    for raw in samples:
        try:
            stamp = float(raw['source_robot_time_s'])
        except (KeyError, TypeError, ValueError):
            stamp = float('nan')
        try:
            point = np.asarray(raw.get('estimated_ball_xy_m') if raw.get('valid_estimate') else [],dtype=float)
        except (TypeError, ValueError):
            point = np.asarray([])
        reason = None
        if not np.isfinite(stamp):
            reason = 'nonfinite_robot_time'
        elif previous_stamp is not None and stamp <= previous_stamp:
            reason = 'nonadvancing_robot_time'
        elif point.shape != (2,) or not np.isfinite(point).all():
            reason = 'invalid_wrench_position'
        elif previous_stamp is not None and stamp - previous_stamp > max_gap_s:
            reason = 'sample_gap_exceeded'
        previous_stamp = stamp if np.isfinite(stamp) else None
        if reason is not None:
            observer = None
            result.append({'source_robot_time_s':stamp if np.isfinite(stamp) else None,
                           'valid':False,'reason':reason,'estimate_xy_m':None,
                           'estimated_velocity_m_s':None,'segment':None})
            continue
        if observer is None:
            observer = DelayedPositionKalman(measurement_std=measurement_std_m,
                                             accel_std=accel_std,max_prediction=max_gap_s)
            segment += 1
        try:
            # A zero prediction horizon is a computational convention for
            # offline smoothing, NOT a claim of zero physical sensor latency.
            state = observer.update(point,stamp=stamp,now=stamp)
        except (ValueError,RuntimeError):
            observer = None
            result.append({'source_robot_time_s':stamp,'valid':False,
                           'reason':'filter_rejected','estimate_xy_m':None,
                           'estimated_velocity_m_s':None,'segment':None})
            continue
        result.append({'source_robot_time_s':stamp,'valid':True,
                       'reason':'offline_static_filter_only',
                       'estimate_xy_m':state['position_m'].tolist(),
                       'estimated_velocity_m_s':state['velocity_m_s'].tolist(),
                       'segment':segment})
    return result


def shadow_replay(empty_path,loaded_path,transform,*,min_normal=.2,
                  measurement_std_m=.002,accel_std=.20,max_gap_s=.15):
    """Perform a static replay using recorded *measured* wrench, not a proxy."""
    raw = replay(empty_path,loaded_path,transform,min_normal=min_normal)
    # replay() discards rejected source records. Do not hide these gaps from
    # the filter: require a clean source stream for this stricter shadow mode.
    if raw['empty_rejected'] or raw['loaded_rejected']:
        raise ValueError('source recordings contain rejected or stale records')
    filtered = filter_samples(raw['samples'],measurement_std_m=measurement_std_m,
                              accel_std=accel_std,max_gap_s=max_gap_s)
    return {'kind':'offline_static_shadow_only',
            'hardware_execution_allowed':False,
            'input_description':'recorded O_F_ext_hat_K, robot state and measured tray transform',
            'time_limitations':'Robot timestamps order samples; sensor age, ROS latency and delay are not measured.',
            'scope':'Stationary fixed tray, ball contact, unchanged payload and no external contacts only.',
            'force_gate_n':min_normal,'filter_settings':{'measurement_std_m':measurement_std_m,
            'assumed_acceleration_process_std':accel_std,'max_sample_gap_s':max_gap_s},
            'baseline':raw['empty_wrench_median_tray'],'samples':filtered,
            'valid_samples':sum(int(x['valid']) for x in filtered),
            'motion_commands_issued':0}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--empty',required=True,help='recorded stationary empty tray JSONL')
    p.add_argument('--loaded',required=True,help='recorded stationary loaded tray JSONL')
    p.add_argument('--ee-to-tray',required=True,help='JSON with measured T_EE_tray_row_major')
    p.add_argument('--output',required=True,help='new JSON output path, never overwritten')
    p.add_argument('--min-normal',type=float,default=.2)
    p.add_argument('--max-gap-s',type=float,default=.15)
    a=p.parse_args()
    transform=json.loads(Path(a.ee_to_tray).read_text())['T_EE_tray_row_major']
    result=shadow_replay(a.empty,a.loaded,transform,min_normal=a.min_normal,max_gap_s=a.max_gap_s)
    with Path(a.output).open('x',encoding='utf-8') as f:
        json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print('Offline stationary estimates: %d/%d valid; physical execution remains unavailable.'
          % (result['valid_samples'],len(result['samples'])))
    return 0 if result['valid_samples'] else 2


if __name__=='__main__':
    raise SystemExit(main())
