#!/usr/bin/env python3
"""Replay a recorded stationary wrench stream, without controlling a robot."""
import argparse
import json
from pathlib import Path
import numpy as np
try:
    from .analyze_static_wrench import matrix,load_recording,wrench_at_tray,_angle
except ImportError:
    from analyze_static_wrench import matrix,load_recording,wrench_at_tray,_angle


def replay(empty_path,loaded_path,transform,min_normal=.2,min_samples=20):
    T=matrix(transform)
    if not np.isfinite(min_normal) or min_normal<=0:raise ValueError('min_normal must be positive')
    _,empty,empty_rejects=load_recording(empty_path,min_samples=min_samples)
    _,loaded,loaded_rejects=load_recording(loaded_path,min_samples=min_samples)
    fields=('F_T_EE','EE_T_K','m_ee','m_load','m_total','F_x_Cee','F_x_Cload',
            'F_x_Ctotal','I_ee','I_load','I_total')
    try:setup={key:np.asarray(empty[0][key],float) for key in fields}
    except KeyError as e:raise ValueError('missing setup field '+str(e))
    transformed=[];ref=None
    def check(record):
        nonlocal ref
        for key, val in setup.items():
            actual=np.asarray(record.get(key,[]),float)
            if actual.shape!=val.shape or not np.isfinite(actual).all() or not np.allclose(actual,val,atol=1e-8,rtol=1e-5):
                raise ValueError('configured tool/load or stiffness frame changed: '+key)
        w,P=wrench_at_tray(record,T)
        if ref is None:ref=P
        if np.linalg.norm(P[:3,3]-ref[:3,3])>.002 or _angle(P[:3,:3],ref[:3,:3])>np.deg2rad(2.):
            raise ValueError('tray pose changed beyond 2mm/2deg preliminary gate')
        return w
    for rec in empty:transformed.append(check(rec))
    baseline=np.median(transformed,axis=0)
    samples=[]
    for rec in loaded:
        diff=check(rec)-baseline;fz=float(diff[2])
        good=bool(abs(fz)>=min_normal)
        est=[float(-diff[4]/fz),float(diff[3]/fz)] if good else None
        samples.append({'source_robot_time_s':rec['source_robot_time_s'],
           'source_header_stamp_s':rec['source_header_stamp_s'],
           'valid_estimate':good,'reason':'above_normal_force_gate' if good else 'insufficient_normal_force',
           'normal_force_n':fz,'estimated_ball_xy_m':est})
    return {'kind':'static_only_replay','empty_file':str(empty_path),'loaded_file':str(loaded_path),
            'T_EE_tray_row_major':T.tolist(),'minimum_absolute_normal_force_n':min_normal,
            'empty_wrench_median_tray':baseline.tolist(),'empty_rejected':empty_rejects,
            'loaded_rejected':loaded_rejects,'accepted_static_records':len(loaded),
            'valid_estimates':sum(s['valid_estimate'] for s in samples),
            'limitations':'Offline, stationary and same pose only. Singularity, calibration accuracy, latency and hardware safety not verified.',
            'samples':samples}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--empty',required=True);p.add_argument('--loaded',required=True)
    p.add_argument('--ee-to-tray',required=True,help='JSON with measured T_EE_tray_row_major')
    p.add_argument('--output',required=True);p.add_argument('--min-normal',type=float,default=.2)
    a=p.parse_args();T=json.loads(Path(a.ee_to_tray).read_text())['T_EE_tray_row_major']
    report=replay(a.empty,a.loaded,T,min_normal=a.min_normal)
    with Path(a.output).open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print('%d/%d samples passed normal-force gate; output %s' % (report['valid_estimates'],report['accepted_static_records'],a.output))
    return 0 if report['valid_estimates']>0 else 2


if __name__=='__main__':raise SystemExit(main())
