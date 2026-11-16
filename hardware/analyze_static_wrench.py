#!/usr/bin/env python3
"""Offline analysis of read-only static Franka wrench recordings.

Requires a measured end-effector-to-tray transform and independently measured
ball positions. Does not connect to a robot, fit a motion policy, or command it.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def matrix(a, column_major=False):
    a=np.asarray(a,dtype=float)
    if a.size != 16 or not np.isfinite(a).all(): raise ValueError('transform must have 16 finite numbers')
    T=a.reshape((4,4),order='F' if column_major else 'C')
    R=T[:3,:3]
    if not np.allclose(T[3],[0,0,0,1],atol=1e-8) or not np.allclose(R.T@R,np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(R),1.,atol=1e-5):
        raise ValueError('invalid rigid homogeneous transform')
    return T


def wrench_at_tray(sample,T_E_T):
    """Rotate wrench in O about K to tray T origin: base sign convention retained."""
    O_E=matrix(sample['O_T_EE'],column_major=True)
    E_K=matrix(sample['EE_T_K'],column_major=True)
    O_K=O_E@E_K
    O_T=O_E@T_E_T
    wrench=np.asarray(sample['O_F_ext_hat_K'],dtype=float)
    if wrench.shape!=(6,) or not np.isfinite(wrench).all():raise ValueError('invalid external wrench')
    R=O_T[:3,:3]; F=wrench[:3]; M=wrench[3:]
    moment_at_tray=M+np.cross(O_K[:3,3]-O_T[:3,3],F)
    return np.r_[R.T@F,R.T@moment_at_tray], O_T


def load_recording(path, min_samples=20, max_joint_speed=.01):
    lines=[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if len(lines)<2 or lines[0].get('schema')!='franka_wrench_observation_v1' or lines[-1].get('kind')!='end':
        raise ValueError('%s is not a complete recorder file' % path)
    if lines[-1].get('errors'):
        raise ValueError('%s recorder reported errors: %s' % (path,lines[-1]['errors']))
    good=[];reasons={};last=None
    for r in lines[1:-1]:
        reason=None
        if r.get('kind')!='sample' or not r.get('candidate_for_offline_analysis',False):reason='invalid_or_stale'
        else:
            dq=np.asarray(r.get('dq',[]),float)
            t=float(r.get('source_robot_time_s',float('nan')))
            if not np.isfinite(t) or (last is not None and t<=last):reason='nonmonotonic_robot_time'
            elif dq.shape!=(7,) or not np.isfinite(dq).all() or np.max(np.abs(dq))>max_joint_speed:reason='robot_moving_or_invalid_velocity'
            else:last=t
        if reason:reasons[reason]=reasons.get(reason,0)+1
        else:good.append(r)
    if len(good)<min_samples:raise ValueError('%s has %d acceptable samples; need %d' % (path,len(good),min_samples))
    return lines[0],good,reasons


def _angle(R1,R2):
    return float(np.arccos(np.clip((np.trace(R1.T@R2)-1)/2,-1.,1.)))


def run_analysis(empty_path, points, T_E_T, *, min_normal=.20, max_translation=.002, max_angle_deg=2.,min_samples=20):
    """Evaluate a static zero-reference geometric model at known ball positions.

    `points` is [(x_m, y_m, path),...], labels are measurement ground truth,
    never used as filter input or as controller input. Rejects inconsistent pose.
    """
    T_E_T=matrix(T_E_T)
    if not points:raise ValueError('at least one independently measured ball position required')
    if not np.isfinite([min_normal,max_translation,max_angle_deg]).all() or min(min_normal,max_translation,max_angle_deg)<=0:
        raise ValueError('analysis limits must be positive and finite')
    eh,empty,empty_rejects=load_recording(empty_path,min_samples=min_samples)
    setup_fields=('F_T_EE','EE_T_K','m_ee','m_load','m_total',
                  'F_x_Cee','F_x_Cload','F_x_Ctotal','I_ee','I_load','I_total')
    try:reference_setup={name:np.asarray(empty[0][name],float) for name in setup_fields}
    except KeyError as e:raise ValueError('recording missing setup field '+str(e))
    def check_setup(sample):
        for name,refval in reference_setup.items():
            val=np.asarray(sample.get(name,[]),float)
            if val.shape!=refval.shape or not np.isfinite(val).all() or not np.allclose(val,refval,atol=1e-8,rtol=1e-5):
                raise ValueError('configured tool/load or stiffness frame changed: '+name)
    for sample in empty:check_setup(sample)
    sample_poses=[wrench_at_tray(s,T_E_T) for s in empty]
    ref=sample_poses[0][1]
    def check_pose(T):
        if np.linalg.norm(T[:3,3]-ref[:3,3])>max_translation or _angle(T[:3,:3],ref[:3,:3])>np.deg2rad(max_angle_deg):
            raise ValueError('recordings do not use the same stationary tray pose within tolerances')
    for _,T in sample_poses:check_pose(T)
    ew=np.asarray([w for w,_ in sample_poses]);base=np.median(ew,axis=0)
    k=max(1,len(ew)//10)
    drift=np.mean(ew[-k:],axis=0)-np.mean(ew[:k],axis=0)
    result={'status':'offline_static_evaluation_only',
      'sensor':'O_F_ext_hat_K in base, about K; transformed to independently supplied tray frame',
      'empty_file':str(empty_path),'empty_label':eh.get('label'),'empty_samples':len(empty),
      'empty_rejected_counts':empty_rejects,'empty_wrench_median_tray':base.tolist(),
      'recorded_configured_tool_and_load':{key:value.tolist() for key,value in reference_setup.items()},
      'empty_wrench_std_tray':np.std(ew,axis=0,ddof=1).tolist(),
      'empty_start_end_mean_drift_tray':drift.tolist(),
      'same_pose_tolerance_m':max_translation,'same_pose_tolerance_deg':max_angle_deg,
      'minimum_absolute_normal_load_n':min_normal,'T_EE_tray_row_major':T_E_T.tolist(),
      'assumptions':['ball contacting horizontal tray without touching robot or other obstacles',
                     'empty and loaded tray share the same pose and mounting',
                     'tray frame origin and orientation were measured independently',
                     'raw wrench might be zero or biased near singularities; no live guarantee'],
      'positions':[]}
    errors=[]
    for x,y,path in points:
        if not np.isfinite([x,y]).all():raise ValueError('ball coordinates must be finite')
        header,records,rejected=load_recording(path,min_samples=min_samples)
        estimates=[]; dropped=0; normals=[]
        for sample in records:
            check_setup(sample);raw,T=wrench_at_tray(sample,T_E_T);check_pose(T)
            f=raw[:3]-base[:3]; tau=raw[3:]-base[3:]
            if abs(f[2])<min_normal: dropped+=1;continue
            estimates.append([-tau[1]/f[2],tau[0]/f[2]])
            normals.append(float(f[2]))
        if len(estimates)<min_samples:
            raise ValueError('%s has only %d samples above normal-force gate' % (path,len(estimates)))
        estimates=np.asarray(estimates); err=estimates-[x,y]; dist=np.linalg.norm(err,axis=1)
        errors.extend(dist.tolist())
        result['positions'].append({'file':str(path),'label':header.get('label'),
             'known_xy_m':[x,y],'accepted_samples':len(estimates),
             'rejected':rejected,'insufficient_normal_force_samples':dropped,
             'normal_force_median_n':float(np.median(normals)),
             'estimated_mean_xy_m':np.mean(estimates,axis=0).tolist(),
             'estimated_std_xy_m':np.std(estimates,axis=0,ddof=1).tolist(),
             'mean_error_xy_m':np.mean(err,axis=0).tolist(),
             'error_rms_m':float(np.sqrt(np.mean(dist**2))),
             'error_p95_m':float(np.percentile(dist,95))})
    result['overall_sample_weighted_error_rms_m']=float(np.sqrt(np.mean(np.square(errors))))
    result['important_limit']='Static, fixed-pose data only; no dynamic latency measurement, generalization, or hardware authorization.'
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--empty',required=True,help='empty tray JSONL from recorder')
    p.add_argument('--point',action='append',nargs=3,metavar=('X_MM','Y_MM','FILE'),required=True,
                   help='independently measured x y ball position on tray and corresponding JSONL; repeat')
    p.add_argument('--ee-to-tray',required=True,help='JSON file with measured T_EE_tray_row_major (4x4)')
    p.add_argument('--output',required=True,help='new JSON path (never overwritten)')
    p.add_argument('--min-normal',type=float,default=.2)
    a=p.parse_args()
    T=json.loads(Path(a.ee_to_tray).read_text())['T_EE_tray_row_major']
    points=[(float(x)/1000,float(y)/1000,path) for x,y,path in a.point]
    result=run_analysis(a.empty,points,T,min_normal=a.min_normal)
    with Path(a.output).open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print('Static RMS estimate error: %.2f mm; output %s' % (result['overall_sample_weighted_error_rms_m']*1000,a.output))


if __name__=='__main__':main()
