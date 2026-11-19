"""Predeclared matched simulation study, no hardware access or ball oracle.

One process per method; fixed noise seeds, tasks, target paths and evaluation
criteria. Results are deliberately checkpointed so failures are retained.
"""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from wrench_control import WrenchController
from robust_control import RobustWrenchController
from wrench_shapes import run as shape_run
from wrench_center_demos import run_one as center_run

# Held-out seeds are not used to choose candidate controller gains.
SEEDS=((401,(.018,-.012)),(607,(-.016,.018)))
TASKS=('circle','rounded_square','figure_eight','centering','carrying')
# (name, measurement delay, synthetic joint torque std, wrench torque bias,
#  multiplicative tray mass error in *observer* model only)
CASES=(('nominal_20ms',.02,.0001,0.,1.),('noisy_50ms',.05,.0005,0.,1.),
       ('noisy_80ms',.08,.001,0.,1.),('biased_50ms',.05,.0001,.001,1.),
       ('mass_mismatch_50ms',.05,.0001,0.,1.10))


def class_for(method):
    if method=='pd': return WrenchController
    if method=='kalman_pd': return RobustWrenchController
    if method=='lqr':
        from lqr_control import LQGWrenchController
        return LQGWrenchController
    if method=='mpc':
        from mpc_control import MPCWrenchController
        return MPCWrenchController
    raise ValueError('unknown method')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',required=True,choices=['pd','kalman_pd','lqr','mpc'])
    p.add_argument('--lqr-kp',type=float,default=.6)
    p.add_argument('--lqr-kd',type=float,default=.8)
    p.add_argument('--tray-mass',type=float,default=.12,
                   help='nominal plant tray mass in kg, used for observer-model comparisons')
    p.add_argument('--only-task',choices=TASKS)
    p.add_argument('--only-case',choices=[c[0] for c in CASES])
    p.add_argument('--only-seed',type=int,choices=[s[0] for s in SEEDS])
    p.add_argument('--output-dir',default='output/compare')
    args=p.parse_args();cls=class_for(args.method)
    root=Path(args.output_dir)/args.method;root.mkdir(parents=True,exist_ok=True)
    results=[]
    for task in TASKS:
        if args.only_task and task!=args.only_task:continue
        for case,delay,noise,bias,mass_factor in CASES:
            if args.only_case and case!=args.only_case:continue
            for seed,offset in SEEDS:
                if args.only_seed and seed!=args.only_seed:continue
                record={'task':task,'case':case,'method':args.method,'seed':seed,
                        'initial_offset_m':list(offset),'delay_s':delay,'joint_torque_std_nm':noise,
                        'joint_velocity_std_rad_s':.0001,'wrench_bias_y_nm':bias,
                        'plant_tray_mass_kg':args.tray_mass,'observer_tray_mass_factor':mass_factor}
                holder=[]
                def factory(model,data,config=None,**kwargs):
                    cfg=dict(config or {})
                    cfg['tray_mass_kg']=float(cfg.get('tray_mass_kg',args.tray_mass))*mass_factor
                    kwargs['torque_bias']=(0.,bias,0.)
                    if args.method!='pd':
                        kwargs.update(measurement_std=.002,accel_std=.3,max_prediction=.15)
                    if args.method=='lqr':
                        # Invert continuous-time 2-state LQR gain equations for
                        # a specified gain pair. R=1; Q terms must be nonnegative.
                        g=9.81/1.4; kp=args.lqr_kp;kd=args.lqr_kd
                        qv=kd*kd-2*kp/g
                        if kp<=0 or kd<=0 or qv < 0:raise ValueError('LQR gain pair not realizable by positive Q')
                        kwargs.update(lqr_q_position=kp*kp,lqr_q_velocity=qv,lqr_r=1.)
                    instance=cls(model,data,config=cfg,**kwargs);holder.append(instance);return instance
                started=time.perf_counter()
                try:
                    if task in ('circle','rounded_square','figure_eight'):
                        r=shape_run(task,config={'ball_initial_offset_m':list(offset),'tray_mass_kg':args.tray_mass},
                            period_s=16. if task=='rounded_square' else 8.,output_dir=root/f'{task}_{case}_{seed}',
                            sensor_model='joint_torque',joint_torque_std=noise,joint_velocity_std=.0001,
                            delay_s=delay,seed=seed,controller_factory=factory)
                        rms=r['plateau_tracking_rms_m']
                        record.update(containment_pass=r['numerical_pass'],rms_tracking_m=rms,
                            performance_pass=bool(r['numerical_pass'] and rms is not None and rms < .005
                                                  and r['max_tray_position_error_m'] < .02),
                            stop_reason=r['stop_reason'],edge_margin_m=r['minimum_full_footprint_margin_m'],
                            estimator_rms_m=r['estimation_error_rms_m'])
                    else:
                        r=center_run(offset,task,scene={'tray_mass_kg':args.tray_mass},sensor_model='joint_torque',joint_torque_std=noise,
                            joint_velocity_std=.0001,delay_s=delay,seed=seed,controller_factory=factory)
                        record.update(containment_pass=bool(r['completed'] and r['minimum_full_footprint_edge_margin_m']>=0),
                            performance_pass=r['success'],last_second_max_m=r['last_second_max_true_error_m'],
                            stop_reason=r['stop_reason'],edge_margin_m=r['minimum_full_footprint_edge_margin_m'],
                            estimator_rms_m=r['estimator_rms_error_m'])
                except Exception as e:
                    record.update(performance_pass=False,containment_pass=False,exception=repr(e))
                record['wall_time_s']=time.perf_counter()-started
                if holder and hasattr(holder[0],'mpc'):
                    times=holder[0].mpc.solve_times
                    record['mpc_solve_count']=len(times)
                    record['mpc_solver_failures']=holder[0].mpc.solver_failures
                    record['mpc_solve_p95_s']=float(np.percentile(times,95)) if times else None
                    record['mpc_solve_max_s']=max(times) if times else None
                results.append(record)
                (root/'results.json').write_text(json.dumps(results,indent=2)+'\n')
                print(f'{args.method} {len(results)} {task} {case} seed {seed} pass={record["performance_pass"]}',flush=True)
    fields=sorted({k for r in results for k in r})
    with (root/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(results)
    manifest={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
              ['compare_methods.py','controller.py','wrench_control.py','robot_scene.py']}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
