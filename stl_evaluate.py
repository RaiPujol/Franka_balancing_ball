"""Offline fixed STL-model checks. No hardware command or truth-based control."""
import argparse,csv,json
from pathlib import Path
from stl_bridge import use_actual_stl_scene
from stl_scene import DEFAULTS, SOURCE_HASH
from lqr_control import LQGWrenchController
from wrench_shapes import run as shape_run
from wrench_center_demos import run_one as center_run

TASKS=('circle','rounded_square','figure_eight','centering','carrying')
SEEDS=((401,(.018,-.012)),(607,(-.016,.018)))
CASES=(('nominal_20ms',.02,.0001,0.,1.),('noisy_50ms',.05,.0005,0.,1.),
       ('noisy_80ms',.08,.001,0.,1.),('wrist_bias_50ms',.05,.0001,.001,1.),
       ('mass_mismatch_50ms',.05,.0001,0.,1.10))


def factory_for(bias,mass_factor):
    def factory(model,data,config=None,**kw):
        c=dict(config or {})
        c['tray_mass_kg']=float(c.get('tray_mass_kg', DEFAULTS['tray_mass_kg']))*mass_factor
        kw.update(torque_bias=(0.,bias,0.),measurement_std=.002,accel_std=.3,max_prediction=.15,
                  lqr_q_position=.36,lqr_q_velocity=.46874617737003077,lqr_r=1.)
        return LQGWrenchController(model,data,config=c,**kw)
    return factory


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',default='output/stl_fixed_suite')
    p.add_argument('--only-task',choices=TASKS)
    p.add_argument('--only-case',choices=[c[0] for c in CASES])
    p.add_argument('--only-seed',type=int,choices=[s[0] for s in SEEDS])
    a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);results=[]
    with use_actual_stl_scene():
        for task in TASKS:
            if a.only_task and a.only_task!=task:continue
            for condition,delay,noise,bias,mass_factor in CASES:
                if a.only_case and a.only_case!=condition:continue
                for seed,offset in SEEDS:
                    if a.only_seed and a.only_seed!=seed:continue
                    x={'task':task,'case':condition,'noise_seed':seed,'initial_offset_m':list(offset),
                       'delay_s':delay,'joint_torque_std_nm':noise,'joint_velocity_std_rad_s':.0001,
                       'wrist_torque_bias_y_nm':bias,'observer_mass_factor':mass_factor,
                       'plant_tray_mass_kg':DEFAULTS['tray_mass_kg'], 'platform_stl_sha256':SOURCE_HASH,
                       'collision_surface':'flat proxy: central dimple omitted',
                       'mount':'gripper-alignment hypothesis, rigid attachment'}
                    try:
                        factory=factory_for(bias,mass_factor)
                        if task in ('circle','rounded_square','figure_eight'):
                            r=shape_run(task,config={'ball_initial_offset_m':list(offset)},
                                period_s=16. if task=='rounded_square' else 8.,
                                output_dir=out/f'{task}_{condition}_{seed}',sensor_model='joint_torque',
                                joint_torque_std=noise,joint_velocity_std=.0001,delay_s=delay,seed=seed,
                                controller_factory=factory)
                            rms=r['plateau_tracking_rms_m']
                            x.update(completed=r['controller_reached_end'],stop_reason=r['stop_reason'],
                                rms_tracking_m=rms,estimator_rms_m=r['estimation_error_rms_m'],
                                min_full_footprint_margin_m=r['minimum_full_footprint_margin_m'],
                                numerical_pass=bool(r['numerical_pass'] and rms is not None and rms<.005
                                                    and r['max_tray_position_error_m']<.02))
                        else:
                            r=center_run(offset,task,sensor_model='joint_torque',joint_torque_std=noise,
                                joint_velocity_std=.0001,delay_s=delay,seed=seed,controller_factory=factory)
                            x.update(completed=r['completed'],stop_reason=r['stop_reason'],
                                last_second_max_error_m=r['last_second_max_true_error_m'],
                                estimator_rms_m=r['estimator_rms_error_m'],
                                min_full_footprint_margin_m=r['minimum_full_footprint_edge_margin_m'],
                                numerical_pass=r['success'])
                    except Exception as e:x.update(numerical_pass=False,exception=repr(e))
                    results.append(x);(out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
                    print(f'{len(results)} {task} {condition} seed {seed}: {"PASS" if x["numerical_pass"] else "FAIL"}',flush=True)
    fields=sorted({k for x in results for k in x})
    with (out/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(results)


if __name__=='__main__':main()
