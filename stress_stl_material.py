"""Offline actual-STL friction/bolt sensitivity, with evaluator-only truth.

Two initial states/noise seeds; friction and bolt masses are hypothetical.
The same assumed tool model is used by plant and observer, so this is not a
robustness test against incorrectly measured bolt inertia. Flat proxy omits
central dimple; attachment is rigidly idealized. No hardware commands.
"""
import argparse, csv, itertools, json, multiprocessing, os, traceback
from pathlib import Path


def trial(item):
    i,task,friction,bolt,delay,seed,offset,out=item
    from stl_bridge import use_actual_stl_scene
    from stl_evaluate import factory_for
    from wrench_shapes import run as shape_run
    from wrench_center_demos import run_one
    scene={'sliding_friction':friction,'bolt_mass_kg':bolt}
    x={'id':i,'task':task,'sliding_friction':friction,'bolt_mass_kg':bolt,
       'delay_s':delay,'seed':seed,'initial_offset_m':list(offset),
       'joint_torque_noise_std_nm':.0005,'joint_velocity_noise_std_rad_s':.0001,
       'observer_model':'same bolt mass and nominal platform inertia as plant',
       'collision_surface':'flat proxy; dimple omitted',
       'limitations':'hypothetical unmeasured friction and bolt mass; bolt is surrogate sphere, real inertia and mount unknown'}
    try:
        with use_actual_stl_scene():
            common=dict(sensor_model='joint_torque',joint_torque_std=.0005,joint_velocity_std=.0001,
                        delay_s=delay,seed=seed,controller_factory=factory_for(0.,1.))
            if task=='circle':
                r=shape_run('circle',config={**scene,'ball_initial_offset_m':list(offset)},
                            output_dir=Path(out)/'detail'/f'{i:02d}',**common)
                rms=r['plateau_tracking_rms_m']
                x.update(completed=r['controller_reached_end'],stop_reason=r['stop_reason'],
                    rms_tracking_m=rms,estimator_rms_m=r['estimation_error_rms_m'],
                    min_full_footprint_margin_m=r['minimum_full_footprint_margin_m'],
                    max_tray_error_m=r['max_tray_position_error_m'],
                    numerical_pass=bool(r['numerical_pass'] and rms is not None and rms<.005 and r['max_tray_position_error_m']<.02))
            else:
                r=run_one(offset,'centering',scene=scene,**common)
                x.update(completed=r['completed'],stop_reason=r['stop_reason'],
                    last_second_max_error_m=r['last_second_max_true_error_m'],
                    estimator_rms_m=r['estimator_rms_error_m'],
                    min_full_footprint_margin_m=r['minimum_full_footprint_edge_margin_m'],
                    max_tray_error_m=r['max_tray_path_error_m'],numerical_pass=r['success'])
    except Exception as e:
        x.update(completed=False,numerical_pass=False,exception=repr(e),traceback=traceback.format_exc())
    Path(out,'trials',f'{i:02d}.json').write_text(json.dumps(x,indent=2)+'\n')
    print(f"{i:02d} {task} friction={friction} bolt={bolt} delay={delay} seed={seed}: {'PASS' if x['numerical_pass'] else 'FAIL'}",flush=True)
    return x


def main():
    p=argparse.ArgumentParser();p.add_argument('--jobs',type=int,default=3);p.add_argument('--output-dir',default='output/stl_material_stress');args=p.parse_args()
    out=Path(args.output_dir).resolve();(out/'trials').mkdir(parents=True,exist_ok=True)
    seeds=((701,(.018,-.012)),(809,(-.016,.018)))
    grid=list(itertools.product(('circle','centering'),(.1,.3,.6),(0.,.01),(.02,.05),seeds))
    items=[(i+1,task,friction,bolt,delay,seed,offset,str(out)) for i,(task,friction,bolt,delay,(seed,offset)) in enumerate(grid)]
    with multiprocessing.get_context('spawn').Pool(args.jobs) as pool: rows=pool.map(trial,items,chunksize=1)
    rows.sort(key=lambda x:x['id']);(out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    fields=sorted({k for x in rows for k in x})
    with (out/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    print(f"Total {sum(x['numerical_pass'] for x in rows)}/{len(rows)} pass",flush=True)

if __name__=='__main__':main()
