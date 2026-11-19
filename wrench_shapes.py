"""Draw shapes using simulated noisy delayed wrist-wrench feedback only.

Simulator true ball state is used *only* by the separate evaluator and plot.
The controller's feedback inputs are the simulated wrist wrench and robot state.
No physical Franka wrench estimator, sensor calibration, or hardware interface.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio
from robot_scene import build_model
from controller import SmoothPath
from shape_paths import ShapePath
from wrench_control import WrenchController


def _font(size):
    try: return ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', size)
    except OSError: return ImageFont.load_default()


def run(shape, *, config=None, radius_m=.02, period_s=8., video=None,
        output_dir='output/wrench_shapes', force_std=.005, torque_std=.0001,
        delay_s=.02, seed=0, position_tau=.01, velocity_tau=.05, accel_tau=.0001,
        feedback_kp=1., feedback_kd=.8, joint_velocity_std=0.,
        sensor_model='cut_wrench', joint_torque_std=.0001, controller_factory=None,
        controller_label=None):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    model,data,meta=build_model(config)
    # Joint servo startup is based on robot joint state only, not ball state.
    qhold=data.qpos[meta['arm_qpos_ids']].copy()
    kp=np.array([600,600,500,500,250,180,100]);kd=np.array([50,50,40,35,20,16,12])
    for _ in range(round(.25/model.opt.timestep)):
        data.ctrl[:7]=np.clip(kp*(qhold-data.qpos[:7])-kd*data.qvel[:7]+data.qfrc_bias[:7],model.actuator_ctrlrange[:7,0],model.actuator_ctrlrange[:7,1])
        mujoco.mj_step(model,data)
    mujoco.mj_forward(model,data)
    target=ShapePath(shape=shape,radius_m=radius_m,period_s=period_s)
    factory=WrenchController if controller_factory is None else controller_factory
    observer_config={k:v for k,v in meta['config'].items() if k!='ball_initial_offset_m'}
    control=factory(model,data,config=observer_config,path=SmoothPath(segment_s=5.),target_path=target,
        force_std=force_std,torque_std=torque_std,delay_s=delay_s,seed=seed,
        position_tau=position_tau,velocity_tau=velocity_tau,accel_tau=accel_tau,
        feedback_kp=feedback_kp,feedback_kd=feedback_kd,joint_velocity_std=joint_velocity_std,
        sensor_model=sensor_model,joint_torque_std=joint_torque_std)
    # Fill the sensor delay queue and settle the observer while holding the
    # robot based on its joint state. No ball ground truth is used.
    for _ in range(round(max(.3,delay_s+.1)/model.opt.timestep)):
        mujoco.mj_forward(model,data)
        control.prime()
        data.ctrl[:7]=np.clip(kp*(qhold-data.qpos[:7])-kd*data.qvel[:7]+data.qfrc_bias[:7],model.actuator_ctrlrange[:7,0],model.actuator_ctrlrange[:7,1])
        mujoco.mj_step(model,data)
    mujoco.mj_forward(model,data)
    start_time=float(data.time)
    duration=target.end_time+3.; dt=float(model.opt.timestep)
    rows=[];contacts=set();reason=None;no_contact=0;next_frame=0.;actual_trace=[];estimate_trace=[];max_tray_error=0.
    renderer=writer=None
    if video:
        Path(video).parent.mkdir(parents=True,exist_ok=True)
        renderer=mujoco.Renderer(model,height=480,width=640)
        camera=mujoco.MjvCamera();camera.lookat[:]=[.32,0.,.46];camera.distance=1.6
        camera.azimuth=135;camera.elevation=-35
        writer=imageio.get_writer(video,fps=24,codec='libx264',macro_block_size=16)
    lim=max(radius_m*1.9,.04); cx,cy,h=800,265,128
    def point(pos): return (int(round(cx+h*pos[0]/lim)),int(round(cy-h*pos[1]/lim)))
    target_curve=[point(radius_m*target.curve(float(x))) for x in np.linspace(0,2*np.pi,401)]
    try:
        for _ in range(round(duration/dt)):
            mujoco.mj_forward(model,data)
            t=float(data.time-start_time)
            try: state=control.step(t)
            except RuntimeError as e:
                reason=str(e)
                break
            # The following is deliberately evaluator-only. None of these
            # simulator ball/contact values is passed back into the controller.
            sid=meta['tray_site_id']; R=data.site_xmat[sid].reshape(3,3)
            actual=(R.T@(data.xpos[meta['ball_body_id']]-data.site_xpos[sid]))[:2].copy()
            estimated=state['ball_pos_local'][:2].copy(); desired=state['ball_target_local'].copy()
            max_tray_error=max(max_tray_error,float(np.linalg.norm(state['target_pos']-data.site_xpos[sid])))
            ball_radius=meta['config']['ball_diameter_m']/2.
            edge=min(meta['config']['tray_length_m']/2-ball_radius-abs(actual[0]),
                     meta['config']['tray_width_m']/2-ball_radius-abs(actual[1]))
            ball_contact=False
            for con in data.contact[:data.ncon]:
                pair={int(con.geom1),int(con.geom2)}
                if pair=={meta['ball_geom_id'],meta['tray_geom_id']}: ball_contact=True
                elif con.dist < -1e-5:
                    contacts.add(tuple(sorted((str(model.geom(con.geom1).name),str(model.geom(con.geom2).name)))))
            if t>=.5 and not ball_contact: no_contact+=1
            rows.append([t,*actual,*estimated,*desired,float(np.linalg.norm(actual-desired)),
                         float(np.linalg.norm(actual-estimated)),float(edge),int(ball_contact)])
            if writer and t+1e-10>=next_frame:
                renderer.update_scene(data,camera=camera)
                canvas=Image.new('RGB',(960,544),'white');canvas.paste(Image.fromarray(renderer.render()),(0,38))
                draw=ImageDraw.Draw(canvas)
                sensor_label='joint-torque proxy' if sensor_model=='joint_torque' else 'ideal flange wrench'
                name=(controller_label+' | ') if controller_label else ''
                draw.text((10,10),f'SIMULATION | {name}{sensor_label} | {shape.replace("_"," ")} | {t:.1f} s',font=_font(16),fill='black')
                draw.text((660,58),'Ball trajectory on tray',font=_font(17),fill='black')
                draw.text((660,81),'tray coordinates (mm)',font=_font(14),fill=(40,40,40))
                draw.rectangle((cx-h,cy-h,cx+h,cy+h),outline=(30,30,30))
                for f in (-.5,0.,.5):
                    z=int(round(h*f));draw.line((cx+z,cy-h,cx+z,cy+h),fill=(220,220,220));draw.line((cx-h,cy+z,cx+h,cy+z),fill=(220,220,220))
                draw.line(target_curve,fill=(0,100,200),width=2)
                actual_trace.append(point(actual));estimate_trace.append(point(estimated))
                if len(actual_trace)>1:draw.line(actual_trace,fill=(220,105,0),width=2)
                if len(estimate_trace)>1:draw.line(estimate_trace,fill=(25,145,75),width=2)
                ax,ay=point(actual);ex,ey=point(estimated)
                draw.ellipse((ax-4,ay-4,ax+4,ay+4),fill=(220,105,0));draw.ellipse((ex-4,ey-4,ex+4,ey+4),fill=(25,145,75))
                draw.text((cx-h,cy+h+6),f'{-lim*1000:.0f}',font=_font(13),fill='black');draw.text((cx+h-23,cy+h+6),f'{lim*1000:.0f}',font=_font(13),fill='black')
                draw.text((660,417),'Blue: target | Green: estimated',font=_font(14),fill='black')
                draw.text((660,438),'Orange: truth for evaluation only',font=_font(14),fill='black')
                draw.text((660,463),f'Tracking error: {np.linalg.norm(actual-desired)*1000:.2f} mm',font=_font(15),fill='black')
                noise_label=(f'Joint torque noise={joint_torque_std:g} Nm, joint v noise={joint_velocity_std:g} rad/s'
                             if sensor_model=='joint_torque' else f'Wrench noise F={force_std:g} N, T={torque_std:g} Nm, joint v={joint_velocity_std:g} rad/s')
                draw.text((10,524),f'Hypothetical {noise_label}; delay={delay_s*1000:.0f} ms',font=_font(12),fill=(40,40,40))
                writer.append_data(np.asarray(canvas));next_frame+=1/24.
            mujoco.mj_step(model,data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                reason='nonfinite simulation';break
    finally:
        if writer and reason and 'canvas' in locals():
            # A paused final frame is visibly labeled; no simulated motion is
            # invented after the controller aborts on its measurement gate.
            paused=canvas.copy();label=ImageDraw.Draw(paused)
            label.rectangle((0,38,640,95),fill=(255,245,240))
            label.text((12,45),'STOPPED: controller or estimator failure',font=_font(17),fill=(150,0,0))
            label.text((12,70),'Frozen frame after stop',font=_font(14),fill=(90,0,0))
            for _ in range(48):writer.append_data(np.asarray(paused))
        if writer:writer.close()
        if renderer:renderer.close()
    # A rejected first observation is a completed experiment with no samples.
    arr=np.asarray(rows,dtype=float).reshape(-1,11)
    mask=(arr[:,0]>=target.plateau_start)&(arr[:,0]<target.plateau_end)
    errors=arr[mask,7]
    summary={'scenario':shape,'classification':'simulation only; noisy delayed joint torque proxy or ideal flange wrench; truth evaluator only',
       'scene':meta['config'],'path':target.__dict__,'duration_requested_s':duration,'duration_completed_s':float(arr[-1,0]) if len(arr) else 0.,
       'controller_reached_end':reason is None and len(arr)==round(duration/dt),'stop_reason':reason,
       'sensor_model':sensor_model,'force_noise_std_n_assumed':force_std if sensor_model=='cut_wrench' else None,
       'torque_noise_std_nm_assumed':torque_std if sensor_model=='cut_wrench' else None,
       'joint_torque_noise_std_nm_assumed':joint_torque_std if sensor_model=='joint_torque' else None,'measurement_delay_s':delay_s,
       'position_filter_tau_s':position_tau,'velocity_filter_tau_s':velocity_tau,'acceleration_filter_tau_s':accel_tau,
       'feedback_kp':feedback_kp,'feedback_kd':feedback_kd,'joint_velocity_noise_std_rad_s_assumed':joint_velocity_std,'noise_seed':seed,
       'plateau_tracking_rms_m':float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
       'plateau_tracking_max_m':float(np.max(errors)) if len(errors) else None,
       'estimation_error_rms_m':float(np.sqrt(np.mean(arr[:,8]**2))) if len(arr) else None,
       'minimum_full_footprint_margin_m':float(np.min(arr[:,9])) if len(arr) else None,
       'max_tray_position_error_m':max_tray_error if len(arr) else None,
       'no_contact_samples_after_half_s':no_contact,'unexpected_geom_contacts':[list(x) for x in sorted(contacts)],
       'warnings':{str(i):int(w.number) for i,w in enumerate(data.warning) if w.number},
       'limitations':'Joint-torque residual proxy generated from ideal simulated wrench; NOT the actual Franka torque estimator. Exact tool model, noise and delay assumptions are uncalibrated; no shelves or physical grasp strength.'}
    summary['numerical_pass']=(summary['controller_reached_end'] and summary['minimum_full_footprint_margin_m']>=0
                               and summary['no_contact_samples_after_half_s']==0 and not contacts and not summary['warnings'])
    with (out/f'{shape}_summary.json').open('w') as f:json.dump(summary,f,indent=2);f.write('\n')
    with (out/f'{shape}_timeseries.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['time_s','truth_x_m','truth_y_m','estimate_x_m','estimate_y_m','target_x_m','target_y_m','truth_tracking_error_m','estimation_error_m','truth_edge_margin_m','truth_contact']);w.writerows(rows)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shape',choices=['circle','rounded_square','figure_eight'],default='circle')
    p.add_argument('--config');p.add_argument('--radius-mm',type=float,default=20.)
    p.add_argument('--period-s',type=float,default=8.);p.add_argument('--video');p.add_argument('--output-dir',default='output/wrench_shapes')
    p.add_argument('--force-std',type=float,default=.005);p.add_argument('--torque-std',type=float,default=.0001);p.add_argument('--delay-ms',type=float,default=20.)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--position-tau',type=float,default=.01);p.add_argument('--velocity-tau',type=float,default=.05)
    p.add_argument('--accel-tau',type=float,default=.0001);p.add_argument('--feedback-kp',type=float,default=1.);p.add_argument('--feedback-kd',type=float,default=.8);p.add_argument('--joint-velocity-std',type=float,default=0.)
    p.add_argument('--sensor-model',choices=['cut_wrench','joint_torque'],default='cut_wrench');p.add_argument('--joint-torque-std',type=float,default=.0001)
    a=p.parse_args();cfg=json.loads(Path(a.config).read_text()) if a.config else None
    print(json.dumps(run(a.shape,config=cfg,radius_m=a.radius_mm/1000,period_s=a.period_s,video=a.video,output_dir=a.output_dir,
        force_std=a.force_std,torque_std=a.torque_std,delay_s=a.delay_ms/1000,seed=a.seed,position_tau=a.position_tau,velocity_tau=a.velocity_tau,
        accel_tau=a.accel_tau,feedback_kp=a.feedback_kp,feedback_kd=a.feedback_kd,joint_velocity_std=a.joint_velocity_std,
        sensor_model=a.sensor_model,joint_torque_std=a.joint_torque_std),indent=2))
