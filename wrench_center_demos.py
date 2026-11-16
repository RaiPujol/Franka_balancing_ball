#!/usr/bin/env python3
"""Offline carrying and random centering with ONLY estimated ball feedback.

The virtual wrist wrench has artificial noise and delay and uses an estimated
robot acceleration for tool compensation. Ground truth is kept in the evaluator
for scoring/rendering only, after each controller action; it is never supplied
to the controller. This idealized model is not measured Franka performance.
"""
import argparse
import csv
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from controller import SmoothPath, smoothstep5
from robot_scene import build_model
from wrench_control import WrenchController


def initial_positions(seed=2026, count=5, radius_m=.025):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0., 2.*np.pi, count)
    radius = radius_m*np.sqrt(rng.uniform(.20, 1., count))
    return np.stack((radius*np.cos(theta), radius*np.sin(theta)), axis=1)


def actual_ball_for_evaluation(model,data,meta):
    sid=meta['tray_site_id']
    R=data.site_xmat[sid].reshape(3,3)
    return R.T @ (data.xpos[meta['ball_body_id']]-data.site_xpos[sid])


class WrenchVideo:
    def __init__(self,path,force_std,torque_std,delay_ms,joint_velocity_std,sensor_model,joint_torque_std):
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        self.writer=imageio.get_writer(str(path),fps=24,codec='libx264',quality=8,macro_block_size=16)
        self.frames=0
        if sensor_model=='joint_torque':
            self.noise=f'joint noise {joint_torque_std:g} N m, dq {joint_velocity_std:g} rad/s | delay {delay_ms:g} ms'
        else:
            self.noise=f'noise: {force_std:g} N, {torque_std:g} N m, dq {joint_velocity_std:g} rad/s | delay {delay_ms:g} ms'
        try:
            self.font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',17)
            self.small=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',14)
        except OSError:
            self.font=self.small=ImageFont.load_default()

    def append(self,renderer,data,camera,actual,estimated,truth_history,estimate_history,cfg,label,t,status=None):
        renderer.update_scene(data,camera=camera)
        canvas=Image.new('RGB',(960,544),'white'); canvas.paste(Image.fromarray(renderer.render()),(0,48))
        d=ImageDraw.Draw(canvas)
        d.text((10,4),'SIMULATION | estimated ball state from virtual wrench',font=self.font,fill='black')
        d.text((10,25),f'{label}  |  {t:.1f} s  |  {self.noise}',font=self.small,fill=(40,40,40))
        x0,y0,x1,y1=660,100,940,380
        d.text((660,55),'Ball position on tray',font=self.font,fill='black')
        d.text((660,78),'Orange: simulation truth for scoring',font=self.small,fill=(40,40,40))
        d.rectangle((x0,y0,x1,y1),fill=(225,237,244),outline=(20,35,45),width=2)
        L,W=cfg['tray_length_m'],cfg['tray_width_m']
        def p(v):return (x0+(v[0]/L+.5)*(x1-x0),y1-(v[1]/W+.5)*(y1-y0))
        bx,by=p(actual[:2]);rx=(x1-x0)*cfg['ball_diameter_m']/(2*L);ry=(y1-y0)*cfg['ball_diameter_m']/(2*W)
        d.ellipse((bx-rx,by-ry,bx+rx,by+ry),fill=(242,242,240),outline=(10,10,10),width=2)
        if len(truth_history)>1:d.line([p(x) for x in truth_history],fill=(220,105,0),width=2)
        if len(estimate_history)>1:d.line([p(x) for x in estimate_history],fill=(20,90,210),width=2)
        cx,cy=p([0,0]);d.line((cx-7,cy,cx+7,cy),fill=(180,20,20),width=2);d.line((cx,cy-7,cx,cy+7),fill=(180,20,20),width=2)
        d.ellipse((bx-3,by-3,bx+3,by+3),fill=(220,105,0))
        if estimated is not None:
            ex,ey=p(estimated[:2]);d.ellipse((ex-4,ey-4,ex+4,ey+4),fill=(20,90,210))
        d.text((660,398),f'Actual offset: {1000*np.linalg.norm(actual[:2]):.1f} mm',font=self.small,fill='black')
        if estimated is not None:d.text((660,419),f'Estimated offset: {1000*np.linalg.norm(estimated[:2]):.1f} mm',font=self.small,fill='black')
        d.text((660,442),'Blue: wrench-based estimate',font=self.small,fill=(20,75,175))
        d.text((660,463),'Orange: truth, scoring only',font=self.small,fill=(170,70,0))
        d.text((660,484),'Red cross: target center',font=self.small,fill=(160,20,20))
        d.text((10,526),'Illustrative virtual sensor; not real Franka measurements',font=self.small,fill=(55,55,55))
        if status:
            d.rectangle((0,464,640,523),fill=(255,240,220));d.text((10,474),status[:75],font=self.font,fill=(150,20,20))
        self.writer.append_data(np.asarray(canvas));self.frames+=1

    def close(self):self.writer.close()


def run_one(offset,task,*,video=None,label='',scene=None,force_std=.005,torque_std=.0001,delay_s=.02,
            position_tau=.01,velocity_tau=.05,seed=0,duration_s=None,output_csv=None,control_ramp_s=1.,joint_velocity_std=.0001,
            sensor_model='cut_wrench',joint_torque_std=.0001,controller_factory=None):
    config=dict(scene or {});config['ball_initial_offset_m']=[float(x) for x in offset]
    model,data,meta=build_model(config)
    # The hidden ball placement is deliberately removed from the controller's
    # configuration. Its starting position must come from the wrench estimate.
    observer_config={k:v for k,v in config.items() if k!='ball_initial_offset_m'}
    if task=='centering':
        path=SmoothPath(offsets=((0.,0.,0.),(0.,0.,0.)),segment_s=1.,start_delay_s=0.)
        duration=9. if duration_s is None else float(duration_s)
    elif task=='carrying':
        path=SmoothPath(segment_s=3.)
        duration=path.end_time+1. if duration_s is None else float(duration_s)
    else:raise ValueError('task must be centering or carrying')
    factory=WrenchController if controller_factory is None else controller_factory
    c=factory(model,data,config=observer_config,path=path,
        force_std=force_std,torque_std=torque_std,delay_s=delay_s,seed=seed,
        position_tau=position_tau,velocity_tau=velocity_tau,joint_velocity_std=joint_velocity_std,
        sensor_model=sensor_model,joint_torque_std=joint_torque_std)
    dt=float(model.opt.timestep)
    # Known robot pose only; no ball information enters this startup hold.
    qhold=data.qpos[:7].copy();kp=np.array([600,600,500,500,250,180,100]);kd=np.array([50,50,40,35,20,16,12])
    for _ in range(int(np.ceil(.4/dt))):
        mujoco.mj_forward(model,data)
        c.prime()
        data.ctrl[:7]=np.clip(kp*(qhold-data.qpos[:7])-kd*data.qvel[:7]+data.qfrc_bias[:7],model.actuator_ctrlrange[:7,0],model.actuator_ctrlrange[:7,1])
        mujoco.mj_step(model,data)
    start=float(data.time)
    renderer=camera=None;next_frame=0.;truth_hist=[];est_hist=[];rows=[];contacts={};misses=0;reason=None
    max_path_error=0.;max_tilt=0.;min_margin=np.inf
    if video:
        renderer=mujoco.Renderer(model,height=480,width=640)
        camera=mujoco.MjvCamera();camera.lookat[:]=[.32,0,.46];camera.distance=1.6;camera.azimuth=45;camera.elevation=-35
    try:
        for _ in range(int(np.ceil(duration/dt))):
            mujoco.mj_forward(model,data);elapsed=float(data.time-start)
            try:
                state=c.step(elapsed)
            except RuntimeError as exc:
                reason='ESTIMATOR STOP: '+str(exc)
                # Controller is not restarted with privileged ball state.
                if video:
                    actual=actual_ball_for_evaluation(model,data,meta)
                    estimate=est_hist[-1] if est_hist else None
                    for _ in range(48):video.append(renderer,data,camera,actual,estimate,truth_hist,est_hist,meta['config'],label,elapsed,reason)
                break
            # Fade in the new robot-state-based command from a joint-pose hold.
            # This avoids a torque discontinuity; the blend does not use truth.
            blend=smoothstep5(elapsed/control_ramp_s)[0] if control_ramp_s>0 else 1.
            hold=np.clip(kp*(qhold-data.qpos[:7])-kd*data.qvel[:7]+data.qfrc_bias[:7],
                         model.actuator_ctrlrange[:7,0],model.actuator_ctrlrange[:7,1])
            data.ctrl[:7]=(1.-blend)*hold+blend*data.ctrl[:7]
            # Only the following evaluator uses the true ball state. It runs
            # after c.step, and is never used to modify/choose a command.
            actual=actual_ball_for_evaluation(model,data,meta)
            estimate=np.asarray(state['ball_pos_local'],float)
            r=meta['config']['ball_diameter_m']/2; L=meta['config']['tray_length_m'];W=meta['config']['tray_width_m']
            margin=min(L/2-r-abs(actual[0]), W/2-r-abs(actual[1]))
            min_margin=min(min_margin,float(margin));max_path_error=max(max_path_error,float(np.linalg.norm(state['target_pos']-data.site_xpos[meta['tray_site_id']])))
            R=data.site_xmat[meta['tray_site_id']].reshape(3,3);max_tilt=max(max_tilt,float(np.degrees(np.arccos(np.clip(R[2,2],-1,1)))))
            on_tray=False
            for contact in data.contact[:data.ncon]:
                ids={int(contact.geom1),int(contact.geom2)}
                if ids=={meta['ball_geom_id'],meta['tray_geom_id']}:on_tray=True
                elif contact.dist < -1e-5:
                    names=tuple(sorted((str(model.geom(contact.geom1).name),str(model.geom(contact.geom2).name))))
                    contacts[' / '.join(names)]=min(float(contact.dist),contacts.get(' / '.join(names),0.))
            if elapsed>.5 and not on_tray:misses+=1
            rows.append((elapsed,float(actual[0]),float(actual[1]),float(estimate[0]),float(estimate[1]),float(np.linalg.norm(actual[:2])),float(np.linalg.norm(estimate[:2])),float(np.linalg.norm(actual[:2]-estimate[:2])),float(margin),int(on_tray)))
            if video and elapsed+1e-10>=next_frame:
                truth_hist.append(actual[:2].copy());est_hist.append(estimate[:2].copy())
                video.append(renderer,data,camera,actual,estimate,truth_hist,est_hist,meta['config'],label,elapsed)
                next_frame+=1./24.
            mujoco.mj_step(model,data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():reason='non-finite simulation state';break
    finally:
        if renderer:renderer.close()
    arr=np.asarray(rows);warnings={str(i):int(w.number) for i,w in enumerate(data.warning) if w.number}
    if not len(rows):
        if output_csv:
            Path(output_csv).write_text('time_s,truth_x_m,truth_y_m,estimated_x_m,estimated_y_m,truth_offset_m,estimated_offset_m,estimation_error_m,full_ball_edge_margin_m,ball_tray_contact\n')
        return {'task':task,'scope':'SIMULATION ONLY: wrench-feedback estimator produced no controlled samples',
                'initial_offset_m_for_evaluation_only':list(map(float,offset)),
                'force_noise_std_n_assumed':force_std,'torque_noise_std_nm_assumed':torque_std,
                'joint_velocity_noise_std_rad_s_assumed':joint_velocity_std,'sensor_delay_s':delay_s,'noise_seed':seed,
                'sensor_model':sensor_model,'joint_torque_noise_std_nm_assumed':joint_torque_std if sensor_model=='joint_torque' else None,
                'position_tau_s':position_tau,'velocity_tau_s':velocity_tau,'requested_duration_s':duration,
                'simulation_duration_s':0.,'completed':False,'stop_reason':reason,
                'initial_true_error_m':None,'final_true_error_m':None,'last_second_max_true_error_m':None,
                'settled_within_3mm_last_second':False,'estimator_rms_error_m':None,
                'minimum_full_footprint_edge_margin_m':None,'max_tray_path_error_m':None,
                'ball_tray_no_contact_steps_after_half_second':0,'unexpected_contacts':{},'warnings':warnings,
                'success':False,'scene':meta['config']}
    last=arr[arr[:,0]>=max(0.,duration-1.)]
    completed=bool(reason is None and len(rows)>=int(np.ceil(duration/dt)))
    settled=bool(completed and len(last)>0 and np.all(last[:,5] <= .003))
    if output_csv:
        with Path(output_csv).open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['time_s','truth_x_m','truth_y_m','estimated_x_m','estimated_y_m','truth_offset_m','estimated_offset_m','estimation_error_m','full_ball_edge_margin_m','ball_tray_contact']);w.writerows(rows)
    return {'task':task,'scope':'SIMULATION ONLY: controller uses estimated ball state from virtual noisy delayed wrench, ground truth for evaluation/visualization only; not hardware validated',
        'initial_offset_m_for_evaluation_only':list(map(float,offset)),'initial_control_delay_warmup_s':.4,'torque_handoff_ramp_s':control_ramp_s,
        'force_noise_std_n_assumed':force_std if sensor_model=='cut_wrench' else None,'torque_noise_std_nm_assumed':torque_std if sensor_model=='cut_wrench' else None,'joint_velocity_noise_std_rad_s_assumed':joint_velocity_std,'sensor_delay_s':delay_s,'noise_seed':seed,
        'sensor_model':sensor_model,'joint_torque_noise_std_nm_assumed':joint_torque_std if sensor_model=='joint_torque' else None,
        'position_tau_s':position_tau,'velocity_tau_s':velocity_tau,
        'requested_duration_s':duration,'simulation_duration_s':float(data.time-start),'completed':completed,'stop_reason':reason,
        'initial_true_error_m':float(arr[0,5]),'final_true_error_m':float(arr[-1,5]),
        'last_second_max_true_error_m':float(last[:,5].max()) if len(last) else None,
        'settled_within_3mm_last_second':settled,'estimator_rms_error_m':float(np.sqrt(np.mean(arr[:,7]**2))),
        'minimum_full_footprint_edge_margin_m':float(min_margin),'max_tray_path_error_m':max_path_error,'max_tilt_deg':max_tilt,
        'ball_tray_no_contact_steps_after_half_second':misses,'unexpected_contacts':contacts,'warnings':warnings,
        'success':bool(completed and settled and min_margin>=0 and misses==0 and max_path_error<.02 and not contacts and not warnings),
        'scene':meta['config']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task',choices=['centering','carrying','both'],default='both')
    p.add_argument('--seed',type=int,default=2026);p.add_argument('--trials',type=int,default=5)
    p.add_argument('--config');p.add_argument('--output-dir',default='output/wrench_demos')
    p.add_argument('--force-std',type=float,default=.005);p.add_argument('--torque-std',type=float,default=.0001);p.add_argument('--delay-ms',type=float,default=20.)
    p.add_argument('--position-tau',type=float,default=.01);p.add_argument('--velocity-tau',type=float,default=.05);p.add_argument('--joint-velocity-std',type=float,default=.0001)
    p.add_argument('--sensor-model',choices=['cut_wrench','joint_torque'],default='cut_wrench');p.add_argument('--joint-torque-std',type=float,default=.0001)
    a=p.parse_args();scene=json.loads(Path(a.config).read_text()) if a.config else {}
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    options={'scene':scene,'force_std':a.force_std,'torque_std':a.torque_std,'delay_s':a.delay_ms/1000,'position_tau':a.position_tau,'velocity_tau':a.velocity_tau,'joint_velocity_std':a.joint_velocity_std,'sensor_model':a.sensor_model,'joint_torque_std':a.joint_torque_std}
    if a.task in ('centering','both'):
        positions=initial_positions(a.seed,a.trials);results=[]
        v=WrenchVideo(out/'SIM_centering_wrench.mp4',a.force_std,a.torque_std,a.delay_ms,a.joint_velocity_std,a.sensor_model,a.joint_torque_std)
        try:
            for i,xy in enumerate(positions):
                r=run_one(xy,'centering',video=v if i<3 else None,label=f'Random start {i+1}/{min(3,a.trials)}',seed=i,output_csv=out/f'center_{i+1}.csv',**options)
                results.append(r);print(f'Center {i+1}/{a.trials}: {r["success"]}; stop={r["stop_reason"]}',flush=True)
        finally:v.close()
        (out/'centering_summary.json').write_text(json.dumps({'placement_seed':a.seed,'video_shows_first_trials':min(3,a.trials),'trials':results},indent=2)+'\n')
    if a.task in ('carrying','both'):
        v=WrenchVideo(out/'SIM_carrying_wrench.mp4',a.force_std,a.torque_std,a.delay_ms,a.joint_velocity_std,a.sensor_model,a.joint_torque_std)
        try:r=run_one((0,0),'carrying',video=v,label='Carry while centering',seed=0,output_csv=out/'carry.csv',**options)
        finally:v.close()
        (out/'carrying_summary.json').write_text(json.dumps(r,indent=2)+'\n')
        print(f'Carrying: {r["success"]}; stop={r["stop_reason"]}',flush=True)

if __name__=='__main__':main()
