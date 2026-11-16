#!/usr/bin/env python3
"""Render four matched, simulation-only controller comparisons.

Controller factories, gains, sensor scenario and hidden initial placement match
the predeclared `compare_methods.py` case. Ball truth appears only in the
evaluation overlay, never as controller input. No hardware interface is used.
"""
import argparse
import json
import os
import textwrap
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from compare_methods import class_for
from wrench_shapes import run as shape_run


LABELS={'pd':'Earlier PD', 'kalman_pd':'Kalman + PD',
        'lqr':'Kalman + LQR (0.6, 0.8)', 'mpc':'Kalman + hybrid MPC/PD'}


def controller_factory(method):
    cls=class_for(method)
    def make(model,data,config=None,**kwargs):
        # Match compare_methods.py: this case has no injected bias and no
        # mismatched observer tray mass.
        cfg=dict(config or {});kwargs['torque_bias']=(0.,0.,0.)
        if method!='pd':
            kwargs.update(measurement_std=.002,accel_std=.3,max_prediction=.15)
        if method=='lqr':
            g=9.81/1.4;kp=.6;kd=.8;qv=kd*kd-2*kp/g
            if qv<0:raise ValueError('LQR weight pair not realizable')
            kwargs.update(lqr_q_position=kp*kp,lqr_q_velocity=qv,lqr_r=1.)
        return cls(model,data,config=cfg,**kwargs)
    return make


def show_exact_stop_reason(path, reason):
    """Annotate the existing frozen stop frames, never inventing movement."""
    path=Path(path);tmp=path.with_name(path.stem+'_annotated.tmp.mp4')
    reader=imageio.get_reader(str(path));meta=reader.get_meta_data()
    frames=reader.count_frames();fps=float(meta['fps'])
    try: font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',15)
    except OSError: font=ImageFont.load_default()
    writer=imageio.get_writer(str(tmp),fps=fps,codec='libx264',macro_block_size=16)
    try:
        for index,frame in enumerate(reader):
            if index>=max(0,frames-48):
                im=Image.fromarray(frame);d=ImageDraw.Draw(im)
                d.rectangle((0,38,640,108),fill=(255,245,240))
                lines=textwrap.wrap('STOPPED: '+str(reason),width=68)[:3]
                for i,line in enumerate(lines):d.text((10,43+19*i),line,font=font,fill=(130,0,0))
                frame=np.asarray(im)
            writer.append_data(frame)
    finally:
        reader.close();writer.close()
    os.replace(tmp,path)


def run_methods(root,methods):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);summaries={}
    for method in methods:
        video=root/(method+'_circle.mp4');sub=root/method
        print('Rendering',LABELS[method],flush=True)
        try:
            r=shape_run('circle',config={'ball_initial_offset_m':[.018,-.012]},
                period_s=8.,output_dir=sub,video=str(video),
                sensor_model='joint_torque',joint_torque_std=.0005,
                joint_velocity_std=.0001,delay_s=.05,seed=401,
                controller_factory=controller_factory(method),controller_label=LABELS[method])
            if r['stop_reason']:
                show_exact_stop_reason(video,r['stop_reason'])
            rms=r['plateau_tracking_rms_m']
            summaries[method]={'label':LABELS[method],'video':video.name,
                'completed':r['controller_reached_end'],'stop_reason':r['stop_reason'],
                'containment_pass':r['numerical_pass'],
                'performance_pass':bool(r['numerical_pass'] and rms is not None and rms<.005
                                        and r['max_tray_position_error_m']<.02),
                'plateau_rms_m':rms,'minimum_full_footprint_margin_m':r['minimum_full_footprint_margin_m'],
                'duration_completed_s':r['duration_completed_s'],'summary_file':str(sub/'circle_summary.json')}
        except Exception as e:
            summaries[method]={'label':LABELS[method],'error':repr(e),'video':None}
        (root/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
        print(json.dumps(summaries[method],indent=2),flush=True)
    return summaries


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',default='output/method_videos')
    p.add_argument('--methods',nargs='+',choices=list(LABELS),default=list(LABELS))
    a=p.parse_args();run_methods(a.output_dir,a.methods)
