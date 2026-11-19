"""Offline geom-distance audit on the selected STL/flat-proxy LQR trials.

Uses MuJoCo geometry distances, including pairs excluded from contacts. This
is a sampled diagnostic, not continuous collision checking or hardware proof.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import mujoco
from stl_bridge import use_actual_stl_scene
from render_methods import controller_factory
from wrench_shapes import run as shape_run
from wrench_center_demos import run_one as center_run


class Audit:
    def __init__(self):self.model=None;self.calls=0;self.mins={};self.start={}
    @staticmethod
    def body(m,i):return m.body(m.geom_bodyid[i]).name
    def init(self,m):
        self.model=m
        enabled=[i for i in range(m.ngeom) if m.geom_contype[i] or m.geom_conaffinity[i]]
        groups={}
        def add(cat,a,b):groups.setdefault(cat,[]).append((a,b))
        plate=m.geom('tray_plate').id;tab=m.geom('tab_collision_proxy').id;ball=m.geom('ball_geom').id;ground=m.geom('ground').id
        finger=[i for i in enabled if self.body(m,i) in ('left_finger','right_finger')]
        robot=[i for i in enabled if self.body(m,i).startswith('link') or self.body(m,i) in ('hand','left_finger','right_finger')]
        for i in finger:add('plate_vs_fingers',plate,i);add('tab_vs_fingers_intentionally_filtered',tab,i)
        for i in robot:
            body=self.body(m,i)
            add('ball_vs_robot',ball,i)
            if body in ('hand','link7'):
                add('plate_vs_hand_and_link7',plate,i);add('tab_vs_hand_and_link7',tab,i)
            if body not in ('hand','left_finger','right_finger','link7'):
                add('plate_vs_other_arm_links',plate,i);add('tab_vs_other_arm_links',tab,i)
            if body!='link0':add('robot_vs_ground_excluding_fixed_base',ground,i)
        add('plate_vs_ground',plate,ground);add('tab_vs_ground',tab,ground)
        def rank(b):
            if b.startswith('link'):return int(b[4:])
            if b=='hand':return 8
            return 9
        for ix,a in enumerate(robot):
            A=self.body(m,a)
            for b in robot[ix+1:]:
                B=self.body(m,b)
                if A==B:continue
                if abs(rank(A)-rank(B))<=1 and {A,B}!={'left_finger','right_finger'}:continue
                add('robot_self_nonadjacent',a,b)
        self.groups=groups
    def sample(self,m,d):
        if self.model is None:self.init(m)
        if m is not self.model:return
        self.calls+=1
        # Every integrator step for the nearly touching plate/finger pair; all
        # other categories at 20 Hz. Distances use transforms at step start.
        for category,pairs in self.groups.items():
            if category!='plate_vs_fingers' and (self.calls-1)%25:continue
            for a,b in pairs:
                dist=float(mujoco.mj_geomDistance(m,d,a,b,.3,None))
                prev=self.mins.get(category)
                if prev is None or dist<prev['distance_m']:
                    self.mins[category]={'distance_m':dist,'distance_capped_at_m':.3,'time_s':max(0.,float(d.time-m.opt.timestep)),
                      'geom_ids':[a,b],'bodies':[self.body(m,a),self.body(m,b)],
                      'geom_names':[m.geom(a).name or f'unnamed_{a}',m.geom(b).name or f'unnamed_{b}']}
                if category not in self.start:self.start[category]=[]
                if self.calls==1:self.start[category].append({'distance_m':dist,'geom_ids':[a,b],'bodies':[self.body(m,a),self.body(m,b)]})


def run(task, output_dir):
    audit=Audit();original=mujoco.mj_step
    def instrument(m,d,*args,**kw):
        result=original(m,d,*args,**kw)
        audit.sample(m,d)
        return result
    mujoco.mj_step=instrument
    try:
        with use_actual_stl_scene():
            factory=controller_factory('lqr')
            if task=='circle':
                result=shape_run('circle',config={'ball_initial_offset_m':[.018,-.012]},period_s=8.,
                    output_dir=output_dir/'circle',sensor_model='joint_torque',joint_torque_std=.0005,
                    joint_velocity_std=.0001,delay_s=.05,seed=401,controller_factory=factory)
                completed=result['controller_reached_end'];reason=result['stop_reason']
            else:
                result=center_run((.018,-.012),'carrying',sensor_model='joint_torque',joint_torque_std=.0005,
                    joint_velocity_std=.0001,delay_s=.05,seed=401,controller_factory=factory)
                completed=result['completed'];reason=result['stop_reason']
    finally:mujoco.mj_step=original
    return {'task':task,'completed':completed,'stop_reason':reason,'integrator_steps':audit.calls,
            'minimum_distances':audit.mins,'initial_plate_finger_pairs':audit.start.get('plate_vs_fingers',[]),
            'notes':['Flat deck and solid tab proxies, not the full STL contact mesh.',
                     'Pairs include filtered collision pairs; tab-finger touching is intentionally filtered.',
                     'Plate-finger sampled each 2 ms integration step; other categories sampled each 50 ms.',
                     'Only modeled ground is checked; physical table, surrounding objects, cable, and human are not modeled.',
                     'Distance and penetration are approximate for the modeled convex geoms; sampling cannot guarantee continuous clearance.',
                     'Adjacent robot bodies are excluded from self-distance category as normally attached.']}

if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('output/stl_distance_audit'))
    output_dir = parser.parse_args().output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    results=[]
    for task in ('circle','carrying'):
        print('Auditing',task,flush=True);r=run(task, output_dir);results.append(r)
        print('completed:',r['completed'],'stop:',r['stop_reason'],flush=True)
        for cat,x in r['minimum_distances'].items():print(cat,round(x['distance_m']*1000,4),'mm',x['bodies'],round(x['time_s'],3),'s',flush=True)
        (output_dir/'distance_audit.json').write_text(json.dumps(results,indent=2)+'\n')
