"""Actual supplied platform STL, offline MuJoCo, not a hardware scene.

The STL has no unit metadata; the user confirmed one CAD unit is one millimetre.
The visual geometry uses the original vertices translated into a tray-top frame;
collision uses a flat top box and a conservative solid box for its mounting tab.
The small central dimple and other non-convex features are NOT in the flat proxy.
It assumes a rigid idealized hand attachment, not a simulated gripper grasp.
The user reported a 92.77 g printed part, but bolt mass, true inertial
distribution, gripping pose, tool transform and friction are unknown.
No hardware commands are present.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import hashlib
import xml.etree.ElementTree as ET
import numpy as np
import mujoco

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'inputs/ball balancing platform.STL'
LOCAL_MESH=ROOT/'assets/platform/platform_tray_frame.stl'
ROBOT_XML=ROOT/'assets/panda/panda_torque_table.xml'
SOURCE_HASH='46457efb2cf70c7f53ea1583df379452e7abe162c1d135bf351783048456c655'
LOCAL_HASH='a70e1d16046b0db5dc50bbefaa8841320a89477e4ebbc82f63d4c8c3009624ff'
DEFAULTS={
 'platform_scale_m_per_cad_unit':.001,
 'tray_mass_kg':.09277,  # user-measured mass of the complete printed STL
 'bolt_mass_kg':0.,  # unmeasured bolt(s), optionally set once weighed
 'mount_offset_m':[.0855,0.,.0805], # illustrative top-center pose in hand frame
 # Orient the 35 mm tab width across the 35 mm jaw opening and point the
 # platform along hand +x, away from the robot base at the default home pose.
 # The attachment pose has NOT been verified against the physical bolted mount.
 'mount_quat_wxyz':[0.,.7071067811865476,.7071067811865476,0.],
 'finger_half_width_m':.0175,
 'ball_diameter_m':.04267,'ball_mass_kg':.04593,
 'ball_initial_offset_m':[0.,0.],'ball_clearance_m':.0002,
 'sliding_friction':.3,'torsional_friction_m':.005,'rolling_friction_m':.0001,
 'timestep_s':.002,'q_home':[0.,-.5,0.,-2.,0.,1.5,.8],
}


def _fmt(values):return ' '.join(f'{float(v):.17g}' for v in values)


def _vec(value,n,name):
    a=np.asarray(value,float)
    if a.shape!=(n,) or not np.isfinite(a).all():raise ValueError(name+' must be a finite '+str(n)+'-vector')
    return a


def _positive(cfg,key,zero=False):
    x=float(cfg[key])
    if not np.isfinite(x) or (x<0 if zero else x<=0):raise ValueError(key+' must be '+('nonnegative' if zero else 'positive'))
    return x


def build_model(config=None):
    """Return (model,data,metadata), matching the earlier `robot_scene` API.

    Mount offset is the tray TOP CENTER in the hand frame, not a verified
    physical mount or earlier placeholder plate-center transform. The tab's
    contact with fingers is excluded because the mesh is rigidly held by fiat.
    """
    supplied={} if config is None else dict(config)
    # Derived keys supplied by prior tools are validated rather than used to
    # resize the STL silently. They are included in metadata for compatibility.
    derived={'tray_length_m','tray_width_m','tray_thickness_m'}
    unknown=set(supplied)-set(DEFAULTS)-derived
    if unknown:raise ValueError('Unknown STL scene parameters: '+str(sorted(unknown)))
    cfg=deepcopy(DEFAULTS);cfg.update({k:v for k,v in supplied.items() if k not in derived})
    scale=_positive(cfg,'platform_scale_m_per_cad_unit');side=150*scale
    for k,expected in (('tray_length_m',side),('tray_width_m',side)):
        if k in supplied and not np.isclose(float(supplied[k]),expected,atol=1e-10):raise ValueError(k+' must match STL dimensions; change STL scale explicitly')
    if not SOURCE.is_file() or hashlib.sha256(SOURCE.read_bytes()).hexdigest()!=SOURCE_HASH:
        raise ValueError('Original STL missing or differs from the inspected mesh')
    if not LOCAL_MESH.is_file() or hashlib.sha256(LOCAL_MESH.read_bytes()).hexdigest()!=LOCAL_HASH:
        raise ValueError('Translated STL asset missing or changed')
    mass=_positive(cfg,'tray_mass_kg');bolt_mass=_positive(cfg,'bolt_mass_kg',True);diam=_positive(cfg,'ball_diameter_m');ballmass=_positive(cfg,'ball_mass_kg')
    clearance=_positive(cfg,'ball_clearance_m',True);dt=_positive(cfg,'timestep_s')
    if dt>.01:raise ValueError('timestep too large')
    friction=[_positive(cfg,k,True) for k in ('sliding_friction','torsional_friction_m','rolling_friction_m')]
    mount=_vec(cfg['mount_offset_m'],3,'mount_offset_m')
    quat=_vec(cfg['mount_quat_wxyz'],4,'mount_quat_wxyz')
    if np.linalg.norm(quat)<1e-12:raise ValueError('mount quaternion must be nonzero')
    quat=quat/np.linalg.norm(quat)
    offset=_vec(cfg['ball_initial_offset_m'],2,'ball_initial_offset_m')
    if np.any(np.abs(offset)+diam/2>=side/2):raise ValueError('ball initially extends beyond square support')
    qhome=_vec(cfg['q_home'],7,'q_home');halfwidth=float(cfg['finger_half_width_m'])
    if not np.isfinite(halfwidth) or not .005<=halfwidth<=.04:raise ValueError('invalid gripper width')
    root=ET.parse(ROBOT_XML).getroot();root.find('compiler').set('meshdir',str(ROBOT_XML.parent/'assets'))
    root.find('option').set('timestep',f'{dt:.17g}')
    assets=root.find('asset');ET.SubElement(assets,'mesh',name='actual_platform_visual',
        file=str(LOCAL_MESH),scale=_fmt([scale]*3),inertia='exact')
    hand=root.find(".//body[@name='hand']");world=root.find('worldbody')
    tray=ET.SubElement(hand,'body',name='tray',pos=_fmt(mount),quat=_fmt(quat))
    # Use exact nonconvex mesh volume under *uniform density* for inertia at
    # the measured mass; real infill and material distribution are unknown.
    ET.SubElement(tray,'geom',name='platform_visual',type='mesh',mesh='actual_platform_visual',
        mass=str(mass),contype='0',conaffinity='0',rgba='.12 .45 .7 1')
    # Flat top approximation exactly matches the measured outer square and
    # z=21 top away from its tiny 2 mm diameter, 1 mm deep central dimple.
    ET.SubElement(tray,'geom',name='tray_plate',type='box',
        size=_fmt([side/2,side/2,.5*scale]),pos=_fmt([0,0,-.5*scale]),
        mass='0',rgba='0 0 0 0',friction=_fmt(friction),condim='6')
    ET.SubElement(tray,'site',name='tray_top',pos='0 0 0',size='.003',rgba='0 1 0 1')
    # Actual extension is concave. This explicit box is a solid approximation,
    # not the detailed hole geometry; it cannot prove physical mount fit.
    tab=ET.SubElement(tray,'body',name='platform_tab')
    ET.SubElement(tab,'geom',name='tab_collision_proxy',type='box',
        pos=_fmt([0,-85.5*scale,-10.5*scale]),size=_fmt([17.5*scale,10.5*scale,10.5*scale]),
        mass='0',rgba='0 0 0 0')
    if bolt_mass:
        # Point-mass surrogate at the tab center. Real bolt locations and
        # inertia are unknown; this must not be interpreted as a bolt CAD model.
        ET.SubElement(tab,'geom',name='bolt_mass_surrogate',type='sphere',
            pos=_fmt([0,-85.5*scale,-10.5*scale]),size='.001',mass=str(bolt_mass),
            contype='0',conaffinity='0',rgba='0 0 0 0')
    contact=root.find('contact')
    if contact is None:contact=ET.SubElement(root,'contact')
    for finger in ('left_finger','right_finger'):
        ET.SubElement(contact,'exclude',body1='platform_tab',body2=finger)
    ET.SubElement(hand,'site',name='wrist_ft',pos='0 0 0',quat='0 1 0 0',size='.003',rgba='1 0 1 1')
    sensors=root.find('sensor')
    if sensors is None:sensors=ET.SubElement(root,'sensor')
    ET.SubElement(sensors,'force',name='tray_cut_force',site='wrist_ft')
    ET.SubElement(sensors,'torque',name='tray_cut_torque',site='wrist_ft')
    ball=ET.SubElement(world,'body',name='ball',pos='0 0 2')
    ET.SubElement(ball,'freejoint',name='ball_joint')
    ET.SubElement(ball,'geom',name='ball_geom',type='sphere',size=str(diam/2),
        mass=str(ballmass),friction=_fmt(friction),condim='6',rgba='.95 .95 .95 1')
    model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    data=mujoco.MjData(model)
    joints=np.array([model.joint(f'joint{i}').id for i in range(1,8)],int)
    qids=model.jnt_qposadr[joints].astype(int);dofs=model.jnt_dofadr[joints].astype(int)
    acts=np.array([model.actuator(f'actuator{i}').id for i in range(1,8)],int)
    if np.any(qhome<model.jnt_range[joints,0]) or np.any(qhome>model.jnt_range[joints,1]):raise ValueError('home outside joint limits')
    data.qpos[qids]=qhome
    for name in ('finger_joint1','finger_joint2'):data.qpos[model.jnt_qposadr[model.joint(name).id]]=halfwidth
    data.ctrl[model.actuator('actuator8').id]=halfwidth;mujoco.mj_forward(model,data)
    site=model.site('tray_top').id;R=data.site_xmat[site].reshape(3,3)
    ballpos=data.site_xpos[site]+R@np.array([*offset,diam/2+clearance])
    bj=model.joint('ball_joint').id;bq=int(model.jnt_qposadr[bj])
    data.qpos[bq:bq+3]=ballpos;data.qpos[bq+3:bq+7]=[1.,0.,0.,0.]
    mujoco.mj_forward(model,data)
    cfg.update(tray_length_m=side,tray_width_m=side,tray_thickness_m=7*scale)
    meta={'config':cfg,'arm_joint_ids':joints,'arm_qpos_ids':qids,'arm_dof_ids':dofs,'arm_actuator_ids':acts,
          'tray_site_id':site,'tray_body_id':model.body('tray').id,'ball_body_id':model.body('ball').id,
          'ball_qpos_address':bq,'ball_dof_address':int(model.jnt_dofadr[bj]),
          'ball_geom_id':model.geom('ball_geom').id,'tray_geom_id':model.geom('tray_plate').id,
          'wrist_ft_site_id':model.site('wrist_ft').id,'force_sensor_address':int(model.sensor_adr[model.sensor('tray_cut_force').id]),
          'torque_sensor_address':int(model.sensor_adr[model.sensor('tray_cut_torque').id]),
          'source_stl_sha256':SOURCE_HASH,'stl_m_per_unit':scale,
          'collision_surface':'flat 150-unit square; central dimple omitted',
          'collision_tab':'conservative solid box; tab undercut not modeled',
          'tab_finger_collision_excluded':True,'attachment_rigid_idealized':True,
          'user_measured_printed_part_mass_kg':.09277,'bolt_mass_unmeasured':True,
          'inertia_model':'exact closed STL under uniform density; actual print infill unknown',
          'physical_mount_and_friction_unverified':True}
    return model,data,meta
