"""Offline MuJoCo Panda, parameterized tray and freely rolling golf ball.

This is a simulation model only. The lab starter uses a Franka Hand, not a
Robotiq gripper. The illustrative bracket and tray are rigidly attached to the
hand; actual finger contact, clamp strength, print flex, robot virtual walls,
shelves, and real material properties are not modeled. Lengths use metres,
masses kilograms, angles radians.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parent
ROBOT_XML = ROOT / 'assets/panda/panda_torque_table.xml'
DEFAULTS = {
    'tray_length_m': .16,      # local x full size
    'tray_width_m': .16,       # local y full size
    'tray_thickness_m': .006,
    'tray_mass_kg': .12,
    'mount_offset_m': [0., 0., .16],  # hand local coordinates, to plate center
    'attachment_mass_kg': .08,
    'attachment_member_m': .014,  # square structural member width (illustrative)
    'attachment_grip_width_m': .060,
    'attachment_grip_depth_m': .028,
    'attachment_grip_length_m': .028,
    'attachment_grip_center_z_m': .091,
    'attachment_side_clearance_m': .009,
    'ball_diameter_m': .04267,
    'ball_mass_kg': .04593,
    'ball_initial_offset_m': [0., 0.], # on tray top in local x,y
    'ball_clearance_m': .0002,
    'sliding_friction': .3,
    'torsional_friction_m': .005,
    'rolling_friction_m': .0001,
    'timestep_s': .002,
    'q_home': [0., -.5, 0., -2., 0., 1.5, .8],
    'finger_half_width_m': .03,
}

def _vec(v, n, key):
    a = np.asarray(v, dtype=float)
    if a.shape != (n,) or not np.isfinite(a).all():
        raise ValueError(f'{key} must be a finite {n}-vector')
    return a

def _fmt(values):
    return ' '.join(f'{x:.17g}' for x in values)

def _positive(cfg, key, *, allow_zero=False):
    x = float(cfg[key])
    if not np.isfinite(x) or (x < 0 if allow_zero else x <= 0):
        raise ValueError(f'{key} must be finite and '+('nonnegative' if allow_zero else 'positive'))
    return x

def build_model(config=None):
    """Build and initialize (model, data, metadata).

    The tray top site +Z is the tray normal and +X/+Y are plate axes. Geometry
    is fixed in the hand frame. The ball is placed once, then moves freely under
    gravity/contact. A config key not in DEFAULTS is rejected to catch typos.
    """
    config = {} if config is None else dict(config)
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown scene parameters: {sorted(unknown)}')
    cfg = deepcopy(DEFAULTS)
    cfg.update(config)
    length = _positive(cfg, 'tray_length_m')
    width = _positive(cfg, 'tray_width_m')
    thick = _positive(cfg, 'tray_thickness_m')
    mass = _positive(cfg, 'tray_mass_kg')
    diam = _positive(cfg, 'ball_diameter_m')
    ballmass = _positive(cfg, 'ball_mass_kg')
    clearance = _positive(cfg, 'ball_clearance_m', allow_zero=True)
    friction = [_positive(cfg, 'sliding_friction', allow_zero=True),
                _positive(cfg, 'torsional_friction_m', allow_zero=True),
                _positive(cfg, 'rolling_friction_m', allow_zero=True)]
    dt = _positive(cfg, 'timestep_s')
    if dt > .01:
        raise ValueError('timestep_s above 0.01 is unsupported')
    mount = _vec(cfg['mount_offset_m'], 3, 'mount_offset_m')
    attachment_mass = _positive(cfg, 'attachment_mass_kg')
    member = _positive(cfg, 'attachment_member_m')
    grip_width = _positive(cfg, 'attachment_grip_width_m')
    grip_depth = _positive(cfg, 'attachment_grip_depth_m')
    grip_length = _positive(cfg, 'attachment_grip_length_m')
    grip_z = _positive(cfg, 'attachment_grip_center_z_m')
    side_clear = _positive(cfg, 'attachment_side_clearance_m', allow_zero=True)
    offset = _vec(cfg['ball_initial_offset_m'], 2, 'ball_initial_offset_m')
    qhome = _vec(cfg['q_home'], 7, 'q_home')
    half_width = float(cfg['finger_half_width_m'])
    if not np.isfinite(half_width) or not .005 <= half_width <= .04:
        raise ValueError('finger_half_width_m must be between .005 and .04')
    tree = ET.parse(ROBOT_XML)
    root = tree.getroot()
    root.find('compiler').set('meshdir', str(ROBOT_XML.parent / 'assets'))
    root.find('option').set('timestep', f'{dt:.17g}')
    hand = root.find(".//body[@name='hand']")
    world = root.find('worldbody')
    # A provisional L-shaped printed support: a block between the finger pads,
    # a high crossbar, an outboard post, and a short support under the plate.
    # All are fixed to the hand rather than dynamically grasped. No finger
    # friction or printed-material strength is inferred from this model.
    # The post sits beyond the +x edge of the tray, leaving the upper surface
    # unobstructed; the lower crossbar overlaps the plate underside by 1 mm.
    if abs(mount[1]) > 1e-9:
        raise ValueError('provisional attachment currently requires mount_offset_m y = 0')
    x_out = mount[0] + length/2 + side_clear + member/2
    x_in = mount[0] + length/2 - .03
    z_high = grip_z + grip_length/2 - member/2
    z_low = mount[2] + thick/2 + member/2 - .001
    if x_out <= 0 or x_in >= x_out or z_low <= z_high:
        raise ValueError('tray offset/size incompatible with this preliminary bracket layout')
    bracket = ET.SubElement(hand, 'body', name='attachment')
    # The bracket is mechanically bonded to the tray and represents an ideal
    # rigidly held gripper insert, so touching surfaces are deliberately
    # collision-excluded; grip force/slip are not simulated.
    contact = root.find('contact')
    if contact is None:
        contact = ET.SubElement(root, 'contact')
    for other in ('tray', 'left_finger', 'right_finger'):
        ET.SubElement(contact, 'exclude', body1='attachment', body2=other)
    def box(name, center, size):
        ET.SubElement(bracket, 'geom', name=name, type='box', pos=_fmt(center),
                      size=_fmt(size), mass=str(attachment_mass/4),
                      rgba='0.96 0.55 0.12 1', friction='0.8 0.005 0.0001')
    box('attachment_grip', [0,0,grip_z], [grip_depth/2,grip_width/2,grip_length/2])
    box('attachment_upper_arm', [x_out/2,0,z_high], [x_out/2,member/2,member/2])
    box('attachment_side_post', [x_out,0,(z_high+z_low)/2], [member/2,member/2,(z_low-z_high)/2 + member/2])
    box('attachment_lower_support', [(x_in+x_out)/2,0,z_low], [(x_out-x_in)/2,member/2,member/2])
    # At the common down-facing hand pose, this half-turn makes local tray +Z
    # point upwards; the measured mounting orientation must replace it later.
    tray = ET.SubElement(hand, 'body', name='tray', pos=_fmt(mount), quat='0 1 0 0')
    ET.SubElement(tray, 'geom', name='tray_plate', type='box',
                  size=_fmt([length/2, width/2, thick/2]), mass=str(mass),
                  friction=_fmt(friction), condim='6', rgba='0.12 0.45 0.7 1')
    ET.SubElement(tray, 'site', name='tray_top', pos=_fmt([0, 0, thick/2]), size='.003', rgba='0 1 0 1')
    # Ideal internal hand/flange wrench, for simulation only. The real Franka
    # estimates external wrench from joint torque sensors; this ideal cut
    # sensor does not model that estimator's noise, latency, or singularities.
    ET.SubElement(hand, 'site', name='wrist_ft', pos='0 0 0', quat='0 1 0 0', size='.003', rgba='1 0 1 1')
    sensors = root.find('sensor')
    if sensors is None:
        sensors = ET.SubElement(root, 'sensor')
    ET.SubElement(sensors, 'force', name='tray_cut_force', site='wrist_ft')
    ET.SubElement(sensors, 'torque', name='tray_cut_torque', site='wrist_ft')
    # The ball is independent and is not welded to the tray.
    ball = ET.SubElement(world, 'body', name='ball', pos='0 0 2')
    ET.SubElement(ball, 'freejoint', name='ball_joint')
    ET.SubElement(ball, 'geom', name='ball_geom', type='sphere', size=str(diam/2),
                  mass=str(ballmass), friction=_fmt(friction), condim='6', rgba='.95 .95 .95 1')
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    data = mujoco.MjData(model)
    joint_ids = np.array([model.joint(f'joint{i}').id for i in range(1,8)], dtype=int)
    qpos_ids = model.jnt_qposadr[joint_ids].astype(int)
    dof_ids = model.jnt_dofadr[joint_ids].astype(int)
    act_ids = np.array([model.actuator(f'actuator{i}').id for i in range(1,8)], dtype=int)
    if np.any(qhome < model.jnt_range[joint_ids, 0]) or np.any(qhome > model.jnt_range[joint_ids, 1]):
        raise ValueError('q_home violates a robot joint limit')
    data.qpos[qpos_ids] = qhome
    for name in ('finger_joint1', 'finger_joint2'):
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = half_width
    data.ctrl[model.actuator('actuator8').id] = half_width
    mujoco.mj_forward(model, data)
    site_id = model.site('tray_top').id
    tray_R = data.site_xmat[site_id].reshape(3,3)
    # Place on plate in its frame. Only this initialization changes ball qpos.
    ball_xyz = data.site_xpos[site_id] + tray_R @ np.array([*offset, diam/2+clearance])
    bjoint = model.joint('ball_joint').id
    bq = int(model.jnt_qposadr[bjoint])
    data.qpos[bq:bq+3] = ball_xyz
    data.qpos[bq+3:bq+7] = [1.,0.,0.,0.]
    mujoco.mj_forward(model, data)
    meta = {'config': cfg, 'arm_joint_ids': joint_ids, 'arm_qpos_ids': qpos_ids,
            'arm_dof_ids': dof_ids, 'arm_actuator_ids': act_ids,
            'tray_site_id': site_id, 'tray_body_id': model.body('tray').id,
            'ball_body_id': model.body('ball').id, 'ball_qpos_address': bq,
            'ball_dof_address': int(model.jnt_dofadr[bjoint]),
            'ball_geom_id': model.geom('ball_geom').id,
            'tray_geom_id': model.geom('tray_plate').id,
            'wrist_ft_site_id': model.site('wrist_ft').id,
            'force_sensor_address': int(model.sensor_adr[model.sensor('tray_cut_force').id]),
            'torque_sensor_address': int(model.sensor_adr[model.sensor('tray_cut_torque').id]),
            'franka_hand_not_robotiq': True,
            'attachment_rigid_idealized': True,
            'attachment_geom_ids': [model.geom(name).id for name in
                ('attachment_grip','attachment_upper_arm','attachment_side_post','attachment_lower_support')]}
    return model, data, meta
