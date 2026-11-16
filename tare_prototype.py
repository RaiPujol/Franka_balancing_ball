"""Experimental simulation-only empty-tool tare for the joint-wrench observer.

The stationary calibration consumes virtual wrench measurements and observed
robot pose; it does not read ball state. One separate empty-tray simulation is
required before the ball is placed. It is an exploratory offset correction,
not hardware code or a substitute for payload identification. A fixed tare is
not generally valid as the robot/tool orientation and acceleration change.
"""
from __future__ import annotations
import numpy as np
import mujoco
from robot_scene import build_model
from joint_wrench_proxy import JointTorqueWrenchObserver


def prepare_empty_scene(actual_config, observed_q):
    """For simulation only: construct a separate empty-tray measurement scene.

    `actual_config` is chosen by the experimenter as the simulated physical
    world, not provided to the controller as a measured physical mass. The
    sphere is collision-disabled and parked away before data generation.
    """
    m,d,meta=build_model(actual_config)
    g=meta['ball_geom_id'];m.geom_contype[g]=0;m.geom_conaffinity[g]=0
    qadr=meta['ball_qpos_address'];d.qpos[qadr:qadr+3]=[5.,5.,5.]
    q=np.asarray(observed_q,float)
    if q.shape!=(9,) or not np.isfinite(q).all():raise ValueError('expected nine observed arm/finger joints')
    joint_names=[m.joint(f'joint{i}').id for i in range(1,8)]+[m.joint('finger_joint1').id,m.joint('finger_joint2').id]
    qidx=[m.jnt_qposadr[j] for j in joint_names];d.qpos[qidx]=q
    d.qvel[:]=0.;mujoco.mj_forward(m,d)
    # Simulate a physically stationary measurement: neither exact acceleration
    # nor an unknown ball state is supplied to the observer.
    d.qacc[:]=0.;mujoco.mj_rnePostConstraint(m,d);mujoco.mj_sensorAcc(m,d)
    return m,d


def stationary_empty_tare(measurement_model,measurement_data,estimated_config,*,
                          joint_torque_std=.0001,joint_velocity_std=.0001,
                          force_bias=(0.,0.,0.),torque_bias=(0.,0.,0.),
                          seed=0,samples=500,damping=.001,min_singular_value=.02):
    """Estimate fixed empty residual (force, moment) at measured wrist frame.

    Inputs are only a virtual sensor stream and observed robot joints; no true
    tool mass or ball position is read from `measurement_model` by this routine.
    Samples here assume stationary q/dq, independent white sensor noise and
    perfect q position; the commanded arm dynamics are NOT calibrated.
    """
    if not isinstance(samples,int) or samples<20 or samples>100000:raise ValueError('20-100000 tare samples required')
    if not np.isfinite([joint_torque_std,joint_velocity_std,damping,min_singular_value]).all() or min(joint_torque_std,joint_velocity_std)<0 or min(damping,min_singular_value)<=0:raise ValueError('invalid noise or numerical parameters')
    fb=np.asarray(force_bias,float);tb=np.asarray(torque_bias,float)
    if fb.shape!=(3,) or tb.shape!=(3,) or not np.isfinite(np.r_[fb,tb]).all():raise ValueError('invalid bias')
    observer=JointTorqueWrenchObserver(estimated_config,joint_torque_std=joint_torque_std,
        joint_velocity_std=joint_velocity_std,damping=damping,min_singular_value=min_singular_value)
    m,d,meta=observer.empty_model,observer.empty,observer.meta
    if (m.nq,m.nv,m.nu)!=(measurement_model.nq,measurement_model.nv,measurement_model.nu):raise ValueError('model dimensions differ')
    q=measurement_data.qpos[observer.qidx].copy();v=measurement_data.qvel[observer.vidx].copy()
    if np.max(np.abs(v[:7]))>.002 or np.max(np.abs(v[7:]))>.001:raise ValueError('calibration requires stationary robot')
    # Read simulated sensor outputs once. They represent a stationary empty tray.
    fa=meta['force_sensor_address'];ta=meta['torque_sensor_address']
    measured_force=measurement_data.sensordata[fa:fa+3].copy()
    measured_torque=measurement_data.sensordata[ta:ta+3].copy()
    if not np.isfinite(np.r_[measured_force,measured_torque]).all():raise ValueError('invalid measured wrench')
    rng=np.random.default_rng(seed);out=[]
    for _ in range(samples):
        d.qpos[observer.qidx]=q
        d.qvel[observer.vidx]=v+rng.normal(0.,joint_velocity_std,len(observer.vidx))
        d.ctrl[:]=measurement_data.ctrl
        mujoco.mj_forward(m,d)
        # Zero acceleration is an assumption for the stationary tare, not a
        # measurement read from the simulator or real robot.
        d.qacc[observer.vidx]=0.;mujoco.mj_rnePostConstraint(m,d);mujoco.mj_sensorAcc(m,d)
        R=d.site_xmat[meta['wrist_ft_site_id']].reshape(3,3)
        jp=np.zeros((3,m.nv));jr=np.zeros((3,m.nv))
        mujoco.mj_jacSite(m,d,jp,jr,meta['wrist_ft_site_id'])
        J=np.vstack((jp[:,meta['arm_dof_ids']],jr[:,meta['arm_dof_ids']]))
        if np.linalg.svd(J,compute_uv=False)[-1]<min_singular_value:raise RuntimeError('tare pose too close to singularity')
        world=np.r_[R@(measured_force-d.sensordata[fa:fa+3]),R@(measured_torque-d.sensordata[ta:ta+3])]
        joint=J.T@world+rng.normal(0.,joint_torque_std,7)
        estimate=np.linalg.solve(J@J.T+damping**2*np.eye(6),J@joint)
        out.append(np.r_[R.T@estimate[:3]+fb,R.T@estimate[3:]+tb])
    data=np.asarray(out);med=np.median(data,axis=0)
    return {'force_bias_correction':(-med[:3]).tolist(), 'torque_bias_correction':(-med[3:]).tolist(),
            'empty_residual_mean_wrist':np.mean(data,axis=0).tolist(),
            'empty_residual_std_wrist':np.std(data,axis=0,ddof=1).tolist(),
            'tare_samples':samples,'simulated_noise_assumed':True,
            'limitation':'Fixed local wrist-frame tare measured once at one stationary pose; dynamic tool model errors and orientation changes are not calibrated.'}
