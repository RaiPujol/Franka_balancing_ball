"""Simulation-only wrench-to-contact estimator; not a real Franka interface.

A MuJoCo ideal internal sensor measures the wrench at the hand/link7 cut.
We subtract the wrench predicted by an unloaded duplicate of the hand+tray,
using the live arm/finger joint positions, velocities, and accelerations.
The duplicate ball is collision-disabled and parked away from the robot. No
ball position or velocity is read by the estimator. A moment shift moves the
wrist residual to the tray top, then a planar center of pressure is obtained.

Ideal compensating-model parameters and exact qacc make this optimistic. An
actual Franka external-wrench estimate comes from the joint torque sensors,
not this ideal cut sensor. Its observability near singularities, calibration,
latency, filtering, model error and noise remain to be measured. Contacts with
anything else can be mistaken for the ball. This estimates *contact location*,
which agrees with projected ball center only while it contacts a flat plate.
"""
from __future__ import annotations
import numpy as np
import mujoco
from robot_scene import build_model
from wrench_frames import wrench_at_tray


def center_of_pressure(force, torque_at_plane, min_normal_force_n=.2,
                       force_std_n=None, torque_std_nm=None):
    """Return planar center of pressure and optional first-order uncertainty.

    Wrench is in plane coordinates, both F and T following the same sign
    convention. Fz must be positive; the caller handles sensor sign. The
    uncertainty assumes independent isotropic Fz, Tx, Ty measurement errors;
    it does not capture bias, rolling contact torque, or compensation error.
    """
    f = np.asarray(force, float)
    t = np.asarray(torque_at_plane, float)
    if f.shape != (3,) or t.shape != (3,) or not np.isfinite(f).all() or not np.isfinite(t).all():
        return {'valid':False, 'reason':'nonfinite_wrench', 'position_m':None, 'std_m':None}
    if not np.isfinite(min_normal_force_n) or min_normal_force_n <= 0:
        raise ValueError('min_normal_force_n must be positive')
    if f[2] < min_normal_force_n:
        return {'valid':False,'reason':'insufficient_normal_force', 'position_m':None,'std_m':None}
    xy = np.array([-t[1]/f[2], t[0]/f[2]])
    std = None
    if force_std_n is not None or torque_std_nm is not None:
        if force_std_n is None or torque_std_nm is None:
            raise ValueError('Specify both force and torque standard deviation')
        sf = float(force_std_n); st = float(torque_std_nm)
        if not np.isfinite(sf) or not np.isfinite(st) or sf < 0 or st < 0:
            raise ValueError('Noise standard deviations must be nonnegative finite')
        std = np.sqrt((st/f[2])**2 + (xy*sf/f[2])**2)
    return {'valid':True,'reason':'ideal_flat_contact_assumed','position_m':xy,'std_m':std}


class IdealWristWrenchEstimator:
    """Estimate plate contact point from ideal cut wrench and robot state.

    Call mj_forward(live_model, live_data) before observe(), so sensor readings
    and qacc correspond to the state. This is an offline numerical experiment.
    noise parameters are illustrative, not specifications of real hardware.
    """
    def __init__(self, config=None, *, min_normal_force_n=.2,
                 force_noise_std_n=0., torque_noise_std_nm=0., seed=0):
        self.empty_model, self.empty_data, self.meta = build_model(config)
        self.min_normal_force_n = float(min_normal_force_n)
        self.force_noise_std_n = float(force_noise_std_n)
        self.torque_noise_std_nm = float(torque_noise_std_nm)
        if self.min_normal_force_n <= 0 or not np.isfinite(self.min_normal_force_n):
            raise ValueError('min_normal_force_n must be positive')
        if min(self.force_noise_std_n,self.torque_noise_std_nm) < 0 or not np.isfinite([self.force_noise_std_n,self.torque_noise_std_nm]).all():
            raise ValueError('noise standard deviations must be finite and nonnegative')
        m=self.empty_model; geo=self.meta['ball_geom_id']
        m.geom_contype[geo] = 0; m.geom_conaffinity[geo] = 0
        self.empty_data.qpos[self.meta['ball_qpos_address']:self.meta['ball_qpos_address']+3]=[5.,5.,5.]
        joints=[m.joint(f'joint{i}').id for i in range(1,8)] + [m.joint('finger_joint1').id,m.joint('finger_joint2').id]
        self.robot_qadr=np.asarray([m.jnt_qposadr[j] for j in joints],int)
        self.robot_dadr=np.asarray([m.jnt_dofadr[j] for j in joints],int)
        self.rng=np.random.default_rng(seed)

    def observe(self, live_model, live_data):
        m=self.empty_model; d=self.empty_data; meta=self.meta
        # Copy ONLY articulated robot data, not the independent ball's state.
        if live_model.nq != m.nq or live_model.nv != m.nv or live_model.nu != m.nu:
            raise ValueError('Model does not match the unloaded compensation model')
        for name in ('body_mass','body_inertia','body_pos','body_quat','geom_size','geom_friction','dof_damping'):
            if not np.array_equal(getattr(live_model,name),getattr(m,name)):
                raise ValueError(f'Compensation model differs from live model: {name}')
        d.qpos[self.robot_qadr]=live_data.qpos[self.robot_qadr]
        d.qvel[self.robot_dadr]=live_data.qvel[self.robot_dadr]
        d.ctrl[:]=live_data.ctrl
        mujoco.mj_forward(m,d)
        # Acceleration is directly available from MuJoCo, but must be measured
        # or estimated and filtered in any physical implementation.
        d.qacc[self.robot_dadr]=live_data.qacc[self.robot_dadr]
        mujoco.mj_rnePostConstraint(m,d)
        mujoco.mj_sensorAcc(m,d)
        fa=meta['force_sensor_address'];ta=meta['torque_sensor_address']
        f=live_data.sensordata[fa:fa+3]-d.sensordata[fa:fa+3]
        t=live_data.sensordata[ta:ta+3]-d.sensordata[ta:ta+3]
        # Add simple illustrative independent sensor noise after compensation.
        f=f+self.rng.normal(0.,self.force_noise_std_n,3)
        t=t+self.rng.normal(0.,self.torque_noise_std_nm,3)
        wrist=meta['wrist_ft_site_id'];top=meta['tray_site_id']
        R=live_data.site_xmat[wrist].reshape(3,3)
        Rt=live_data.site_xmat[top].reshape(3,3)
        delta=live_data.site_xpos[top]-live_data.site_xpos[wrist]
        ftray,ttop=wrench_at_tray(f,t,R,Rt,delta)
        result=center_of_pressure(ftray,ttop,self.min_normal_force_n)
        if result['valid']:
            # Linear propagation of independent isotropic wrist-axis force and
            # torque noise through the moment shift AND wrist-to-tray rotation.
            # This excludes systematic bias and model error.
            r=R.T@delta;cross=np.array([[0.,-r[2],r[1]],[r[2],0.,-r[0]],[-r[1],r[0],0.]])
            A=Rt.T@R;M=-A@cross;fz=ftray[2]
            jf=np.vstack((-M[1]/fz+ttop[1]*A[2]/fz**2,
                           M[0]/fz-ttop[0]*A[2]/fz**2))
            jt=np.vstack((-A[1]/fz,A[0]/fz))
            sf=self.force_noise_std_n;st=self.torque_noise_std_nm
            result['std_m']=np.sqrt(sf**2*np.sum(jf*jf,axis=1)+st**2*np.sum(jt*jt,axis=1))
        result.update({'force_n':ftray.copy(),'torque_at_tray_top_nm':ttop.copy(),
                       'noise_model':'independent zero-mean sensor noise only; excludes model error',
                       'source':'ideal simulated hand-flange wrench and exact robot state'})
        return result
