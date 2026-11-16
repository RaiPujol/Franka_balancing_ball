"""Synthetic Panda joint-torque residual -> wrench approximation.

The MuJoCo cut wrench is projected into 7 joint torques via J.T, corrupted by
illustrative joint-torque noise, and reconstructed by a damped least-squares
pseudoinverse. This is a measurement surrogate, not Franka's real estimator;
joint-torque biases, friction, torque filtering, stiffness behavior, and model
errors are not measured or calibrated. No ball state is used for estimation.
"""
import numpy as np
import mujoco
from force_estimator import center_of_pressure
from wrench_control import WrenchObserver
from wrench_frames import wrench_at_tray


class JointTorqueWrenchObserver(WrenchObserver):
    def __init__(self,config=None,*,joint_torque_std=.0001,
                 min_singular_value=.02,damping=.001,**kwargs):
        # Only synthetic joint torque noise is applied in this measurement mode.
        kwargs['force_std']=0.;kwargs['torque_std']=0.
        super().__init__(config,**kwargs)
        self.joint_torque_std=float(joint_torque_std)
        self.min_singular_value=float(min_singular_value)
        self.damping=float(damping)
        if not np.isfinite([self.joint_torque_std,self.min_singular_value,self.damping]).all() or self.joint_torque_std<0 or min(self.min_singular_value,self.damping)<=0:
            raise ValueError('invalid joint-torque proxy settings')
        self.torque_rng=np.random.default_rng(np.random.SeedSequence([kwargs.get('seed',0),4102]))

    def observe(self,live_model,live_data):
        m,d,meta=self.empty_model,self.empty,self.meta
        if (live_model.nq,live_model.nv,live_model.nu)!=(m.nq,m.nv,m.nu):raise ValueError('model dimensions disagree')
        t=float(live_data.time)
        if self.prev_time is not None and t<self.prev_time-1e-10:raise ValueError('time moved backwards')
        if self.prev_time is not None and abs(t-self.prev_time)<=1e-12:return self.last_estimate
        v=live_data.qvel[self.vidx].copy()+self.joint_rng.normal(0.,self.joint_velocity_std,len(self.vidx))
        if self.prev_time is not None:
            dt=t-self.prev_time;derivative=(v-self.prev_vel)/dt
            self.acc+=(1-np.exp(-dt/self.accel_tau))*(derivative-self.acc)
        self.prev_vel=v;self.prev_time=t
        d.qpos[self.qidx]=live_data.qpos[self.qidx];d.qvel[self.vidx]=v;d.ctrl[:]=live_data.ctrl
        mujoco.mj_forward(m,d);d.qacc[self.vidx]=self.acc
        mujoco.mj_rnePostConstraint(m,d);mujoco.mj_sensorAcc(m,d)
        fa=meta['force_sensor_address'];ta=meta['torque_sensor_address']
        cut_force=live_data.sensordata[fa:fa+3]-d.sensordata[fa:fa+3]
        cut_torque=live_data.sensordata[ta:ta+3]-d.sensordata[ta:ta+3]
        wrist=meta['wrist_ft_site_id'];top=meta['tray_site_id']
        R=d.site_xmat[wrist].reshape(3,3)
        jp=np.zeros((3,m.nv));jr=np.zeros((3,m.nv))
        mujoco.mj_jacSite(m,d,jp,jr,wrist)
        J=np.vstack((jp[:,meta['arm_dof_ids']],jr[:,meta['arm_dof_ids']]))
        singular_values=np.linalg.svd(J,compute_uv=False)
        if singular_values[-1]<self.min_singular_value:
            self.last_estimate=None
            raise RuntimeError('joint torque wrench estimate unavailable near singularity')
        ideal_wrench_world=np.r_[R@cut_force,R@cut_torque]
        # This is an idealized, noise-corrupted synthetic *joint residual*,
        # not the actual Franka internal dynamics observer or measured signal.
        tau_ext=J.T@ideal_wrench_world+self.torque_rng.normal(0.,self.joint_torque_std,7)
        w_world=np.linalg.solve(J@J.T+self.damping**2*np.eye(6),J@tau_ext)
        force=R.T@w_world[:3]+self.fb
        torque=R.T@w_world[3:]+self.tb
        Rt=d.site_xmat[top].reshape(3,3)
        tray_force,torque_plane=wrench_at_tray(force,torque,R,Rt,d.site_xpos[top]-d.site_xpos[wrist])
        obs=center_of_pressure(tray_force,torque_plane,self.min_force)
        obs.update(force_n=tray_force.copy(),torque_at_tray_top_nm=torque_plane.copy(),
                   sample_time=t,minimum_singular_value=float(singular_values[-1]))
        self.pending.append(obs);result=None
        while self.pending and self.pending[0]['sample_time']<=t-self.delay_s+1e-10:result=self.pending.popleft()
        if result is None:return None
        self.last=result
        if not result['valid']:
            self.last_estimate=None
            raise RuntimeError('wrench feedback unavailable: '+result['reason'])
        z=result['position_m'];stamp=result['sample_time']
        if self.pos is None:self.pos=z.copy();self.vel=np.zeros(2);self.filter_time=stamp
        elif stamp>self.filter_time+1e-12:
            dt=stamp-self.filter_time;old=self.pos.copy()
            self.pos+=(1-np.exp(-dt/self.position_tau))*(z-self.pos)
            self.vel+=(1-np.exp(-dt/self.velocity_tau))*((self.pos-old)/dt-self.vel)
            self.filter_time=stamp
        self.last_estimate={'position_m':(self.pos+self.vel*self.delay_s).copy(),
                'velocity_m_s':self.vel.copy(),'raw_position_m':z.copy(),
                'force_n':result['force_n'],'sample_time':stamp,'delay_s':t-stamp,
                'minimum_singular_value':result['minimum_singular_value']}
        return self.last_estimate
