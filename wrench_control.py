"""Simulation-only ball-state feedback from noisy, delayed flange wrench.

No true ball pose, velocity, contact, or acceleration is accessed.  This is an
ideal MuJoCo flange-cut wrench plus synthetic noise/bias/delay, NOT a model of
Franka joint-torque-to-wrench estimation. The tool geometry and inertial model
are assumed perfectly known. Robot position and velocity are assumed observed
without error. Acceleration is differentiated from observed joint velocities.
"""
from collections import deque
import numpy as np
import mujoco

from controller import TrayController
from force_estimator import center_of_pressure
from robot_scene import build_model
from wrench_frames import wrench_at_tray


class WrenchObserver:
    """Estimate ball contact point from a simulated sensor and robot state.

    `force_bias` and `torque_bias` are added in wrist-site sensor axes before
    shifting the wrench to the tray origin and rotating into tray axes.
    """
    def __init__(self, config=None, *, force_std=.005, torque_std=.0001,
                 delay_s=.02, seed=0, force_bias=(0.,0.,0.),
                 torque_bias=(0.,0.,0.), accel_tau=.0001,
                 position_tau=.010, velocity_tau=.050, min_normal_force_n=.2,
                 joint_velocity_std=0.):
        config = dict(config or {})
        # The unknown ball's starting point must never initialize this observer.
        config.pop('ball_initial_offset_m',None)
        self.empty_model,self.empty,self.meta=build_model(config)
        m,d=self.empty_model,self.empty
        b=self.meta['ball_geom_id'];m.geom_contype[b]=0;m.geom_conaffinity[b]=0
        d.qpos[self.meta['ball_qpos_address']:self.meta['ball_qpos_address']+3]=[5.,5.,5.]
        joints=[m.joint(f'joint{i}').id for i in range(1,8)]+[m.joint('finger_joint1').id,m.joint('finger_joint2').id]
        self.qidx=np.asarray([m.jnt_qposadr[j] for j in joints],int)
        self.vidx=np.asarray([m.jnt_dofadr[j] for j in joints],int)
        self.force_std=float(force_std);self.torque_std=float(torque_std);self.delay_s=float(delay_s)
        self.accel_tau=float(accel_tau);self.position_tau=float(position_tau);self.velocity_tau=float(velocity_tau)
        self.min_force=float(min_normal_force_n)
        self.joint_velocity_std=float(joint_velocity_std)
        self.fb=np.asarray(force_bias,float);self.tb=np.asarray(torque_bias,float)
        vals=[self.force_std,self.torque_std,self.delay_s,self.accel_tau,self.position_tau,self.velocity_tau,self.min_force,self.joint_velocity_std]
        if not np.isfinite(vals).all() or min(self.force_std,self.torque_std,self.delay_s,self.joint_velocity_std)<0 or min(self.accel_tau,self.position_tau,self.velocity_tau,self.min_force)<=0 or self.fb.shape!=(3,) or self.tb.shape!=(3,) or not np.isfinite(self.fb).all() or not np.isfinite(self.tb).all():
            raise ValueError('invalid wrench observation settings')
        self.rng=np.random.default_rng(seed)
        self.joint_rng=np.random.default_rng(np.random.SeedSequence([seed, 240913]))
        self.pending=deque();self.prev_vel=None;self.prev_time=None;self.acc=np.zeros(len(self.vidx))
        self.pos=None;self.vel=np.zeros(2);self.filter_time=None;self.last=None;self.last_estimate=None

    def observe(self, live_model, live_data):
        """Return delayed estimate, or None while the delay buffer is filling.

        Call after mj_forward; uses only flange sensor data, robot q/dq, command,
        and kinematics derived from them. The delay applies after the measured
        wrench is compensated at the measurement timestamp.
        """
        m,d=self.empty_model,self.empty; meta=self.meta
        if (live_model.nq,live_model.nv,live_model.nu)!=(m.nq,m.nv,m.nu):
            raise ValueError('robot state dimensions disagree with tool model')
        t=float(live_data.time)
        if self.prev_time is not None and t < self.prev_time-1e-10:
            raise ValueError('simulation time went backwards')
        if self.prev_time is not None and abs(t-self.prev_time) <= 1e-12:
            # An extra call at the same physical time must not fabricate a
            # second sensor sample or a new velocity derivative.
            return self.last_estimate
        v=live_data.qvel[self.vidx].copy()+self.joint_rng.normal(0.,self.joint_velocity_std,len(self.vidx))
        if self.prev_time is not None and t > self.prev_time+1e-12:
            dt=t-self.prev_time
            derivative=(v-self.prev_vel)/dt
            self.acc += (1-np.exp(-dt/self.accel_tau))*(derivative-self.acc)
        self.prev_vel=v;self.prev_time=t
        d.qpos[self.qidx]=live_data.qpos[self.qidx]
        d.qvel[self.vidx]=v
        d.ctrl[:]=live_data.ctrl
        mujoco.mj_forward(m,d)
        d.qacc[self.vidx]=self.acc
        mujoco.mj_rnePostConstraint(m,d);mujoco.mj_sensorAcc(m,d)
        fa=meta['force_sensor_address'];ta=meta['torque_sensor_address']
        f=live_data.sensordata[fa:fa+3]-d.sensordata[fa:fa+3]+self.fb+self.rng.normal(0,self.force_std,3)
        tau=live_data.sensordata[ta:ta+3]-d.sensordata[ta:ta+3]+self.tb+self.rng.normal(0,self.torque_std,3)
        wrist=meta['wrist_ft_site_id'];top=meta['tray_site_id']
        # Relative wrist-to-tray transform belongs to the known rigid tool model.
        Rw=d.site_xmat[wrist].reshape(3,3);Rt=d.site_xmat[top].reshape(3,3)
        tray_force,plane_tau=wrench_at_tray(f,tau,Rw,Rt,d.site_xpos[top]-d.site_xpos[wrist])
        observation=center_of_pressure(tray_force,plane_tau,self.min_force)
        observation.update(force_n=tray_force.copy(),torque_at_tray_top_nm=plane_tau.copy(),sample_time=t)
        self.pending.append(observation)
        result=None
        while self.pending and self.pending[0]['sample_time']<=t-self.delay_s+1e-10:
            result=self.pending.popleft()
        if result is None:return None
        self.last=result
        if not result['valid']:
            self.last_estimate=None
            raise RuntimeError('wrench feedback unavailable: '+result['reason'])
        z=result['position_m']; stamp=result['sample_time']
        if self.pos is None:
            self.pos=z.copy();self.vel=np.zeros(2);self.filter_time=stamp
        elif stamp>self.filter_time+1e-12:
            dt=stamp-self.filter_time;old=self.pos.copy()
            self.pos += (1-np.exp(-dt/self.position_tau))*(z-self.pos)
            diff=(self.pos-old)/dt
            self.vel += (1-np.exp(-dt/self.velocity_tau))*(diff-self.vel)
            self.filter_time=stamp
        # Constant-velocity latency compensation, not a read of simulated ball.
        pred=self.pos+self.vel*self.delay_s
        self.last_estimate={'position_m':pred.copy(),'velocity_m_s':self.vel.copy(),
                'raw_position_m':z.copy(),'force_n':result['force_n'],
                'sample_time':stamp,'delay_s':t-stamp}
        return self.last_estimate


class WrenchController(TrayController):
    """Trajectory controller whose ball feedback comes solely from the wrench."""
    def __init__(self,model,data,config=None,*,path=None,target_path=None,
                 force_std=.005,torque_std=.0001,delay_s=.02,seed=0,
                 force_bias=(0.,0.,0.),torque_bias=(0.,0.,0.),
                 accel_tau=.0001,position_tau=.010,velocity_tau=.050,
                 feedback_kp=1.,feedback_kd=.8,joint_velocity_std=0.,
                 sensor_model='cut_wrench',joint_torque_std=.0001,
                 min_singular_value=.02,damping=.001,**kwargs):
        kwargs.pop('mode',None)
        super().__init__(model,data,path=path,mode='estimated_feedback',
                         feedback_kp=feedback_kp,feedback_kd=feedback_kd,**kwargs)
        self.target_path=target_path
        observer_kwargs=dict(force_std=force_std,torque_std=torque_std,
                delay_s=delay_s,seed=seed,force_bias=force_bias,torque_bias=torque_bias,
                accel_tau=accel_tau,position_tau=position_tau,velocity_tau=velocity_tau,
                joint_velocity_std=joint_velocity_std)
        if sensor_model=='cut_wrench':
            self.observer=WrenchObserver(config,**observer_kwargs)
        elif sensor_model=='joint_torque':
            # This sensor mode synthesizes joint torque residuals from an ideal
            # flange wrench; force_std/torque_std are not applied in this mode.
            from joint_wrench_proxy import JointTorqueWrenchObserver
            self.observer=JointTorqueWrenchObserver(config,joint_torque_std=joint_torque_std,
                min_singular_value=min_singular_value,damping=damping,**observer_kwargs)
        else:
            raise ValueError('sensor_model must be cut_wrench or joint_torque')
        self.sensor_model=sensor_model
        self.last_observation=None
    def desired_ball_state(self,t):
        return self.target_path.sample(t) if self.target_path is not None else super().desired_ball_state(t)
    def prime(self):
        """Call while holding the arm before the trajectory starts."""
        try:
            self.last_observation=self.observer.observe(self.model,self.data)
        except RuntimeError:
            # Contact may not have formed in the first few simulated steps.
            self.last_observation=None
        return self.last_observation
    def ball_state(self,jp,jr):
        estimate=self.observer.observe(self.model,self.data)
        if estimate is None:raise RuntimeError('wrench delay buffer not ready: prime during stationary hold')
        self.last_observation=estimate
        return np.r_[estimate['position_m'],0.],np.r_[estimate['velocity_m_s'],0.]
