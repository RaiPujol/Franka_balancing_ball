"""Receding-horizon tray-slope control with PD stabilizer, simulation only.

Uses wrench-derived position/velocity from RobustWrenchController. The MPC
models rolling ball dynamics and a first-order plate orientation response. It
solves a finite-horizon bounded optimization, re-planning at a fixed interval.
The optimizer predicts a blend of its slope sequence with the current PD
command frozen over its horizon. In execution the PD part changes at each
control step, so this is an approximation, not a full closed-loop prediction.
It is a heuristic, not a collision checker or physical motion controller.
SciPy optimizer iterations have no enforceable wall-clock deadline here.
"""
from dataclasses import dataclass
import time
import numpy as np
import mujoco
from scipy.optimize import minimize
from controller import orientation_error, rotation_vector
from controller import desired_tray_rotation
from robust_control import RobustWrenchController


@dataclass
class MPCParameters:
    dt: float = .05
    horizon: int = 14
    lag_s: float = .25
    position_weight: float = 25.
    velocity_weight: float = 10.
    tilt_weight: float = .7
    delta_weight: float = 20.
    edge_penalty: float = 30000.
    edge_margin_m: float = .008
    max_tilt_rad: float = .10
    max_iterations: int = 150
    predictive_blend: float = .1

    def validate(self):
        a=np.array([self.dt,self.lag_s,self.position_weight,self.velocity_weight,self.tilt_weight,
                    self.delta_weight,self.edge_penalty,self.edge_margin_m,self.max_tilt_rad],float)
        if (not isinstance(self.horizon, int) or isinstance(self.horizon, bool)
            or not isinstance(self.max_iterations,int) or isinstance(self.max_iterations,bool)
            or not np.isfinite(a).all() or np.any(a<=0) or self.dt>0.2 or self.horizon<2 or self.horizon>80 or self.max_iterations<1):
            raise ValueError('invalid MPC parameters')
        if self.dt/self.lag_s>1:
            raise ValueError('model timestep exceeds plate lag')
        if not np.isfinite(self.predictive_blend) or not 0<=self.predictive_blend<=1:
            raise ValueError('predictive blend must be in [0,1]')


class BallSlopeMPC:
    """Axis-separable finite-horizon optimizer with bounded slope and soft edge cost.

    State = ball position, velocity, plate slope. Plate slope follows the
    command via first-order lag. Ball acceleration includes gravity and the
    specified Cartesian tray acceleration. Bounds constrain each slope axis;
    the command is additionally norm-limited. Edge penalties are soft and do
    not guarantee containment. No measured physical parameter is claimed.
    """
    def __init__(self,params=None,half_extent=(.08,.08),ball_radius=.021335):
        self.params=params or MPCParameters();self.params.validate()
        self.half_extent=np.asarray(half_extent,float)
        self.ball_radius=float(ball_radius)
        if self.half_extent.shape!=(2,) or not np.isfinite(self.half_extent).all() or not np.isfinite(self.ball_radius) or self.ball_radius<=0 or min(self.half_extent)<=self.ball_radius+self.params.edge_margin_m:
            raise ValueError('tray too small for MPC margin')
        p=self.params;h=p.dt;g=9.81/1.4;self.geff=g
        self.A=np.array([[1,h,-.5*g*h*h],[0,1,-g*h],[0,0,1-h/p.lag_s]])
        self.B=np.array([0.,0.,h/p.lag_s]);self.E=np.array([-.5*g*h*h/9.81,-g*h/9.81,0.])
        self.G=np.eye(p.horizon)-np.eye(p.horizon,k=-1)
        self.last_command=np.zeros(2);self.last_applied=np.zeros(2);self.last_sequence=np.zeros((2,p.horizon))
        self.solve_times=[];self.solver_failures=0;self.last_status=None

    def _matrices(self,x,disturbance,baseline_slope=0.,blend=1.):
        n=self.params.horizon; state=x.copy();sens=np.zeros((3,n));base=np.empty(n);D=np.empty((n,n));vbase=np.empty(n);V=np.empty((n,n))
        for k in range(n):
            state=self.A@state+self.E*disturbance[k]+self.B*(1.-blend)*baseline_slope
            sens=self.A@sens;sens[:,k]+=self.B*blend
            base[k]=state[0];D[k]=sens[0];vbase[k]=state[1];V[k]=sens[1]
        return base,D,vbase,V

    def solve(self,pos,vel,slopes,target_xy,accel_xy,target_velocity_xy=None,baseline_slope=None):
        p=self.params;n=p.horizon;pos=np.asarray(pos,float);vel=np.asarray(vel,float);slopes=np.asarray(slopes,float)
        target=np.asarray(target_xy,float);accel=np.asarray(accel_xy,float)
        target_velocity=np.zeros_like(target) if target_velocity_xy is None else np.asarray(target_velocity_xy,float)
        baseline=np.zeros(2) if baseline_slope is None else np.asarray(baseline_slope,float)
        if any(a.shape!=(2,) for a in (pos,vel,slopes,baseline)) or target.shape!=(n,2) or accel.shape!=(n,2) or target_velocity.shape!=(n,2) or not all(np.isfinite(a).all() for a in (pos,vel,slopes,target,accel,target_velocity,baseline)):
            raise ValueError('nonfinite or malformed MPC observations')
        results=[];max_bound=p.max_tilt_rad/np.sqrt(2.);start=time.perf_counter()
        for ax in range(2):
            alpha=p.predictive_blend
            base,D,vbase,V=self._matrices(np.array([pos[ax],vel[ax],slopes[ax]]),accel[:,ax],baseline[ax],alpha)
            diff0=np.r_[self.last_applied[ax],np.zeros(n-1)]
            edge=self.half_extent[ax]-self.ball_radius-p.edge_margin_m
            def f(u):
                x=base+D@u;err=x-target[:,ax];verr=vbase+V@u-target_velocity[:,ax]
                applied=(1.-alpha)*baseline[ax]+alpha*u;du=self.G@applied-diff0
                spill=np.maximum(np.abs(x)-edge,0.)*np.sign(x)
                cost=p.position_weight*np.dot(err,err)+p.velocity_weight*np.dot(verr,verr)+p.tilt_weight*np.dot(applied,applied)+p.delta_weight*np.dot(du,du)+p.edge_penalty*np.dot(spill,spill)
                grad=2*(p.position_weight*(D.T@err)+p.velocity_weight*(V.T@verr)+alpha*p.tilt_weight*applied+alpha*p.delta_weight*(self.G.T@du)+p.edge_penalty*(D.T@spill))
                return cost,grad
            initial=np.r_[self.last_sequence[ax,1:],self.last_sequence[ax,-1]]
            opt=minimize(f,initial,method='L-BFGS-B',jac=True,bounds=[(-max_bound,max_bound)]*n,
                         options={'maxiter':p.max_iterations,'ftol':1e-8,'gtol':1e-6})
            if not opt.success or not np.isfinite(opt.x).all():
                self.solver_failures+=1;self.last_status=str(opt.message)
                raise RuntimeError('MPC optimizer did not converge: '+str(opt.message))
            self.last_sequence[ax]=opt.x
            results.append(float(opt.x[0]))
        self.solve_times.append(time.perf_counter()-start)
        self.last_command=np.asarray(results);self.last_status='converged'
        return self.last_command.copy()


class MPCWrenchController(RobustWrenchController):
    """Blended MPC/PD controller fed by the wrench-based ball estimator."""
    def __init__(self,model,data,config=None,*,mpc_params=None,**kwargs):
        super().__init__(model,data,config=config,**kwargs)
        cfg=dict(config or {})
        self.mpc=BallSlopeMPC(mpc_params,(float(cfg.get('tray_length_m',.16))/2,float(cfg.get('tray_width_m',.16))/2),float(cfg.get('ball_diameter_m',.04267))/2)
        self.next_plan_t=None;self.command_slope=np.zeros(2)

    def step(self,t,dt=None):
        m,d=self.model,self.data;dt=m.opt.timestep if dt is None else float(dt)
        if dt<=0:raise ValueError('dt must be positive')
        p,R=d.site_xpos[self.site_id],d.site_xmat[self.site_id].reshape(3,3)
        jp,jr=np.zeros((3,m.nv)),np.zeros((3,m.nv));mujoco.mj_jacSite(m,d,jp,jr,self.site_id)
        bp,bv=self.ball_state(jp,jr)
        displacement,v,a=self.path.sample(t);desired_position=self.initial_position+displacement
        ball_target,ball_target_vel,ball_target_accel=self.desired_ball_state(t)
        baseline=desired_tray_rotation(self.initial_rotation,a,'estimated_feedback',
                bp[:2]-ball_target,bv[:2]-ball_target_vel,self.mpc.params.max_tilt_rad,
                kp=self.feedback_kp,kd=self.feedback_kd,relative_accel_xy=ball_target_accel)
        relative=self.initial_rotation.T@baseline
        baseline_slope=np.array([relative[2,0],relative[2,1]])
        if self.next_plan_t is None or t>=self.next_plan_t-1e-9:
            rel=self.initial_rotation.T@R;slopes=np.array([rel[2,0],rel[2,1]])
            ts=t+self.mpc.params.dt*np.arange(1,self.mpc.params.horizon+1)
            refs=np.array([self.desired_ball_state(float(tk))[0] for tk in ts])
            vrefs=np.array([self.desired_ball_state(float(tk))[1] for tk in ts])
            accels=np.array([(self.initial_rotation.T@self.path.sample(float(tk))[2])[:2] for tk in ts])
            self.command_slope=self.mpc.solve(bp[:2],bv[:2],slopes,refs,accels,vrefs,baseline_slope)
            self.next_plan_t=t+self.mpc.params.dt
        alpha=self.mpc.params.predictive_blend
        command=(1-alpha)*baseline_slope+alpha*self.command_slope
        magnitude=np.linalg.norm(command)
        if magnitude>self.mpc.params.max_tilt_rad:command*=self.mpc.params.max_tilt_rad/magnitude
        self.mpc.last_applied=command.copy()
        desired_rotation=self.initial_rotation@rotation_vector([command[1],-command[0],0.])
        twist=np.r_[v+3*(desired_position-p),3.5*orientation_error(R,desired_rotation)]
        twist[:3]=np.clip(twist[:3],-.20,.20);twist[3:]=np.clip(twist[3:],-.50,.50)
        J=np.vstack((jp[:,self.dofs],jr[:,self.dofs]))
        dq=J.T@np.linalg.solve(J@J.T+self.damping**2*np.eye(6),twist)
        dq=np.clip(dq,-self.max_qdot,self.max_qdot);self.qref+=dt*dq
        q=d.qpos[self.qids];self.qref=np.clip(self.qref,q-.15,q+.15)
        tau=self.kp*(self.qref-q)+self.kd*(dq-d.qvel[self.dofs])+d.qfrc_bias[self.dofs]
        for i,act in enumerate(self.actuators):
            if m.actuator_ctrllimited[act]:tau[i]=np.clip(tau[i],*m.actuator_ctrlrange[act])
        d.ctrl[self.actuators]=tau
        self.last_state={'ball_pos_local':bp.copy(),'ball_vel_local':bv.copy(),'ball_target_local':ball_target.copy(),
                         'ball_target_vel_local':ball_target_vel.copy(),'target_pos':desired_position.copy(),
                         'target_rotation':desired_rotation.copy(),'qdot_cmd':dq.copy(),'tau_cmd':tau.copy()}
        return self.last_state
