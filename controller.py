"""Simulation-only differential-IK and ball-on-tray controllers.

No ROS or hardware interfaces. `oracle_feedback` reads the simulated ball pose;
that state is not measured by a physical tray. Parameters and gains require
identification and independent validation before any hardware work.
"""
from dataclasses import dataclass
import numpy as np
import mujoco


def smoothstep5(u):
    """Quintic unit displacement and first two derivatives w.r.t. u."""
    if u <= 0.0:
        return 0.0, 0.0, 0.0
    if u >= 1.0:
        return 1.0, 0.0, 0.0
    return (10*u**3 - 15*u**4 + 6*u**5,
            30*u**2 - 60*u**3 + 30*u**4,
            60*u - 180*u**2 + 120*u**3)


@dataclass
class SmoothPath:
    """Cartesian displacements from the initial pose, with full stops at nodes."""
    offsets: tuple = ((0., 0., 0.), (0., .10, 0.), (.08, .10, 0.), (.08, 0., 0.), (0., 0., 0.))
    segment_s: float = 3.0
    start_delay_s: float = 1.0

    @property
    def end_time(self):
        return self.start_delay_s + self.segment_s * (len(self.offsets)-1)

    def sample(self, t):
        if self.segment_s <= 0 or len(self.offsets) < 2:
            raise ValueError('path requires at least two offsets and positive segment duration')
        offsets = np.asarray(self.offsets, float)
        elapsed = max(0., t - self.start_delay_s)
        if elapsed >= self.segment_s*(len(offsets)-1):
            return offsets[-1].copy(), np.zeros(3), np.zeros(3)
        i = min(int(elapsed // self.segment_s), len(offsets)-2)
        s, ds, dds = smoothstep5((elapsed-i*self.segment_s)/self.segment_s)
        delta = offsets[i+1]-offsets[i]
        return offsets[i]+s*delta, ds*delta/self.segment_s, dds*delta/self.segment_s**2


def rotation_vector(v):
    """Exponential map from small rotation vector to 3x3 rotation matrix."""
    v = np.asarray(v, dtype=float)
    angle = np.linalg.norm(v)
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = v / angle
    K = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K@K)


def orientation_error(current, target):
    """World-frame infinitesimal orientation error, valid near desired pose."""
    return .5 * sum(np.cross(current[:,i], target[:,i]) for i in range(3))


def desired_tray_rotation(reference, world_accel, mode, ball_xy=None, ball_vxy=None,
                          max_tilt_rad=np.deg2rad(7.), kp=2.0, kd=1.4,
                          relative_accel_xy=None, inertia_ratio=0.4):
    """Compute desired tray orientation; feedback reads simulation ball state.

    The linearized surface slopes in reference tray axes are -a/g for
    inertial compensation. Positive ball displacement requires a positive
    slope towards that side so gravity provides a restoring acceleration.
    """
    if mode not in ('level', 'feedforward', 'oracle_feedback', 'estimated_feedback'):
        raise ValueError('unknown controller mode')
    if mode == 'level':
        return reference.copy()
    if max_tilt_rad <= 0:
        raise ValueError('max tilt must be positive')
    slope = -(reference.T @ np.asarray(world_accel, float))[:2] / 9.81
    if mode in ('oracle_feedback', 'estimated_feedback'):
        if ball_xy is None or ball_vxy is None:
            raise ValueError('oracle feedback requires ball position and velocity')
        slope += kp*np.asarray(ball_xy) + kd*np.asarray(ball_vxy)
        if relative_accel_xy is not None:
            # Rolling sphere, I = inertia_ratio * m*r^2; a model assumption.
            slope -= (1.0 + inertia_ratio)*np.asarray(relative_accel_xy)/9.81
    # Limit norm, not each axis, so combined tilt remains bounded.
    mag = np.linalg.norm(slope)
    if mag > max_tilt_rad:
        slope *= max_tilt_rad / mag
    # local rotation about x raises +y; rotation about y lowers +x.
    tilt = np.array([slope[1], -slope[0], 0.])
    return reference @ rotation_vector(tilt)


class TrayController:
    """Torque-controlled seven-axis Panda using task-space velocity IK.

    Ball states in oracle mode are simulation ground truth. This simple
    controller is for offline experiments and does not certify collision,
    actuator limits, or real-world robustness.
    """
    def __init__(self, model, data, site='tray_top', ball_body='ball',
                 arm_qpos=None, arm_dofs=None, arm_actuators=None,
                 path=None, mode='level', max_tilt_rad=np.deg2rad(7.),
                 feedback_kp=2., feedback_kd=1.4):
        self.model, self.data = model, data
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        self.ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ball_body)
        if self.site_id < 0 or self.ball_id < 0:
            raise ValueError('tray site or ball body not found in model')
        self.qids = np.asarray(range(7) if arm_qpos is None else arm_qpos, int)
        self.dofs = np.asarray(range(7) if arm_dofs is None else arm_dofs, int)
        self.actuators = np.asarray(range(7) if arm_actuators is None else arm_actuators, int)
        if any(len(x)!=7 for x in (self.qids,self.dofs,self.actuators)):
            raise ValueError('controller expects seven arm joints and actuators')
        self.path = path or SmoothPath()
        self.mode = mode
        self.max_tilt_rad = max_tilt_rad
        self.feedback_kp = float(feedback_kp)
        self.feedback_kd = float(feedback_kd)
        if not np.isfinite([self.feedback_kp,self.feedback_kd]).all() or min(self.feedback_kp,self.feedback_kd)<0:
            raise ValueError('ball feedback gains must be nonnegative finite')
        self.initial_position = data.site_xpos[self.site_id].copy()
        self.initial_rotation = data.site_xmat[self.site_id].reshape(3,3).copy()
        self.qref = data.qpos[self.qids].copy()
        self.kp = np.array([300.,300.,250.,220.,100.,80.,60.])
        self.kd = np.array([30.,30.,25.,22.,12.,10.,8.])
        self.max_qdot = np.array([.45,.45,.45,.45,.65,.65,.65])
        self.damping = .08
        self.last_state = {}

    def desired_ball_state(self, t):
        """Stationary target at tray center; subclasses may override in simulation."""
        return np.zeros(2), np.zeros(2), np.zeros(2)

    def ball_state(self, jp, jr):
        d,m = self.data,self.model
        rot = d.site_xmat[self.site_id].reshape(3,3)
        rel_world = d.xpos[self.ball_id] - d.site_xpos[self.site_id]
        jac = np.zeros((3,m.nv))
        mujoco.mj_jacBody(m,d,jac,None,self.ball_id)
        ball_world_v = jac @ d.qvel
        tray_v, tray_w = jp@d.qvel, jr@d.qvel
        local_p = rot.T @ rel_world
        local_v = rot.T @ (ball_world_v-tray_v-np.cross(tray_w,rel_world))
        return local_p, local_v

    def step(self, t, dt=None):
        m,d = self.model,self.data
        dt = m.opt.timestep if dt is None else float(dt)
        if dt <= 0:
            raise ValueError('dt must be positive')
        p, R = d.site_xpos[self.site_id],d.site_xmat[self.site_id].reshape(3,3)
        jp, jr = np.zeros((3,m.nv)), np.zeros((3,m.nv))
        mujoco.mj_jacSite(m,d,jp,jr,self.site_id)
        bp,bv = self.ball_state(jp,jr)
        displacement,v,a = self.path.sample(t)
        desired_position = self.initial_position + displacement
        ball_target, ball_target_vel, ball_target_accel = self.desired_ball_state(t)
        desired_rotation = desired_tray_rotation(
            self.initial_rotation, a, self.mode, bp[:2]-ball_target,
            bv[:2]-ball_target_vel, self.max_tilt_rad, kp=self.feedback_kp, kd=self.feedback_kd,
            relative_accel_xy=ball_target_accel)
        # Velocity reference combines trajectory feedforward and task error.
        twist = np.r_[v + 3.0*(desired_position-p), 3.5*orientation_error(R,desired_rotation)]
        twist[:3] = np.clip(twist[:3],-.20,.20)
        twist[3:] = np.clip(twist[3:],-.50,.50)
        J = np.vstack((jp[:,self.dofs],jr[:,self.dofs]))
        dq = J.T @ np.linalg.solve(J@J.T + self.damping**2*np.eye(6),twist)
        dq = np.clip(dq,-self.max_qdot,self.max_qdot)
        self.qref += dt*dq
        # Integrator anti-windup: cap error relative to measured joints.
        q = d.qpos[self.qids]
        self.qref = np.clip(self.qref,q-.15,q+.15)
        tau = self.kp*(self.qref-q) + self.kd*(dq-d.qvel[self.dofs]) + d.qfrc_bias[self.dofs]
        # Respect any declared actuator torque limits.
        for i, act in enumerate(self.actuators):
            if m.actuator_ctrllimited[act]:
                tau[i] = np.clip(tau[i],*m.actuator_ctrlrange[act])
        d.ctrl[self.actuators] = tau
        self.last_state = {'ball_pos_local': bp.copy(), 'ball_vel_local':bv.copy(),
                           'ball_target_local': ball_target.copy(),
                           'ball_target_vel_local':ball_target_vel.copy(),
                           'target_pos':desired_position.copy(), 'target_rotation':desired_rotation.copy(),
                           'qdot_cmd':dq.copy(), 'tau_cmd':tau.copy()}
        return self.last_state
