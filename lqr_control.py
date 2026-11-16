"""Simulation-only delayed-wrench LQG candidate; no physical motion interface.

The ball dynamics and LQR are deliberately approximate; this is not a safety
controller, and the simulated sensor is not calibrated to the lab Franka.
"""
import numpy as np
from scipy.linalg import solve_continuous_are
from robust_control import RobustWrenchController


def lqr_gains(q_position, q_velocity, r_tilt, inertia_ratio=.4, gravity=9.81):
    """Return positive (position, velocity) slope gains for rolling sphere.

    Per axis the small-angle model is x_dot=v, v_dot=-(g/(1+I/mr²))*slope.
    We minimize the continuous-time integral q_position*x²+q_velocity*v²+
    r_tilt*slope².  In this two-state model LQR is a PD feedback law; Q/R
    weights select the gains. This is not a constrained or global optimum.
    """
    q_position=float(q_position);q_velocity=float(q_velocity);r_tilt=float(r_tilt)
    inertia_ratio=float(inertia_ratio);gravity=float(gravity)
    if not np.isfinite([q_position,q_velocity,r_tilt,inertia_ratio,gravity]).all() or min(q_position,r_tilt,gravity)<=0 or min(q_velocity,inertia_ratio)<0:
        raise ValueError('LQR costs and dynamics must be finite and physically valid')
    b=gravity/(1.+inertia_ratio)
    A=np.array([[0.,1.],[0.,0.]])
    B=np.array([[0.],[-b]])
    Q=np.diag([q_position,q_velocity]);R=np.array([[r_tilt]])
    P=solve_continuous_are(A,B,Q,R)
    K=np.linalg.solve(R,B.T@P)
    gains=-K[0]
    if not np.isfinite(gains).all() or np.min(gains)<0:
        raise ValueError('LQR solver returned invalid gains')
    return float(gains[0]),float(gains[1])


class LQGWrenchController(RobustWrenchController):
    """LQR position/velocity gains plus delayed virtual-wrench Kalman filter.

    Ball measurements come from the inherited joint-wrench observer, not from
    MuJoCo ball state. Actuation is the existing task-space arm controller.
    """
    def __init__(self,model,data,config=None,*,lqr_q_position=.36,
                 lqr_q_velocity=.46874617737003077,lqr_r=1.,**kwargs):
        kp,kd=lqr_gains(lqr_q_position,lqr_q_velocity,lqr_r)
        # Shared benchmark runners supply feedback gains; LQR owns these two.
        kwargs.pop('feedback_kp',None);kwargs.pop('feedback_kd',None)
        super().__init__(model,data,config=config,feedback_kp=kp,feedback_kd=kd,**kwargs)
        self.lqr_weights={'q_position':float(lqr_q_position),
                          'q_velocity':float(lqr_q_velocity),'r_tilt':float(lqr_r)}
        self.lqr_gains=(kp,kd)
