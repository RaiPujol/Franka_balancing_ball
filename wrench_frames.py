"""Frame utilities for simulated wrench observations, independent of ball pose."""
import numpy as np


def wrench_at_tray(force_wrist,torque_wrist,world_R_wrist,world_R_tray,world_wrist_to_tray):
    """Shift a wrist-origin wrench to tray origin and rotate into tray axes.

    Rotation matrices map respective local axes to world; `world_wrist_to_tray`
    is the world vector from the wrist-origin to tray-origin. Both output arrays
    are force and moment expressed in tray coordinates. This operation uses no
    ball pose, and does not establish physical frame calibration.
    """
    f=np.asarray(force_wrist,float);tau=np.asarray(torque_wrist,float)
    Rw=np.asarray(world_R_wrist,float);Rt=np.asarray(world_R_tray,float);r=np.asarray(world_wrist_to_tray,float)
    if f.shape!=(3,) or tau.shape!=(3,) or r.shape!=(3,) or Rw.shape!=(3,3) or Rt.shape!=(3,3):
        raise ValueError('wrench and frame sizes are invalid')
    if not all(np.isfinite(x).all() for x in (f,tau,Rw,Rt,r)):
        raise ValueError('wrench and frame must be finite')
    # tau about tray origin = tau about wrist origin - (tray-wrist) × force.
    torque_at_tray_wrist=tau-np.cross(Rw.T@r,f)
    wrist_to_tray=Rt.T@Rw
    return wrist_to_tray@f,wrist_to_tray@torque_at_tray_wrist
