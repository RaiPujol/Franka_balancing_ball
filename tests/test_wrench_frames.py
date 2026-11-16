"""Wrench and contact-point frames must follow rotated CAD tray axes."""
import unittest
import numpy as np
import mujoco
from wrench_frames import wrench_at_tray
from force_estimator import center_of_pressure
from stl_bridge import use_actual_stl_scene
from joint_wrench_proxy import JointTorqueWrenchObserver


def rz(a):
    c,s=np.cos(a),np.sin(a);return np.array([[c,-s,0],[s,c,0],[0,0,1.]])


class WrenchFrameTests(unittest.TestCase):
    def test_shift_and_rotate_known_contact(self):
        Rw=rz(.31);Rt=rz(.31+np.pi/2);delta=np.array([.04,-.02,.11])
        ftray=np.array([.1,-.08,.45]);point=np.array([.020,-.015,0.]);ttray=np.cross(point,ftray)
        fw=Rw.T@Rt@ftray;tw=Rw.T@Rt@ttray+np.cross(Rw.T@delta,fw)
        f2,t2=wrench_at_tray(fw,tw,Rw,Rt,delta)
        np.testing.assert_allclose(f2,ftray,atol=1e-12);np.testing.assert_allclose(t2,ttray,atol=1e-12)
        np.testing.assert_allclose(center_of_pressure(f2,t2)['position_m'],point[:2],atol=1e-12)

    def test_sensor_bias_shifts_and_rotates_with_the_wrench(self):
        Rw=rz(.4);Rt=rz(-.9);delta=np.array([.06,-.03,.12])
        f=np.array([.1,.2,.45]);t=np.array([.02,-.01,.03])
        fb=np.array([.01,-.02,.03]);tb=np.array([.002,-.001,.0005])
        a,b=wrench_at_tray(f,t,Rw,Rt,delta)
        c,d=wrench_at_tray(f+fb,t+tb,Rw,Rt,delta)
        np.testing.assert_allclose(c-a,Rt.T@Rw@fb,atol=1e-12)
        expected=Rt.T@Rw@(tb-np.cross(Rw.T@delta,fb))
        np.testing.assert_allclose(d-b,expected,atol=1e-12)

    def test_joint_wrench_estimates_in_actual_rotated_stl_axes(self):
        with use_actual_stl_scene() as build:
            m,d,meta=build({'ball_initial_offset_m':[.02,-.015]})
            q0=d.qpos[:7].copy();kp=np.array([600,600,500,500,250,180,100]);kd=np.array([50,50,40,35,20,16,12])
            for _ in range(250):
                d.ctrl[:7]=np.clip(kp*(q0-d.qpos[:7])-kd*d.qvel[:7]+d.qfrc_bias[:7],m.actuator_ctrlrange[:7,0],m.actuator_ctrlrange[:7,1]);mujoco.mj_step(m,d)
            mujoco.mj_forward(m,d)
            obs=JointTorqueWrenchObserver(meta['config'],joint_torque_std=0.,joint_velocity_std=0.,delay_s=0.)
            result=obs.observe(m,d)
            sid=meta['tray_site_id'];Rt=d.site_xmat[sid].reshape(3,3)
            actual=Rt.T@(d.xpos[meta['ball_body_id']]-d.site_xpos[sid])
            # Truth is used only by this independent offline test to score the estimate.
            np.testing.assert_allclose(result['position_m'],actual[:2],atol=.001)


if __name__=='__main__':unittest.main()
