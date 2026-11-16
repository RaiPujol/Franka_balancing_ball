"""Checks for ball-state isolation, delay, and robot acceleration estimation."""
import unittest
import numpy as np
import mujoco
from robot_scene import build_model
from wrench_control import WrenchObserver


def settled_scene():
    m,d,meta=build_model({'ball_initial_offset_m':[.015,-.01]})
    q0=d.qpos[:7].copy();kp=np.array([600,600,500,500,250,180,100]);kd=np.array([50,50,40,35,20,16,12])
    for _ in range(round(.25/m.opt.timestep)):
        d.ctrl[:7]=np.clip(kp*(q0-d.qpos[:7])-kd*d.qvel[:7]+d.qfrc_bias[:7],m.actuator_ctrlrange[:7,0],m.actuator_ctrlrange[:7,1])
        mujoco.mj_step(m,d)
    mujoco.mj_forward(m,d)
    return m,d,meta


class ObserverTests(unittest.TestCase):
    def test_does_not_read_ball_state_or_live_acceleration(self):
        m,d,meta=settled_scene()
        a=WrenchObserver(force_std=0,torque_std=0,delay_s=0)
        b=WrenchObserver(force_std=0,torque_std=0,delay_s=0)
        one=a.observe(m,d)
        # Keep the sensor reading fixed but corrupt fields forbidden to the
        # controller. Its result must remain identical.
        ball=meta['ball_qpos_address'];d.qpos[ball:ball+3]=[10.,20.,30.]
        d.qacc[:]=12345.
        two=b.observe(m,d)
        self.assertIsNotNone(one);self.assertIsNotNone(two)
        np.testing.assert_allclose(one['position_m'],two['position_m'],atol=1e-12)
        np.testing.assert_allclose(one['velocity_m_s'],two['velocity_m_s'],atol=1e-12)

    def test_delay_buffer_returns_only_old_sensor_sample(self):
        m,d,_=settled_scene();o=WrenchObserver(force_std=0,torque_std=0,delay_s=.02)
        now=d.time
        self.assertIsNone(o.observe(m,d))
        result=None
        for _ in range(20):
            mujoco.mj_step(m,d);mujoco.mj_forward(m,d)
            try:result=o.observe(m,d)
            except RuntimeError:result=None
            if result is not None:break
        self.assertIsNotNone(result)
        self.assertGreaterEqual(d.time-result['sample_time'],.02-1e-9)
        self.assertAlmostEqual(result['sample_time'],now,places=8)

    def test_invalid_noise_settings_are_rejected(self):
        with self.assertRaises(ValueError):WrenchObserver(joint_velocity_std=-.001)
        with self.assertRaises(ValueError):WrenchObserver(delay_s=-.001)


if __name__=='__main__':unittest.main()
