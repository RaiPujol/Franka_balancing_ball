"""Isolation and singularity gating for synthetic joint-torque observer."""
import unittest
import numpy as np
from test_wrench_control import settled_scene
from joint_wrench_proxy import JointTorqueWrenchObserver
from wrench_control import WrenchController


class JointTorqueObserverTests(unittest.TestCase):
    def test_does_not_read_live_ball_or_acceleration(self):
        m,d,meta=settled_scene()
        a=JointTorqueWrenchObserver(joint_torque_std=0,delay_s=0)
        b=JointTorqueWrenchObserver(joint_torque_std=0,delay_s=0)
        one=a.observe(m,d)
        adr=meta['ball_qpos_address'];d.qpos[adr:adr+3]=[10.,20.,30.];d.qacc[:]=12345.
        two=b.observe(m,d)
        np.testing.assert_allclose(one['position_m'],two['position_m'],atol=1e-12)
        np.testing.assert_allclose(one['velocity_m_s'],two['velocity_m_s'],atol=1e-12)

    def test_singularity_gate_fails_closed(self):
        m,d,_=settled_scene()
        observer=JointTorqueWrenchObserver(joint_torque_std=0,delay_s=0,min_singular_value=100.)
        with self.assertRaisesRegex(RuntimeError,'near singularity'):
            observer.observe(m,d)

    def test_controller_selects_explicit_sensor_model(self):
        m,d,meta=settled_scene()
        ctrl=WrenchController(m,d,meta['config'],sensor_model='joint_torque')
        self.assertIsInstance(ctrl.observer,JointTorqueWrenchObserver)
        self.assertEqual(ctrl.mode,'estimated_feedback')


if __name__=='__main__':unittest.main()
