"""LQR equation checks and feedback isolation, entirely offline."""
import unittest
import numpy as np
from lqr_control import lqr_gains,LQGWrenchController
from robot_scene import build_model


class LqrTests(unittest.TestCase):
    def test_scale_invariance(self):
        np.testing.assert_allclose(lqr_gains(.16,.526,1.),lqr_gains(16.,52.6,100.),rtol=1e-9)

    def test_closed_loop_is_stable(self):
        kp,kd=lqr_gains(.16,.526,1.)
        A=np.array([[0.,1.],[-9.81/1.4*kp,-9.81/1.4*kd]])
        self.assertTrue(np.all(np.real(np.linalg.eigvals(A))<0))

    def test_bad_weights_rejected(self):
        for weights in ((0,1,1),(1,-1,1),(1,1,0),(np.nan,1,1)):
            with self.assertRaises(ValueError):lqr_gains(*weights)

    def test_sensor_backends(self):
        model,data,meta=build_model()
        c=LQGWrenchController(model,data,config=meta['config'],sensor_model='joint_torque')
        self.assertEqual(c.mode,'estimated_feedback')
        self.assertEqual(c.observer.__class__.__name__,'JointTorqueWrenchObserver')
        self.assertGreater(c.lqr_gains[0],0.)


if __name__=='__main__':unittest.main()
