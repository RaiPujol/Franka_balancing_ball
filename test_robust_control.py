import unittest
import numpy as np
from robust_control import DelayedPositionKalman


class DelayedFilterTests(unittest.TestCase):
    def test_delay_prediction_does_not_modify_posterior(self):
        f=DelayedPositionKalman(measurement_std=.001,accel_std=.1)
        f.update([0,0],0.,0.)
        for i in range(1,51):
            t=i*.01;f.update([.02*t,0],t,t)
        first=f.update([.01,0],.5,.55)
        second=f.update([.01,0],.5,.55)
        self.assertTrue(np.allclose(first['position_m'],second['position_m']))
        self.assertGreater(first['position_m'][0],.01)
        self.assertLess(abs(first['velocity_m_s'][0]-.02),.01)

    def test_stale_and_invalid_observations_fail_closed(self):
        f=DelayedPositionKalman(max_prediction=.1)
        with self.assertRaises(RuntimeError): f.update([0,0],0.,.11)
        with self.assertRaises(ValueError): f.update([np.nan,0],0.,0.)
        f.update([0,0],1.,1.)
        with self.assertRaises(RuntimeError):f.update([0,0],.9,1.)
        with self.assertRaisesRegex(RuntimeError,'gap'):f.update([0,0],1.2,1.2)

    def test_covariance_remains_symmetric_positive(self):
        f=DelayedPositionKalman()
        for i in range(300):f.update([.01*np.sin(i/50),.01*np.cos(i/50)],i*.002,i*.002+.02)
        np.testing.assert_allclose(f.P,f.P.T,atol=1e-12)
        self.assertGreaterEqual(np.linalg.eigvalsh(f.P).min(),0.)

    def test_prediction_uncertainty_includes_process_noise(self):
        quiet=DelayedPositionKalman(accel_std=.01)
        uncertain=DelayedPositionKalman(accel_std=2.)
        a=quiet.update([0.,0.],0.,.1)
        b=uncertain.update([0.,0.],0.,.1)
        self.assertTrue(np.all(b['position_std_m']>a['position_std_m']))


if __name__=='__main__':unittest.main()
