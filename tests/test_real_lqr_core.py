import unittest

import numpy as np

from hardware.real_lqr_core import BallLQRCore, ControllerFault


def core():
    return BallLQRCore(max_tilt_rad=.03, max_age_s=.08, min_normal_n=.2,
                       edge_margin_m=.01)


class RealLQRCoreTests(unittest.TestCase):
    def test_center_error_and_saturation(self):
        ctl = core()
        result = ctl.update([.01, -.005], .45, sample_time_s=1., now_s=1.)
        np.testing.assert_allclose(result['slope_xy_rad'], [.006, -.003])
        ctl = core()
        result = ctl.update([.07, 0], .45, sample_time_s=1., now_s=1.)
        self.assertAlmostEqual(np.linalg.norm(result['slope_xy_rad']), .03)

    def test_feedforward_sign_matches_tray_model(self):
        ctl = core()
        result = ctl.update([0., 0.], .45, sample_time_s=1., now_s=1.,
                            target_accel_m_s2=[.1, 0.],
                            tray_accel_xy_m_s2=[0., .2])
        np.testing.assert_allclose(result['slope_xy_rad'], [-.14 / 9.81, -.2 / 9.81])

    def test_stale_and_force_faults_latch(self):
        for args in (([0., 0.], .1, 1., 1.), ([0., 0.], .45, 1., 1.1)):
            ctl = core()
            with self.assertRaises(ControllerFault):
                ctl.update(args[0], args[1], sample_time_s=args[2], now_s=args[3])
            with self.assertRaises(ControllerFault):
                ctl.update([0., 0.], .45, sample_time_s=2., now_s=2.)

    def test_duplicate_time_and_edge_fault(self):
        ctl = core()
        ctl.update([0., 0.], .45, sample_time_s=1., now_s=1.)
        with self.assertRaises(ControllerFault):
            ctl.update([0., 0.], .45, sample_time_s=1., now_s=1.01)
        ctl = core()
        with self.assertRaises(ControllerFault):
            ctl.update([.09, 0.], .45, sample_time_s=1., now_s=1.)

    def test_slope_rotation_matches_simulator_with_feedforward_and_clamp(self):
        try:
            from controller import desired_tray_rotation, rotation_vector
        except ImportError:
            self.skipTest('MuJoCo is required for the simulator parity test')

        reference = rotation_vector(np.array([.2, -.1, .3]))
        tray_accel = np.array([.1, -.2])
        target_accel = np.array([.02, -.03])
        target = np.array([.005, -.003])
        target_velocity = np.array([-.004, .006])

        for estimated_position in (np.array([.01, -.01]), np.array([.06, .04])):
            ctl = core()
            estimated_velocity = np.array([.03, -.02])

            class FixedFilter:
                def update(self, measurement, stamp, now):
                    return {'position_m': estimated_position.copy(),
                            'velocity_m_s': estimated_velocity.copy()}

            ctl.filter = FixedFilter()
            result = ctl.update(estimated_position, .45, sample_time_s=1., now_s=1.,
                                target_xy_m=target,
                                target_velocity_m_s=target_velocity,
                                target_accel_m_s2=target_accel,
                                tray_accel_xy_m_s2=tray_accel)
            slope = result['slope_xy_rad']
            actual_rotation = reference @ rotation_vector(np.array([slope[1], -slope[0], 0.]))
            expected_rotation = desired_tray_rotation(
                reference, reference @ np.r_[tray_accel, 0.], 'estimated_feedback',
                ball_xy=estimated_position-target,
                ball_vxy=estimated_velocity-target_velocity,
                kp=.6, kd=.8, max_tilt_rad=.03,
                relative_accel_xy=target_accel)
            np.testing.assert_allclose(actual_rotation, expected_rotation, atol=1e-12)
            self.assertLessEqual(np.linalg.norm(slope), .03 + 1e-12)

    def test_reject_future_observation(self):
        ctl = core()
        with self.assertRaises(ControllerFault):
            ctl.update([0., 0.], .45, sample_time_s=1., now_s=.99)


if __name__ == '__main__':
    unittest.main()
