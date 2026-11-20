import unittest

from hardware.calib_observability import summarize


class ObservabilityTests(unittest.TestCase):
    def test_holdout_recovers_affine_map_on_spanning_data(self):
        points = []
        for x, y in [(-0.04, -0.04), (-0.04, 0.04), (0.04, -0.04), (0.04, 0.04), (0, 0)]:
            points.append({'known_xy_m': [x, y],
                           'estimated_mean_xy_m': [2 * x + 0.01, 0.5 * y - 0.02],
                           'estimated_std_xy_m': [0.001, 0.001]})
        result = summarize({'status': 'offline_static_evaluation_only', 'positions': points})
        self.assertLess(result['leave_one_location_out_max_mm'], 1e-10)
        self.assertEqual(result['point_count'], 5)

    def test_collinear_positions_are_rejected(self):
        points = [{'known_xy_m': [i * .01, 0],
                   'estimated_mean_xy_m': [i * .01, 0],
                   'estimated_std_xy_m': [.001, .001]} for i in range(5)]
        with self.assertRaisesRegex(ValueError, 'collinear'):
            summarize({'status': 'offline_static_evaluation_only', 'positions': points})
