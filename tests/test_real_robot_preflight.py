"""Checks for the read-only robot state diagnostic."""
import unittest

from hardware.real_robot_preflight import summarize


class PreflightSummaryTests(unittest.TestCase):
    def sample(self, robot_time, stamp, receipt, fresh=True):
        return {
            'source_robot_time_s': robot_time,
            'source_header_stamp_s': stamp,
            'receipt_monotonic_s': receipt,
            'source_is_fresh': fresh,
            'all_zero_estimated_wrench': True,
            'robot_mode': 2,
            'm_load': 0.1625,
            'm_total': 0.9,
        }

    def test_nonadvancing_source_and_unfresh_are_counted(self):
        samples = [self.sample(1., 10., 20.),
                   self.sample(1.01, 10.01, 20.02),
                   self.sample(1.01, 10.01, 20.03, fresh=False)]
        result = summarize(samples, invalid_count=1)
        self.assertEqual(result['nonadvancing_robot_time_intervals'], 1)
        self.assertEqual(result['nonadvancing_header_intervals'], 1)
        self.assertEqual(result['source_fresh_flags_false'], 1)
        self.assertEqual(result['invalid_samples'], 1)
        self.assertAlmostEqual(result['mean_robot_time_interval_s'], .01)
        self.assertNotIn('latency_s', result)

    def test_no_observations_is_not_an_available_stream(self):
        result = summarize([], invalid_count=2)
        self.assertFalse(result['stream_observed'])
        self.assertEqual(result['invalid_samples'], 2)


if __name__ == '__main__':
    unittest.main()
