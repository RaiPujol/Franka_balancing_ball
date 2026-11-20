import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from hardware.torque_baseline import summarize


def recording(path, *, moving=False, drift=False, invalid=False, load_change=False):
    rows = [{'kind': 'metadata', 'schema': 'franka_wrench_observation_v1',
             'label': 'empty'}]
    for i in range(30):
        q = [0.] * 7
        if drift:
            q[1] = i * .001
        dq = [0.] * 7
        if moving and i == 10:
            dq[1] = .02
        tau = [1., 2., 3., 4., 5. + i * .001, 6., 7.]
        sample = {'kind': 'sample', 'candidate_for_offline_analysis': not (invalid and i == 0),
                  'source_robot_time_s': 1. + i * .01, 'q': q, 'dq': dq, 'tau_J': tau,
                  'm_ee': .73, 'm_load': .1625 if load_change and i == 10 else 0., 'm_total': .73,
                  'F_T_EE': np.eye(4).flatten(order='F').tolist(),
                  'EE_T_K': np.eye(4).flatten(order='F').tolist()}
        for name in ('F_x_Cee', 'F_x_Cload', 'F_x_Ctotal'):
            sample[name] = [0.] * 3
        for name in ('I_ee', 'I_load', 'I_total'):
            sample[name] = [0.] * 9
        rows.append(sample)
    rows.append({'kind': 'end', 'errors': []})
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


class TorqueBaselineTests(unittest.TestCase):
    def test_stationary_baseline_and_invalid_sample_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'baseline.jsonl'
            recording(path, invalid=True)
            result = summarize(path)
            self.assertEqual(result['accepted_samples'], 29)
            self.assertEqual(result['rejected_counts']['invalid_or_stale'], 1)
            self.assertEqual(result['joint_reference_median_rad'], [0.] * 7)
            self.assertEqual(result['configured_tool_and_load']['m_load'], 0.)
            self.assertEqual(len(result['configured_tool_and_load']['F_T_EE']), 16)
            np.testing.assert_allclose(result['joint_5_6_median_nm'], [5.015, 6.])
            self.assertGreater(result['joint_5_6_std_nm'][0], 0)
            self.assertAlmostEqual(result['joint_5_6_std_nm'][1], 0)

    def test_recording_motion_and_cumulative_joint_drift_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'baseline.jsonl'
            recording(path, moving=True)
            with self.assertRaisesRegex(ValueError, 'moving'):
                summarize(path)
            recording(path, drift=True)
            with self.assertRaisesRegex(ValueError, 'position range'):
                summarize(path)

    def test_changing_configured_load_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'baseline.jsonl'
            recording(path, load_change=True)
            with self.assertRaisesRegex(ValueError, 'tool/load or stiffness frame changed: m_load'):
                summarize(path)

    def test_changed_stiffness_frame_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'baseline.jsonl'
            recording(path)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[11]['EE_T_K'][12] = .001
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            with self.assertRaisesRegex(ValueError, 'stiffness frame changed: EE_T_K'):
                summarize(path)


if __name__ == '__main__':
    unittest.main()
