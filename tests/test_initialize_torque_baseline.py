import json
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
import tempfile

from hardware.initialize_torque_baseline import initialize


class InitializeTorqueBaselineTests(TestCase):
    def test_recording_and_report_are_created_separately_for_each_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)

            def recorder(command, **kwargs):
                self.assertIn('hardware.record_wrench_readonly', command)
                path = Path(command[command.index('--output') + 1])
                path.write_text('fake complete recording')
                return SimpleNamespace(returncode=0)

            report = {'joint_5_6_median_nm': [1., 2.], 'joint_5_6_std_nm': [.1, .2]}
            with patch('hardware.initialize_torque_baseline.subprocess.run', side_effect=recorder), \
                 patch('hardware.initialize_torque_baseline.summarize', return_value=report):
                first = initialize('/state', directory)
                second = initialize('/state', directory)
            self.assertNotEqual(first[0], second[0])
            self.assertEqual(json.loads(first[1].read_text()), report)
            self.assertTrue(first[0].exists())
            self.assertTrue(second[1].exists())

    def test_failed_recording_never_creates_baseline_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch('hardware.initialize_torque_baseline.subprocess.run',
                       return_value=SimpleNamespace(returncode=1)), \
                 patch('hardware.initialize_torque_baseline.summarize') as analyze:
                with self.assertRaisesRegex(RuntimeError, 'recording failed'):
                    initialize('/state', tmp)
            analyze.assert_not_called()
            self.assertEqual(list(Path(tmp).glob('*_baseline.json')), [])

    def test_invalid_analysis_retains_raw_without_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            def recorder(command, **kwargs):
                Path(command[command.index('--output') + 1]).write_text('partial')
                return SimpleNamespace(returncode=0)

            with patch('hardware.initialize_torque_baseline.subprocess.run', side_effect=recorder), \
                 patch('hardware.initialize_torque_baseline.summarize', side_effect=ValueError('motion')):
                with self.assertRaisesRegex(RuntimeError, 'analysis failed'):
                    initialize('/state', tmp)
            self.assertEqual(len(list(Path(tmp).glob('*.jsonl'))), 1)
            self.assertEqual(list(Path(tmp).glob('*_baseline.json')), [])
