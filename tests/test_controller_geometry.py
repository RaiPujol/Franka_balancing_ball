import tempfile
import unittest

from stl_bridge import use_actual_stl_scene
from stl_scene import DEFAULTS
from wrench_center_demos import run_one
from wrench_shapes import run


class ControllerGeometryTests(unittest.TestCase):
    def test_runners_expose_tray_dimensions_and_withhold_ball_start(self):
        class ConfigurationCaptured(Exception):
            pass

        def check_controller_config(model, data, config, **kwargs):
            # The live collision deck gives an independent expected size.
            deck = model.geom('tray_plate')
            self.assertAlmostEqual(config['tray_length_m'], 2 * deck.size[0])
            self.assertAlmostEqual(config['tray_width_m'], 2 * deck.size[1])
            self.assertAlmostEqual(config['tray_mass_kg'], DEFAULTS['tray_mass_kg'])
            self.assertNotIn('ball_initial_offset_m', config)
            raise ConfigurationCaptured

        with use_actual_stl_scene(), tempfile.TemporaryDirectory() as output:
            with self.assertRaises(ConfigurationCaptured):
                run('circle', config={'ball_initial_offset_m': [.01, -.015]},
                    output_dir=output, controller_factory=check_controller_config)
            with self.assertRaises(ConfigurationCaptured):
                run_one((.01, -.015), 'carrying', controller_factory=check_controller_config)

    def test_rejected_first_wrench_records_zero_samples(self):
        class RejectFirstWrench:
            def __init__(self, *args, **kwargs):
                pass

            def prime(self):
                pass

            def step(self, t):
                raise RuntimeError('wrench feedback unavailable: insufficient_normal_force')

        with use_actual_stl_scene(), tempfile.TemporaryDirectory() as output:
            result = run('circle', output_dir=output, controller_factory=RejectFirstWrench)

        self.assertFalse(result['numerical_pass'])
        self.assertEqual(result['duration_completed_s'], 0.)
        self.assertIsNone(result['estimation_error_rms_m'])
        self.assertIsNone(result['minimum_full_footprint_margin_m'])
        self.assertIn('insufficient_normal_force', result['stop_reason'])
