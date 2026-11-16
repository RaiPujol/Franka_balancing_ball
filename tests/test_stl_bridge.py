import unittest
import stl_scene
import wrench_control
import wrench_shapes
import wrench_center_demos
from stl_bridge import use_actual_stl_scene


class SceneBridgeTests(unittest.TestCase):
    def test_swaps_and_restores_even_on_error(self):
        old=(wrench_control.build_model,wrench_shapes.build_model,wrench_center_demos.build_model)
        with self.assertRaisesRegex(RuntimeError,'injected failure'):
            with use_actual_stl_scene() as build:
                self.assertIs(build,stl_scene.build_model)
                self.assertIs(wrench_control.build_model,build)
                self.assertIs(wrench_shapes.build_model,build)
                raise RuntimeError('injected failure')
        self.assertEqual((wrench_control.build_model,wrench_shapes.build_model,wrench_center_demos.build_model),old)

    def test_plant_and_unloaded_observer_match_stl(self):
        with use_actual_stl_scene() as build:
            model,data,meta=build({'ball_initial_offset_m':[.01,-.01]})
            control=wrench_control.WrenchController(model,data,config=meta['config'],sensor_model='joint_torque')
            self.assertEqual(meta['source_stl_sha256'],control.observer.meta['source_stl_sha256'])
            self.assertEqual(control.observer.meta['config']['ball_initial_offset_m'],[0.,0.])
            self.assertAlmostEqual(control.observer.meta['config']['tray_mass_kg'],.09277)

    def test_rejects_unvalidated_variants(self):
        with self.assertRaises(ValueError):
            with use_actual_stl_scene('unknown'):pass


if __name__=='__main__':unittest.main()
