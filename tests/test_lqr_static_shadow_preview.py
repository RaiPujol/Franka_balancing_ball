"""Offline preview tests. No ROS, robot connection, or motion dispatch."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

from hardware.lqr_static_shadow_preview import (static_center_preview,run_preview,
    MAX_TILT_RAD,dispatch_to_robot,HardwareExecutionLocked)


def sample(t,loaded):
    T=np.eye(4).flatten(order='F').tolist()
    # Known x=20mm, y=-10mm for static fz=-.45 N.
    w=[0.,0.,-.45,.0045,.009,0.] if loaded else [0.]*6
    result={'kind':'sample','source_robot_time_s':t,'source_header_stamp_s':100+t,
            'candidate_for_offline_analysis':True,'dq':[0.]*7,'O_T_EE':T,'F_T_EE':T,
            'EE_T_K':T,'O_F_ext_hat_K':w,'m_ee':1.,'m_load':.2,'m_total':1.2}
    for key in ('F_x_Cee','F_x_Cload','F_x_Ctotal'):result[key]=[0.]*3
    for key in ('I_ee','I_load','I_total'):result[key]=[0.]*9
    return result


def write_record(path,rows):
    vals=[{'kind':'metadata','schema':'franka_wrench_observation_v1','label':'synthetic'}]+rows+[{'kind':'end','errors':[]}]
    Path(path).write_text(''.join(json.dumps(x)+'\n' for x in vals))


class StaticPreviewTests(unittest.TestCase):
    def test_known_feedback_and_tilt_bound(self):
        entries=[{'source_robot_time_s':1.,'valid':True,'estimate_xy_m':[.02,-.01],
                  'estimated_velocity_m_s':[.01,-.02]},
                 {'source_robot_time_s':1.01,'valid':True,'estimate_xy_m':[1.,1.],
                  'estimated_velocity_m_s':[0.,0.]}]
        out=static_center_preview(entries)
        np.testing.assert_allclose(out[0]['preview_slope_xy_rad'],[.02,-.022],atol=1e-12)
        self.assertAlmostEqual(np.linalg.norm(out[1]['preview_slope_xy_rad']),MAX_TILT_RAD)
        self.assertIn('NOT A ROBOT COMMAND',out[0]['reason'])

    def test_invalid_stale_and_nonfinite_do_not_output_slope(self):
        def row(t,valid=True,pos=(.01,.01),vel=(0.,0.)):
            return {'source_robot_time_s':t,'valid':valid,'estimate_xy_m':pos,'estimated_velocity_m_s':vel}
        rows=static_center_preview([row(1.),row(1.),row(1.4),row(1.41,False),row(1.42,pos=(float('nan'),0))])
        self.assertEqual([x['preview_valid'] for x in rows],[True,False,False,False,False])
        self.assertEqual([x['reason'] for x in rows[1:]],['nonadvancing_robot_time','sample_gap_exceeded','invalid_observation','nonfinite_or_malformed_state'])
        self.assertTrue(all(x['preview_slope_xy_rad'] is None for x in rows[1:]))

    def test_recorded_wrench_pipeline_and_execution_lock(self):
        with tempfile.TemporaryDirectory() as folder:
            empty=Path(folder)/'empty.jsonl';loaded=Path(folder)/'loaded.jsonl'
            write_record(empty,[sample(1+i*.01,False) for i in range(25)])
            write_record(loaded,[sample(2+i*.01,True) for i in range(25)])
            result=run_preview(empty,loaded,np.eye(4))
            self.assertFalse(result['hardware_execution_allowed'])
            self.assertEqual(result['motion_commands_issued'],0)
            self.assertEqual(result['valid_previews'],25)
            np.testing.assert_allclose(result['samples'][-1]['preview_slope_xy_rad'],[.012,-.006],atol=1e-12)
            with self.assertRaises(HardwareExecutionLocked):dispatch_to_robot(True,approved=True)

    def test_module_import_requires_neither_ros_nor_mujoco(self):
        code=('import sys; import hardware.lqr_static_shadow_preview; '
              'assert "rospy" not in sys.modules; assert "mujoco" not in sys.modules')
        subprocess.run([sys.executable,'-c',code],check=True)


if __name__=='__main__':unittest.main()
