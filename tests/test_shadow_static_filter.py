"""Offline static shadow checks; imports no ROS and never controls equipment."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from hardware.shadow_static_filter import (HardwareExecutionLocked, dispatch_to_robot,
                                           filter_samples, shadow_replay)


def record(path, values):
    rows=[{'kind':'metadata','schema':'franka_wrench_observation_v1','label':'offline test'}]
    rows.extend(values)
    rows.append({'kind':'end','errors':[]})
    Path(path).write_text(''.join(json.dumps(v)+'\n' for v in rows))


def sample(t, with_ball):
    T=np.eye(4).flatten(order='F').tolist()
    # fz=-0.45; tau_y=+0.009 and tau_x=+0.0045 means x=.02,y=-.01.
    wrench=[0,0,-.45,.0045,.009,0] if with_ball else [0.]*6
    r={'kind':'sample','source_robot_time_s':t,'source_header_stamp_s':100+t,
       'candidate_for_offline_analysis':True,'dq':[0.]*7,
       'O_T_EE':T,'EE_T_K':T,'F_T_EE':T,'O_F_ext_hat_K':wrench,
       'm_ee':1.,'m_load':.2,'m_total':1.2}
    for k in ('F_x_Cee','F_x_Cload','F_x_Ctotal'):r[k]=[0.]*3
    for k in ('I_ee','I_load','I_total'):r[k]=[0.]*9
    return r


class ShadowFilterTests(unittest.TestCase):
    def test_execution_is_never_available(self):
        with self.assertRaises(HardwareExecutionLocked):
            dispatch_to_robot('any', approved=True, force=True)

    def test_bad_time_invalid_sample_and_gap_reset_the_filter(self):
        def p(t,x=[.02,-.01],valid=True):
            return {'source_robot_time_s':t,'valid_estimate':valid,'estimated_ball_xy_m':x}
        rows=filter_samples([p(1.),p(1.01),p(1.01),p(1.02),p(1.5),p(1.51),
                             p(1.52,valid=False),p(1.53)])
        self.assertEqual([v['valid'] for v in rows],[True,True,False,True,False,True,False,True])
        self.assertEqual([v['segment'] for v in rows],[1,1,None,2,None,3,None,4])
        self.assertEqual(rows[2]['reason'],'nonadvancing_robot_time')
        self.assertEqual(rows[4]['reason'],'sample_gap_exceeded')
        self.assertEqual(rows[6]['reason'],'invalid_wrench_position')

    def test_complete_recording_uses_only_recorded_wrench(self):
        with tempfile.TemporaryDirectory() as folder:
            empty=Path(folder)/'empty.jsonl';loaded=Path(folder)/'loaded.jsonl'
            record(empty,[sample(1+i*.01,False) for i in range(25)])
            record(loaded,[sample(2+i*.01,True) for i in range(25)])
            result=shadow_replay(empty,loaded,np.eye(4))
            self.assertFalse(result['hardware_execution_allowed'])
            self.assertEqual(result['motion_commands_issued'],0)
            self.assertEqual(result['valid_samples'],25)
            np.testing.assert_allclose(result['samples'][-1]['estimate_xy_m'],[.02,-.01],atol=1e-12)

    def test_rejected_source_samples_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            empty=Path(folder)/'empty.jsonl';loaded=Path(folder)/'loaded.jsonl'
            baseline=[sample(1+i*.01,False) for i in range(25)]
            points=[sample(2+i*.01,True) for i in range(26)]
            points[2]['candidate_for_offline_analysis']=False
            record(empty,baseline);record(loaded,points)
            with self.assertRaisesRegex(ValueError,'rejected or stale'):
                shadow_replay(empty,loaded,np.eye(4))


if __name__=='__main__':
    unittest.main()
