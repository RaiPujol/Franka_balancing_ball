"""Offline checks: no ROS import and no robot connection."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np
from hardware.record_wrench_readonly import parse_message,FIELDS
from hardware.analyze_static_wrench import run_analysis,wrench_at_tray,matrix
from hardware.replay_static_wrench import replay


def rotz(theta):
    c,s=np.cos(theta),np.sin(theta)
    return np.array([[c,-s,0],[s,c,0],[0,0,1.]])


def pose(R,p):
    T=np.eye(4);T[:3,:3]=R;T[:3,3]=p;return T


def sample_at(offset=None, robot_time=1.):
    O_E=pose(rotz(.5),[.35,-.12,.5]);E_K=pose(rotz(-.22),[.02,.01,.03])
    E_T=pose(rotz(.3),[.04,-.01,.13]);O_T=O_E@E_T;O_K=O_E@E_K
    base=np.array([.3,-.1,1.2,.03,.05,-.02])
    if offset is not None:
        x,y=offset;fz=-.45;base+=np.array([0,0,fz,y*fz,-x*fz,0])
    force=O_T[:3,:3]@base[:3];mt=O_T[:3,:3]@base[3:]
    mk=mt-np.cross(O_K[:3,3]-O_T[:3,3],force)
    out={'kind':'sample','source_robot_time_s':robot_time,'source_header_stamp_s':100+robot_time,
      'candidate_for_offline_analysis':True,'dq':[0]*7,
      'O_T_EE':O_E.flatten(order='F').tolist(),'EE_T_K':E_K.flatten(order='F').tolist(),
      'F_T_EE':np.eye(4).flatten(order='F').tolist(), 'O_F_ext_hat_K':np.r_[force,mk].tolist(),
      'm_ee':1.,'m_load':.2,'m_total':1.2}
    for key in ('F_x_Cee','F_x_Cload','F_x_Ctotal'):out[key]=[0,0,0]
    for key in ('I_ee','I_load','I_total'):out[key]=[0]*9
    return out,E_T,base


def write_rec(path, samples, errors=None):
    rows=[{'kind':'metadata','schema':'franka_wrench_observation_v1','label':'synthetic'}]
    rows+=samples;rows+=[{'kind':'end','errors':errors or []}]
    Path(path).write_text(''.join(json.dumps(row)+'\n' for row in rows))


class ReadOnlyTests(unittest.TestCase):
    def test_parse_requires_fields_and_rejects_invalid(self):
        msg=SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(to_sec=lambda:100.),seq=10,frame_id=''),
                            robot_time=5.,is_fresh=True,robot_mode=2,
                            m_ee=1.,m_load=.2,m_total=1.2,gripper_width=.06,control_command_success_rate=1.)
        for name,n in FIELDS.items():setattr(msg,name,[.1]*n)
        r=parse_message(msg,1.,2.)
        self.assertEqual(len(r['O_F_ext_hat_K']),6)
        self.assertEqual(r['source_robot_time_s'],5.)
        msg.O_F_ext_hat_K=[float('nan')]*6
        with self.assertRaises(ValueError):parse_message(msg,1.,2.)

    def test_nonidentity_transform_and_signed_wrench(self):
        row,T,expected=sample_at((.023,-.013))
        actual,_=wrench_at_tray(row,T)
        np.testing.assert_allclose(actual,expected,atol=1e-12)
        with self.assertRaises(ValueError):matrix(np.zeros((4,4)))

    def test_static_estimate_replay_and_setup_change_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty=Path(tmp)/'empty.jsonl';loaded=Path(tmp)/'loaded.jsonl'
            baseline=[sample_at(robot_time=1+i*.01)[0] for i in range(25)]
            values=[sample_at((.023,-.013),robot_time=1+i*.01)[0] for i in range(25)]
            T=sample_at()[1];write_rec(empty,baseline);write_rec(loaded,values)
            r=run_analysis(empty,[(.023,-.013,loaded)],T)
            self.assertLess(r['overall_sample_weighted_error_rms_m'],1e-12)
            p=replay(empty,loaded,T)
            self.assertEqual(p['valid_estimates'],25)
            np.testing.assert_allclose(p['samples'][0]['estimated_ball_xy_m'],[.023,-.013],atol=1e-12)
            values[2]['m_load']+=.1;write_rec(loaded,values)
            with self.assertRaisesRegex(ValueError,'tool/load'):run_analysis(empty,[(.023,-.013,loaded)],T)
            with self.assertRaisesRegex(ValueError,'tool/load'):replay(empty,loaded,T)

    def test_rejects_incomplete_or_moving_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'rec.jsonl';rows=[sample_at(robot_time=1+i*.01)[0] for i in range(25)]
            rows[0]['dq'][0]=.2
            write_rec(p,rows,errors=['stream went silent'])
            with self.assertRaisesRegex(ValueError,'recorder reported errors'):replay(p,p,sample_at()[1])


if __name__=='__main__':unittest.main()
