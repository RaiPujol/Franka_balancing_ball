"""Offline sensor tests; ideal wrench simulation is not hardware validation."""
import sys, unittest
from pathlib import Path
import numpy as np
import mujoco
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from robot_scene import build_model
from force_estimator import center_of_pressure,IdealWristWrenchEstimator

class WrenchTests(unittest.TestCase):
    def test_algebra_signs_and_gate(self):
        # F=(0,0,.45) through (x,y)=(.03,-.02): T=r cross F.
        f=np.array([0.,0.,.45]);r=np.array([.03,-.02,0.]);torque=np.cross(r,f)
        est=center_of_pressure(f,torque,force_std_n=.01,torque_std_nm=.002)
        np.testing.assert_allclose(est['position_m'],r[:2],atol=1e-12)
        self.assertGreater(est['std_m'][0],.002/.45)
        self.assertFalse(center_of_pressure([0,0,.1],[0,0,0])['valid'])
        self.assertFalse(center_of_pressure([0,0,-.45],[0,0,0])['valid'])
    def test_offcenter_ball_recovered_from_wrist_wrench(self):
        for xy in ([.03,-.02],[-.025,.035],[0,0]):
            cfg={'ball_initial_offset_m':xy,'tray_length_m':.2,'tray_width_m':.2}
            m,d,meta=build_model(cfg);sensor=IdealWristWrenchEstimator(cfg)
            q=d.qpos[:7].copy();kp=np.array([600,600,500,500,250,180,100]);kd=np.array([50,50,40,35,20,16,12])
            for _ in range(300):
                d.ctrl[:7]=np.clip(kp*(q-d.qpos[:7])-kd*d.qvel[:7]+d.qfrc_bias[:7],m.actuator_ctrlrange[:7,0],m.actuator_ctrlrange[:7,1]);mujoco.mj_step(m,d)
            mujoco.mj_forward(m,d);out=sensor.observe(m,d)
            self.assertTrue(out['valid'])
            site=meta['tray_site_id'];truth=d.site_xmat[site].reshape(3,3).T@(d.xpos[meta['ball_body_id']]-d.site_xpos[site])
            np.testing.assert_allclose(out['position_m'],truth[:2],atol=.00015)
    def test_no_ball_contact_gates_estimate(self):
        m,d,meta=build_model();sensor=IdealWristWrenchEstimator()
        adr=meta['ball_qpos_address'];d.qpos[adr:adr+3]=[0,0,2]
        mujoco.mj_forward(m,d);out=sensor.observe(m,d)
        self.assertFalse(out['valid']);self.assertEqual(out['reason'],'insufficient_normal_force')
    def test_mismatched_model_rejected(self):
        m,d,_=build_model({'tray_width_m':.2});sensor=IdealWristWrenchEstimator()
        mujoco.mj_forward(m,d)
        with self.assertRaises(ValueError):sensor.observe(m,d)

if __name__=='__main__':unittest.main()
