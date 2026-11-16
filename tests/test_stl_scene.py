"""Geometry and contact checks for the provisional STL simulation scene."""
import unittest
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial import cKDTree
import stl_scene


class ActualPlatformSceneTests(unittest.TestCase):
    def test_measured_mass_dimensions_and_free_ball(self):
        m,d,meta=stl_scene.build_model()
        self.assertAlmostEqual(m.body_mass[meta['tray_body_id']],.09277,places=10)
        np.testing.assert_allclose(m.geom_size[meta['tray_geom_id']],[.075,.075,.0005])
        self.assertEqual(int(m.jnt_type[m.joint('ball_joint').id]),int(mujoco.mjtJoint.mjJNT_FREE))
        R=d.site_xmat[meta['tray_site_id']].reshape(3,3)
        self.assertGreater(R[2,2],.99999)
        np.testing.assert_allclose(R.T@(d.xpos[meta['ball_body_id']]-d.site_xpos[meta['tray_site_id']]),[0.,0.,.04267/2+.0002],atol=1e-10)
        self.assertEqual(d.ncon,0,'nominal ball/robot should start without contact')

    def test_imported_mesh_alignment_after_mujoco_recentering(self):
        m,d,meta=stl_scene.build_model();mid=m.mesh('actual_platform_visual').id;gid=m.geom('platform_visual').id
        a,n=int(m.mesh_vertadr[mid]),int(m.mesh_vertnum[mid]);v=m.mesh_vert[a:a+n]
        mesh_world=v@d.geom_xmat[gid].reshape(3,3).T+d.geom_xpos[gid]
        raw=Path(stl_scene.LOCAL_MESH).read_bytes()
        rec=np.frombuffer(raw[84:],dtype=[('normal','<f4',(3,)),('v','<f4',(3,3)),('attr','<u2')])
        local=np.unique(rec['v'].reshape(-1,3),axis=0)*meta['config']['platform_scale_m_per_cad_unit']
        body_world=local@d.xmat[meta['tray_body_id']].reshape(3,3).T+d.xpos[meta['tray_body_id']]
        self.assertLess(float(cKDTree(mesh_world).query(body_world)[0].max()),3e-8)

    def test_tab_proxies_reach_both_modeled_fingers(self):
        m,d,_=stl_scene.build_model();tab=m.geom('tab_collision_proxy').id
        for name in ('left_finger','right_finger'):
            geoms=[i for i in range(m.ngeom) if m.body(int(m.geom_bodyid[i])).name==name and m.geom_contype[i]]
            nearest=min(mujoco.mj_geomDistance(m,d,tab,i,1.,np.zeros(6)) for i in geoms)
            self.assertLessEqual(nearest,.0001)

    def test_stationary_control_keeps_free_ball_supported_briefly(self):
        m,d,meta=stl_scene.build_model();q0=d.qpos[meta['arm_qpos_ids']].copy();kp=np.array([600,600,500,500,250,180,100]);kd=np.array([50,50,40,35,20,16,12]);n=0
        for _ in range(500):
            d.ctrl[:7]=np.clip(kp*(q0-d.qpos[:7])-kd*d.qvel[:7]+d.qfrc_bias[:7],m.actuator_ctrlrange[:7,0],m.actuator_ctrlrange[:7,1]);mujoco.mj_step(m,d)
            n+=int(any({int(c.geom1),int(c.geom2)}=={meta['tray_geom_id'],meta['ball_geom_id']} for c in d.contact[:d.ncon]))
        self.assertGreater(n,450)
        self.assertFalse(any(x.number for x in d.warning))

    def test_bolt_mass_is_optional_and_separate(self):
        m,_,_=stl_scene.build_model({'bolt_mass_kg':.005});self.assertAlmostEqual(m.body_mass[m.body('platform_tab').id],.005,places=10)


if __name__=='__main__':unittest.main()
