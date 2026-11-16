"""MPC numeric checks; these do not certify physical safety."""
import unittest
from unittest.mock import patch
import numpy as np
from mpc_control import MPCParameters,BallSlopeMPC


class MPCUnitTests(unittest.TestCase):
    def test_returns_bounded_restoring_command(self):
        x=BallSlopeMPC();ref=np.zeros((x.params.horizon,2))
        u=x.solve([.01,0],[0,0],[0,0],ref,ref)
        self.assertGreater(u[0],0)
        self.assertLessEqual(np.linalg.norm(u),x.params.max_tilt_rad+1e-9)
        self.assertTrue(np.isfinite(u).all())
        self.assertTrue(x.solve_times and x.solve_times[-1]>0)

    def test_invalid_parameters_rejected(self):
        for kw in ({'ball_radius':0},{'ball_radius':float('nan')},{'half_extent':(.01,.01)}):
            with self.assertRaises(ValueError): BallSlopeMPC(**kw)
        for p in (MPCParameters(horizon=2.5),MPCParameters(max_iterations=1.2),MPCParameters(predictive_blend=1.1)):
            with self.assertRaises(ValueError):BallSlopeMPC(p)

    def test_optimizer_failure_propagates(self):
        from types import SimpleNamespace
        f=SimpleNamespace(success=False,x=np.zeros(14),message='synthetic failure')
        p=BallSlopeMPC();z=np.zeros((14,2))
        with patch('mpc_control.minimize',return_value=f):
            with self.assertRaisesRegex(RuntimeError,'optimizer did not converge'):
                p.solve([0,0],[0,0],[0,0],z,z)

    def test_prediction_models_frozen_pd_blend(self):
        p=BallSlopeMPC(); n=p.params.horizon
        state=np.array([.01,-.02,.003]);dist=np.linspace(-.1,.1,n)
        u=np.linspace(-.03,.04,n); base,D,vbase,V=p._matrices(state,dist,.02,.1)
        model=state.copy();poses=[];velocities=[]
        for k in range(n):
            applied=.9*.02+.1*u[k]
            model=p.A@model+p.E*dist[k]+p.B*applied
            poses.append(model[0]);velocities.append(model[1])
        np.testing.assert_allclose(base+D@u,poses,atol=1e-12)
        np.testing.assert_allclose(vbase+V@u,velocities,atol=1e-12)

    def test_soft_edge_is_not_a_safety_guarantee(self):
        p=BallSlopeMPC();z=np.zeros((14,2));u=p.solve([.055,0],[.5,0],[0,0],z,z)
        # Even for an optimizer solution, an outward-moving ball already near
        # the edge has a predicted first-step position beyond the soft margin.
        base,D,_,_=p._matrices(np.array([.055,.5,0.]),np.zeros(14))
        pred=float(base[0]+D[0]@p.last_sequence[0])
        self.assertGreater(pred,p.half_extent[0]-p.ball_radius-p.params.edge_margin_m)

if __name__=='__main__':unittest.main()
