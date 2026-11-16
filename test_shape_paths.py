import unittest
import numpy as np
from shape_paths import ShapePath


class ShapePathTests(unittest.TestCase):
    def test_periodic_and_shapes_distinct(self):
        curves=[]
        for name in ('circle','rounded_square','figure_eight'):
            path=ShapePath(shape=name)
            self.assertTrue(np.allclose(path.curve(0),path.curve(2*np.pi),atol=1e-12))
            curves.append(path.curve(np.pi/4))
        self.assertFalse(np.allclose(curves[0],curves[1]))
        self.assertFalse(np.allclose(curves[0],curves[2]))

    def test_position_and_velocity_derivatives(self):
        for name in ('circle','rounded_square','figure_eight'):
            path=ShapePath(shape=name)
            for t in (3.,7.1,13.4,22.):
                eps=1e-4
                p,v,a=path.sample(t)
                pplus,vplus,_=path.sample(t+eps)
                pminus,vminus,_=path.sample(t-eps)
                np.testing.assert_allclose((pplus-pminus)/(2*eps),v,atol=2e-7)
                np.testing.assert_allclose((vplus-vminus)/(2*eps),a,atol=2e-6)

    def test_starts_and_ends_at_center(self):
        for name in ('circle','rounded_square','figure_eight'):
            path=ShapePath(shape=name)
            for time in (0.,path.end_time+1.):
                for item in path.sample(time):self.assertTrue(np.all(item==0))


if __name__=='__main__':unittest.main()
