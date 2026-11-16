"""NumPy-only fixed-lag 2-D filter, reusable for offline measurement replay.

Timestamps must share a validated monotonic clock. A message header alone does
not establish acquisition time. This filter does not command hardware.
"""
from __future__ import annotations
import numpy as np


class DelayedPositionKalman:
    """Constant-velocity Kalman filter updating at measurement time.

    Predicts to current time without moving the measurement-time posterior.
    Measurement variance is fixed. Random acceleration process noise is an
    approximation; systematic bias, ball slip, and impacts are not modeled.
    """
    def __init__(self, measurement_std=.002, accel_std=.20, max_prediction=.15):
        if not np.isfinite([measurement_std,accel_std,max_prediction]).all() or min(measurement_std,accel_std,max_prediction)<=0:
            raise ValueError('filter parameters must be finite and positive')
        self.measurement_std=float(measurement_std)
        self.accel_std=float(accel_std)
        self.max_prediction=float(max_prediction)
        self.x=None;self.P=None;self.stamp=None

    def _transition(self,dt):
        F=np.eye(4);F[:2,2:]=dt*np.eye(2)
        q=self.accel_std**2
        Q=np.block([[dt**3/3*np.eye(2),dt**2/2*np.eye(2)],
                    [dt**2/2*np.eye(2),dt*np.eye(2)]])*q
        return F,Q

    def update(self,measurement,stamp,now):
        z=np.asarray(measurement,float);stamp=float(stamp);now=float(now)
        if z.shape!=(2,) or not np.isfinite(z).all() or not np.isfinite([stamp,now]).all():
            raise ValueError('invalid wrench position observation')
        delay=now-stamp
        if delay < -1e-9 or delay > self.max_prediction:
            raise RuntimeError('position measurement timestamp is stale or in the future')
        if self.x is None:
            self.x=np.r_[z,[0.,0.]];self.P=np.diag([self.measurement_std**2]*2+[.1**2]*2);self.stamp=stamp
        elif stamp < self.stamp-1e-10:
            raise RuntimeError('out-of-order position measurement')
        elif stamp > self.stamp+1e-12:
            dt=stamp-self.stamp;F,Q=self._transition(dt)
            if dt>self.max_prediction:
                raise RuntimeError('gap between position measurements exceeded limit')
            x=F@self.x;P=F@self.P@F.T+Q
            H=np.hstack((np.eye(2),np.zeros((2,2))))
            R=np.eye(2)*self.measurement_std**2
            S=H@P@H.T+R; K=np.linalg.solve(S,H@P).T
            self.x=x+K@(z-H@x)
            I=np.eye(4)-K@H
            self.P=I@P@I.T+K@R@K.T
            self.stamp=stamp
        F,Q=self._transition(max(0.,delay))
        pred=F@self.x
        return {'position_m':pred[:2].copy(),'velocity_m_s':pred[2:].copy(),
                'raw_position_m':z.copy(),'sample_time':stamp,'delay_s':delay,
                'position_std_m':np.sqrt(np.maximum(np.diag((F@self.P@F.T+Q))[:2],0.))}
