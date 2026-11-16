"""Offline fixed-lag estimation for a ball on a moving tray.

The measured position is a delayed wrench-derived center of pressure.  The
filter does not read the ball's simulated state or simulator acceleration.
This is an experimental estimator, not a hardware safety controller.
"""
from __future__ import annotations

import numpy as np
from wrench_control import WrenchController


from position_filter import DelayedPositionKalman


class RobustWrenchController(WrenchController):
    """PD/trajectory controller using the measurement-time Kalman posterior.

    All ball feedback comes from the parent's simulated wrench observer. No
    true simulator ball state or joint acceleration is read by this controller.
    """
    def __init__(self,*args,measurement_std=.002,accel_std=.20,max_prediction=.15,**kwargs):
        super().__init__(*args,**kwargs)
        self.ball_filter=DelayedPositionKalman(measurement_std,accel_std,max_prediction)

    def _filter(self,observation):
        if observation is None:return None
        estimate=self.ball_filter.update(observation['raw_position_m'],observation['sample_time'],self.data.time)
        if 'force_n' in observation:
            estimate['force_n']=np.asarray(observation['force_n'],float).copy()
        return estimate

    def prime(self):
        observation=super().prime()
        if observation is None:
            self.last_observation=None
        else:
            self.last_observation=self._filter(observation)
        return self.last_observation

    def ball_state(self,jp,jr):
        observation=self.observer.observe(self.model,self.data)
        if observation is None:raise RuntimeError('wrench delay buffer not ready')
        self.last_observation=self._filter(observation)
        return np.r_[self.last_observation['position_m'],0.],np.r_[self.last_observation['velocity_m_s'],0.]
