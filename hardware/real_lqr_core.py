"""Transport-independent candidate for measured-wrench ball control.

This module has no robot connection or arm command. The caller must supply a
position estimate from a validated dynamic force/torque model in tray axes.
The static empty-tray subtraction in replay_static_wrench is valid only at
its recorded pose, and is not a dynamic estimator. Acquisition and current
timestamps must be in the same characterized monotonic clock domain.
"""
import numpy as np

from position_filter import DelayedPositionKalman


class ControllerFault(RuntimeError):
    """The candidate is latched off and must be replaced before another run."""


class BallLQRCore:
    """Calculate the tested candidate's tray-frame slope from a ball estimate.

    The tilt and timing limits must be chosen for the measured robot setup.
    Creating this core does not enable any robot motion or certify an estimator.
    """

    def __init__(self, *, max_tilt_rad, max_age_s, min_normal_n,
                 edge_margin_m, ball_radius_m,
                 tray_half_width_m=0.233 / 2, max_gap_s=0.15,
                 measurement_std_m=0.002, accel_std=0.3):
        vals = [max_tilt_rad, max_age_s, min_normal_n, edge_margin_m,
                ball_radius_m, tray_half_width_m, max_gap_s,
                measurement_std_m, accel_std]
        if not np.isfinite(vals).all() or min(vals) <= 0:
            raise ValueError('all dimensions, gates and limits must be finite and positive')
        if ball_radius_m + edge_margin_m >= tray_half_width_m:
            raise ValueError('the permitted ball area must fit within the tray')
        if max_age_s > max_gap_s:
            raise ValueError('maximum age cannot exceed the filter prediction limit')
        self.max_tilt_rad = float(max_tilt_rad)
        self.max_age_s = float(max_age_s)
        self.min_normal_n = float(min_normal_n)
        self.safe_half_width_m = float(tray_half_width_m - ball_radius_m - edge_margin_m)
        self.max_gap_s = float(max_gap_s)
        self.filter = DelayedPositionKalman(measurement_std_m, accel_std, max_gap_s)
        self.last_stamp = None
        self.last_now = None
        self.fault = None

    def _stop(self, reason):
        self.fault = reason
        raise ControllerFault(reason)

    def update(self, position_xy_m, normal_force_n, *, sample_time_s, now_s,
               target_xy_m=(0., 0.), target_velocity_m_s=(0., 0.),
               target_accel_m_s2=(0., 0.), tray_accel_xy_m_s2=(0., 0.)):
        """Return a bounded slope calculation, never a robot motion command.

        Positive slope rises toward positive tray x or y. Inputs are in the
        independently measured tray frame. Normal force is positive into the
        ball under the estimator's verified sign convention. Ball and target
        positions and velocities use tray axes. Tray acceleration must be
        expressed in the level reference-tray axes used by the simulator.
        """
        if self.fault is not None:
            raise ControllerFault('controller fault is latched: ' + self.fault)
        try:
            position = np.asarray(position_xy_m, float)
            target = np.asarray(target_xy_m, float)
            velocity = np.asarray(target_velocity_m_s, float)
            accel = np.asarray(target_accel_m_s2, float)
            tray_accel = np.asarray(tray_accel_xy_m_s2, float)
            force = float(normal_force_n)
            stamp = float(sample_time_s)
            now = float(now_s)
        except (TypeError, ValueError):
            self._stop('malformed observation')
        vectors = (position, target, velocity, accel, tray_accel)
        if any(v.shape != (2,) or not np.isfinite(v).all() for v in vectors):
            self._stop('invalid vector')
        if not np.isfinite([force, stamp, now]).all():
            self._stop('invalid force or timestamp')
        if force < self.min_normal_n:
            self._stop('insufficient normal force')
        if now < stamp or now - stamp > self.max_age_s:
            self._stop('stale or future observation')
        if self.last_stamp is not None and stamp <= self.last_stamp:
            self._stop('nonadvancing observation timestamp')
        if self.last_now is not None and now <= self.last_now:
            self._stop('nonadvancing control timestamp')
        if self.last_stamp is not None and stamp - self.last_stamp > self.max_gap_s:
            self._stop('observation gap exceeded')
        if np.max(np.abs(position)) >= self.safe_half_width_m:
            self._stop('observed ball outside permitted area')
        if np.max(np.abs(target)) >= self.safe_half_width_m:
            self._stop('target outside permitted area')
        try:
            estimate = self.filter.update(position, stamp=stamp, now=now)
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            self._stop('Kalman filter rejected observation')
        estimated_xy = estimate['position_m']
        if not np.isfinite(estimated_xy).all() or not np.isfinite(estimate['velocity_m_s']).all():
            self._stop('nonfinite Kalman state')
        if np.max(np.abs(estimated_xy)) >= self.safe_half_width_m:
            self._stop('predicted ball outside permitted area')
        # Match desired_tray_rotation and the evaluated gains. The desired
        # reference tray frame must be known before conversion into robot pose.
        slope = (0.6 * (estimated_xy - target)
                 + 0.8 * (estimate['velocity_m_s'] - velocity)
                 - tray_accel / 9.81 - 1.4 * accel / 9.81)
        if not np.isfinite(slope).all():
            self._stop('nonfinite slope')
        magnitude = np.linalg.norm(slope)
        if not np.isfinite(magnitude):
            self._stop('nonfinite slope norm')
        if magnitude > self.max_tilt_rad:
            slope *= self.max_tilt_rad / magnitude
        self.last_stamp = stamp
        self.last_now = now
        return {'slope_xy_rad': slope.copy(),
                'ball_xy_m': estimated_xy.copy(),
                'ball_velocity_m_s': estimate['velocity_m_s'].copy(),
                'measurement_age_s': now - stamp}
