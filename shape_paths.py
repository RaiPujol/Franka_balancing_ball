"""Smooth two-dimensional reference paths on the tray, in metres and seconds.

These are offline simulation targets; tracking them requires ball sensing.
The square is deliberately rounded because a perfectly sharp corner would
require an abrupt change of velocity for a continuously moving ball.
"""
from dataclasses import dataclass
import math
import numpy as np
from controller import smoothstep5


@dataclass
class ShapePath:
    shape: str = 'circle'
    radius_m: float = .02
    period_s: float = 8.
    ramp_s: float = 3.
    cycles: int = 2
    center_hold_s: float = 2.

    def __post_init__(self):
        if self.shape not in ('circle', 'rounded_square', 'figure_eight'):
            raise ValueError('unsupported shape')
        if not np.isfinite([self.radius_m,self.period_s,self.ramp_s,self.center_hold_s]).all():
            raise ValueError('trajectory parameters must be finite')
        if self.radius_m <= 0 or self.period_s <= 0 or self.ramp_s <= 0 or self.center_hold_s < 0 or not isinstance(self.cycles,int) or self.cycles < 1:
            raise ValueError('invalid trajectory parameters')

    @property
    def plateau_start(self): return self.center_hold_s+self.ramp_s
    @property
    def plateau_end(self): return self.plateau_start+self.cycles*self.period_s
    @property
    def end_time(self): return self.plateau_end+self.ramp_s

    def curve(self, phi):
        c,s=math.cos(phi),math.sin(phi)
        if self.shape=='circle': return np.array([c,s])
        if self.shape=='figure_eight': return np.array([s, 0.5*math.sin(2*phi)])
        # Superellipse: smooth, rounded-square boundary with half-width 1.
        d=(c**8+s**8)**(1./8.)
        return np.array([c/d,s/d])

    def sample(self, t):
        u=t-self.center_hold_s
        if u <= 0 or t >= self.end_time:
            return np.zeros(2),np.zeros(2),np.zeros(2)
        if t < self.plateau_start:
            h,hd,hdd=smoothstep5(u/self.ramp_s)
            hd/=self.ramp_s;hdd/=self.ramp_s**2
        elif t <= self.plateau_end:
            h,hd,hdd=1.,0.,0.
        else:
            f,fd,fdd=smoothstep5((t-self.plateau_end)/self.ramp_s)
            h,hd,hdd=1.-f,-fd/self.ramp_s,-fdd/self.ramp_s**2
        omega=2*math.pi/self.period_s
        phi=omega*u
        q=self.curve(phi)
        # Central finite differences of the analytic smooth curve.
        eps=1.e-4; plus=self.curve(phi+eps);minus=self.curve(phi-eps)
        dq=(plus-minus)/(2*eps); ddq=(plus-2*q+minus)/(eps*eps)
        return (self.radius_m*h*q,
                self.radius_m*(hd*q+h*omega*dq),
                self.radius_m*(hdd*q+2*hd*omega*dq+h*omega*omega*ddq))
