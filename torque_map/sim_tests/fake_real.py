import sys, numpy as np
sys.argv = [sys.argv[0], 'y'] + [a for a in sys.argv[1:] if a != 'y']
import fake_rail as F
from fake_rail import *
class RealishIO(FakeIO):
    """Like the Nov 20 20:44-20:52 logs: joint 5 torque follows its own motion (+-0.3 Nm with the
    direction, with memory), joint 3 is quantised (24 mNm steps), 25 mNm still, more while moving;
    ball seen at 80 % on joint 3."""
    def __init__(self, seed=1):
        super().__init__(seed); self.h5 = 0.0
    def read(self):
        st = super().read(); tau = st["tau"]
        v5 = self.dq[4]
        self.h5 += 0.15 * (0.3 * np.tanh(v5 / 0.004) - self.h5)          # friction with memory
        tau[4] += self.h5
        if self.ball is not None:                                          # ball at 80 % on joint 3 (super gave 90 %)
            _, Jb = kin.fk(self.q, rail._T_point('y', self.ball)); tau[2] -= 0.1 * kin.BALL_MASS * kin.G * Jb[2, 2]
        tau[2] += self.rng.normal(0, 0.015) + self.rng.normal(0, 1) * min(0.05, 2.0 * abs(v5))
        tau[2] = np.round(tau[2] / 0.024) * 0.024
        return st
if __name__ == '__main__':
    ch = [int(j) for j in sys.argv[2].split(',')] if len(sys.argv) > 2 else [2]
    smooth = sys.argv[3] if len(sys.argv) > 3 else 'medium'
    io = RealishIO(); ctx = Ctx(); c = tm.Calib(); c.q0 = np.array(kin.HOME_POSE)
    rail.set_cfg(c, {"axis": "y", "channels": ch, "smooth": smooth}); print('cfg channels', rail.cfg_of(c)['channels'], smooth)
    rail.job_rail_map(io, c, ctx, seconds=3)
    rail.job_rail_sweep(io, c, ctx, seconds=40)
    reqs = {}
    ctx.anchor_request = lambda: reqs.pop(io.t_step, None) if hasattr(io, 't_step') else None
    orig = io.place_ball
    def place(ball, hold=False):
        orig(ball, hold)
    log = rail.job_rail_balance(io, c, ctx, seconds=40, gain=0.6, goal_mm=0, anchor_mm=0)
    tr = np.array([t for t in log['truth'] if t is not None]); est = np.array(log['p_est']); meas = np.array(log['p_meas']); n = len(tr)
    print(f'  ball truth rms 2nd half {np.sqrt(np.mean(tr[n//2:]**2))*1000:.1f} mm | estimate - truth rms {np.sqrt(np.mean((est-tr)**2))*1000:.1f} mm, '
          f'raw - truth {np.std(meas-tr)*1000:.0f} mm | estimate jitter (sample to sample) {np.std(np.diff(est))*1000:.2f} mm')
