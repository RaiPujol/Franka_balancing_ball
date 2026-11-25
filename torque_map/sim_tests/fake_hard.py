import sys, numpy as np
args = sys.argv[1:]; sys.argv = [sys.argv[0], 'y']
from fake_real import RealishIO, Ctx, tm, kin, rail
class HardIO(RealishIO):
    """RealishIO plus: arm reaches 85 % of the command, 50 ms extra delay; ball rolls on rails
    at 0.6 g sin(tilt) instead of 5/7 g; noisier joint 3 (40 mNm)."""
    def __init__(self, seed=2):
        super().__init__(seed); self.queue = []
    def command(self, q):
        q = np.array(q, float); q[4] = self.q0[4] + 0.85 * (q[4] - self.q0[4])
        self.queue.append(q); self.qd = self.queue.pop(0) if len(self.queue) > 5 else self.queue[0]; self._sim(self.dt)
    def _sim(self, T):
        g = kin.ROLL_ACCEL; kin.ROLL_ACCEL = 0.6 * 9.81 if self.ball is not None else g
        try:
            super()._sim(T)
        finally:
            kin.ROLL_ACCEL = g
    def read(self):
        st = super().read(); st["tau"][2] += self.rng.normal(0, 0.03); return st
    def command(self, q, _c=command):
        _c(self, q)
        if getattr(self, 'push_at', None) is not None and self.t >= self.push_at:
            self.ball = 0.05; self.v = 0.0; self.push_at = None; self.t_push = self.t   # moved to +50 mm and released
import fake_rail
def run(cfg, seconds=60, start=0.04):
    io = HardIO(); ctx = Ctx(); c = tm.Calib(); c.q0 = np.array(kin.HOME_POSE)
    rail.set_cfg(c, dict(axis="y", **cfg))
    import io as _io, contextlib
    with contextlib.redirect_stdout(_io.StringIO()):
        rail.job_rail_map(io, c, ctx, seconds=3); rail.job_rail_sweep(io, c, ctx, seconds=40)
        orig = io.place_ball
        def place(ball, hold=False):
            orig(ball, hold)
            if ball is not None and not hold: io.push_at = io.t + 10.0
        io.place_ball = place
        log = rail.job_rail_balance(io, c, ctx, seconds=seconds, gain=0.6, goal_mm=0, anchor_mm=0)
        # push the ball 4 cm off right after the anchor, like a disturbance
    tr = np.array([t for t in log['truth'] if t is not None]); n = len(tr)
    tl = np.degrees(np.array(log['tilt'])[-n:, 0])
    i0 = int(round((io.t_push - (io.t - n * 0.01)) / 0.01)) if hasattr(io, 't_push') else 0
    after = tr[i0:]
    over = -after.min() * 1000                                      # how far past centre after the +50 mm release
    inside = np.abs(after) < 0.01
    settle = next((k * 0.01 for k in range(len(after)) if inside[k:].all()), float('nan'))
    return n * 0.01, np.sqrt(np.mean(tr[-2000:] ** 2)) * 1000, over, np.abs(tl[i0:]).max(), settle
if __name__ == "__main__":
    for name, cfg in (("NEW default", {}), ("LQR", {"ctrl": "lqr"}), ("kp .5 kv 1.5", {"kp": 0.5, "kv": 1.5})):
        T, rms, over, tmax, settle = run(cfg)
        print(f"{name:20s} ran {T:3.0f} s | release at +50 mm: overshoot past C {over:5.1f} mm, max tilt {tmax:.2f} deg | last 20 s rms {rms:4.1f} mm")
