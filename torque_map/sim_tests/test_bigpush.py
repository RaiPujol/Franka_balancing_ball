"""A hard push on a rail like the real one (ends at +-115 mm, map points +-100 mm; the fake ball falls off if it hits
an end faster than 20 mm/s): once the ball is within 15 mm of the centre after 10 s it is shoved toward the back end
at 120 mm/s (PUSH=<m/s> to change; ~120 mm/s is the most this rail can catch, the brake off drops it). With the emergency brake (brake tilt 2.5 deg) it must be caught; the same push
with the brake off is reported for comparison."""
import numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import common
exec(common.yaw_prelude())
BREAK, ROLLF, LEAN = np.radians(0.1), np.radians(0.05), 0.0
lim = 0.115                                     # the fake's rail end (yaw_test reads this global)


class BrunoFake(GenFake):
    """GenFake made like iam-bruno: 100 Hz state and wrist joints that follow freely (GenFake imitates the earlier
    arm: 28 Hz state and sticky wrist joints, which delayed the paddle ~0.5 s and made every hard push a drop)."""
    def read(self):
        if getattr(self, "_c", None) is None or self.t - self._ct >= 0.0099:
            st = H.HardIO.read(self)
            st["tau"][[1, 3]] += self.rng.normal(0, 0.012, 2)
            self._c, self._ct = st, self.t
        return {k: np.array(v, copy=True) for k, v in self._c.items()}

    def _sim(self, T):
        h = 0.001
        for _ in range(int(round(T / h))):
            if self.streaming:
                for j in (4, 5):
                    tq = self.K[j] * (self.qd[j] - self.q[j]) - self.D[j] * self.dq[j]
                    self.dq[j] += tq / 0.02 * h; self.q[j] += self.dq[j] * h
            if self.ball is not None and not self.hold:
                T_, _ = kin.fk(self.q, kin.paddle_transform())
                g_t = -0.6 * 9.81 * (T_[:3, 1][2] + np.sin(LEAN))
                if abs(self.v) < 0.002 and abs(g_t) <= 0.6 * 9.81 * np.sin(BREAK):
                    self.v = 0.0
                else:
                    self.v += (g_t - np.sign(self.v if self.v else g_t) * 0.6 * 9.81 * np.sin(ROLLF)) * h
                self.ball += self.v * h
                if abs(self.ball) >= lim:
                    if abs(self.v) > 0.02: self.ball = None; self.v = 0.0
                    else: self.ball = np.sign(self.ball) * lim; self.v = 0.0
            self.t += h
PUSH = float(os.environ.get("PUSH", 0.12))
out = {}
for brake in (0.0, 2.5):
    io = BrunoFake(4); c = common.setup(io, rail, kin, tm, H, points=[-100.0, -50.0, 0.0, 50.0, 100.0], brake_tilt_deg=brake)
    logs = []
    class C(H.Ctx):
        t0 = None
        def log(s, m): logs.append(m)
        def tune_version(s):
            if s.t0 is None: s.t0 = io.t
            if (io.t - s.t0 > 10 and not getattr(s, "pushed", False) and io.ball is not None
                    and abs(io.ball) < 0.015 and abs(io.v) < 0.02):
                s.pushed = True; io.v = PUSH
            return 0
    with common.quiet():
        log = rail.job_rail_balance(io, c, C(), seconds=25, anchor_mm=0)
    out[brake] = (sum("DROPPED" in m for m in logs), float(np.max(np.array(log.get("emergency", [0])))), getattr(C, "pushed", None))
ok = out[2.5][0] == 0
print(("PASS" if ok else "FAIL") + f": {PUSH * 1000:.0f} mm/s push -> drops with emergency brake {out[2.5][0]} "
      f"(brake used: {'yes' if out[2.5][1] else 'no'}), without it {out[0.0][0]}")
