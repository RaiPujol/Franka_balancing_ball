"""Shared set-up for the fake-robot tests: a GenFake arm (yaw_test.py) at the turned-hand pose, rail along y,
sensing on joints 2 + 4, mapped and swept like on the robot. Import this AFTER exec-ing yaw_test.py's top part."""
import contextlib, io as _io
HERE = __import__("os").path.dirname(__import__("os").path.abspath(__file__))


def yaw_prelude():
    """The source of yaw_test.py up to its trial() function: GenFake, H (fake_hard), rail, kin, tm."""
    return open(__import__("os").path.join(HERE, "yaw_test.py")).read().split("def trial(")[0]


BRUNO_FAKE = '''
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
'''


def bruno_prelude():
    """yaw_test's GenFake plus BrunoFake (100 Hz state, freely following wrist joints, like iam-bruno)."""
    return yaw_prelude() + "\n" + BRUNO_FAKE


def setup(io, rail, kin, tm, H, points=None, **cfg):
    c = tm.Calib(); c.q0 = __import__("numpy").array(kin.HOME_POSE)
    if points is not None:
        c.rail_points = list(points)
    rail.set_cfg(c, dict(axis="y", hand_yaw_deg=-90.0))
    rail.set_cfg(c, dict(rail.PRESETS["smooth"], balance_stops=False, **cfg))
    with contextlib.redirect_stdout(_io.StringIO()):
        rail.job_rail_map(io, c, H.Ctx(), seconds=2)
        rail.job_rail_sweep(io, c, H.Ctx(), seconds=25)
    return c


def quiet():
    return contextlib.redirect_stdout(_io.StringIO())
