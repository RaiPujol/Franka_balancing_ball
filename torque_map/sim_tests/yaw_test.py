import sys, contextlib, io as _io, numpy as np
sys.argv = ['x']
import fake_hard as H
rail, kin, tm = H.rail, H.kin, H.tm
lim = kin.PLATE_HALF - kin.BALL_RADIUS
BREAK, ROLLF, LEAN = np.radians(1.6), np.radians(0.3), np.radians(0.6)
class GenFake(H.HardIO):
    """Pose-independent ball physics along the paddle y rail (gravity along the rail's real direction),
    rail breakaway 1.6 deg, rolling friction 0.3 deg, deck lean 0.6 deg, ball falls off at speed;
    static friction on BOTH wrist joints 5 and 6; 28 Hz state; vibration on joints 2, 3, 4 with joint accel."""
    acc_f = 0.0
    def read(self):
        if getattr(self, '_c', None) is None or self.t - self._ct >= 0.035:
            st = super().read()
            vib = abs(np.degrees(self.acc_f))
            st["tau"][2] += self.rng.normal(0, 1) * 0.0005 * vib
            st["tau"][[1, 3]] += self.rng.normal(0, 1, 2) * 0.00025 * vib
            self._c, self._ct = st, self.t
        return {k: np.array(v, copy=True) for k, v in self._c.items()}
    def _sim(self, T):
        h = 0.001
        for _ in range(int(round(T / h))):
            if self.streaming:
                for j in (4, 5):
                    tq = self.K[j] * (self.qd[j] - self.q[j]) - self.D[j] * self.dq[j]
                    if abs(self.dq[j]) < 1e-4 and abs(tq) < 5.0:
                        self.dq[j] = 0.0; self.acc[j] = 0.0
                    else:
                        a = (tq - 2.0 * np.sign(self.dq[j] if abs(self.dq[j]) > 1e-4 else tq)) / 0.02
                        v0 = self.dq[j]; self.dq[j] += a * h
                        if v0 != 0 and np.sign(self.dq[j]) != np.sign(v0): self.dq[j] = 0.0
                        self.q[j] += self.dq[j] * h; self.acc[j] = a
                self.acc_f += (h / 0.05) * (max(abs(self.acc[4]), abs(self.acc[5])) - self.acc_f)
            if self.ball is not None and not self.hold:
                T_, J_ = kin.fk(self.q, kin.paddle_transform())
                rail_dir = T_[:3, 1]                                    # paddle +y in base
                g_t = -0.6 * 9.81 * (rail_dir[2] + np.sin(LEAN))        # gravity along the rail (+ lean toward +y)
                if abs(self.v) < 0.002:
                    if abs(g_t) > 0.6 * 9.81 * np.sin(BREAK):
                        self.v += (g_t - np.sign(g_t) * 0.6 * 9.81 * np.sin(ROLLF)) * h
                    else:
                        self.v = 0.0
                else:
                    self.v += (g_t - np.sign(self.v) * 0.6 * 9.81 * np.sin(ROLLF)) * h
                self.ball += self.v * h
                if abs(self.ball) >= lim:
                    if abs(self.v) > 0.02: self.ball = None; self.v = 0.0        # falls off
                    else: self.ball = np.sign(self.ball) * lim; self.v = 0.0
            self.t += h
    def goto(self, q, duration):
        super().goto(q, duration); self._c = None
def trial(yaw, seed, seconds=90):
    io = GenFake(seed); c = tm.Calib(); c.q0 = np.array(kin.HOME_POSE)
    cfg = dict(rail.PRESETS["pd"], balance_stops=False)
    rail.set_cfg(c, dict(axis="y", hand_yaw_deg=yaw)); rail.set_cfg(c, cfg)
    with contextlib.redirect_stdout(_io.StringIO()):
        rail.job_rail_map(io, c, H.Ctx(), seconds=3); rail.job_rail_sweep(io, c, H.Ctx(), seconds=40)
    fit = c.railmap
    drops = [0]; prev = [True]
    class C(H.Ctx):
        def tune_version(s):
            if io.ball is None and prev[0]: drops[0] += 1
            prev[0] = io.ball is not None
            return 0
    logs = []
    class C2(C):
        def log(s, m): logs.append(m)
    with contextlib.redirect_stdout(_io.StringIO()):
        log = rail.job_rail_balance(io, c, C2(), seconds=seconds, anchor_mm=0)
    tr = np.array([x if x is not None else np.nan for x in log['truth']])
    return drops[0], np.sqrt(np.nanmean(tr ** 2)) * 1000, fit["channels"], [round(r, 2) for r in fit["ratio"]], fit["verdict"], c.railmap.get("moving_noise_mm")
for yaw in (0.0, -90.0):
    res = [trial(yaw, s) for s in (2, 3, 4)]
    print(f"hand yaw {yaw:+.0f}: sensing joints {[j + 1 for j in res[0][2]]}, slope ratio {res[0][3]}, map {res[0][4]}, noise while rocking {res[0][5]:.0f} mm | drops per 90 s {np.mean([r[0] for r in res]):.1f} {[r[0] for r in res]} | ball rms while on {np.mean([r[1] for r in res]):.0f} mm", flush=True)
