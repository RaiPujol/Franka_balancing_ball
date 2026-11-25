import sys, os, numpy as np
sys.path.insert(0, os.environ.get('TM_PKG', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import kin, tm_core as tm, rail
SP = os.path.dirname(os.path.abspath(__file__))
rail.LOG_DIR = tm.LOG_DIR = os.path.join(SP, 'fake_logs'); os.makedirs(rail.LOG_DIR, exist_ok=True)
tm.Calib.save = lambda self, path=None: None
AXIS = sys.argv[1] if len(sys.argv) > 1 else 'y'

class FakeIO:
    """Wrist joints follow their targets through the impedance (inertia 0.02, no stiction).
    Ball rolls along the rail axis. Torques: kinematic ball weight (90 %), hand weight and
    inertia with the moving joint, noise like the Nov 20 logs."""
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.q = np.array(kin.HOME_POSE); self.q0 = self.q.copy(); self.dq = np.zeros(7); self.qd = self.q.copy()
        self.t = 0.0; self.dt = 0.02; self.ball = None; self.v = 0.0; self.hold = False; self.streaming = False
        self.acc = np.zeros(7); self.K, self.D = kin.K_GAINS.copy(), kin.D_GAINS.copy()
        _, J = kin.fk(self.q0, kin.paddle_transform()); self.J0 = J
    def set_rate(self, r): self.dt = 1.0 / r
    def clock(self): return self.t
    def _tilt(self):  # [a, b]
        return np.array([self.J0[3, 4] * (self.q[4] - self.q0[4]), self.J0[4, 5] * (self.q[5] - self.q0[5])])
    def _sim(self, T):
        h = 0.001
        for _ in range(int(round(T / h))):
            if self.streaming:
                for j in (4, 5):
                    a = (self.K[j] * (self.qd[j] - self.q[j]) - self.D[j] * self.dq[j]) / 0.02
                    self.dq[j] += a * h; self.q[j] += self.dq[j] * h; self.acc[j] = a
            if self.ball is not None and not self.hold:
                a, b = self._tilt()
                acc = kin.ROLL_ACCEL * (np.sin(b) if AXIS == 'x' else -np.sin(a))
                jj, ii = (5, 0) if AXIS == 'x' else (4, 1)
                acc -= 5.0 / 7.0 * self.J0[ii, jj] * self.acc[jj]        # the deck swings sideways under the ball
                self.v += acc * h; self.ball += self.v * h
                lim = kin.PLATE_HALF - kin.BALL_RADIUS
                if abs(self.ball) > lim: self.ball = np.sign(self.ball) * lim; self.v = -0.3 * self.v
            self.t += h
    def read(self):
        q = self.q
        tau = np.array([0.0, -5.0, 0.3, 8.0, 0.5, 2.5, 0.0])
        tau += np.array([0, 0.9, 0.3, -1.4, 0.4, 0.6, 0]) * ((q[4] - self.q0[4]) + (q[5] - self.q0[5]))
        tau += np.array([0, 0.002, 0.002, -0.003, 0.004, 0.004, 0]) * (self.acc[4] + self.acc[5])
        if self.ball is not None:
            T = rail._T_point(AXIS, self.ball)
            _, Jb = kin.fk(q, T); tau += 0.9 * kin.BALL_MASS * kin.G * Jb[2]
        tau += self.rng.normal(0, 1, 7) * np.array([0.03, 0.035, 0.03, 0.03, 0.01, 0.01, 0.02])
        return {"q": q + self.rng.normal(0, 2e-5, 7), "dq": self.dq + self.rng.normal(0, 2e-4, 7), "tau": tau, "wrench": np.zeros(6)}
    def ball_truth(self): return None if self.ball is None else self.ball
    def goto(self, q, duration): self.q = np.array(q, float); self.qd = self.q.copy(); self.dq[:] = 0; self._sim(0.1)
    def start_stream(self, q, k=None, d=None):
        self.streaming = True; self.qd = np.array(q, float)
        if k is not None: self.K, self.D = np.array(k), np.array(d)
    def command(self, q): self.qd = np.array(q, float); self._sim(self.dt)
    def stop_stream(self): self.streaming = False
    def wait(self, s): self._sim(s)
    def place_ball(self, ball, hold=False):
        self.ball = None if ball is None else float(ball[0] if AXIS == 'x' else ball[1]); self.v = 0.0; self.hold = hold

class Ctx:
    def status(self, m): pass
    def log(self, m):
        if 'log saved' not in m: print('  LOG', m)
    def cancelled(self): return False
    def ask(self, m): pass
    def answered(self): return True
    def live(self, **k): pass

if __name__ == '__main__':
    io = FakeIO(); ctx = Ctx(); c = tm.Calib(); c.q0 = np.array(kin.HOME_POSE)
    rail.set_cfg(c, {"axis": AXIS})
    if len(sys.argv) > 2: rail.set_cfg(c, {"channels": [int(j) for j in sys.argv[2].split(',')]})
    print('cfg', rail.cfg_of(c))
    rail.job_rail_track(io, c, ctx, seconds=12)
    rail.job_rail_map(io, c, ctx, seconds=2)
    rail.job_rail_sweep(io, c, ctx, seconds=30)
    orig = io.place_ball
    def place(ball, hold=False):
        orig(ball, hold)
        if ball is not None and not hold: io.ball = 0.04
    io.place_ball = place
    log = rail.job_rail_balance(io, c, ctx, seconds=40, gain=0.6, goal_mm=0)
    tr = np.array([t for t in log['truth'] if t is not None]); est = np.array(log['p_est']); n = len(tr)
    print(f'truth rms 2nd half {np.sqrt(np.mean(tr[n//2:]**2))*1000:.1f} mm, est-truth rms {np.sqrt(np.mean((est-tr)**2))*1000:.1f} mm')
    c.rail_cfg["goal"] = None
    log = rail.job_rail_balance(io, c, ctx, seconds=30, gain=0.6, goal_mm=30)
    tr = np.array([t for t in log['truth'] if t is not None]); n = len(tr)
    print(f'goal +30 mm: truth mean 2nd half {np.mean(tr[n//2:])*1000:.1f} mm')
