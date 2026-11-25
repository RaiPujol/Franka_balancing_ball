"""Joint 4's zero creeps up 1.5 mNm/s (as on Nov 24: ~45 mNm per run) while joint 2 stays put. With the drift
correction (drift_comp_s 10) the estimate must stay closer to the true ball than without it."""
import numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import common
exec(common.bruno_prelude())
lim = 0.115
BREAK, ROLLF, LEAN = np.radians(0.1), np.radians(0.05), 0.0
class Drift4(BrunoFake):
    on = None
    def read(self):
        st = super().read()
        if self.on is not None: st["tau"][3] += 0.0015 * (self.t - self.on)
        return st
err = {}
for comp in (0.0, 10.0):
    io = Drift4(2); c = common.setup(io, rail, kin, tm, H, points=[-100.0, -50.0, 0.0, 50.0, 100.0], drift_comp_s=comp)
    class C(H.Ctx):
        t0 = None
        def log(s, m): pass
        def tune_version(s):
            if s.t0 is None: s.t0 = io.t; io.on = io.t
            return 0
    with common.quiet():
        log = rail.job_rail_balance(io, c, C(), seconds=45, anchor_mm=0)
    pe = np.array(log["p_est"]); tr = np.array([np.nan if x is None else float(x) for x in log["truth"]])
    n = min(len(pe), len(tr)); last = slice(n - 1500, n)
    err[comp] = float(np.nanmean(pe[last] - tr[last])) * 1000
ok = abs(err[10.0]) < abs(err[0.0]) and abs(err[10.0]) < 10
print(("PASS" if ok else "FAIL") + f": estimate bias over the last 15 s with joint 4 drifting: without correction {err[0.0]:+.0f} mm, with it {err[10.0]:+.0f} mm")
