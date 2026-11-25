"""Joint 2's zero drifts 4 mNm/s: the joint-gap check must warn. Without drift it must stay quiet."""
import numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import common
exec(common.yaw_prelude())
BREAK, ROLLF, LEAN = np.radians(0.1), np.radians(0.05), 0.0
class Drift(GenFake):
    drift_on = None; rate = 0.0
    def read(self):
        st = super().read()
        if self.drift_on is not None: st["tau"][1] += self.rate * (self.t - self.drift_on)
        return st
res = {}
for rate in (0.0, 0.004):
    io = Drift(2); io.rate = rate; c = common.setup(io, rail, kin, tm, H)
    logs = []
    class C(H.Ctx):
        t0 = None
        def log(s, m): logs.append(m)
        def tune_version(s):
            if s.t0 is None: s.t0 = io.t; io.drift_on = io.t
            return 0
    with common.quiet():
        rail.job_rail_balance(io, c, C(), seconds=30, anchor_mm=0)
    res[rate] = sum(m.startswith("drift") for m in logs)
ok = res[0.0] == 0 and res[0.004] >= 1
print(("PASS" if ok else "FAIL") + f": drift warnings without drift {res[0.0]} (want 0), with 4 mNm/s drift {res[0.004]} (want >= 1)")
