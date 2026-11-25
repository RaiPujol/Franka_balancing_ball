"""A 4 Nm press on joint 4 for 0.3 s at 10 s: handled as CONTACT (level, ball off, ball back) and the run goes on."""
import numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import common
exec(common.yaw_prelude())
BREAK, ROLLF, LEAN = np.radians(0.1), np.radians(0.05), 0.0
class Push(GenFake):
    t_push = None
    def read(self):
        st = super().read()
        if self.t_push is not None and 0 < self.t - self.t_push < 0.3: st["tau"][3] += 4.0
        return st
io = Push(2); c = common.setup(io, rail, kin, tm, H)
logs = []
class C(H.Ctx):
    t0 = None
    def log(s, m): logs.append(m)
    def tune_version(s):
        if s.t0 is None: s.t0 = io.t; io.t_push = io.t + 10
        return 0
with common.quiet():
    log = rail.job_rail_balance(io, c, C(), seconds=25, anchor_mm=0)
contact = [m for m in logs if "CONTACT" in m]
ok = len(contact) == 1 and "finished" in log["stop_reason"][0]
print(("PASS" if ok else "FAIL") + f": contact events {len(contact)} (want 1), run {log['stop_reason'][0]}")
