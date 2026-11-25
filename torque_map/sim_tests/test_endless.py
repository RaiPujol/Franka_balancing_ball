"""seconds = 0 balances until STOP: the run must keep going past any time limit, save its log in parts
(LOG_PART_SECONDS, 5 s here) and end cleanly when STOP is pressed (after 12 s here)."""
import glob, numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import common
exec(common.yaw_prelude())
BREAK, ROLLF, LEAN = np.radians(0.1), np.radians(0.05), 0.0
rail.LOG_PART_SECONDS = 5.0
io = GenFake(2); c = common.setup(io, rail, kin, tm, H)
logs = []
class C(H.Ctx):
    t0 = None
    def log(s, m): logs.append(m)
    def tune_version(s):
        if s.t0 is None: s.t0 = io.t
        return 0
    def cancelled(s):
        return s.t0 is not None and io.t - s.t0 > 12.0
with common.quiet():
    log = rail.job_rail_balance(io, c, C(), seconds=0, anchor_mm=0)
parts = [m for m in logs if m.startswith("log part saved")]
stopped = any("stopped from the UI" in m for m in logs)
ok = len(parts) == 2 and stopped
print(("PASS" if ok else "FAIL") + f": endless run saved {len(parts)} log parts (want 2 in 12 s), ended by STOP: {stopped}")
