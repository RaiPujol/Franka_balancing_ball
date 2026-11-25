"""The ball falls off at 20 s: the run must detect the drop, re-zero, ask for the ball, and keep balancing."""
import numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import common
exec(common.yaw_prelude())
BREAK, ROLLF, LEAN = np.radians(0.1), np.radians(0.05), 0.0
io = GenFake(2); c = common.setup(io, rail, kin, tm, H)
logs = []
class C(H.Ctx):
    t0 = None
    def log(s, m): logs.append(m)
    def tune_version(s):
        if s.t0 is None: s.t0 = io.t
        if io.t - s.t0 > 20 and not getattr(s, "dropped", False): s.dropped = True; io.ball = None
        return 0
with common.quiet():
    log = rail.job_rail_balance(io, c, C(), seconds=35, anchor_mm=0)
drops = [m for m in logs if "DROPPED" in m]; rez = [m for m in logs if m.startswith("re-zeroed")]
ok = len(drops) >= 1 and len(rez) >= 1 and "finished" in log["stop_reason"][0]
print(("PASS" if ok else "FAIL") + f": drop detected {len(drops)}x, re-zeroed {len(rez)}x, run {log['stop_reason'][0]}")
