"""Roll test on a fake ball with a known rolling gain (0.6 g / (5/7 g) = 0.84) and lean (0.3 deg): the prompts must
name the right marks and the result must land within ~20 %."""
import numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import common
exec(common.yaw_prelude())
BREAK, ROLLF, LEAN = np.radians(0.1), np.radians(0.02), np.radians(0.3)
io = GenFake(3); c = common.setup(io, rail, kin, tm, H, points=[-60.0, -30.0, 0.0, 30.0, 60.0])
logs, prompts = [], []
class C(H.Ctx):
    def log(s, m): logs.append(m)
    def ask(s, m):
        prompts.append(m)
        if "BACK end" in m: io.ball, io.v = +0.050, 0.0
        elif "FRONT end" in m: io.ball, io.v = -0.050, 0.0
        elif "OFF" in m: io.ball = None
        return super().ask(m) if hasattr(super(), "ask") else None
with common.quiet():
    out = rail.job_rail_rolltest(io, c, C(), deg=1.0, rolls=2)
good_prompts = sum("-50 mm mark" in p for p in prompts) == 2 and sum("+50 mm mark" in p for p in prompts) == 2
rs, lean = out.get("roll_scale", 0), out.get("lean_deg", 0)
ok = good_prompts and 0.65 < rs < 1.0 and 0.15 < abs(lean) < 0.45
print(("PASS" if ok else "FAIL") + f": prompts {'ok' if good_prompts else 'WRONG'}, rolling gain {rs:.2f} (true 0.84), lean {lean:+.2f} deg (true 0.30)")
