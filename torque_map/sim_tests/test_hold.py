"""Still-ball hold: a sticky rail (0.4 deg breakaway) and sensor noise. With the hold on, the paddle must dither
less around a still ball than with it off, and the ball must stay on."""
import numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import common
exec(common.yaw_prelude())
BREAK, ROLLF, LEAN = np.radians(0.4), np.radians(0.05), np.radians(0.1)
out = {}
for hold in (0.0, 12.0):
    io = GenFake(5); c = common.setup(io, rail, kin, tm, H, hold_mm=hold)
    logs = []
    class C(H.Ctx):
        def log(s, m): logs.append(m)
    with common.quiet():
        log = rail.job_rail_balance(io, c, C(), seconds=40, anchor_mm=0)
    tt = np.degrees(np.array(log["tilt_target"]))[1000:]
    out[hold] = (float(np.std(tt)), sum("DROPPED" in m for m in logs))
ok = out[12.0][0] < out[0.0][0] and out[12.0][1] == 0
print(("PASS" if ok else "FAIL") + f": tilt dither rms hold off {out[0.0][0]:.2f} deg, hold on {out[12.0][0]:.2f} deg; drops with hold {out[12.0][1]}")
