"""One line per log file in torque_map/logs: kind, start, length, why it ended, drops. Usage: index_logs.py logs_dir out.md"""
import glob, os, sys, numpy as np
rows = ["# Log index (torque_map/logs)", "",
        "Per-cycle arrays: t, q, tau, tilt, tilt_cmd, q_target, dq, fresh. Per-balancing-step arrays (rail balance/probe): p_meas, "
        "p_est, v_est, bias, tilt_target, d, offset, v_star, p_ahead, ... and (from Nov 21 17:00) cycle_t, which matches the per-cycle t. "
        "In older logs the per-step arrays do NOT line up with the per-cycle ones after a drop prompt (prompt cycles are in between).", "",
        "| file | kind | log length s | balancing s | drops | ended | notes |", "|---|---|---|---|---|---|---|"]
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.npz")), key=os.path.getmtime):
    try:
        d = np.load(f)
        kind = os.path.basename(f).rsplit("_", 2)[0]
        T = d["t"][-1] if "t" in d.files and len(d["t"]) else 0
        bal = len(d["p_est"]) * 0.01 if "p_est" in d.files else (len(d["est"]) * 0.02 if "est" in d.files else 0)
        drops = len(d["drop_t"]) if "drop_t" in d.files else ""
        why = str(d["stop_reason"][0]).splitlines()[0][:70].replace("|", "/") if "stop_reason" in d.files else ""
        note = ""
        if "sweep" in kind and "tilt" in d.files:
            tl = np.degrees(d["tilt"]); note = "tilt range %+.1f..%+.1f deg" % (tl.min(), tl.max())
        rows.append(f"| {os.path.basename(f)} | {kind} | {T:.0f} | {bal:.0f} | {drops} | {why} | {note} |")
    except Exception as e:
        rows.append(f"| {os.path.basename(f)} | ? | | | | unreadable: {e} | |")
open(sys.argv[2], "w").write("\n".join(rows) + "\n"); print("indexed", len(rows) - 6, "logs")
