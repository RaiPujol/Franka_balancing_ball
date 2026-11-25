"""Drive the balancer UI from here: tune, start, watch, analyse.
  rb.py state
  rb.py tune '{"lqr_gain": 1.0}'
  rb.py start rail_balance '{"seconds": 45}'      (or rail_sweep / rail_probe)
  rb.py watch [seconds]                            (prints status/prompt changes until idle)
  rb.py analyze [logfile]                          (newest railbalance log by default)"""
import sys, json, time, urllib.request, glob, os
import numpy as np
URL = "http://localhost:8000"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def get():
    return json.loads(urllib.request.urlopen(URL + "/api/state", timeout=3).read())
def post(p, b):
    r = urllib.request.urlopen(urllib.request.Request(URL + p, json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=5)
    return r.read().decode()
cmd = sys.argv[1]
if cmd == "state":
    s = get(); c = s["rail"]["cfg"]
    print("busy:", s["busy"], "| status:", s["status"], "| prompt:", s["prompt"], "| stale code:", s.get("code_stale"))
    print({k: c[k] for k in ("ctrl", "lqr_gain", "lqr_qv", "kp", "kv", "vmax_mm_s", "tilt_max_deg", "lead_s", "wc", "rate_max_dps", "kf_accel", "kf_r_scale", "balance_stops", "unstick", "goal_mm")})
    m = s["rail"]["map"] or {}; print("map", m.get("verdict"), "| sweep", m.get("moving_verdict"), m.get("moving_noise_mm"), "| sweep stale", s["rail"]["sweep_stale"])
    for l in s["logs"][-6:]: print("  ", l[:220])
elif cmd == "tune":
    print(post("/api/tune", {"cfg": json.loads(sys.argv[2])}))
elif cmd == "start":
    print(post("/api/run", {"job": sys.argv[2], "params": json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}}))
elif cmd == "watch":
    # print changes until the job that is running now (or starts) has finished, or the time is up
    t_end = time.time() + (float(sys.argv[2]) if len(sys.argv) > 2 else 600)
    last = None; seen = set(get()["logs"][-30:]); was_busy = False
    while time.time() < t_end:
        s = get()
        key = (s["busy"], s["prompt"])
        if key != last:
            print(time.strftime("%H:%M:%S"), "busy:", s["busy"], ("| WAITING FOR YOU: " + s["prompt"]) if s["prompt"] else "", flush=True); last = key
        for l in s["logs"][-30:]:
            if l not in seen: print("   log:", l[:230], flush=True); seen.add(l)
        was_busy = was_busy or s["busy"] is not None
        if was_busy and s["busy"] is None:
            break
        time.sleep(0.5)
elif cmd == "analyze":
    f = sys.argv[2] if len(sys.argv) > 2 else max(glob.glob(os.path.join(HERE, "logs/railbalance_y_*")), key=os.path.getmtime)
    d = np.load(f); n = len(d["p_est"]); dt = 0.01
    pe = np.array(d["p_est"]) * 1000; goal = np.array(d["goal"]) * 1000; err = pe - goal
    tl = np.degrees(d["tilt"][-n:, 0]); tt = np.degrees(d["tilt_target"]); dd = np.abs(d["d"][:, 0])
    pm = np.array(d["p_meas"]) * 1000
    print(os.path.basename(f), "| ran %.1f s | %s" % (n * dt, str(d["stop_reason"][0])[:160] if "stop_reason" in d.files else "?"))
    seg = lambda a, b: slice(int(a / dt), int(b / dt))
    for a in range(0, int(n * dt), 10):
        s_ = seg(a, min(a + 10, n * dt))
        print("  %3d-%3d s: ball err rms %5.1f mm, mean %+5.1f, max %5.1f | at an end %3.0f %% | tilt rms %.2f max %.2f deg | target at limit %3.0f %%"
              % (a, min(a + 10, n * dt), np.sqrt(np.mean(err[s_] ** 2)), err[s_].mean(), np.abs(err[s_]).max(),
                 100 * np.mean(np.abs(pe[s_]) > 78), np.sqrt(np.mean(tl[s_] ** 2)), np.abs(tl[s_]).max(),
                 100 * np.mean(np.abs(tt[s_]) >= 0.98 * np.abs(tt).max())))
    zc = np.sum(np.diff(np.sign(err)) != 0) / (n * dt)
    print("  overall: err rms %.1f mm | crossings of the goal %.2f /s | |d| 99%% %.2f Nm | resyncs %d | stall time %.1f s, unstick %.1f s | est-raw rms %.0f mm"
          % (np.sqrt(np.mean(err ** 2)), zc, np.percentile(dd, 99), d["resyncs"][-1] if "resyncs" in d.files else -1,
             np.sum(d["pinned"]) * dt, np.sum(d["unstick"]) * dt, np.sqrt(np.mean((pe - pm) ** 2))))
    # dominant oscillation
    F = np.abs(np.fft.rfft(err - err.mean())); fr = np.fft.rfftfreq(n, dt)
    if n > 300: print("  ball oscillation peak %.2f Hz (period %.1f s)" % (fr[np.argmax(F[2:]) + 2], 1 / fr[np.argmax(F[2:]) + 2]))
