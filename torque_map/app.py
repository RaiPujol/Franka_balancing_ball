"""Web UI for the torque-map balancer. Python standard library only (nothing to install).

On the robot (in the lab docker container, terminal E):
    python3 app.py
    then open  http://localhost:8000  in Firefox on the user PC.

On a laptop, against the simulator (needs mujoco):
    python3 app.py --sim

The robot runs in this process's main thread (rospy needs it); the web server runs in a
background thread and only queues jobs, so a slow browser can never stall the robot.
"""

import argparse
import json
import os
import queue
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

import kin
import tm_core as tm

HERE = os.path.dirname(os.path.abspath(__file__))
LIVE_SECONDS = 20.0


class UIState:
    """Everything the page shows, plus the operator's answers. Shared between threads."""

    def __init__(self, calib):
        self.lock = threading.Lock()
        self.calib = calib
        self.busy = None
        self.status_msg = "idle"
        self.prompt_msg = None
        self.answer = threading.Event()
        self.cancel = threading.Event()
        self.logs = []
        self.series = {k: [] for k in ("t", "tau5", "tau6", "err5", "err6", "estx", "esty", "measx", "measy",
                                     "tilta", "tiltb")}
        self.last = {}
        self.t0 = time.time()
        self._last_push = 0.0
        self.mode = "ROBOT"

    # ---- what jobs call (the ctx interface of tm_core)
    def status(self, msg):
        with self.lock:
            self.status_msg = msg

    def log(self, msg):
        line = time.strftime("%H:%M:%S ") + msg
        print(line, flush=True)
        with self.lock:
            self.logs = (self.logs + [line])[-300:]

    def cancelled(self):
        return self.cancel.is_set()

    def ask(self, msg):
        self.answer.clear()
        with self.lock:
            self.prompt_msg = msg

    def answered(self):
        if self.answer.is_set() or self.cancel.is_set():
            with self.lock:
                self.prompt_msg = None
            return True
        return False

    def live(self, tau=None, q=None, est=None, meas=None, err=None, tilt=None):
        now = time.time()
        if now - self._last_push < 0.04:            # 25 Hz is plenty for the page
            return
        self._last_push = now
        t = now - self.t0
        c = self.calib
        with self.lock:
            L = self.series
            L["t"].append(t)
            ch = c.channels
            empty = np.array(c.empty["mean"]) if c.empty else None
            d = (np.asarray(tau) - empty)[ch] if (tau is not None and empty is not None) else [None, None]
            L["tau5"].append(None if d[0] is None else float(d[0]))
            L["tau6"].append(None if d[1] is None else float(d[1]))
            if err is None and c.fit and d[0] is not None:
                err = np.asarray(d) - np.array(c.fit["a"])          # vs the centre value
            for k, v in (("err", err), ("est", est), ("meas", meas)):
                a, b = (None, None) if v is None else (float(v[0]), float(v[1]))
                L[k + ("5" if k == "err" else "x")].append(a)
                L[k + ("6" if k == "err" else "y")].append(b)
            if meas is None and c.fit and d[0] is not None and self.busy is None:
                # idle and still: show where the map puts the ball right now
                p = np.linalg.solve(np.array(c.fit["B"]), np.asarray(d) - np.array(c.fit["a"]))
                L["measx"][-1], L["measy"][-1] = float(p[0]), float(p[1])
            ta, tb = (None, None) if tilt is None else (float(np.degrees(tilt[0])), float(np.degrees(tilt[1])))
            L["tilta"].append(ta)
            L["tiltb"].append(tb)
            while L["t"] and L["t"][0] < t - LIVE_SECONDS:
                for v in L.values():
                    v.pop(0)
            if q is not None:
                self.last["q"] = np.asarray(q).tolist()
                self.last["tau"] = None if tau is None else np.asarray(tau).tolist()

    # ---- what the page reads
    def snapshot(self):
        with self.lock:
            c = self.calib
            q = self.last.get("q")
            return {
                "mode": self.mode, "busy": self.busy, "status": self.status_msg, "prompt": self.prompt_msg,
                "logs": self.logs[-60:], "live": self.series,
                "q": q, "tau": self.last.get("tau"),
                "pose": tm.pose_report(q) if q is not None else None,
                "calib": {
                    "has_pose": c.q0 is not None, "points": c.points,
                    "readings": {k: {"x": v["x"], "y": v["y"],
                                     "d": None if not c.empty else
                                     ((np.array(v["mean"]) - np.array(c.empty["mean"]))[c.channels]).tolist()}
                                 for k, v in c.readings.items()},
                    "empty": c.empty is not None, "empty_end": c.empty_end is not None,
                    "fit": c.fit, "swept": c.tare_W is not None, "tilt_max_deg": c.tilt_max_deg,
                    "ball_mass": c.ball_mass, "plate_half": kin.PLATE_HALF, "ball_radius": kin.BALL_RADIUS,
                    "grip_to_c_mm": kin.GRIP_TO_C * 1000.0, "wrist_scale": c.wrist_scale,
                },
            }


def make_handler(state, jobs):
    page = os.path.join(HERE, "ui.html")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json", extra=None):
            data = body if isinstance(body, bytes) else body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                with open(page, "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            elif self.path == "/api/state":
                self._send(200, json.dumps(state.snapshot()))
            elif self.path == "/api/calib.json":
                self._send(200, json.dumps(state.calib.to_json(), indent=1), extra={
                    "Content-Disposition": f"attachment; filename=tm_calib_{time.strftime('%m%d_%H%M')}.json"})
            elif self.path == "/api/map.csv":
                c = state.calib
                rows = ["name,x_mm,y_mm," + ",".join(f"dtau_j{j + 1}_Nm" for j in range(7))]
                if c.empty:
                    e0 = np.array(c.empty["mean"])
                    for k, v in c.readings.items():
                        d = np.array(v["mean"]) - e0
                        rows.append(f"{k},{v['x']},{v['y']}," + ",".join(f"{x:.5f}" for x in d))
                self._send(200, "\n".join(rows) + "\n", "text/csv", extra={
                    "Content-Disposition": f"attachment; filename=torque_map_{time.strftime('%m%d_%H%M')}.csv"})
            else:
                self._send(404, "{}")

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0) or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/api/run":
                if state.busy is not None:
                    return self._send(409, json.dumps({"error": f"busy with {state.busy}"}))
                state.cancel.clear()
                state.busy = body.get("job")             # mark busy right away (no double clicks)
                jobs.put((body.get("job"), body.get("params", {})))
                self._send(200, "{}")
            elif self.path == "/api/continue":
                state.answer.set()
                self._send(200, "{}")
            elif self.path == "/api/cancel":
                state.cancel.set()
                state.answer.set()
                self._send(200, "{}")
            elif self.path == "/api/points":
                if state.busy is not None:
                    return self._send(409, json.dumps({"error": "busy"}))
                pts = [{"name": str(p["name"])[:12], "x": float(p["x"]), "y": float(p["y"])} for p in body["points"]]
                c = state.calib
                with state.lock:
                    c.points = pts
                    keep = {p["name"]: p for p in pts}
                    c.readings = {k: v for k, v in c.readings.items()
                                  if k in keep and abs(keep[k]["x"] - v["x"]) < 1e-6 and abs(keep[k]["y"] - v["y"]) < 1e-6}
                    c.fit = None if len(c.readings) < 3 else c.fit
                    c.save()
                self._send(200, "{}")
            elif self.path == "/api/settings":
                c = state.calib
                if "tilt_max_deg" in body:
                    c.tilt_max_deg = float(np.clip(float(body["tilt_max_deg"]), 1.0, 6.0))
                if "wrist_scale" in body:
                    c.wrist_scale = float(tm.GAIN_SCALES.get(str(body["wrist_scale"]), 1.0))
                    c.tare_W = c.tare_sigma = None      # the tilt sweep depends on the gains
                c.save()
                self._send(200, "{}")
            else:
                self._send(404, "{}")

    return Handler


def run_job(io, state, name, params):
    c = state.calib
    f = {
        "go_home": lambda: tm.job_go_home(io, c, state),
        "guide": lambda: tm.job_guide(io, c, state, float(params.get("seconds", 30))),
        "use_current": lambda: tm.job_use_current(io, c, state),
        "grip": lambda: io.grip(True),
        "release": lambda: io.grip(False),
        "record_empty": lambda: tm.job_record_empty(io, c, state, params.get("which", "start"),
                                                    float(params.get("seconds", 3))),
        "record_point": lambda: tm.job_record_point(io, c, state, params["name"], float(params.get("seconds", 3))),
        "fit_map": lambda: tm.job_fit_map(io, c, state),
        "track": lambda: tm.job_track(io, c, state),
        "tilt_sweep": lambda: tm.job_tilt_sweep(io, c, state, float(params.get("seconds", 60))),
        "balance": lambda: tm.job_balance(io, c, state, float(params.get("seconds", 30)),
                                          float(params.get("gain", 0.6)), params.get("axis", "both")),
    }.get(name)
    if f is None:
        state.log(f"unknown job {name}")
        return
    state.status(name.replace("_", " "))
    try:
        f()
    except kin.SafetyStop as e:
        state.log(f"STOPPED: {e}")
    except ValueError as e:                 # frankapy: unreachable target / virtual walls
        state.log(f"frankapy refused the motion: {e}")
    except Exception as e:                  # keep the UI alive, show what broke
        state.log(f"ERROR in {name}: {e!r}")
        traceback.print_exc()


def main():
    ap = argparse.ArgumentParser(description="Torque-map ball balancer with a web UI.")
    ap.add_argument("--sim", action="store_true", help="Run against the MuJoCo simulator (laptop).")
    ap.add_argument("--view", action="store_true", help="With --sim: open the MuJoCo viewer.")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to reach the page from another computer.")
    args = ap.parse_args()

    calib = tm.Calib.load()
    if args.sim:
        import sim_io
        io = sim_io.SimIO(rate_hz=calib.rate_hz, view=args.view, realtime=True)
    else:
        import robot_io
        io = robot_io.FrankaIO(calib.rate_hz)
    state = UIState(calib)
    state.mode = "SIMULATOR" if args.sim else "ROBOT"
    jobs = queue.Queue()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state, jobs))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    state.log(f"UI ready: open http://localhost:{args.port} in a browser ({'SIMULATOR' if args.sim else 'ROBOT'})")
    try:
        while True:
            try:
                name, params = jobs.get(timeout=0.05)
            except queue.Empty:
                s = io.read()                       # idle: keep the live readout going
                state.live(tau=s["tau"], q=s["q"])
                io.wait(0.05)
                continue
            state.busy = name
            run_job(io, state, name, params)
            with state.lock:
                state.busy, state.prompt_msg = None, None
                state.status_msg = "idle"
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
