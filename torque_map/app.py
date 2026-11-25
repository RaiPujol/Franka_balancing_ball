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
import rail

HERE = os.path.dirname(os.path.abspath(__file__))
LIVE_SECONDS = 20.0
CODE_FILES = ("app.py", "rail.py", "kin.py", "tm_core.py", "robot_io.py")


def _code_stamp():
    return {f: os.path.getmtime(os.path.join(HERE, f)) for f in CODE_FILES if os.path.exists(os.path.join(HERE, f))}


CODE_AT_START = _code_stamp()          # python loads the code once: edits need a restart of app.py


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
                                     "tilta", "tiltb", "tiltcmda", "tiltcmdb", "railidle")}
        self.last = {}
        self.t0 = time.time()
        self._last_push = 0.0
        self.mode = "ROBOT"
        self.anchor_req = None                  # "the ball is at p (m) now", from the page, for the rail balancer
        self.tune_ver = 0                       # bumped by /api/tune; the rail balancer re-reads its tuning
        self._idle_avg = None

    # ---- what jobs call (the ctx interface of tm_core)
    def status(self, msg):
        with self.lock:
            self.status_msg = msg

    def log(self, msg):
        line = time.strftime("%H:%M:%S ") + msg
        print(line, flush=True)
        try:                                     # keep a copy on disk: the page and terminal scroll away
            os.makedirs(tm.LOG_DIR, exist_ok=True)
            with open(os.path.join(tm.LOG_DIR, time.strftime("session_%m%d.log")), "a") as f:
                f.write(line + "\n")
        except OSError:
            pass
        with self.lock:
            self.logs = (self.logs + [line])[-300:]

    def tune_version(self):
        return self.tune_ver

    def anchor_request(self):
        with self.lock:
            p, self.anchor_req = self.anchor_req, None
        return p

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

    def live(self, tau=None, q=None, est=None, meas=None, err=None, tilt=None, tiltcmd=None):
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
            L["tiltcmda"].append(None if tiltcmd is None else float(np.degrees(tiltcmd[0])))
            L["tiltcmdb"].append(None if tiltcmd is None else float(np.degrees(tiltcmd[1])))
            ri = None
            rm, raw = c.railmap, c.railmap_raw
            if rm and raw and tau is not None and self.busy is None:
                # idle: where the rail map puts the ball (approximate: idle is not the streaming controller)
                dr = (np.asarray(tau) - np.array(raw["empty"]))[rm["channels"]]
                ri = rail.combine(rm, dr, rm["sigma"])[0]
                if abs(ri) > kin.PLATE_HALF + 0.03:
                    ri = None                    # no ball on the paddle (or not a valid reading)
                else:                            # smooth the idle reading (about 0.5 s) so it does not jump
                    self._idle_avg = ri if self._idle_avg is None else self._idle_avg + 0.08 * (ri - self._idle_avg)
                    ri = self._idle_avg
            if ri is None:
                self._idle_avg = None
            L["railidle"].append(ri)
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
                "mode": self.mode, "busy": self.busy,
                "code_stale": sorted(f for f, t in _code_stamp().items() if CODE_AT_START.get(f) != t), "status": self.status_msg, "prompt": self.prompt_msg,
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
                    "grip_to_c_mm": kin.GRIP_TO_C * 1000.0, "wrist_scale": c.wrist_scale, "joint_watch": c.joint_watch,
                },
                "rail": {
                    "cfg": rail.cfg_of(c), "points": rail.points_of(c), "map": c.railmap,
                    "axes": {k: {kk: (vv.tolist() if hasattr(vv, "tolist") else vv) for kk, vv in g.items()
                                 if kk in ("name", "plus", "minus", "joint", "allowed", "channels")}
                             for k in rail.AXES for g in [rail.geometry(c, k, rail.cfg_of(c)["hand_yaw_deg"])]},
                    "presets": rail.PRESETS,
                    "mapped": c.railmap_raw is not None, "swept": c.railtare_W is not None,
                    "sweep_stale": bool(c.railtare_cfg) and c.railtare_cfg != rail._sweep_key(rail.cfg_of(c), c),
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
            elif self.path == "/api/tune":
                # live tuning: allowed while a job runs (only the TUNABLE keys; axis, joints, rate need a stop)
                c = state.calib
                new = {k: v for k, v in body.get("cfg", {}).items() if k in rail.TUNABLE}
                with state.lock:
                    rail.set_cfg(c, new)
                    state.tune_ver += 1
                try:
                    c.save()
                except OSError:
                    pass
                self._send(200, json.dumps({"applied": sorted(new), "live": state.busy in ("rail_balance", "rail_probe")}))
            elif self.path == "/api/anchor":
                with state.lock:
                    state.anchor_req = float(body.get("p_mm", 0.0)) / 1000.0
                self._send(200, "{}")
            elif self.path == "/api/railsettings":
                if state.busy is not None:
                    return self._send(409, json.dumps({"error": "busy"}))
                c = state.calib
                with state.lock:
                    rail.set_cfg(c, body.get("cfg", {}))
                    if "points" in body:
                        pts = sorted(float(v) for v in body["points"])
                        if len(pts) >= 2:
                            c.rail_points = pts
                    if "wrist_scale" in body:
                        c.wrist_scale = float(tm.GAIN_SCALES.get(str(body["wrist_scale"]), 1.0))
                    try:
                        rail.sync(c, state)          # sensing joints may have changed: refit from the recordings
                    except kin.SafetyStop as e:
                        state.log(f"rail settings: {e}")
                    c.save()
                self._send(200, "{}")
            elif self.path == "/api/settings":
                c = state.calib
                if "tilt_max_deg" in body:
                    c.tilt_max_deg = float(np.clip(float(body["tilt_max_deg"]), 1.0, 6.0))
                if body.get("joint_watch") in ("warn", "stop"):
                    c.joint_watch = body["joint_watch"]
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
        "goto_saved": lambda: tm.job_goto_saved(io, c, state),
        "guide": lambda: tm.job_guide(io, c, state, float(params.get("seconds", 30))),
        "use_current": lambda: tm.job_use_current(io, c, state),
        "grip": lambda: io.grip(True),
        "release": lambda: io.grip(False),
        "record_empty": lambda: tm.job_record_empty(io, c, state, params.get("which", "start"),
                                                    float(params.get("seconds", 3))),
        "record_point": lambda: tm.job_record_point(io, c, state, params["name"], float(params.get("seconds", 3))),
        "fit_map": lambda: tm.job_fit_map(io, c, state),
        "track": lambda: tm.job_track(io, c, state),
        "nudge": lambda: tm.job_nudge(io, c, state, params.get("axis", "x"), float(params.get("deg", 0.2))),
        "tilt_sweep": lambda: tm.job_tilt_sweep(io, c, state, float(params.get("seconds", 60))),
        "rail_track": lambda: rail.job_rail_track(io, c, state, float(params.get("seconds", 20)),
                                                  float(params.get("amp_deg", 1.5))),
        "rail_map": lambda: rail.job_rail_map(io, c, state, float(params.get("seconds", 3))),
        "rail_fit": lambda: (rail.rail_fit(c, state), rail.refit_sweep_from_log(c, state) if c.railtare_W is not None else None),
        "rail_sweep": lambda: rail.job_rail_sweep(io, c, state, float(params.get("seconds", 40))),
        "rail_rolltest": lambda: rail.job_rail_rolltest(io, c, state, float(params.get("deg", 1.0)), int(params.get("rolls", 3))),
        "rail_probe": lambda: rail.job_rail_probe(io, c, state, float(params.get("seconds", 40)),
                                                  float(params.get("anchor_mm", 0.0)), bool(params.get("anchor", True))),
        "rail_balance": lambda: (rail.set_cfg(c, {k: v for k, v in params.get("ctrl", {}).items() if k in rail.TUNABLE}),
                                 c.save(),
                                 rail.job_rail_balance(io, c, state, float(params.get("seconds", 60)),
                                                       anchor_mm=float(params.get("anchor_mm", 0.0)),
                                                       anchor=bool(params.get("anchor", True)))),
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
        for line in traceback.format_exc().rstrip().splitlines()[-6:]:
            state.log("  " + line)
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
    if hasattr(io, "state_rate"):
        state.log(f"robot state: {io.state_rate(1.0):.0f} messages/s on the state topic (was ~28 new states/s via the service)")
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
