"""MuJoCo stand-in for the robot (laptop only). Reuses the simulator of the earlier
torque_balance package with this setup: 233 mm deck (162.5 g), tab grip, deck centre 125 mm
out, solid 74.5 mm 193 g ball, torque noise measured on the lab Franka on Nov 17.

Needs mujoco and the Lab 2 Panda model next to the torque_balance folder (../Lab2)."""

import os
import sys

import numpy as np

import kin

HERE = os.path.dirname(os.path.abspath(__file__))
for cand in ("torque_balance", "BallBalance"):
    path = os.path.join(os.path.dirname(HERE), cand)
    if os.path.exists(os.path.join(path, "sim_ball_balance.py")):
        sys.path.insert(0, path)
        break
else:
    raise ImportError("sim_io needs ../torque_balance/sim_ball_balance.py")

import ball_core as _bc                                      # noqa: E402

_bc.paddle_transform = lambda *a, **k: kin.paddle_transform()  # the scene uses this mount
_bc.PLATE_HALF = kin.PLATE_HALF
import sim_ball_balance as _sb                               # noqa: E402

_sb.BALL_RADIUS = kin.BALL_RADIUS
_sb.PLATE_MASS = kin.PLATE_MASS
_sb.SCENE_XML = _sb.HERE / "panda_tm_scene.xml"


class SimIO(_sb.SimIO):
    def __init__(self, rate_hz=50.0, noise=1.0, view=False, seed=0, rail="none", realtime=False):
        super().__init__(rate_hz=rate_hz, noise=noise, ball_mass=kin.BALL_MASS, view=view,
                         realtime=view or realtime, seed=seed, verbose=False, rail=rail)

        self._pace = realtime and not view
        self._wall = None

    def _advance(self, n):
        super()._advance(n)
        if getattr(self, "_pace", False):                    # keep the UI plots in real time
            import time
            if self._wall is None:
                self._wall = time.perf_counter() - self.data.time
            delay = self.data.time - (time.perf_counter() - self._wall)
            if delay > 0:
                time.sleep(delay)
            elif delay < -0.5:
                self._wall = time.perf_counter() - self.data.time

    def grip(self, close=True):
        pass

    def guide(self, seconds):
        self.wait(min(seconds, 1.0))                         # nobody to push the simulated arm

    def place_ball(self, ball, hold=False):
        self.ask("", ball=ball, hold=hold)
