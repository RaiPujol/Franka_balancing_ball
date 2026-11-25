"""Shared set-up for the fake-robot tests: a GenFake arm (yaw_test.py) at the turned-hand pose, rail along y,
sensing on joints 2 + 4, mapped and swept like on the robot. Import this AFTER exec-ing yaw_test.py's top part."""
import contextlib, io as _io
HERE = __import__("os").path.dirname(__import__("os").path.abspath(__file__))


def yaw_prelude():
    """The source of yaw_test.py up to its trial() function: GenFake, H (fake_hard), rail, kin, tm."""
    return open(__import__("os").path.join(HERE, "yaw_test.py")).read().split("def trial(")[0]


def setup(io, rail, kin, tm, H, points=None, **cfg):
    c = tm.Calib(); c.q0 = __import__("numpy").array(kin.HOME_POSE)
    if points is not None:
        c.rail_points = list(points)
    rail.set_cfg(c, dict(axis="y", hand_yaw_deg=-90.0))
    rail.set_cfg(c, dict(rail.PRESETS["smooth"], balance_stops=False, **cfg))
    with contextlib.redirect_stdout(_io.StringIO()):
        rail.job_rail_map(io, c, H.Ctx(), seconds=2)
        rail.job_rail_sweep(io, c, H.Ctx(), seconds=25)
    return c


def quiet():
    return contextlib.redirect_stdout(_io.StringIO())
