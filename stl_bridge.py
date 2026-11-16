"""Explicit single-process adapter for legacy simulator runners only.

The original experimental scripts capture `build_model` at import time, both
for the physical simulation and the observer's unloaded tool model. This
context swaps each reference atomically and restores it even after exceptions.
It is intended for sequential offline simulation only, never hardware.
"""
from contextlib import contextmanager
import threading

import stl_scene

_LOCK=threading.RLock()


@contextmanager
def use_actual_stl_scene(surface='flat'):
    """Bind legacy scene references to the inspected STL scene temporarily.

    At present the `flat` surface omits the tiny central dimple. Unsupported
    variants fail instead of silently using a different collision model.
    Plant and observer both receive the same builder; the observer remains
    free to have an intentionally mismatched mass in stress tests.
    """
    if surface!='flat':raise ValueError('Only flat collision proxy is currently validated')
    import wrench_control
    import wrench_shapes
    import wrench_center_demos
    import run_sim
    import force_estimator
    import tare_prototype
    targets=(wrench_control,wrench_shapes,wrench_center_demos,run_sim,force_estimator,tare_prototype)
    with _LOCK:
        prior=[(module,module.build_model) for module in targets]
        try:
            for module,_ in prior:module.build_model=stl_scene.build_model
            yield stl_scene.build_model
        finally:
            for module,previous in prior:module.build_model=previous
