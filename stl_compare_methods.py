"""Run matched controller comparisons with the actual STL scene (offline only)."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import compare_methods
from stl_bridge import use_actual_stl_scene
from stl_scene import DEFAULTS, SOURCE, LOCAL_MESH

if __name__ == '__main__':
    # Match the active CAD scene in both the plant and the nominal observer.
    if not any(arg == '--tray-mass' or arg.startswith('--tray-mass=') for arg in sys.argv[1:]):
        sys.argv.extend(['--tray-mass', str(DEFAULTS['tray_mass_kg'])])
    with use_actual_stl_scene():
        compare_methods.main()
    # Capture the actual STL scene and mesh as well as controller sources.
    # compare_methods' legacy manifest alone only identifies the placeholder.
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--method', required=True)
    p.add_argument('--output-dir', default='output/compare')
    args, _ = p.parse_known_args()
    root = Path(__file__).resolve().parent
    files = ('stl_scene.py', 'stl_bridge.py', 'stl_compare_methods.py',
             'compare_methods.py', str(LOCAL_MESH.relative_to(root)),
             str(SOURCE.relative_to(root)), 'actual_tray_config.json', 'controller.py',
             'wrench_control.py', 'wrench_shapes.py', 'wrench_center_demos.py',
             'lqr_control.py', 'mpc_control.py', 'robust_control.py',
             'joint_wrench_proxy.py', 'wrench_frames.py')
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
              for name in files}
    (Path(args.output_dir) / args.method / 'actual_scene_manifest.json').write_text(
        json.dumps(hashes, indent=2) + '\n')
