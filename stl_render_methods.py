"""Render matched circle-controller comparisons in the actual STL scene.

The STL is visual geometry; the tray contact surface remains a flat proxy.
All controllers receive only estimated, noisy and delayed wrench feedback.
The simulator ball state is used for scoring and video overlays only.
"""
import argparse
import render_methods
from stl_bridge import use_actual_stl_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='output/stl_comparison_videos')
    parser.add_argument('--methods', nargs='+', choices=list(render_methods.LABELS),
                        default=list(render_methods.LABELS))
    args = parser.parse_args()
    for key, label in list(render_methods.LABELS.items()):
        render_methods.LABELS[key] = 'STL visual / flat contact | ' + label
    with use_actual_stl_scene():
        render_methods.run_methods(args.output_dir, args.methods)


if __name__ == '__main__':
    main()
