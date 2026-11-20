#!/usr/bin/env python3
"""Record and analyze a stationary empty-tray torque baseline.

The operator must ensure that the tray is empty and the arm is stationary.
This starts the existing read-only ROS subscriber and writes analysis files.
It never creates a FrankaArm, commands motion or changes robot settings.
A software baseline does not zero physical sensors, and advancing timestamps
do not prove acquisition freshness or latency.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

from hardware.torque_baseline import summarize


def initialize(topic, output_dir, duration_s=20):
    if not topic or not isinstance(topic, str):
        raise ValueError('a published robot-state topic must be supplied')
    if not 1 <= duration_s <= 300:
        raise ValueError('duration must be between 1 and 300 seconds')
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        raise ValueError('output directory must already exist')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    prefix = 'empty_torque_' + stamp + '_' + uuid4().hex
    recording = output_dir / (prefix + '.jsonl')
    baseline = output_dir / (prefix + '_baseline.json')
    command = [
        sys.executable, '-m', 'hardware.record_wrench_readonly',
        '--topic', topic, '--output', str(recording), '--label', 'empty_torque_baseline',
        '--duration', str(duration_s),
    ]
    print('Recording empty tray to ' + str(recording), flush=True)
    try:
        result = subprocess.run(command, timeout=duration_s + 40, check=False)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError('recording timed out; the incomplete file was retained at ' +
                           str(recording)) from exc
    if result.returncode:
        raise RuntimeError('recording failed; inspect the retained file at ' + str(recording))
    try:
        report = summarize(recording)
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
        raise RuntimeError('baseline analysis failed; recording retained at ' +
                           str(recording) + ': ' + str(exc)) from exc
    with baseline.open('x', encoding='utf-8') as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write('\n')
    return recording, baseline, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--topic', required=True, help='verified ROS RobotState topic')
    parser.add_argument('--output-dir', required=True, help='existing directory for new recordings')
    parser.add_argument('--duration', type=float, default=20., help='recording seconds; default 20')
    args = parser.parse_args()
    try:
        recording, baseline, report = initialize(args.topic, args.output_dir, args.duration)
    except (RuntimeError, ValueError, OSError) as exc:
        parser.exit(1, 'Initialization failed: ' + str(exc) + '\n')
    print('Raw recording: ' + str(recording))
    print('Baseline analysis: ' + str(baseline))
    for joint, median, std in zip((5, 6), report['joint_5_6_median_nm'],
                                  report['joint_5_6_std_nm']):
        print('Joint %d: median %.6f N m, standard deviation %.6f N m' %
              (joint, median, std))
    print('Digital baseline recorded; no physical torque sensor was zeroed.')


if __name__ == '__main__':
    main()
