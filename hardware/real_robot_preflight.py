#!/usr/bin/env python3
"""Read-only ROS state stream diagnostic for the ball-balancing project.

No robot interface, service client, publisher, motion or gripper command is
created. Source timestamps and receipts are reported separately because their
clock relationship and sensor acquisition latency have not been calibrated.
"""
import argparse
import json
import math
import threading
import time

from hardware.record_wrench_readonly import parse_message


def summarize(records, invalid_count=0):
    """Summarize parsed samples without assuming header timestamps are acquisition times."""
    if not records:
        return {'samples': 0, 'invalid_samples': invalid_count, 'stream_observed': False}
    intervals = [b['source_robot_time_s'] - a['source_robot_time_s']
                 for a, b in zip(records, records[1:])]
    positive = [x for x in intervals if x > 0]
    receipts = [b['receipt_monotonic_s'] - a['receipt_monotonic_s']
                for a, b in zip(records, records[1:])]
    return {
        'samples': len(records),
        'invalid_samples': invalid_count,
        'stream_observed': True,
        'source_fresh_flags_false': sum(not r['source_is_fresh'] for r in records),
        'nonadvancing_robot_time_intervals': sum(x <= 0 for x in intervals),
        'nonadvancing_header_intervals': sum(
            b['source_header_stamp_s'] <= a['source_header_stamp_s']
            for a, b in zip(records, records[1:])),
        'mean_robot_time_interval_s': sum(positive) / len(positive) if positive else None,
        'largest_robot_time_interval_s': max(positive) if positive else None,
        'largest_receipt_interval_s': max(receipts) if receipts else None,
        'all_zero_wrench_samples': sum(r['all_zero_estimated_wrench'] for r in records),
        'latest_robot_mode': records[-1]['robot_mode'],
        'configured_load_mass_kg': records[-1]['m_load'],
        'configured_total_mass_kg': records[-1]['m_total'],
        'caveat': ('Advancing timestamps do not establish the acquisition latency, '
                   'sensor validity, calibrated tray transform, or permission to move.'),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--topic', help='published franka_interface_msgs/RobotState topic')
    parser.add_argument('--seconds', type=float, default=5.,
                        help='wall-clock observation duration, 1 to 30 seconds')
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 1 <= args.seconds <= 30:
        parser.error('--seconds must be between 1 and 30')
    try:
        import numpy
        import rospy
        import rosgraph
        import frankapy
        from franka_interface_msgs.msg import RobotState
    except ImportError as exc:
        parser.error('source the lab ROS workspace and install its Python dependencies: ' + str(exc))
    print('Python dependencies available; numpy ' + numpy.__version__)
    if not args.topic:
        print('No topic requested. This import check does not connect to the robot.')
        return 0
    if not rosgraph.is_master_online():
        print('ROS master is unavailable.')
        return 2
    topics = dict(rosgraph.Master('/ball_preflight_inspect').getPublishedTopics(''))
    expected = 'franka_interface_msgs/RobotState'
    if topics.get(args.topic) != expected:
        print(json.dumps({'topic': args.topic, 'published_type': topics.get(args.topic),
                          'expected_type': expected, 'state_topic_available': False}))
        return 2
    rospy.init_node('ball_balancing_readonly_preflight', anonymous=True, disable_signals=True)
    records = []
    invalid = [0]
    lock = threading.Lock()

    def on_state(message):
        try:
            sample = parse_message(message, time.monotonic(), time.time())
        except (ValueError, TypeError, AttributeError):
            with lock:
                invalid[0] += 1
            return
        with lock:
            records.append(sample)

    subscriber = rospy.Subscriber(args.topic, RobotState, on_state, queue_size=100)
    try:
        until = time.monotonic() + args.seconds
        while time.monotonic() < until and not rospy.is_shutdown():
            time.sleep(min(.05, max(0., until - time.monotonic())))
    finally:
        subscriber.unregister()
    with lock:
        snapshot, invalid_count = list(records), invalid[0]
    report = summarize(snapshot, invalid_count)
    report['topic'] = args.topic
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report['samples'] >= 2 else 2


if __name__ == '__main__':
    raise SystemExit(main())
