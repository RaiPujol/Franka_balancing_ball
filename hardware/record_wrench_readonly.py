#!/usr/bin/env python3
"""Record Franka ROS state only. This script publishes and commands nothing.

Run on the lab PC only after verifying the actual ROS topic and message type.
Never use as a feedback source or motion interface: timestamps/rates need to
be characterized, and a ROS message stamped 'fresh' can still be old.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from queue import Empty, Full, Queue
import threading
import time

FIELDS = {'O_F_ext_hat_K':6,'K_F_ext_hat_K':6,'O_T_EE':16,'F_T_EE':16,'EE_T_K':16,
          'q':7,'dq':7,'tau_J':7,'tau_ext_hat_filtered':7,
          'F_x_Cee':3,'F_x_Cload':3,'F_x_Ctotal':3,'I_ee':9,'I_load':9,'I_total':9}


def vector(msg, name, length):
    values = [float(x) for x in getattr(msg, name)]
    if len(values) != length or not all(math.isfinite(x) for x in values):
        raise ValueError('%s must have %d finite values' % (name, length))
    return values


def parse_message(msg, receipt_monotonic_s, receipt_wall_s):
    """Pure parser: no rospy, no robot connection. Invalid schema raises."""
    if not hasattr(msg, 'is_fresh') or not hasattr(msg, 'robot_time'):
        raise ValueError('RobotState missing freshness or robot_time')
    time_value = float(msg.robot_time)
    if not math.isfinite(time_value):
        raise ValueError('robot_time must be finite')
    h = msg.header
    stamp = h.stamp
    stamp_s = float(stamp.to_sec())
    if not math.isfinite(stamp_s) or stamp_s<=0:raise ValueError('source header timestamp must be positive and finite')
    data = {'kind':'sample','receipt_monotonic_s':float(receipt_monotonic_s),
            'receipt_wall_s':float(receipt_wall_s), 'source_header_seq':int(h.seq),
            'source_header_stamp_s':stamp_s,'source_frame_id':str(h.frame_id),
            'source_robot_time_s':time_value,'source_is_fresh':bool(msg.is_fresh),
            'robot_mode':int(msg.robot_mode)}
    for name,n in FIELDS.items():
        data[name] = vector(msg,name,n)
    for name in ('m_ee','m_load','m_total','gripper_width','control_command_success_rate'):
        value = float(getattr(msg,name))
        if not math.isfinite(value): raise ValueError('%s must be finite' % name)
        data[name] = value
    # Preserve zero rather than infer validity: the external estimate can be
    # zero at singularities, and may also legitimately be small.
    data['all_zero_estimated_wrench'] = all(x == 0.0 for x in data['O_F_ext_hat_K'])
    return data


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--topic', required=True, help='verified live RobotState topic, not a command topic')
    p.add_argument('--output',required=True, help='new JSONL path; existing files are not overwritten')
    p.add_argument('--label',required=True,help='plain label, e.g. empty or position_A')
    p.add_argument('--duration',type=float,default=20.,help='wall-clock collection seconds (1–300)')
    p.add_argument('--first-message-timeout',type=float,default=8.)
    p.add_argument('--silence-timeout',type=float,default=2.)
    a=p.parse_args()
    if not (1 <= a.duration <= 300 and 1 <= a.first_message_timeout <= 30 and .2 <= a.silence_timeout <= 30):
        p.error('duration must be 1–300s, first-message-timeout 1–30s, silence-timeout .2–30s')
    # Import only when invoked on a ROS machine; importing this file does not
    # initialize a node, connect to ROS, or command a robot.
    try:
        import rospy
        from franka_interface_msgs.msg import RobotState
    except ImportError as e:
        p.error('ROS1 and franka_interface_msgs are required on the lab PC: %s' % e)
    path=Path(a.output)
    if not path.parent.is_dir(): p.error('output parent directory does not exist')
    queue=Queue(maxsize=10000); errors=[]; overflow=threading.Event()
    def callback(msg):
        try: item=parse_message(msg,time.monotonic(),time.time())
        except Exception as e:
            item={'kind':'invalid_message','reason':str(e),'receipt_monotonic_s':time.monotonic()}
        try: queue.put_nowait(item)
        except Full: overflow.set()
    count=0;valid=0;prev_robot_time=None;prev_header_stamp=None;prev_seq=None;last_receive=None;last_candidate=None
    with path.open('x',encoding='utf-8') as f:
        header={'kind':'metadata','schema':'franka_wrench_observation_v1','label':a.label,'topic':a.topic,
                'utc_started':datetime.now(timezone.utc).isoformat(),
                'wrench_frame':'O_F_ext_hat_K: expressed in base frame O, moment about stiffness frame K',
                'pose_layout':'O_T_EE and EE_T_K are column-major homogeneous transforms',
                'timestamp_warning':'header stamp is publisher/snapshot time; robot_time is separate; candidate flag cannot prove freshness, latency or absence of replay',
                'calibration_warning':'record only; no motion or configuration commands'}
        f.write(json.dumps(header,allow_nan=False)+'\n');f.flush()
        rospy.init_node('franka_wrench_readonly_recorder',anonymous=True,disable_signals=True)
        sub=rospy.Subscriber(a.topic,RobotState,callback,queue_size=100)
        start=time.monotonic(); deadline=start+a.duration; first_deadline=start+a.first_message_timeout
        try:
            while time.monotonic()<deadline and not rospy.is_shutdown():
                if overflow.is_set(): errors.append('local queue overflow');break
                try: item=queue.get(timeout=.2)
                except Empty:
                    if valid==0 and time.monotonic()>first_deadline:
                        errors.append('no candidate RobotState messages received');break
                    if last_candidate is not None and time.monotonic()-last_candidate>a.silence_timeout:
                        errors.append('no advancing candidate RobotState samples');break
                    continue
                count+=1
                if item['kind']=='sample':
                    rt=item['source_robot_time_s']
                    item['robot_time_advanced']=(prev_robot_time is None or rt>prev_robot_time)
                    stamp=item['source_header_stamp_s'];seq=item['source_header_seq']
                    item['header_stamp_advanced']=(prev_header_stamp is None or stamp>prev_header_stamp)
                    item['header_seq_advanced']=(prev_seq is None or 0 < (seq-prev_seq) % (2**32) < 2**31)
                    item['candidate_for_offline_analysis']=bool(item['source_is_fresh'] and item['robot_time_advanced'] and
                        item['header_stamp_advanced'] and item['header_seq_advanced'] and rt>0)
                    if prev_robot_time is not None and rt<prev_robot_time:
                        errors.append('robot_time moved backwards or robot restarted')
                    if prev_header_stamp is not None and stamp<prev_header_stamp:
                        errors.append('source header time moved backwards')
                    prev_robot_time=rt;prev_header_stamp=stamp;prev_seq=seq
                    if item['candidate_for_offline_analysis']:
                        valid+=1;last_candidate=item['receipt_monotonic_s']
                    last_receive=item['receipt_monotonic_s']
                f.write(json.dumps(item,allow_nan=False)+'\n')
                if count%50==0:f.flush()
                if errors:break
                if valid==0 and time.monotonic()>first_deadline:
                    errors.append('no candidate RobotState messages received');break
                if last_candidate is not None and time.monotonic()-last_candidate>a.silence_timeout:
                    errors.append('no advancing candidate RobotState samples');break
        except KeyboardInterrupt:
            errors.append('operator interrupted')
        finally:
            if rospy.is_shutdown():errors.append('ROS node shut down before collection completed')
            sub.unregister()
            summary={'kind':'end','records':count,'valid_records':valid,'errors':errors,
                     'elapsed_monotonic_s':time.monotonic()-start,
                     'last_receive_monotonic_s':last_receive}
            f.write(json.dumps(summary,allow_nan=False)+'\n');f.flush()
    print('Saved %s: %d records, %d marked valid. %s' % (path,count,valid,', '.join(errors)))
    return 0 if valid>0 and not errors else 1


if __name__=='__main__':raise SystemExit(main())
