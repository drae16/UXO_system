#!/usr/bin/env python3
"""
Standalone test for GPSNode.run_task_for():
  ScanArea -> go_prone -> Reconstruct -> stand_up -> enable_move

No Nav2 / GPS / fromLL needed. Runs the sequence once and exits.

Usage:
  ros2 run <your_pkg> run_task_tester.py
  ros2 run <your_pkg> run_task_tester.py --ros-args -p send_motion:=false
"""
import sys
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus

from go2_interfaces.msg import WebRtcReq
from nav_search.action import ScanArea, Reconstruct

SPORT_TOPIC = 'rt/api/sport/request'
API_ENABLE_MOVE = 1002
API_STAND_UP = 1004
API_PRONE = 1005


class RunTaskTester(Node):
    def __init__(self):
        super().__init__('run_task_tester')

        # Same defaults as the original node
        self.declare_parameter('start_angle', -3.14)
        self.declare_parameter('end_angle', 3.14)
        self.declare_parameter('num_steps', 10)
        self.declare_parameter('scan_min_confidence', 0.7)
        self.declare_parameter('reconstruct_min_confidence', 0.4)
        # Test controls
        self.declare_parameter('send_motion', True)       # False = log only, don't move the Go2
        self.declare_parameter('skip_reconstruct', False)
        self.declare_parameter('action_timeout', 0.0)   # sec; <= 0 waits forever

        self.scan_client = ActionClient(self, ScanArea, '/scan_area')
        self.reconstruct_client = ActionClient(self, Reconstruct, '/reconstruct_target')
        self.cmd_pub = self.create_publisher(WebRtcReq, '/webrtc_req', 10)

    def p(self, name):
        return self.get_parameter(name).value

    # ---------- action helper ----------
    def _run_action(self, client: ActionClient, goal, name: str):
        if not client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(f'{name} action server not available')
            return None

        send_future = client.send_goal_async(
            goal,
            feedback_callback=lambda fb: self.get_logger().debug(f'{name} feedback: {fb.feedback}'),
        )
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warn(f'{name} goal rejected')
            return None
        self.get_logger().info(f'{name} goal accepted')

        timeout = self.p('action_timeout')
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(
            self, result_future, timeout_sec=timeout if timeout > 0 else None
        )
        if not result_future.done():
            self.get_logger().error(f'{name} timed out after {timeout:.0f}s, canceling')
            cancel_future = goal_handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=3.0)
            return None

        wrapped = result_future.result()
        if wrapped.status != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().warn(f'{name} finished with status {wrapped.status}')
        return wrapped.result

    # ---------- same calls as GPSNavigator ----------
    def call_scan_area(self) -> Optional[ScanArea.Result]:
        goal = ScanArea.Goal()
        goal.start_angle = float(self.p('start_angle'))
        goal.end_angle = float(self.p('end_angle'))
        goal.num_steps = int(self.p('num_steps'))
        goal.min_confidence = float(self.p('scan_min_confidence'))

        result = self._run_action(self.scan_client, goal, 'ScanArea')
        if result is None:
            return None
        if result.found:
            self.get_logger().info(
                f'ScanArea: target found at base (x={result.x_base:.2f}, y={result.y_base:.2f})')
        else:
            self.get_logger().info('ScanArea: no target found')
        return result

    def call_3d_reconstruction(self) -> Optional[Reconstruct.Result]:
        goal = Reconstruct.Goal()
        goal.min_confidence = float(self.p('reconstruct_min_confidence'))

        result = self._run_action(self.reconstruct_client, goal, 'Reconstruct')
        if result is None:
            return None
        self.get_logger().info(
            'Reconstruction complete' if result.success else 'Reconstruction failed')
        return result

    # ---------- same motion commands as GPSNode ----------
    def _send_sport(self, api_id: int, label: str):
        if not self.p('send_motion'):
            self.get_logger().info(f'[dry run] {label} (api_id={api_id})')
            return
        # Wait briefly for the WebRTC bridge to discover us so the first msg isn't dropped
        deadline = time.time() + 2.0
        while self.cmd_pub.get_subscription_count() == 0 and time.time() < deadline:
            time.sleep(0.05)
        if self.cmd_pub.get_subscription_count() == 0:
            self.get_logger().warn(f'No subscribers on /webrtc_req, {label} may be lost')

        msg = WebRtcReq()
        msg.api_id = api_id
        msg.topic = SPORT_TOPIC
        self.cmd_pub.publish(msg)
        self.get_logger().info(label)

    def go_prone(self):
        self._send_sport(API_PRONE, 'Going prone')

    def stand_up(self):
        self._send_sport(API_STAND_UP, 'Standing up')

    def enable_move(self):
        self._send_sport(API_ENABLE_MOVE, 'Ready to move')

    # ---------- mirror of run_task_for ----------
    def run_task(self) -> bool:
        self.get_logger().info('Running task (test, no waypoint)')

        scan = self.call_scan_area()
        if scan is None or not scan.found:
            return False

        self.go_prone()
        time.sleep(1.0)

        if self.p('skip_reconstruct'):
            self.get_logger().info('Skipping reconstruction (skip_reconstruct=true)')
            construct_ok = True
        else:
            construct = self.call_3d_reconstruction()
            construct_ok = construct is not None and construct.success

        # Original stands up whenever a result comes back; stand regardless here
        # so a failed reconstruct doesn't leave the dog lying down.
        self.stand_up()
        time.sleep(2.0)
        self.enable_move()
        return construct_ok


def main(args=None):
    rclpy.init(args=args)
    node = RunTaskTester()
    ok = False
    try:
        ok = node.run_task()
        node.get_logger().info(f'Task test finished: {"SUCCESS" if ok else "FAILED"}')
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()