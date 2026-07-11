#!/usr/bin/env python3

import sys

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from nav_search.action import ScanArea


class ArmSearchTestClient(Node):
    def __init__(self):
        super().__init__('arm_test')

        self.declare_parameter('start_angle', -3.1)
        self.declare_parameter('end_angle', 3.1)
        self.declare_parameter('num_steps', 10)
        self.declare_parameter('min_confidence', 0.6)

        self.start_angle = self.get_parameter('start_angle').value
        self.end_angle = self.get_parameter('end_angle').value
        self.num_steps = self.get_parameter('num_steps').value
        self.min_confidence = self.get_parameter('min_confidence').value

        self._client = ActionClient(self, ScanArea, 'scan_area')

    def send_goal(self):
        self.get_logger().info('Waiting for scan_area action server...')
        if not self._client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(
                'scan_area action server not available after 5s. '
                'Is arm_search_tracking running?'
            )
            rclpy.shutdown()
            sys.exit(1)

        goal_msg = ScanArea.Goal()
        goal_msg.start_angle = float(self.start_angle)
        goal_msg.end_angle = float(self.end_angle)
        goal_msg.num_steps = int(self.num_steps)
        goal_msg.min_confidence = float(self.min_confidence)

        self.get_logger().info(
            f'Sending ScanArea goal: start={goal_msg.start_angle:.2f} '
            f'end={goal_msg.end_angle:.2f} steps={goal_msg.num_steps} '
            f'min_conf={goal_msg.min_confidence:.2f}'
        )

        send_goal_future = self._client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('ScanArea goal was rejected by server')
            rclpy.shutdown()
            return

        self.get_logger().info('ScanArea goal accepted, waiting for result...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def feedback_callback(self, feedback_msg):
        fb = feedback_msg.feedback
        self.get_logger().info(
            f'Feedback: progress={fb.progress:.1f}% current_angle={fb.current_angle:.3f} rad'
        )

    def result_callback(self, future):
        result = future.result().result
        status = future.result().status

        # status codes: 4=SUCCEEDED, 5=CANCELED, 6=ABORTED (action_msgs/GoalStatus)
        if status == 4:
            self.get_logger().info(
                f'Scan SUCCEEDED: found={result.found} '
                f'x_base={result.x_base:.3f} y_base={result.y_base:.3f}'
            )
        elif status == 6:
            self.get_logger().error('Scan ABORTED')
        elif status == 5:
            self.get_logger().warn('Scan CANCELED')
        else:
            self.get_logger().error(f'Scan ended with status code {status}')

        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = ArmSearchTestClient()
    node.send_goal()
    rclpy.spin(node)


if __name__ == '__main__':
    main()