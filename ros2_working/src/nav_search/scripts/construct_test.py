#!/usr/bin/env python3
import sys
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav_search.action import Reconstruct


class ReconstructTestClient(Node):
    def __init__(self, min_confidence):
        super().__init__("reconstruct_test_client")
        self.min_confidence = min_confidence
        self.client = ActionClient(self, Reconstruct, "reconstruct_target")

    def send_goal(self):
        if not self.client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("reconstruct_target action server not available")
            rclpy.shutdown()
            return

        goal_msg = Reconstruct.Goal()
        goal_msg.min_confidence = self.min_confidence

        self.get_logger().info(
            f"Sending reconstruct goal: min_confidence={self.min_confidence:.2f}")

        send_future = self.client.send_goal_async(
            goal_msg, feedback_callback=self.on_feedback)
        send_future.add_done_callback(self.on_goal_response)

    def on_goal_response(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Goal was rejected by the server")
            rclpy.shutdown()
            return
        self.get_logger().info("Goal accepted; waiting for result...")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.on_result)

    def on_feedback(self, feedback_msg):
        fb = feedback_msg.feedback
        self.get_logger().info(f"Feedback: progress={fb.progress:.1f}")

    def on_result(self, future):
        result = future.result().result
        self.get_logger().info(f"Result: success={result.success}")
        rclpy.shutdown()   # one-shot: done after the result arrives


def main(args=None):
    rclpy.init(args=args)

    # Optional first arg: min_confidence (default 0.7 to match calibration usage).
    min_conf = 0.4
    if len(sys.argv) > 1:
        try:
            min_conf = float(sys.argv[1])
        except ValueError:
            print(f"Could not parse min_confidence '{sys.argv[1]}', using default 0.7")

    node = ReconstructTestClient(min_conf)
    node.send_goal()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()