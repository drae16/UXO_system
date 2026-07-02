import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav_search.action import ScanArea 


class ArmTest(Node):
    def __init__(self):
        super().__init__("gps_node")
        self.test_client = ActionClient(self, ScanArea, '/scan_area')
        while not self.test_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info('Waiting for scan client to start..')
        
    def call_scan_area(self):
        if not self.test_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Scan action not available')
            return None
        
        goal = ScanArea.Goal()
        goal.start_angle = -3.14
        goal.end_angle = 3.14
        goal.num_steps = 10
        goal.min_confidence = 0.6

        send_future = self.test_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)

        goal_handle = send_future.result()
        if not goal_handle or not goal_handle.accepted:
            self.get_logger().warn('ScanArea goal was rejected')
            return None

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        wrapped_result = result_future.result()
        if wrapped_result is None or wrapped_result.result is None:
            self.get_logger().warn('ScanArea returned no result message')
            return None

        result = wrapped_result.result

        if result.found:
            self.get_logger().info(
                f'ScanArea: target found at base (x={result.x_base:.2f}, y={result.y_base:.2f})'
            )
        else:
            self.get_logger().info('ScanArea: no target found')

        return result
    

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = ArmTest()
        node.call_scan_area()
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()