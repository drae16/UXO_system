
#!/usr/bin/env python3
import math, time, threading
from collections import deque
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from rclpy.action import ActionClient
from go2_interfaces.msg import WebRtcReq
from geometry_msgs.msg import PoseStamped, Quaternion, Vector3
from nav2_msgs.action import NavigateToPose
from robot_localization.srv import FromLL
from std_msgs.msg import Empty
from nav_search.action import ScanArea 

def yaw_to_quat(yaw: float) -> Quaternion:
    q = Quaternion()
    q.w = math.cos(yaw * 0.5)
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw * 0.5)
    return q


class GPSNavigator:
    """Navigator wrapper that uses a raw ActionClient (no BasicNavigator)."""

    def __init__(self, node: Node):
        self.node = node
        self.client = ActionClient(node, NavigateToPose, "navigate_to_pose")
        self.scan_client = ActionClient(node, ScanArea, '/scan_area')
        # Ensure action server is up
        while not self.client.wait_for_server(timeout_sec=1.0):
            node.get_logger().info("Waiting for NavigateToPose action server...")

        self.current_goal_handle = None
        self._lock = threading.Lock()

    def go_to_pose(self, pose: PoseStamped) -> bool:
        """Send a NavigateToPose goal and wait for result (blocking)."""
        goal = NavigateToPose.Goal()

        goal.pose = pose

        send_future = self.client.send_goal_async(goal, feedback_callback=self._on_feedback)
       
        while rclpy.ok() and not send_future.done():
            time.sleep(0.01)

        goal_handle = send_future.result()
        if not goal_handle or not goal_handle.accepted:
            self.node.get_logger().warn("Goal rejected")
            return False

        with self._lock:
            self.current_goal_handle = goal_handle

       
        result_future = goal_handle.get_result_async()
        while rclpy.ok() and not result_future.done():
            time.sleep(0.1)

        result = result_future.result()
        with self._lock:
            self.current_goal_handle = None

        if result and result.status == 4:  # GoalStatus.STATUS_SUCCEEDED
            return True
        return False

    def cancel(self):
        with self._lock:
            if self.current_goal_handle:
                self.node.get_logger().info("Canceling current goal")
                fut = self.current_goal_handle.cancel_goal_async()
                while rclpy.ok() and not fut.done():
                    time.sleep(0.01)

    def call_scan_area(self) -> ScanArea.Result | None:
        # Wait for server
        if not self.scan_client.wait_for_server(timeout_sec=5.0):
            self.node.get_logger().error('ScanArea action server not available')
            return None

        goal = ScanArea.Goal()
        goal.start_angle = -1.4 # -0.5      # or whatever you want
        goal.end_angle   =  1.4 #0.5
        goal.num_steps   = 4
        goal.min_confidence = 0.6

        # Send goal
        send_future = self.scan_client.send_goal_async(goal)

        # Busy-wait like you do for Nav2
        while rclpy.ok() and not send_future.done():
            time.sleep(0.01)

        goal_handle = send_future.result()
        if not goal_handle or not goal_handle.accepted:
            self.node.get_logger().warn('ScanArea goal was rejected')
            return None

        # Wait for result
        result_future = self.scan_client._get_result_async(goal_handle)
        while rclpy.ok() and not result_future.done():
            time.sleep(0.05)

        wrapped_result = result_future.result()
        result = wrapped_result.result

        if result is None:
            self.node.get_logger().warn('ScanArea returned no result message')
            return None

        if result.found:
            self.node.get_logger().info(
                f'ScanArea: target found at base (x={result.x_base:.2f}, y={result.y_base:.2f})'
            )
        else:
            self.node.get_logger().info('ScanArea: no target found')

        return result

    def _on_feedback(self, feedback_msg):
        fb = feedback_msg.feedback
        try:
            d = fb.distance_remaining
            self.node.get_logger().info(f"Distance remaining: {d:.2f}")
            time.sleep(2)
        except Exception:
            pass


class GPSNode(Node):


    def __init__(self):
        super().__init__("gps_node")

        # Service client
        self.localizer = self.create_client(FromLL, "/fromLL")
        while not self.localizer.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Waiting for /fromLL ...")

        qos = QoSProfile(
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
        )
        self.sub = self.create_subscription(Vector3, "gps_targets", self._cb_target, qos)

        self.publisher_command = self.create_publisher(WebRtcReq, '/webrtc_req', 10)

        self.publisher_completion = self.create_publisher(Empty, 'input_at_waypoint/input',10)


        self.queue = deque()
        self._lock = threading.Lock()

        # Navigator wrapper
        self.navigator = GPSNavigator(self)

        # Worker thread
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

    def _cb_target(self, msg: Vector3):
        with self._lock:
            self.queue.append((msg.x, msg.y, msg.z))  # (lat, lon, yaw in radians)
        self.get_logger().info(f"Queued waypoint: {msg.x}, {msg.y}, {msg.z}")

    def _worker_loop(self):
        while rclpy.ok():
            if not self.queue:
                time.sleep(0.1)
                continue

            with self._lock:
                lat, lon, yaw = self.queue.popleft()

            pose = self._convert_gps(lat, lon, yaw)
            
            if pose is None:
                continue

            ok = self.navigator.go_to_pose(pose)
            if ok:
                self.get_logger().info("Reached goal, running task...")
                self.run_task_for(pose)  
            else:
                self.get_logger().warn("Navigation failed/canceled")

    def _convert_gps(self, lat: float, lon: float, yaw: float) -> Optional[PoseStamped]:
        req = FromLL.Request()
        req.ll_point.latitude = lat
        req.ll_point.longitude = lon
        req.ll_point.altitude = 0.0

        fut = self.localizer.call_async(req)
        while rclpy.ok() and not fut.done():
            time.sleep(0.01)

        resp = fut.result()
        if not resp:
            self.get_logger().error("fromLL failed")
            return None

        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position = resp.map_point
        pose.pose.orientation = yaw_to_quat(yaw)
        return pose
    
    def stretch(self):
        msg = WebRtcReq()
        msg.api_id = 1017
        msg.topic = 'rt/api/sport/request'
        self.publisher_command.publish(msg)
        self.get_logger().info('Performing stretch')

    def run_task_for(self,pose: PoseStamped):
        x = pose.pose.position.x
        y = pose.pose.position.y
        self.get_logger().info(f'Running task at waypoint ({x:.2f}, {y:.2f})')

        scan = self.navigator.call_scan_area()
        time.sleep(2)

    time.sleep(8)

    

def main():
    rclpy.init()
    node = GPSNode()
    try:
        exec = MultiThreadedExecutor()
        exec.add_node(node)
        exec.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
