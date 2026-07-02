#!/usr/bin/env python3
import time
import cameratransform as ct
import numpy as np
import math
import rclpy
from rclpy.node import Node
from ensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
from moveit.planning import MoveItPy
from moveit.core.robot_state import RobotState


class ArmSearchNode(Node):
    def __init__(self):
        super().__init__("arm_search_node")

        # === Parameters (set these via YAML or command line) ===
        self.declare_parameter("planning_group", "interbotix_arm")
        self.declare_parameter("base_joint_name", "base_roll_joint")
        self.declare_parameter("start_angle", 0.0)     # rad
        self.declare_parameter("end_angle", 3.14)        # rad
        self.declare_parameter("num_steps", 10)
        self.declare_parameter("yolo_model_path", "yolov11n.pt")
        self.declare_parameter("conf_threshold", 0.4)

        self.planning_group = (
            self.get_parameter("planning_group").get_parameter_value().string_value
        )
        self.base_joint_name = (
            self.get_parameter("base_joint_name").get_parameter_value().string_value
        )
        self.start_angle = (
            self.get_parameter("start_angle").get_parameter_value().double_value
        )
        self.end_angle = (
            self.get_parameter("end_angle").get_parameter_value().double_value
        )
        self.num_steps = (
            self.get_parameter("num_steps").get_parameter_value().integer_value
        )
        self.camera_topic = (
            self.get_parameter("camera_topic").get_parameter_value().string_value
        )
        self.yolo_model_path = (
            self.get_parameter("yolo_model_path").get_parameter_value().string_value
        )
        self.conf_threshold = (
            self.get_parameter("conf_threshold").get_parameter_value().double_value
        )

        self.get_logger().info(f"Planning group: {self.planning_group}")
        self.get_logger().info(f"base joint: {self.base_joint_name}")
        self.get_logger().info(f"Camera topic: {self.camera_topic}")
        self.get_logger().info(f"YOLO model: {self.yolo_model_path}")

        # === YOLO model ===
        self.model = YOLO(self.yolo_model_path)

        # === MoveItPy setup ===
        self.moveit = MoveItPy(node_name="moveit_py_node")
        self.arm = self.moveit.get_planning_component(self.planning_group)
        self.robot_model = self.moveit.get_robot_model()

        #camera setup
        self.FOCAL_LENGTH = 13.6 # Focal length in mm
        self.IMAGE_SIZE = (640,512) # Image size in pixels
        self.SENSOR_SIZE = (7.68, 6.144)
        self.POS_X = 0 # x location of camera in meters (relative frame of reference for image info)
        self.POS_Y = 0
        ELEVATION = 1 # Camera elevation in meters
        TILT = 90 # Tilt angle in degrees, 0 is facing ground, 90 is parallel to ground, 180 is facing upward
        HEADING = 0
        ROLL = 0
        objheight = 0 

        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            self.get_logger().info("No camera is open")

        # Start search after a short delay to let everything connect
        self.search_started = False
        self.start_timer = self.create_timer(2.0, self.start_search_once)

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------


    def start_search_once(self):
        # Run the search loop once
        if self.search_started:
            return
        self.search_started = True
        self.get_logger().info("Starting arm search sweep...")
        self.search_loop()
    
    #def locate_target(self):
        #ELEVATION =  # Camera elevation in meters

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------
    def search_loop(self):
        # Generate sweep angles
        angles = np.linspace(self.start_angle, self.end_angle, self.num_steps)

        for angle in angles:
            self.get_logger().info(f"Moving base to angle {angle:.3f} rad")

            success = self.move_base_to(angle)
            if not success:
                self.get_logger().warn("Planning failed for this step, skipping.")
                continue

            # Allow a little time for vibration to settle and image to update
            rclpy.spin_once(self, timeout_sec=0.1)
            time.sleep(0.3)

          '''  ret, self.last_image = self.camera.read()

            if self.last_image is None:
                self.get_logger().warn("No camera image received yet, skipping YOLO.")
                continue

            # Run YOLO on the most recent image
            detected = self.run_yolo(self.last_image)

            if detected:
                self.get_logger().info(
                    f"Detection found at base angle {angle:.3f} rad. Stopping search."
                )
                break
'''
        self.get_logger().info("Search loop complete.")

    # ------------------------------------------------------------------
    # MoveIt helper
    # ------------------------------------------------------------------
    def move_base_to(self, target_angle: float) -> bool:
        """
        Plan and execute a motion where only the base joint is set to target_angle.
        Other joints follow current state (approximation; adjust as needed).
        """
        # Build a goal RobotState from the current model
        robot_state = RobotState(self.robot_model)
        robot_state.set_to_default_values()

        # Get current joint positions from the planning scene monitor
        # (fallback to default if that’s not hooked up)
        # NOTE: You can refine this by reading current state from planning_scene_monitor.
        joint_group = self.robot_model.get_joint_model_group(self.planning_group)
        joint_names = joint_group.get_variable_names()

        # Initialize with default or current values
        current_positions = {name: 0.0 for name in joint_names}
        # Only override the base joint
        if self.base_joint_name not in current_positions:
            self.get_logger().error(
                f"base joint '{self.base_joint_name}' not in group '{self.planning_group}'"
            )
            return False

        current_positions[self.base_joint_name] = target_angle
        robot_state.joint_positions = current_positions

        # Plan from current state to this goal
        self.arm.set_start_state_to_current_state()
        self.arm.set_goal_state(robot_state=robot_state)

        plan_result = self.arm.plan()
        if not plan_result:
            self.get_logger().warn("No plan found.")
            return False

        self.arm.execute(plan_result)
        return True

    # ------------------------------------------------------------------
    # YOLO helper
    # ------------------------------------------------------------------
    '''def run_yolo(self, image) -> bool:
        """
        Run YOLO on the given OpenCV BGR image.
        Returns True if any detection above threshold is found.
        """
        results = self.model(image)[0]
        boxes = results.boxes

        if boxes is None or len(boxes) == 0:
            self.get_logger().info("YOLO: no detections.")
            return False

        # Filter by confidence
        high_conf = [b for b in boxes if float(b.conf[0]) >= self.conf_threshold]
        if len(high_conf) == 0:
            self.get_logger().info("YOLO: detections below confidence threshold.")
            return False

        self.get_logger().info(f"YOLO: {len(high_conf)} detections above threshold.")
        # You can also inspect b.xyxy, b.cls, etc. here.
        return True
'''

def main(args=None):
    rclpy.init(args=args)
    node = ArmSearchNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()