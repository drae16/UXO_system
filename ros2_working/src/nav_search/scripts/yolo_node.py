#!/usr/bin/env python3
import time
import math
import os
import threading
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from sensor_msgs.msg import Image
from tf2_ros import Buffer, TransformListener
from nav_search.action import DetectTarget
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory
import cv2 as cv
import cameratransform as ct
import numpy as np


def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
        return roll_x, pitch_y, yaw_z # in radians


class YoloDetectNode(Node):
    def __init__(self):
        super().__init__("yolo_node")

        model_path = os.path.join(
        get_package_share_directory("nav_search"),
        "models",
        "pipes.pt"
        )
        self.model = YOLO(model_path)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.camera = cv.VideoCapture(2)
        self.camera.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*"MJPG"))
        self.camera.set(cv.CAP_PROP_FRAME_WIDTH,  640)
        self.camera.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv.CAP_PROP_BUFFERSIZE, 1)

        actual_w = self.camera.get(cv.CAP_PROP_FRAME_WIDTH)
        actual_h = self.camera.get(cv.CAP_PROP_FRAME_HEIGHT)
        self.get_logger().info(f"Resolution: {actual_w} x {actual_h}")

        # writer records every frame from the camera thread. size must match the
        # frames actually returned or write() silently drops them -> empty file.
        self.fourcc = cv.VideoWriter_fourcc(*"mp4v")
        self.out = cv.VideoWriter("/home/drl/output.mp4", self.fourcc, 20.0,
                                  (int(actual_w), int(actual_h)))

        self.last_image = None

        # camera thread owns the capture: the only place read()/write() happen.
        self.running = True
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()

        self.server = ActionServer(
            self,
            DetectTarget,
            "detect_target",
            self.execute_cb,
        )

        self.FOCAL_LENGTH = 508.3
        self.IMAGE_SIZE = (640,480) # Image size in pixels
        self.lens = ct.BrownLensDistortion(0.0510, -0.386, 0.0)
        self.POS_X = 0 # x location of camera in meters (relative frame of reference for image info)
        self.POS_Y = 0
        self.ELEVATION = None # Camera elevation in meters
        self.TILT = 90 # Tilt angle in degrees, 0 is facing ground, 90 is parallel to ground, 180 is facing upward
        self.HEADING = 0
        self.ROLL = 0
        self.objheight = 0
        self.BLUR_THRESHOLD = 800.0
        self.SHARP_TIMEOUT = 1.0 # seconds to wait for a sharp frame before giving up

    def camera_loop(self):
        # continuously drain the camera so last_image is always current and the
        # recording gets every frame. rebind is atomic, so no lock needed.
        while self.running:
            ret, frame = self.camera.read()
            if not ret:
                continue
            self.last_image = frame
            self.out.write(frame)

    def is_blurry(self, image, threshold=800.0):
        """
        Detect if an image is blurry using the Laplacian variance method.

        Args:
            image (numpy.ndarray): The input image.
            threshold (float): Variance threshold below which the image is considered blurry.

        Returns:
            bool: True if the image is blurry, False otherwise.
            float: The variance of the Laplacian.
        """
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        laplacian = cv.Laplacian(gray, cv.CV_64F)
        variance = laplacian.var()
        return laplacian, variance < threshold, variance

    def get_sharp_image(self):
        # pull frames from the stream and return the first one that passes the
        # blur test. returns immediately if the current frame is already crisp.
        # on timeout, return the sharpest frame seen so far (None if none arrived).
        deadline = time.time() + self.SHARP_TIMEOUT
        best_img = None
        best_var = -1.0
        while time.time() < deadline:
            frame = None if self.last_image is None else self.last_image.copy()
            if frame is None:
                time.sleep(0.005)
                continue
            _, blur, var = self.is_blurry(frame, self.BLUR_THRESHOLD)
            if not blur:
                return frame
            if var > best_var:
                best_var = var
                best_img = frame
            time.sleep(0.005) # let the camera thread post a new frame
        return best_img


    # Inputs:
    #   pos - an array [x,y] of the center of the object, in pixels. [0,0] is at top left of image.
    #       x and y are # of pixels right and down, respectively
    #   pitch, roll, yaw - pitch, roll, yaw of the camera in degrees
    #   knownParamType - the known parameter of the object
            # Acceptable inputs: "Z", "D"
    #   knownParamValue - The value of the known parameter, in meters
    # Outputs: [x,y,z] of the object (ndarray)
    #   (0,0,0) is the point directly underneath the camera (i.e. z=0 is ground)
    # Note: HEADING for camera should be compass heading (world frame), clockwise is positive, 0 degrees faces north
    #   Then, x and y from the output tell you how far east and north the object is from the camera, respectively
    def spatial_transformation(self, points, knownParamType, knownParamValue):
        # points: either a single [u, v] or a list of [u, v] points
        cam = ct.Camera(ct.RectilinearProjection(focallength_px=self.FOCAL_LENGTH,
                                                 image=self.IMAGE_SIZE),
                        lens=self.lens,
                        orientation=ct.SpatialOrientation(pos_x_m=self.POS_X,
                                               pos_y_m=self.POS_Y,
                                               elevation_m=self.ELEVATION,
                                               tilt_deg=self.TILT,
                                               roll_deg=self.ROLL,
                                               heading_deg=self.HEADING))

        pts = np.array(points)

        if knownParamType == "Z":
            objectPos = cam.spaceFromImage(pts, Z=knownParamValue)
        elif knownParamType == "D":
            objectPos = cam.spaceFromImage(pts, D=knownParamValue)
        else:
            objectPos = np.full(pts.shape[:-1] + (3,), -999.0)

        return objectPos


    def execute_cb(self, goal_handle):
        min_conf = goal_handle.request.min_confidence
        feedback = DetectTarget.Feedback()
        feedback.progress = 0.0
        goal_handle.publish_feedback(feedback)

        img = self.get_sharp_image()

        if img is None:
            self.get_logger().info(f"failure")
            result = DetectTarget.Result()
            result.found = False
            result.x_base = 0.0
            result.y_base = 0.0
            result.confidence = 0.0
            goal_handle.abort()
            return result

        self.get_logger().info(f"image received")

        # 1) YOLO detect
        results = self.model.predict(img,conf=0.7,show=True)[0]
        boxes = results.boxes

        if boxes is None or len(boxes) == 0:
            result = DetectTarget.Result()
            result.found = False
            result.x_base = 0.0
            result.y_base = 0.0
            result.confidence = 0.0
            goal_handle.succeed()
            return result

        # pick best box above threshold
        best = None
        best_conf = 0.0
        for b in boxes:
            conf = float(b.conf[0])
            if conf > min_conf and conf > best_conf:
                best = b
                best_conf = conf

        result = DetectTarget.Result()
        if best is None:
            result.found = False
            result.x_base = 0.0
            result.y_base = 0.0
            result.confidence = 0.0
            goal_handle.succeed()
            return result

        try:
            tf = self.tf_buffer.lookup_transform(
                "base_footprint", "vx300s/camera_link", rclpy.time.Time())
            q = tf.transform.rotation
            position = tf.transform.translation
            qx, qy, qz, qw = q.x, q.y, q.z, q.w

            roll, tilt, heading = euler_from_quaternion(qx, qy, qz, qw)
            self.ROLL = roll * 180/math.pi
            self.TILT = 90 - (tilt *180/math.pi) -2
            self.HEADING= heading *-180/math.pi
            self.ELEVATION = position.z
            self.POS_X = -position.y
            self.POS_Y = position.x

        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            result.found = False
            result.x_base = 0.0
            result.y_base = 0.0
            result.confidence = best_conf
            goal_handle.succeed()
            return result

        # 3) compute (x, y) of target
        self.get_logger().info(f"camera pos = z={self.ELEVATION}, tilt = {self.TILT} , heading = {self.HEADING}")
        self.get_logger().info(f"camera pos = height ={self.ELEVATION}, X = {self.POS_Y}, Y = {-self.POS_X}")
        x_min, y_min, x_max, y_max = best.xyxy[0].tolist()
        u = (x_min + x_max) / 2.0
        v = (y_min + y_max) / 2.0

        all_points_px = [
            [u, v],           # center
            [x_min, y_max],   # bottom-left
            [x_max, y_max],   # bottom-right
            [x_min, y_min],   # top-left
            [x_max, y_min],   # top-right
        ]

        self.get_logger().info(f"camera pos = z={self.ELEVATION}, tilt = {self.TILT}, heading = {self.HEADING}")

        all_positions = self.spatial_transformation(all_points_px, "Z", 0)


        x_base, y_base, z_base = all_positions[0]
        result.found = True
        result.x_base = float(y_base)
        result.y_base = float(-1 * x_base)
        dist = math.sqrt(result.x_base**2 + result.y_base**2)
        self.get_logger().info(f"Coordinates found x= {result.x_base},y = {result.y_base},z= {z_base}")
        self.get_logger().info(f"Coordinates found distance= {dist}")
        result.confidence = best_conf


        corner_candidates = []
        for (cx_base, cy_base, cz_base) in all_positions[1:]:
            out_x = float(cy_base)
            out_y = float(-1 * cx_base)
            c_dist = math.sqrt(out_x**2 + out_y**2)
            corner_candidates.append((c_dist, out_x, out_y))

        corner_candidates.sort(key=lambda c: c[0])

        result.near_x1 = corner_candidates[0][1]
        result.near_y1 = corner_candidates[0][2]
        result.near_x2 = corner_candidates[1][1]
        result.near_y2 = corner_candidates[1][2]
        self.get_logger().info(
            f"Nearest corners: ({result.near_x1:.3f},{result.near_y1:.3f}) dist={corner_candidates[0][0]:.3f}, "
            f"({result.near_x2:.3f},{result.near_y2:.3f}) dist={corner_candidates[1][0]:.3f}"
        )

        feedback.progress = 100.0
        goal_handle.publish_feedback(feedback)
        goal_handle.succeed()
        return result

    def destroy_node(self):
        self.running = False
        self.camera_thread.join(timeout=2.0)
        self.out.release() 
        self.camera.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()