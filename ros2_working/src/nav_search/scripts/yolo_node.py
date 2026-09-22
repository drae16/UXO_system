#!/usr/bin/env python3

import time
import math
import os
import threading
import datetime
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from sensor_msgs.msg import Image, CompressedImage
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


SFM_START = "sfm_start"
SFM_STOP = "sfm_stop"


class YoloDetectNode(Node):
    def __init__(self):
        super().__init__("yolo_node")

        model_path = os.path.join(
        get_package_share_directory("nav_search"),
        "models",
        "pipes.pt"
        )
        self.model = YOLO(model_path)

        self.date_time = datetime.datetime.now()
        self.save_path = f"/home/drl/Data/3D_const_images/{self.date_time}"
        self.img_num = 0
        
    
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

                # ---- capture at full sensor resolution, held for the whole run ----
        # SfM gets these full-res frames; detection resizes down to IMAGE_SIZE.
        # Set these to your sensor's true max supported MJPG mode, and keep the
        # aspect ratio ~4:3 to match the calibration (IMAGE_SIZE). Verify against
        # the actual resolution printed below — the driver silently substitutes
        # a mode it supports if the request isn't available.
        self.CAPTURE_WIDTH = 2048
        self.CAPTURE_HEIGHT = 1536
        self.IMAGE_SIZE = (2048,1536)

        self.CAPTURE_WIDTH_2 = 640
        self.CAPTURE_HEIGHT_2 = 512
        self.IMAGE_SIZE_2 = (640,512)

        self.camera = cv.VideoCapture(2)
        self.camera2 = cv.VideoCapture(4)
        self.eo_cam = None
        self.ir_cam = None

        try:
            self.camera.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*"MJPG"))
            self.camera.set(cv.CAP_PROP_FRAME_WIDTH,  self.CAPTURE_WIDTH)
            self.camera.set(cv.CAP_PROP_FRAME_HEIGHT, self.CAPTURE_HEIGHT)
            self.camera.set(cv.CAP_PROP_BUFFERSIZE, 1)
            self.eo_cam = self.camera
        except:
            self.camera.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*"MJPG"))
            self.camera.set(cv.CAP_PROP_FRAME_WIDTH,  self.CAPTURE_WIDTH_2)
            self.camera.set(cv.CAP_PROP_FRAME_HEIGHT, self.CAPTURE_HEIGHT_2)
            self.camera.set(cv.CAP_PROP_BUFFERSIZE, 1)
            self.ir_cam = self.camera

        if self.eo_cam == None:
            self.camera2.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*"MJPG"))
            self.camera2.set(cv.CAP_PROP_FRAME_WIDTH,  self.CAPTURE_WIDTH)
            self.camera2.set(cv.CAP_PROP_FRAME_HEIGHT, self.CAPTURE_HEIGHT)
            self.camera2.set(cv.CAP_PROP_BUFFERSIZE, 1)
            self.eo_cam = self.camera2
        else:
            self.camera2.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*"MJPG"))
            self.camera2.set(cv.CAP_PROP_FRAME_WIDTH,  self.CAPTURE_WIDTH_2)
            self.camera2.set(cv.CAP_PROP_FRAME_HEIGHT, self.CAPTURE_HEIGHT_2)
            self.camera2.set(cv.CAP_PROP_BUFFERSIZE, 1)
            self.ir_cam = self.camera2



        actual_w = self.eo_cam.get(cv.CAP_PROP_FRAME_WIDTH)
        actual_h = self.eo_cam.get(cv.CAP_PROP_FRAME_HEIGHT)
        self.get_logger().info(f"Capture resolution: {actual_w} x {actual_h}")

        # warn if the driver gave a non-4:3 mode -- the detection resize would
        # then distort and break the calibrated pixel model.


        self.img_pub = self.create_publisher(CompressedImage, "sfm_img", 10)

        self.ir_pub = self.create_publisher(CompressedImage, "ir_img", 10)

        self.server = ActionServer(
            self,
            DetectTarget,
            "detect_target",
            self.execute_cb,
        )

        self.FOCAL_LENGTH = 1626.56 #1016.6 #508.3
        self.lens = ct.BrownLensDistortion(0.0510, -0.386, 0.0)
        self.POS_X = 0 # x location of camera in meters (relative frame of reference for image info)
        self.POS_Y = 0
        self.ELEVATION = None # Camera elevation in meters
        self.TILT = 90 # Tilt angle in degrees, 0 is facing ground, 90 is parallel to ground, 180 is facing upward
        self.HEADING = 0
        self.ROLL = 0
        self.objheight = 0
        self.BLUR_THRESHOLD = 900.0
        self.SHARP_TIMEOUT = 5.0 # seconds to wait for a sharp frame before giving up


        os.mkdir(self.save_path)

    def is_blurry(self, image, threshold=1000.0):
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
        # read frames on demand and return the first that passes the blur test.
        # on timeout, return the sharpest frame seen so far (None if none arrived).
        deadline = time.time() + self.SHARP_TIMEOUT
        best_img = None
        best_var = -1.0
        while time.time() < deadline:
            ret, frame = self.eo_cam.read()
            ret, frame = self.eo_cam.read()
            if not ret:
                time.sleep(0.005)
                continue
            _, blur, var = self.is_blurry(frame, self.BLUR_THRESHOLD)
            if not blur:
                return frame
            if var > best_var:
                best_var = var
                best_img = frame
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

    def get_ir_image(self):
        ret, frame = self.ir_cam.read()

        if ret:
            return frame
        
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

    # ---- Dispatch on goal.mode: "move" | "cal" | "img" ----
    def execute_cb(self, goal_handle):
        mode = goal_handle.request.mode
        self.get_logger().info(f"execute_cb mode='{mode}'")

        if mode == "move":
            return self._do_move(goal_handle)
        elif mode == "cal":
            return self._do_cal(goal_handle)
        elif mode == "img":
            return self._do_img(goal_handle)
        elif mode == "start":
            return self._do_start(goal_handle)
        elif mode == "stop":
            return self._do_stop(goal_handle)
        else:
            self.get_logger().error(f"unknown mode '{mode}'")
            result = self._blank_result()
            goal_handle.abort()
            return result

    # Union result with every field zeroed; each mode fills what it needs.
    def _blank_result(self):
        result = DetectTarget.Result()
        result.found = False
        result.x_base = 0.0
        result.y_base = 0.0
        result.confidence = 0.0
        result.coverage = 0.0
        result.near_x1 = 0.0
        result.near_y1 = 0.0
        result.near_x2 = 0.0
        result.near_y2 = 0.0
        return result

    # ---- move: metric detection + nearest corners (arm base frame) ----
    def _do_move(self, goal_handle):
        min_conf = goal_handle.request.min_confidence
        feedback = DetectTarget.Feedback()
        feedback.progress = 0.0
        goal_handle.publish_feedback(feedback)

        result = self._blank_result()

        img = self.get_sharp_image()
        if img is None:
            self.get_logger().info(f"failure")
            goal_handle.abort()
            return result

        self.get_logger().info(f"image received")


        results = self.model.predict(img, conf=min_conf, show=False)[0]        
        boxes = results.boxes

        if boxes is None or len(boxes) == 0:
            goal_handle.succeed()
            return result

        best = None
        best_conf = 0.0
        for b in boxes:
            conf = float(b.conf[0])
            if conf > min_conf and conf > best_conf:
                best = b
                best_conf = conf

        if best is None:
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
            result.confidence = best_conf
            goal_handle.succeed()
            return result

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

    # ---- cal: pixel center + frame coverage ----
    def _do_cal(self, goal_handle):
        min_conf = goal_handle.request.min_confidence
        feedback = DetectTarget.Feedback()
        feedback.progress = 0.0
        goal_handle.publish_feedback(feedback)

        result = self._blank_result()

        img = self.get_sharp_image()
        if img is None:
            self.get_logger().info(f"failure")
            goal_handle.abort()
            return result

        self.get_logger().info(f"image received")


        results = self.model.predict(img, conf=min_conf, show=True)[0]        
        boxes = results.boxes

        if boxes is None or len(boxes) == 0:
            goal_handle.succeed()
            return result

        best = None
        best_conf = 0.0
        for b in boxes:
            conf = float(b.conf[0])
            if conf > min_conf and conf > best_conf:
                best = b
                best_conf = conf

        if best is None:
            goal_handle.succeed()
            return result

        x_min, y_min, x_max, y_max = best.xyxy[0].tolist()
        w_px, h_px = self.IMAGE_SIZE
        coverage = ((x_max - x_min) * (y_max - y_min)) / (w_px * h_px)
        print("PIxel Coverage:",(x_max - x_min) ,(y_max - y_min))
        u = (x_min + x_max) / 2.0
        v = (y_min + y_max) / 2.0

        result.found = True
        result.x_base = float(u)          # pixel u (see .action note)
        result.y_base = float(v)          # pixel v
        result.confidence = best_conf
        result.coverage = float(coverage)

        self.get_logger().info(f"target in frame, center at u= {u}, v = {v}")
        self.get_logger().info(f"Target consuming {coverage} of frame")

        feedback.progress = 100.0
        goal_handle.publish_feedback(feedback)
        goal_handle.succeed()
        return result

    # ---- img: grab a sharp frame, PNG-compress, publish on sfm_img ----
        # ---- img: grab a sharp frame, PNG-compress, publish on sfm_img ----
    def _do_img(self, goal_handle):
        feedback = DetectTarget.Feedback()
        feedback.progress = 0.0
        goal_handle.publish_feedback(feedback)

        result = self._blank_result()

        img = self.get_sharp_image()
        ir_img = self. get_ir_image()



        if img is None:
            self.get_logger().info("img: no sharp frame")
            goal_handle.abort()
            return result

        seq = self.img_num  # index for THIS frame; advanced only on success
        cv.imwrite(f"{self.save_path}/img{seq}.png", img)
    

        ok, buf = cv.imencode(".png", img)

        ir_ok, buf_ir = cv.imencode(".png",ir_img)

        if not ok:
            self.get_logger().error("img: PNG encode failed")
            goal_handle.abort()
            return result

        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = str(seq)
        msg.format = "png"
        msg.data = buf.tobytes()
        self.img_pub.publish(msg)


        msg_ir = CompressedImage()
        msg_ir.header.stamp = self.get_clock().now().to_msg()
        msg_ir.header.frame_id = str(seq)
        msg_ir.format = "png"
        msg_ir.data = buf_ir.tobytes()
        self.ir_pub.publish(msg_ir)

        self.img_num += 1
        result.found = True
        self.get_logger().info(f"img: published seq={seq}, {len(msg.data)} bytes on sfm_img")

        feedback.progress = 100.0
        goal_handle.publish_feedback(feedback)
        goal_handle.succeed()
        return result
        # ---- start: reset the per-target counter, signal a new collection ----
    def _do_start(self, goal_handle):
        self.img_num = 0
        self._publish_control(SFM_START)
        result = self._blank_result()
        result.found = True
        self.get_logger().info("start: published sfm_start on sfm_img")
        goal_handle.succeed()
        return result

    # ---- stop: signal end of collection so the GCS reconstructs ----
    def _do_stop(self, goal_handle):
        self._publish_control(SFM_STOP)
        result = self._blank_result()
        result.found = True
        self.get_logger().info("stop: published sfm_stop on sfm_img")
        goal_handle.succeed()
        return result

    # blank control frame; the GCS branches on frame_id before touching data.
    def _publish_control(self, frame_id):
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.format = "png"
        self.img_pub.publish(msg)

    def destroy_node(self):
        self.running = False
        self.eo_cam.release()
        self.ir_cam.release()
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