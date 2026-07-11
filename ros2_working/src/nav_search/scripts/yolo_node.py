#!/usr/bin/env python3
import time
import math
import os
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
import math
 
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

        self.camera = cv.VideoCapture(2)


        # verify it actually set correctly — the camera may not support the resolutdistanceion
        actual_w = self.camera.get(cv.CAP_PROP_FRAME_WIDTH)
        actual_h = self.camera.get(cv.CAP_PROP_FRAME_HEIGHT)

        self.get_logger().info(f"Resolution: {actual_w} x {actual_h}")
        self.camera.set(cv.CAP_PROP_BUFFERSIZE, 1)

        self.last_image = None

        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.server = ActionServer(
            self,
            DetectTarget,
            "detect_target",
            self.execute_cb,
        )

        self.FOCAL_LENGTH = 453.4608 #flir values 13.6 # Focal length in mm
        self.IMAGE_SIZE = (640,480) # Image size in pixels
        self.POS_X = 0 # x location of camera in meters (relative frame of reference for image info)
        self.POS_Y = 0
        self.ELEVATION = None # Camera elevation in meters
        self.TILT = 90 # Tilt angle in degrees, 0 is facing ground, 90 is parallel to ground, 180 is facing upward
        self.HEADING = 0
        self.ROLL = 0
        self.objheight = 0 

    def image_cb(self):
        time.sleep(0.4)
        #reading twice to clear buffer and get most recent frame
        ret,self.last_image = self.camera.read()
        ret,self.last_image = self.camera.read()



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
    def spatial_transformation(self,pos,knownParamType, knownParamValue):
        # Declare a camera object
                # Camera object used for transformations
        cam = ct.Camera(ct.RectilinearProjection(focallength_px=self.FOCAL_LENGTH, 
                                                 image=self.IMAGE_SIZE),
                        
                         ct.SpatialOrientation(pos_x_m = self.POS_X,
                                               pos_y_m = self.POS_Y,
                                               elevation_m=self.ELEVATION, 
                                               tilt_deg = self.TILT, 
                                               roll_deg = self.ROLL, 
                                               heading_deg = self.HEADING))

        # Do the transform
        if knownParamType == "Z": # Height is known
            objectPos = cam.spaceFromImage(pos, Z=knownParamValue)
        elif knownParamType == "D": # Distance is known
            objectPos = cam.spaceFromImage(pos, D=knownParamValue)
        else:
            # Error
            objectPos = np.array([-999, -999, -999])
        
        return objectPos


    def execute_cb(self, goal_handle):
        min_conf = goal_handle.request.min_confidence
        feedback = DetectTarget.Feedback()
        feedback.progress = 0.0
        goal_handle.publish_feedback(feedback)

        self.image_cb()

        if self.last_image is None:
            self.get_logger().info(f"failure")
            result = DetectTarget.Result()
            result.found = False
            result.x_base = 0.0
            result.y_base = 0.0
            result.confidence = 0.0
            goal_handle.abort()
            return result

        img = self.last_image.copy()

        self.get_logger().info(f"image received")


        
        # 1) YOLO detect
        results = self.model.predict(img,conf=0.7,show=False)[0]
        boxes = results.boxes

        #remove later

        try:
            tf = self.tf_buffer.lookup_transform(
                "base_footprint", "vx300s/camera_link", rclpy.time.Time())
            q = tf.transform.rotation
            position = tf.transform.translation
            qx, qy, qz, qw = q.x, q.y, q.z, q.w

            roll, tilt, heading = euler_from_quaternion(qx, qy, qz, qw)
            ROLL = roll * 180/math.pi
            TILT = 90 - (tilt *180/math.pi)
            HEADING= heading *180/math.pi 
            X_POS = position.x
            Y_POS = position.y
            self.get_logger().info(f"camera pos = height ={position.z}, X = {X_POS}", Y = {Y_POS})
        except:
            pass


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
            self.TILT = 90 - (tilt *180/math.pi)
            self.HEADING= heading *-180/math.pi 
            self.ELEVATION = position.z 
            self.POS_X = position.x
            self.Y_POS = position.y
            


        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            result.found = False
            result.x_base = 0.0
            result.y_base = 0.0
            result.confidence = best_conf
            goal_handle.succeed()
            return result

        # 3) compute (x, y) of target using YOUR existing function
        # center of bbox:
        self.get_logger().info(f"camera pos = z={self.ELEVATION}, tilt = {self.TILT} , heading = {self.HEADING}")
        x_min, y_min, x_max, y_max = best.xyxy[0].tolist()
        u = (x_min + x_max) / 2.0
        v = (y_min + y_max) / 2.0

        Img_location = [u,v]
        self.get_logger().info(f"position = {u,v}")

      
        # call your function here, e.g.:
        # x_base, y_base = self.compute_xy_from_pixel(u, v, tf, intrinsics, ...)
        x_base, y_base, z_base = self.spatial_transformation(Img_location,"Z",0)
        self.get_logger().info(f"Coordinates found x= {x_base},y = {y_base},z= {z_base}")
        result.found = True
        result.x_base = float(y_base)
        result.y_base = float(-1*x_base)
        result.confidence = best_conf

        feedback.progress = 100.0
        goal_handle.publish_feedback(feedback)
        goal_handle.succeed()
        return result

def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()

