"""
Heading-corrected odometry using the Yost 3-Space IMU (ENU) as the yaw reference.

For every odom message:
  offset   = wrap(imu_yaw(t) - odom_yaw(t))          # odom heading error vs IMU
  pos     += rot(offset_mid) @ (odom_xy(t) - odom_xy(t_prev))
  yaw_out  = odom_yaw + offset (= IMU yaw)

Position starts at (0, 0) when the node starts; heading is absolute ENU from the IMU
(set zero_yaw:=true to also start heading at 0).

heading_offset_deg is added to the IMU yaw (e.g. -9.0 for Oahu declination, or a mount
correction). It can be changed while running and takes effect on the next odom message:
    ros2 param set /odom_calibration heading_offset_deg -9.0
Changes only affect motion from that point on; the path already integrated is kept.
Use either this or the driver's magnetic_declination_deg for declination, not both.

IMU yaw is interpolated to each odom stamp. If the stamps can't be matched (e.g. the
two sources use different clocks), the latest IMU sample is used instead.
"""
import bisect
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from rclpy.qos import QoSProfile, DurabilityPolicy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64


def stamp_sec(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def quat_yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def yaw_quat(theta):
    return (0.0, 0.0, math.sin(theta / 2.0), math.cos(theta / 2.0))


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def rot(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


class OdomImuCalibration(Node):
    def __init__(self):
        super().__init__('odom_calibration')

        self.odom_topic = self.declare_parameter('odom_topic', '/odom_corrected').value
        self.imu_topic = self.declare_parameter('imu_topic', '/mavros/imu/data').value
        self.out_topic = self.declare_parameter('output_topic', '/odom_calibrated').value
        self.max_pair_dt = self.declare_parameter('max_pair_dt', 0.05).value      # s
        self.imu_buffer_sec = self.declare_parameter('imu_buffer_sec', 2.0).value  # s
        self.offset_alpha = self.declare_parameter('offset_alpha', 0.1).value      # 1.0 = no smoothing
        self.zero_yaw = self.declare_parameter('zero_yaw', False).value
        self.heading_offset = math.radians(
            self.declare_parameter('heading_offset_deg', -7.1).value)              # added to IMU yaw
        self.add_on_set_parameters_callback(self.on_params)

        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 50)
        self.imu_sub = self.create_subscription(Imu, self.imu_topic, self.imu_callback, 200)

        self.cal_pub = self.create_publisher(Odometry, self.out_topic, 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.theta_pub = self.create_publisher(Float64, '/calibration/theta', latched)

        self.imu_t = []          # IMU stamps (sorted)
        self.imu_yaw = []        # matching yaw (rad, ENU)

        self.offset = None       # smoothed imu_yaw - odom_yaw
        self.yaw0 = 0.0          # heading subtracted when zero_yaw is set
        self.pos = None          # corrected position, starts at (0, 0)
        self.prev = None         # (t, x, y, offset) of previous odom sample
        self.warned_clock = False

        self.get_logger().info(
            f'Correcting {self.odom_topic} heading with {self.imu_topic} -> {self.out_topic} '
            f'(heading_offset_deg={math.degrees(self.heading_offset):+.2f})')

    def on_params(self, params):
        for prm in params:
            if prm.name == 'heading_offset_deg':
                try:
                    val = float(prm.value)
                except (TypeError, ValueError):
                    return SetParametersResult(successful=False, reason='heading_offset_deg must be a number')
                old = math.degrees(self.heading_offset)
                self.heading_offset = math.radians(val)
                self.get_logger().info(f'heading_offset_deg changed {old:+.2f} -> {val:+.2f}')
            elif prm.name == 'offset_alpha':
                self.offset_alpha = float(prm.value)
            elif prm.name == 'max_pair_dt':
                self.max_pair_dt = float(prm.value)
        return SetParametersResult(successful=True)

    # ---------------------------------------------------------------- IMU
    def imu_callback(self, msg):
        t = stamp_sec(msg)
        if self.imu_t and t <= self.imu_t[-1]:
            return
        self.imu_t.append(t)
        self.imu_yaw.append(quat_yaw(msg.orientation))

        cut = bisect.bisect_left(self.imu_t, t - self.imu_buffer_sec)
        if cut:
            del self.imu_t[:cut]
            del self.imu_yaw[:cut]

    def imu_yaw_at(self, t):
        """IMU yaw interpolated to time t, or None if t isn't covered by the buffer."""
        if not self.imu_t:
            return None
        k = bisect.bisect_left(self.imu_t, t)
        if k == 0:
            return self.imu_yaw[0] if self.imu_t[0] - t <= self.max_pair_dt else None
        if k == len(self.imu_t):
            return self.imu_yaw[-1] if t - self.imu_t[-1] <= self.max_pair_dt else None
        t0, t1 = self.imu_t[k - 1], self.imu_t[k]
        y0, y1 = self.imu_yaw[k - 1], self.imu_yaw[k]
        if t1 - t0 > 2.0 * self.max_pair_dt:
            return None
        a = (t - t0) / (t1 - t0)
        return wrap(y0 + a * wrap(y1 - y0))

    # --------------------------------------------------------------- odom
    def odom_callback(self, msg):
        if not self.imu_t:
            return

        t = stamp_sec(msg)
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        odom_yaw = quat_yaw(q)

        imu_yaw = self.imu_yaw_at(t)
        if imu_yaw is None:
            # Stamps don't line up (different clocks or IMU dropout): use the latest IMU yaw.
            if not self.warned_clock:
                self.get_logger().warn(
                    f'Could not match odom stamp {t:.3f} to IMU stamps '
                    f'[{self.imu_t[0]:.3f}, {self.imu_t[-1]:.3f}]; using latest IMU sample. '
                    'Check that both topics use the same clock.')
                self.warned_clock = True
            imu_yaw = self.imu_yaw[-1]

        raw_offset = wrap(imu_yaw - odom_yaw)
        if self.offset is None:
            self.offset = raw_offset
            self.yaw0 = wrap(imu_yaw + self.heading_offset) if self.zero_yaw else 0.0
            self.get_logger().info(
                f'Initial heading offset {math.degrees(raw_offset):.2f} deg; '
                f'IMU yaw {math.degrees(imu_yaw):.2f} deg (ENU)')
        else:
            self.offset = wrap(self.offset + self.offset_alpha * wrap(raw_offset - self.offset))

        if self.prev is None:
            self.pos = np.zeros(2)
        else:
            t_prev, x_prev, y_prev, off_prev = self.prev
            if t - t_prev <= 0.0:
                return
            step = np.array([p.x - x_prev, p.y - y_prev])
            off_mid = off_prev + 0.5 * wrap(self.offset - off_prev)
            self.pos = self.pos + rot(off_mid + self.heading_offset - self.yaw0) @ step
        self.prev = (t, p.x, p.y, self.offset)

        correction = self.offset + self.heading_offset - self.yaw0
        qx, qy, qz, qw = quat_mul(yaw_quat(correction), (q.x, q.y, q.z, q.w))

        out = Odometry()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = msg.header.frame_id
        out.child_frame_id = msg.child_frame_id
        out.pose.pose.position.x = float(self.pos[0])
        out.pose.pose.position.y = float(self.pos[1])
        out.pose.pose.position.z = p.z
        out.pose.pose.orientation.x = qx
        out.pose.pose.orientation.y = qy
        out.pose.pose.orientation.z = qz
        out.pose.pose.orientation.w = qw
        out.pose.covariance = msg.pose.covariance
        out.twist = msg.twist          # body-frame twist is unaffected by a heading correction
        self.cal_pub.publish(out)

        self.theta_pub.publish(Float64(data=math.degrees(wrap(self.offset + self.heading_offset))))


def main(args=None):
    rclpy.init(args=args)
    node = OdomImuCalibration()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()