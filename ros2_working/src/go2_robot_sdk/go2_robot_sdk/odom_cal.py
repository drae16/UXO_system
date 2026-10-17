import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from nav_msgs.msg import Odometry
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
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def circular_mean(angles):
    return math.atan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))


def rot(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


class OdomCalibration(Node):
    def __init__(self):
        super().__init__('odom_calibration')

        self.window_duration = 10.0       # s of stationary averaging
        self.min_samples = 20             # valid antenna/odom pairs required
        self.max_pair_dt = 0.05           # s, antenna-to-odom stamp tolerance
        self.outlier_k = 3.0              # reject |d - mean| > k * robust sigma
        self.outlier_floor_deg = 0.5      # minimum robust sigma
        self.timer_period = 0.1
        self.drift_rate = math.radians(2.8) / 60.0   # rad per s of walking, CCW odom drift
        self.walk_speed = 0.1             # m/s, step speed above which walking time accumulates

        self.odom_sub = self.create_subscription(
            Odometry, '/odom_corrected', self.odom_callback, 50)
        self.ant_sub = self.create_subscription(
            Odometry, '/odometry/antennas', self.antenna_callback, 10)

        self.cal_pub = self.create_publisher(Odometry, '/odom_calibrated', 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.theta_pub = self.create_publisher(Float64, '/calibration/theta', latched)

        self.state = 'WAIT'
        self.window_start = None
        self.have_odom = False
        self.have_ant = False

        self.odom_buf = []      # (t, yaw, x, y)
        self.ant_buf = []       # (t, yaw, x, y)

        self.theta = None
        self.odom_origin = None
        self.base_origin = None

        self.pos = None         # dead-reckoned position in the antenna frame
        self.prev = None        # (t, yaw, x, y) of the previous odom sample
        self.walk_time = 0.0

        self.timer = self.create_timer(self.timer_period, self.tick)

    def odom_callback(self, msg):
        if self.state == 'COLLECTING':
            p = msg.pose.pose.position
            self.odom_buf.append((stamp_sec(msg), quat_yaw(msg.pose.pose.orientation), p.x, p.y))
        elif self.state == 'DONE':
            self.publish_calibrated(msg)
        elif self.state == 'WAIT':
            self.have_odom = True

    def antenna_callback(self, msg):
        if self.state == 'COLLECTING':
            p = msg.pose.pose.position
            self.ant_buf.append((stamp_sec(msg), quat_yaw(msg.pose.pose.orientation), p.x, p.y))
        elif self.state == 'WAIT':
            self.have_ant = True

    def tick(self):
        if self.state == 'WAIT':
            if self.have_odom and self.have_ant:
                self.state = 'COLLECTING'
                self.window_start = self.get_clock().now()
                self.get_logger().info(
                    f'Averaging antenna heading for {self.window_duration:.0f} s; keep the robot still')

        elif self.state == 'COLLECTING':
            elapsed = (self.get_clock().now() - self.window_start).nanoseconds * 1e-9
            if elapsed >= self.window_duration:
                self.fit()

    def pairs(self):
        odom_t = np.array([s[0] for s in self.odom_buf])
        out = []
        for t, ant_yaw, ax, ay in self.ant_buf:
            k = int(np.searchsorted(odom_t, t))
            candidates = [c for c in (k - 1, k) if 0 <= c < len(odom_t)]
            if not candidates:
                continue
            i = min(candidates, key=lambda c: abs(odom_t[c] - t))
            if abs(odom_t[i] - t) <= self.max_pair_dt:
                out.append((wrap(ant_yaw - self.odom_buf[i][1]), ant_yaw, ax, ay,
                            self.odom_buf[i][2], self.odom_buf[i][3]))
        return np.array(out)

    def fit(self):
        P = self.pairs() if self.odom_buf and self.ant_buf else np.empty((0, 6))

        if len(P) >= 3:
            d = P[:, 0]
            med = circular_mean(d)
            dev = np.abs(wrap(d - med))
            sigma = max(1.4826 * float(np.median(dev)), math.radians(self.outlier_floor_deg))
            keep = dev <= self.outlier_k * sigma
            P = P[keep]
            rejected = int((~keep).sum())
        else:
            rejected = 0

        if len(P) < self.min_samples:
            self.get_logger().warn(
                f'{len(P)} valid samples after {self.window_duration:.0f} s '
                f'({len(self.ant_buf)} antenna msgs, {rejected} outliers); need '
                f'{self.min_samples}, extending window')
            self.window_start = self.get_clock().now()
            return

        self.theta = circular_mean(P[:, 0])
        spread = math.degrees(float(np.std(wrap(P[:, 0] - self.theta))))

        self.base_origin = np.zeros(2)
        self.odom_origin = np.array(self.odom_buf[0][2:4])

        self.get_logger().info(
            f'theta = {math.degrees(self.theta):.3f} deg (std {spread:.3f} deg, '
            f'n={len(P)}, {rejected} outliers rejected)')
        self.theta_pub.publish(Float64(data=math.degrees(self.theta)))
        self.state = 'DONE'

    def publish_calibrated(self, msg):
        t = stamp_sec(msg)
        p = msg.pose.pose.position
        xy_odom = np.array([p.x, p.y])
        yaw = quat_yaw(msg.pose.pose.orientation)

        if self.prev is None:
            self.pos = rot(self.theta) @ (xy_odom - self.odom_origin) + self.base_origin
        else:
            t_prev, yaw_prev, x_prev, y_prev = self.prev
            dt = t - t_prev
            if dt <= 0.0:
                return
            step = xy_odom - np.array([x_prev, y_prev])
            yaw_mid = yaw_prev + 0.5 * wrap(yaw - yaw_prev)
            walking = np.hypot(step[0], step[1]) / dt > self.walk_speed
            if walking:
                self.walk_time += dt
            body_step = rot(-yaw_mid) @ step
            heading_mid = yaw_mid + self.theta - self.drift_rate * self.walk_time
            self.pos = self.pos + rot(heading_mid) @ body_step
        self.prev = (t, yaw, p.x, p.y)

        xy = self.pos
        correction = self.theta - self.drift_rate * self.walk_time
        q = msg.pose.pose.orientation
        qx, qy, qz, qw = quat_mul(yaw_quat(correction), (q.x, q.y, q.z, q.w))

        out = Odometry()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = msg.header.frame_id
        out.child_frame_id = msg.child_frame_id
        out.pose.pose.position.x = float(xy[0])
        out.pose.pose.position.y = float(xy[1])
        out.pose.pose.position.z = p.z
        out.pose.pose.orientation.x = qx
        out.pose.pose.orientation.y = qy
        out.pose.pose.orientation.z = qz
        out.pose.pose.orientation.w = qw
        out.twist = msg.twist
        self.cal_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = OdomCalibration()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()