import math

import numpy as np
import rclpy
from rclpy.node import Node
from message_filters import Subscriber, ApproximateTimeSynchronizer
from sensor_msgs.msg import NavSatFix, NavSatStatus
from nav_msgs.msg import Odometry

WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3


def geodetic_to_ecef(lat, lon, alt):
    lat, lon = math.radians(lat), math.radians(lon)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * math.sin(lat) ** 2)
    return np.array([
        (n + alt) * math.cos(lat) * math.cos(lon),
        (n + alt) * math.cos(lat) * math.sin(lon),
        (n * (1.0 - WGS84_E2) + alt) * math.sin(lat),
    ])


def ecef_to_enu_matrix(lat, lon):
    la, lo = math.radians(lat), math.radians(lon)
    return np.array([
        [-math.sin(lo), math.cos(lo), 0.0],
        [-math.sin(la) * math.cos(lo), -math.sin(la) * math.sin(lo), math.cos(la)],
        [math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)],
    ])


class AntennaOdometry(Node):
    def __init__(self):
        super().__init__('antenna_odometry')

        self.lead_topic = '/fix'
        self.trail_topic = '/fix2'
        self.baseline = 0.60
        self.baseline_tol = 0.05
        self.sync_slop = 0.05
        self.min_status = NavSatStatus.STATUS_GBAS_FIX
        self.frame_id = 'odom'
        self.child_frame_id = 'base_link'

        self.datum_ecef = None
        self.enu_rot = None

        self.lead_sub = Subscriber(self, NavSatFix, self.lead_topic)
        self.trail_sub = Subscriber(self, NavSatFix, self.trail_topic)
        self.sync = ApproximateTimeSynchronizer(
            [self.lead_sub, self.trail_sub], queue_size=20, slop=self.sync_slop)
        self.sync.registerCallback(self.fix_callback)

        self.pub = self.create_publisher(Odometry, '/odometry/antennas', 10)

    def fix_callback(self, lead, trail):
        if lead.status.status < self.min_status or trail.status.status < self.min_status:
            return

        lead_ecef = geodetic_to_ecef(lead.latitude, lead.longitude, lead.altitude)
        trail_ecef = geodetic_to_ecef(trail.latitude, trail.longitude, trail.altitude)

        sep = float(np.linalg.norm(lead_ecef - trail_ecef))
        if abs(sep - self.baseline) > self.baseline_tol:
            self.get_logger().warn(
                f'Antenna separation {sep:.3f} m outside baseline tolerance',
                throttle_duration_sec=1.0)
            return

        mid_ecef = (lead_ecef + trail_ecef) / 2.0
        if self.datum_ecef is None:
            self.datum_ecef = mid_ecef
            self.enu_rot = ecef_to_enu_matrix(lead.latitude, lead.longitude)
            self.get_logger().info('ENU datum set at first valid antenna midpoint')

        mid_enu = self.enu_rot @ (mid_ecef - self.datum_ecef)
        d_enu = self.enu_rot @ (lead_ecef - trail_ecef)
        heading = math.atan2(d_enu[1], d_enu[0])

        out = Odometry()
        out.header.stamp = lead.header.stamp
        out.header.frame_id = self.frame_id
        out.child_frame_id = self.child_frame_id
        out.pose.pose.position.x = float(mid_enu[0])
        out.pose.pose.position.y = float(mid_enu[1])
        out.pose.pose.position.z = float(mid_enu[2])
        out.pose.pose.orientation.z = math.sin(heading / 2.0)
        out.pose.pose.orientation.w = math.cos(heading / 2.0)
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = AntennaOdometry()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()