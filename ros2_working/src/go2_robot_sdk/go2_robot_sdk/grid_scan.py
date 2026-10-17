#!/usr/bin/env python3
"""
GPS-only (RTK) lawnmower survey node. No IMU or odometry is used.

The whole pattern is planned from the first valid fix (the origin):
  - Lane 0 runs north for `north_distance` m. Waypoints are spaced
    `waypoint_spacing` m apart.
  - At each lane end, the lane switch is also a set of GPS waypoints: a
    semicircular arc (diameter = `lane_spacing`) that carries the robot
    `lane_spacing` m west and turns it to face the opposite direction.
  - Lanes alternate north/south until `west_distance` m of westward offset is
    covered, then the robot stops.

Heading comes only from RTK motion (course over ground): the bearing of the
chord between the newest fix and an older fix at least `course_baseline` m
away. Because heading only exists while the robot moves, the robot never
rotates in place. Turns are done while walking slowly forward. Before the
first heading estimate, it creeps straight ahead, so start it roughly facing
north.

Arrival: a waypoint counts as reached when the robot is within its tolerance
OR has crossed the line through the waypoint perpendicular to the segment
leading to it. The crossing rule stops the robot from orbiting a waypoint it
narrowly missed.

Subscribes:  /fix      sensor_msgs/NavSatFix
Publishes:   /cmd_vel  geometry_msgs/Twist

Run:
  ros2 run <pkg> lawnmower_nav --ros-args -p north_distance:=20.0 -p west_distance:=3.0
"""
import math
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import NavSatFix, NavSatStatus
from geometry_msgs.msg import Twist

EARTH_R = 6378137.0  # WGS-84 equatorial radius [m]


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class LawnmowerNav(Node):
    def __init__(self):
        super().__init__('gridscan')
        p = self.declare_parameter

        # Pattern geometry [m]
        self.north_dist = float(p('north_distance', 10.0).value)
        self.west_dist = float(p('west_distance', 10.0).value)
        self.wp_spacing = float(p('waypoint_spacing', 3.0).value)
        self.lane_spacing = float(p('lane_spacing', 0.3).value)
        self.headland = float(p('headland', 0.0).value)        # extra straight run past lane end before turning
        self.turn_pts = int(p('turn_points', 2).value)          # waypoints along the turn arc

        # Arrival tolerances [m]
        self.arrive_tol = float(p('arrival_tolerance', 0.5).value)  # along-lane waypoints
        self.turn_tol = float(p('turn_tolerance', 0.08).value)      # lane ends, turn arc, lane starts

        # Speeds / gains
        self.cruise = float(p('cruise_speed', 0.5).value)
        self.turn_speed = float(p('turn_speed', 0.1).value)    # forward speed while turning (must stay > 0)
        self.creep_speed = float(p('creep_speed', 0.15).value)  # straight creep when heading is unknown
        self.k_lin = float(p('k_linear', 0.5).value)            # slow-down gain approaching lane ends
        self.max_ang = float(p('max_angular', 0.6).value)
        self.k_ang = float(p('k_angular', 1.5).value)
        self.slow_turn_angle = math.radians(float(p('slow_turn_deg', 30.0).value))

        # GPS course (heading) estimator
        self.course_baseline = float(p('course_baseline', 0.10).value)  # min chord length [m]
        self.course_window = float(p('course_window', 3.0).value)       # fix history kept [s]
        self.course_hold = float(p('course_hold', 1.0).value)           # reuse last heading this long [s]
        self.lag_gain = float(p('lag_comp_gain', 1.0).value)            # chord-lag correction (0 = off)

        self.fix_timeout = float(p('fix_timeout', 1.0).value)
        rate = float(p('control_rate', 20.0).value)

        self.n_lanes = int(math.floor(self.west_dist / self.lane_spacing + 1e-6)) + 1

        r_needed = self.lane_spacing / 2.0
        r_min = self.turn_speed / self.max_ang
        if r_min > r_needed:
            self.get_logger().warn(
                f'Turn radius at turn_speed/max_angular is {r_min:.2f} m but the lane switch needs '
                f'{r_needed:.2f} m. Lower turn_speed or raise max_angular, or the robot will overshoot the arc.')

        # State
        self.origin = None
        self.pos = None
        self.last_fix_time = None
        self.hist = deque()          # (t, east, north)
        self.last_heading = None
        self.last_heading_time = None
        self.last_w = 0.0
        self.plan = []
        self.idx = 0
        self.prev_wp = None
        self.done = False

        self.create_subscription(NavSatFix, '/fix', self.fix_cb, qos_profile_sensor_data)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_timer(1.0 / rate, self.control_loop)

        self.get_logger().info(
            f'Lawnmower (GPS-only): {self.n_lanes} lanes x {self.north_dist} m, '
            f'lane spacing {self.lane_spacing} m. Waiting for RTK fix...')

    # ------------------------------------------------------------------ planning
    def lane_x(self, i):
        return -i * self.lane_spacing

    def lane_dir(self, i):
        return 1.0 if i % 2 == 0 else -1.0

    def build_plan(self):
        plan = []
        s = self.lane_spacing
        for i in range(self.n_lanes):
            x, d = self.lane_x(i), self.lane_dir(i)
            y0 = 0.0 if d > 0 else self.north_dist
            y1 = self.north_dist if d > 0 else 0.0

            if i > 0:
                plan.append((x, y0, 'turn'))  # lane start (trivially reached when headland = 0)

            k = 1
            while True:
                y = y0 + d * k * self.wp_spacing
                if (y1 - y) * d <= 1e-9:
                    plan.append((x, y1, 'lane_end'))
                    break
                plan.append((x, y, 'lane'))
                k += 1

            if i + 1 < self.n_lanes:
                # Semicircle from this lane to the next lane (always to the west)
                R = s / 2.0
                cx, cy = x - R, y1 + d * self.headland
                if self.headland > 0.0:
                    plan.append((x, cy, 'turn'))
                n = max(1, self.turn_pts)
                for j in range(1, n + 1):
                    phi = math.pi * j / n
                    plan.append((cx + R * math.cos(phi), cy + d * R * math.sin(phi), 'turn'))
        return plan

    # ----------------------------------------------------------------- callbacks
    def fix_cb(self, msg: NavSatFix):
        if msg.status.status < NavSatStatus.STATUS_FIX:
            return
        if math.isnan(msg.latitude) or math.isnan(msg.longitude):
            return

        if self.origin is None:
            lat0, lon0 = msg.latitude, msg.longitude
            self.origin = (lat0, lon0, math.cos(math.radians(lat0)))
            self.plan = build = self.build_plan()
            self.prev_wp = (0.0, 0.0)
            self.get_logger().info(
                f'Origin {lat0:.8f}, {lon0:.8f}. Planned {len(build)} waypoints. '
                f'First {self.fmt(build[0])}')

        lat0, lon0, coslat0 = self.origin
        east = math.radians(msg.longitude - lon0) * EARTH_R * coslat0
        north = math.radians(msg.latitude - lat0) * EARTH_R
        self.pos = (east, north)

        now = self.now_s()
        self.last_fix_time = now
        self.hist.append((now, east, north))

    # ------------------------------------------------------- heading from motion
    def estimate_heading(self, now):
        while self.hist and now - self.hist[0][0] > self.course_window:
            self.hist.popleft()

        if len(self.hist) >= 2:
            tn, xn, yn = self.hist[-1]
            for t, x, y in reversed(self.hist):
                if math.hypot(xn - x, yn - y) >= self.course_baseline:
                    h = math.atan2(yn - y, xn - x)
                    # Chord bearing ~ heading at the chord midpoint; push it forward
                    # by the commanded turn over half the chord time.
                    h += self.lag_gain * self.last_w * (tn - t) / 2.0
                    self.last_heading = wrap(h)
                    self.last_heading_time = now
                    return self.last_heading

        if self.last_heading is not None and now - self.last_heading_time < self.course_hold:
            return self.last_heading
        return None

    # ------------------------------------------------------------------- control
    def control_loop(self):
        if self.done or self.pos is None or not self.plan:
            self.stop()
            return

        now = self.now_s()
        if now - self.last_fix_time > self.fix_timeout:
            self.get_logger().warn('Fix stale, holding.', throttle_duration_sec=2.0)
            self.hist.clear()          # heading must be re-earned after a pause
            self.last_heading = None
            self.stop()
            return

        # Advance past any reached waypoints
        while not self.done and self.reached(self.plan[self.idx]):
            wx, wy, kind = self.plan[self.idx]
            self.get_logger().info(f'Reached {kind} {self.fmt(self.plan[self.idx])}')
            self.prev_wp = (wx, wy)
            self.idx += 1
            if self.idx >= len(self.plan):
                self.done = True
                self.get_logger().info('Survey complete. Stopping.')
        if self.done:
            self.stop()
            return

        wx, wy, kind = self.plan[self.idx]
        dx, dy = wx - self.pos[0], wy - self.pos[1]
        dist = math.hypot(dx, dy)

        yaw = self.estimate_heading(now)
        cmd = Twist()
        if yaw is None:
            cmd.linear.x = self.creep_speed   # walk straight until a heading is known
            self.last_w = 0.0
            self.cmd_pub.publish(cmd)
            return

        err = wrap(math.atan2(dy, dx) - yaw)
        w = clamp(self.k_ang * err, -self.max_ang, self.max_ang)

        if kind == 'turn' or abs(err) > self.slow_turn_angle:
            v = self.turn_speed
        elif kind == 'lane_end':
            v = clamp(self.k_lin * dist, self.turn_speed, self.cruise)
        else:
            v = self.cruise

        cmd.linear.x = v
        cmd.angular.z = w
        self.last_w = w
        self.cmd_pub.publish(cmd)

    def reached(self, wp):
        wx, wy, kind = wp
        tol = self.arrive_tol if kind == 'lane' else self.turn_tol
        rx, ry = self.pos[0] - wx, self.pos[1] - wy
        if math.hypot(rx, ry) < tol:
            return True
        # Crossed the perpendicular line through the waypoint
        sx, sy = wx - self.prev_wp[0], wy - self.prev_wp[1]
        return (sx * sx + sy * sy) > 1e-9 and (rx * sx + ry * sy) > 0.0

    # ------------------------------------------------------------------- helpers
    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def stop(self):
        self.last_w = 0.0
        self.cmd_pub.publish(Twist())

    @staticmethod
    def fmt(wp):
        return f'(E {wp[0]:+.2f}, N {wp[1]:+.2f})'


def main():
    rclpy.init()
    node = LawnmowerNav()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()