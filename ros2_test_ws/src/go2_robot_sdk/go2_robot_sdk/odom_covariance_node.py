
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from nav_msgs.msg import Odometry
from go2_interfaces.msg import Go2State  # velocity.x, velocity.y, velocity.z

class OdomCovAndVelFuser(Node):
    """
    Build a corrected Odometry message:
      - Pose/orientation copied from /odom.
      - Pose covariance added/overwritten.
      - Linear velocity from /go2_states (vector3).
      - Angular velocity set to zero.
      - Twist covariance added.
    Publishes on /odom_corrected.
    """

    def __init__(self):
        super().__init__('odom_cov_and_vel_fuser')

        # Parameters
        self.in_odom_topic = self.declare_parameter('in_odom', '/odom').get_parameter_value().string_value
        self.states_topic  = self.declare_parameter('states_topic', '/go2_states').get_parameter_value().string_value
        self.out_odom_topic = self.declare_parameter('out_odom', '/odom_corrected').get_parameter_value().string_value
        self.max_velocity_age = self.declare_parameter('max_velocity_age', 0.3).get_parameter_value().double_value
        self.keep_in_pose_cov = self.declare_parameter('keep_in_pose_covariance', False).get_parameter_value().bool_value

        # Covariance defaults
        self.pose_var_x     = self.declare_parameter('pose_variance_x',   0.05).get_parameter_value().double_value
        self.pose_var_y     = self.declare_parameter('pose_variance_y',   0.05).get_parameter_value().double_value
        self.pose_var_z     = self.declare_parameter('pose_variance_z', 1000.0).get_parameter_value().double_value
        self.pose_var_roll  = self.declare_parameter('pose_variance_roll', 1000.0).get_parameter_value().double_value
        self.pose_var_pitch = self.declare_parameter('pose_variance_pitch',1000.0).get_parameter_value().double_value
        self.pose_var_yaw   = self.declare_parameter('pose_variance_yaw',  0.10).get_parameter_value().double_value

        self.twist_var_vx   = self.declare_parameter('twist_variance_vx', 0.05).get_parameter_value().double_value
        self.twist_var_vy   = self.declare_parameter('twist_variance_vy', 0.05).get_parameter_value().double_value
        self.twist_var_vz   = self.declare_parameter('twist_variance_vz', 1000.0).get_parameter_value().double_value
        self.twist_var_wx   = self.declare_parameter('twist_variance_wx', 1000.0).get_parameter_value().double_value
        self.twist_var_wy   = self.declare_parameter('twist_variance_wy', 1000.0).get_parameter_value().double_value
        self.twist_var_wz   = self.declare_parameter('twist_variance_wz', 0.10).get_parameter_value().double_value

        # Publishers / subscribers
        self.pub = self.create_publisher(Odometry, self.out_odom_topic, 10)
        self.create_subscription(Odometry, self.in_odom_topic, self._odom_cb, 50)
        self.create_subscription(Go2State, self.states_topic, self._states_cb, 50)

        # Cache latest linear velocity
        self._last_vel = None
        self._last_vel_time = None

        self.get_logger().info(
            f"Listening to {self.in_odom_topic} and {self.states_topic}, publishing {self.out_odom_topic}"
        )

    def _states_cb(self, msg: Go2State):
        try:
            vx = float(msg.velocity.x)
            vy = float(msg.velocity.y)
            vz = float(msg.velocity.z)
        except AttributeError:
            # fallback if velocity is a plain list
            vx, vy, vz = float(msg.velocity[0]), float(msg.velocity[1]), float(msg.velocity[2])
        self._last_vel = (vx, vy, vz)
        self._last_vel_time = self.get_clock().now()

    def _odom_cb(self, msg: Odometry):
        out = Odometry()
        out.header = msg.header
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = msg.header.frame_id or 'odom'
        out.child_frame_id = msg.child_frame_id or 'base_link'

        # Pose
        out.pose.pose = msg.pose.pose

        # Pose covariance
        pose_cov_in = list(msg.pose.covariance)
        if self.keep_in_pose_cov and any(abs(x) > 0.0 for x in pose_cov_in):
            out.pose.covariance = pose_cov_in
        else:
            pc = [0.0] * 36
            pc[0]  = self.pose_var_x
            pc[7]  = self.pose_var_y
            pc[14] = self.pose_var_z
            pc[21] = self.pose_var_roll
            pc[28] = self.pose_var_pitch
            pc[35] = self.pose_var_yaw
            out.pose.covariance = pc

        # Twist.linear from go2_states if fresh
        vx = vy = vz = 0.0
        if self._last_vel and self._last_vel_time:
            if (self.get_clock().now() - self._last_vel_time) < Duration(seconds=self.max_velocity_age):
                vx, vy, vz = self._last_vel
        out.twist.twist.linear.x = vx
        out.twist.twist.linear.y = vy
        out.twist.twist.linear.z = vz

        # Twist.angular zeros
        out.twist.twist.angular.x = 0.0
        out.twist.twist.angular.y = 0.0
        out.twist.twist.angular.z = 0.0

        # Twist covariance
        tc = [0.0] * 36
        tc[0]  = self.twist_var_vx
        tc[7]  = self.twist_var_vy
        tc[14] = self.twist_var_vz
        tc[21] = self.twist_var_wx
        tc[28] = self.twist_var_wy
        tc[35] = self.twist_var_wz
        out.twist.covariance = tc

        self.pub.publish(out)


def main():
    rclpy.init()
    rclpy.spin(OdomCovAndVelFuser())
    rclpy.shutdown()

if __name__ == '__main__':
    main()