import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix

class FixOriginPublisher(Node):
    def __init__(self):
        super().__init__("fix_origin_publisher")

        # Subscription to raw GPS fixes
        self.sub = self.create_subscription(
            NavSatFix, "/fix", self.cb, 10
        )

        # Publisher for the origin (with transient_local QoS so late subscribers get it)
        qos = rclpy.qos.QoSProfile(depth=1)
        qos.durability = rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL
        self.pub = self.create_publisher(NavSatFix, "/gps/fix/origin", qos)

        self.origin_msg = None

    def cb(self, msg: NavSatFix):
        # Only capture the first valid fix
        if self.origin_msg is None and msg.status.status >= 0:
            self.get_logger().info(
                f"Got first GPS fix, setting origin: {msg.latitude}, {msg.longitude}"
            )
            self.origin_msg = msg
            self.pub.publish(self.origin_msg)

        elif self.origin_msg:
            # Keep re-publishing stored origin so it's always available
            self.pub.publish(self.origin_msg)


def main(args=None):
    rclpy.init(args=args)
    node = FixOriginPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()