import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Vector3

from std_msgs.msg import String


class subscriber(Node):
    def listener_callback(self, msg: Vector3): 
        self.get_logger().info(f"Received:{msg}")

    def __init__(self):

        super().__init__('gps_subscribe')
        gps_qos = QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST,
        )
        self.subscription = self.create_subscription(
                Vector3,
                '/gps_targets',
                self.listener_callback,
                gps_qos
        )



def main(args=None):
    rclpy.init(args=args)
    subscriber_node = subscriber()
    rclpy.spin(subscriber_node)

if __name__ == '__main__':
    main()



