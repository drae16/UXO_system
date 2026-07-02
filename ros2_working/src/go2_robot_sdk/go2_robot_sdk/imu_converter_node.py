import rclpy
from rclpy.node import Node
from go2_interfaces.msg import IMU
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Quaternion, Vector3



class ImuConverterNode(Node):

    def __init__(self):
        super().__init__('imu_converter_node')

        self.subscription = self.create_subscription(
            IMU,
            '/imu',
            self.imu_callback,
            10
        )

        self.publisher = self.create_publisher(Imu, 'imu/data', 10)

        self.frame_id = 'imu'  # Change if needed

    def imu_callback(self, msg):
        imu_msg = Imu()
        imu_msg.header.stamp = self.get_clock().now().to_msg()
        imu_msg.header.frame_id = self.frame_id

        # Orientation (quaternion)
        imu_msg.orientation = Quaternion(
            x=float(msg.quaternion[1]),
            y=float(msg.quaternion[2]),
            z=float(msg.quaternion[3]),
            w=float(msg.quaternion[0])
        )

        # Angular velocity (gyroscope)
        imu_msg.angular_velocity = Vector3(
            x=float(msg.gyroscope[0]),
            y=float(msg.gyroscope[1]),
            z=float(msg.gyroscope[2])
        )

        # Linear acceleration (accelerometer)
        imu_msg.linear_acceleration = Vector3(
            x=float(msg.accelerometer[0]),
            y=float(msg.accelerometer[1]),
            z=float(msg.accelerometer[2])
        )

        # Covariances – rough defaults (tune later)
        imu_msg.orientation_covariance = [0.01, 0.0, 0.0,
                                          0.0, 0.01, 0.0,
                                          0.0, 0.0, 0.01]

        imu_msg.angular_velocity_covariance = [0.01, 0.0, 0.0,
                                               0.0, 0.01, 0.0,
                                               0.0, 0.0, 0.01]

        imu_msg.linear_acceleration_covariance = [0.1, 0.0, 0.0,
                                                  0.0, 0.1, 0.0,
                                                  0.0, 0.0, 0.1]

        self.publisher.publish(imu_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ImuConverterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
