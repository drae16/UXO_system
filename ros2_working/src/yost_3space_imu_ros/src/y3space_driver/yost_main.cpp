// Plain rclcpp entry point (replaces the am_utils lifecycle/YostNode/YostStats setup).
#include <memory>
#include <y3space_driver/Y3SpaceDriver.h>

rclcpp::Node::SharedPtr am::Node::node = nullptr;

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);

  am::Node::node = std::make_shared<rclcpp::Node>("y3space_driver");

  auto driver = std::make_shared<am::Y3SpaceDriver>();
  driver->initialize();  // reads params, opens serial, configures IMU, starts publish timer

  ROS_INFO_STREAM(am::Node::node->get_name() << ": running...");
  rclcpp::spin(am::Node::node);

  driver.reset();  // destructor closes the serial port
  am::Node::node.reset();
  rclcpp::shutdown();
  return 0;
}
