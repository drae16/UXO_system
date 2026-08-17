// odom_tf_republisher_node.cpp
//
// Broadcasts the odom->base_link transform from the /odom TOPIC, in C++, so it
// is not subject to the Go2 driver's Python GIL contention (WebRTC crypto).
//
// The driver still publishes the /odom topic, but should NO LONGER broadcast
// odom->base itself (remove its sendTransform call). This node turns that topic
// into the transform promptly: it stamps the TF with the odom message's own
// stamp and sends it the instant the message arrives, adding no GIL-race delay.
//
// Frames and topic are parameters; defaults match a standard single-robot Go2.

#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>

class OdomTfRepublisher : public rclcpp::Node
{
public:
  OdomTfRepublisher()
  : rclcpp::Node("odom_tf_republisher")
  {
    odom_topic_ = this->declare_parameter<std::string>("odom_topic", "/odom");
    odom_frame_ = this->declare_parameter<std::string>("odom_frame", "odom");
    base_frame_ = this->declare_parameter<std::string>("base_frame", "base_link");
    // If true, trust the message's child_frame_id over the base_frame param.
    use_msg_child_frame_ =
      this->declare_parameter<bool>("use_msg_child_frame", false);

    broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(*this);

    // Match the driver's odom QoS (reliable, depth 10).
    sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      odom_topic_, rclcpp::QoS(10),
      std::bind(&OdomTfRepublisher::odomCallback, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(),
      "odom_tf_republisher up: '%s' -> broadcast %s->%s",
      odom_topic_.c_str(), odom_frame_.c_str(), base_frame_.c_str());
  }

private:
  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    geometry_msgs::msg::TransformStamped tf;

    // Stamp with the odom message's OWN stamp -- keeps odom->base aligned on the
    // same clock as everything downstream and avoids now()-vs-source skew.
    tf.header.stamp    = msg->header.stamp;
    tf.header.frame_id = odom_frame_;
    tf.child_frame_id  = use_msg_child_frame_ ? msg->child_frame_id : base_frame_;

    tf.transform.translation.x = msg->pose.pose.position.x;
    tf.transform.translation.y = msg->pose.pose.position.y;
    tf.transform.translation.z = msg->pose.pose.position.z;
    tf.transform.rotation       = msg->pose.pose.orientation;

    broadcaster_->sendTransform(tf);
  }

  std::string odom_topic_, odom_frame_, base_frame_;
  bool use_msg_child_frame_{false};

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_;
  std::shared_ptr<tf2_ros::TransformBroadcaster> broadcaster_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<OdomTfRepublisher>());
  rclcpp::shutdown();
  return 0;
}