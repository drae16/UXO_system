// This node is used to gate GPS odometry readings when performing the local arm scan
// doing so allows both ekfs to operate solely off local odometry readings, removing error from erroneous GPS

#include <atomic>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_srvs/srv/set_bool.hpp>

class GpsGateNode : public rclcpp::Node
{
public:
  GpsGateNode()
  : rclcpp::Node("gps_gate")
  {
    // --- Parameters (exposed for field tuning) ---
    const std::string input_topic =
      this->declare_parameter<std::string>("input_topic", "/odometry/gps");
    const std::string output_topic =
      this->declare_parameter<std::string>("output_topic", "/odometry/gps_gated");
    const std::string service_name =
      this->declare_parameter<std::string>("service_name", "set_gps_gate");
    // Initial gate state. true = pass-through, false = gated (drop).
    gate_gps_.store(
      this->declare_parameter<bool>("initial_pass_through", true));

    // Match navsat_transform / EKF default QoS: RELIABLE, depth 10.
    // Do NOT switch this to sensor_data (best-effort): a reliable EKF
    // subscriber downstream would then receive nothing.
    const rclcpp::QoS qos(10);

    pub_ = this->create_publisher<nav_msgs::msg::Odometry>(output_topic, qos);

    sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      input_topic, qos,
      std::bind(&GpsGateNode::odomCallback, this, std::placeholders::_1));

    srv_ = this->create_service<std_srvs::srv::SetBool>(
      service_name,
      std::bind(&GpsGateNode::gateCallback, this,
                std::placeholders::_1, std::placeholders::_2));

    RCLCPP_INFO(
      this->get_logger(),
      "gps_gate up: '%s' -> '%s' | service '%s' | initial state: %s",
      input_topic.c_str(), output_topic.c_str(), service_name.c_str(),
      gate_gps_.load() ? "PASS-THROUGH" : "GATED");
  }

private:
  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    // Flag true  -> republish (pass through).
    // Flag false -> drop silently. The absence of a message is the freeze;
    //               never resend a stale fix here.
    if (gate_gps_.load(std::memory_order_relaxed)) {
      pub_->publish(*msg);
    }
  }

  void gateCallback(
    const std::shared_ptr<std_srvs::srv::SetBool::Request> req,
    std::shared_ptr<std_srvs::srv::SetBool::Response> resp)
  {
    gate_gps_.store(req->data, std::memory_order_relaxed);

    // Per spec: response carries the resulting flag state (not the usual
    // "operation succeeded" meaning of SetBool::success).
    resp->success = gate_gps_.load(std::memory_order_relaxed);
    resp->message = resp->success ? "gps pass-through ENABLED"
                                  : "gps GATED (dropping messages)";
    RCLCPP_INFO(this->get_logger(), "%s", resp->message.c_str());
  }

  // Single-threaded executor serializes callbacks, but atomic keeps this
  // correct under a multi-threaded executor too, at zero practical cost.
  std::atomic<bool> gate_gps_{true};

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr srv_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<GpsGateNode>());
  rclcpp::shutdown();
  return 0;
}