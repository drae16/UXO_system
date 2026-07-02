

#include <chrono>
#include <cmath>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "control_msgs/msg/joint_jog.hpp"

// Custom service
#include "nav_search/srv/track_target.hpp"

#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Vector3.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"

using namespace std::chrono_literals;

static inline double clamp(double v, double lo, double hi) { return std::max(lo, std::min(hi, v)); }
static inline double wrap_pi(double a) {
  while (a > M_PI) a -= 2.0 * M_PI;
  while (a < -M_PI) a += 2.0 * M_PI;
  return a;
}
static inline double yaw_from_vec(const tf2::Vector3& v) { return std::atan2(v.y(), v.x()); }
static inline double pitch_from_vec(const tf2::Vector3& v) {
  const double h = std::hypot(v.x(), v.y());
  return std::atan2(-v.z(), h);
}

class CameraServoTracker : public rclcpp::Node {
public:
  CameraServoTracker()
  : Node("camera_servo_tracker"),
    tf_buffer_(this->get_clock()),
    tf_listener_(tf_buffer_)
  {
    base_frame_ = declare_parameter<std::string>("base_frame", "vx300s/base_link");
    ee_frame_   = declare_parameter<std::string>("ee_frame",   "vx300s/ee_gripper_link");

    joint_waist_    = declare_parameter<std::string>("joint_waist", "waist");
    joint_wrist_    = declare_parameter<std::string>("joint_wrist_angle", "wrist_angle");

    rate_hz_ = declare_parameter<double>("rate_hz", 100.0);
    deadband_rad_ = declare_parameter<double>("deadband_rad", 0.01);
    k_yaw_   = declare_parameter<double>("k_yaw",   1.5);
    k_pitch_ = declare_parameter<double>("k_pitch", 1.5);

    max_vel_waist_    = declare_parameter<double>("max_vel_waist",    0.8);
    max_vel_wrist_    = declare_parameter<double>("max_vel_wrist",    1.0);

    // Publish to MoveIt Servo input (remap/namespace as needed)
    jog_pub_ = create_publisher<control_msgs::msg::JointJog>("delta_joint_cmds", 10);

    // Service: set target + enable/disable
    srv_ = create_service<nav_search::srv::TrackTarget>(
      "track_target",
      [this](const std::shared_ptr<nav_search::srv::TrackTarget::Request> req,
             std::shared_ptr<nav_search::srv::TrackTarget::Response> res)
      {
        tracking_enabled_ = req->enable;

        if (tracking_enabled_) {
          target_ = req->target;
          target_.header.stamp = rclcpp::Time(0, 0, this->get_clock()->get_clock_type());
          have_target_ = true;
          res->accepted = true;
          res->message = "Tracking ENABLED and target set";
          RCLCPP_INFO(get_logger(), "%s (target frame: %s)",
                      res->message.c_str(), target_.header.frame_id.c_str());
        } else {
          // Stop publishing entirely (Servo will halt after timeout)
          res->accepted = true;
          res->message = "Tracking DISABLED";
          RCLCPP_INFO(get_logger(), "%s", res->message.c_str());
        }
      });

    const auto period = std::chrono::duration<double>(1.0 / std::max(1.0, rate_hz_));
    timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      std::bind(&CameraServoTracker::on_timer, this));
  }

private:
  void on_timer() {
    if (!tracking_enabled_ || !have_target_) return;

    // TF base->ee
    geometry_msgs::msg::TransformStamped tf_base_ee;
    try {
      tf_base_ee = tf_buffer_.lookupTransform(base_frame_, ee_frame_, tf2::TimePointZero);
    } catch (const tf2::TransformException &ex) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "TF base->ee unavailable: %s", ex.what());
      return;
    }

    tf2::Vector3 cam_pos(
      tf_base_ee.transform.translation.x,
      tf_base_ee.transform.translation.y,
      tf_base_ee.transform.translation.z);

    tf2::Quaternion q;
    tf2::fromMsg(tf_base_ee.transform.rotation, q);
    q.normalize();
    tf2::Matrix3x3 R(q);

    // Current camera forward axis in base (assume +X is forward)
    const tf2::Vector3 CAM_FWD(1.0, 0.0, 0.0);
    tf2::Vector3 f = R * CAM_FWD;
    if (f.length2() < 1e-9) return;
    f.normalize();

    // Transform service-provided target into base
    geometry_msgs::msg::PointStamped tgt_base;
    try {
      tgt_base = tf_buffer_.transform(target_, base_frame_, tf2::durationFromSec(0.05));
    } catch (const tf2::TransformException &ex) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "TF target->base failed: %s", ex.what());
      return;
    }

    tf2::Vector3 tgt_pos(tgt_base.point.x, tgt_base.point.y, tgt_base.point.z);

    // Desired direction camera->target
    tf2::Vector3 d = tgt_pos - cam_pos;
    if (d.length2() < 1e-9) return;
    d.normalize();

    // Errors between current forward and desired direction
    double yaw_err   = wrap_pi(yaw_from_vec(d)   - yaw_from_vec(f));
    double pitch_err = wrap_pi(pitch_from_vec(d) - pitch_from_vec(f));
    double desried_pitch = wrap_pi(pitch_from_vec(d));
    double curr_pitch = wrap_pi(pitch_from_vec(f));

    RCLCPP_INFO(get_logger(), "yaw error %.3f , pitch error %.3f)", yaw_err, pitch_err);
    RCLCPP_INFO(get_logger(), "pitch wanted %.3f , pitch now %.3f)", desried_pitch, curr_pitch);

    if (std::abs(yaw_err) < deadband_rad_) yaw_err = 0.0;
    if (std::abs(pitch_err) < deadband_rad_) pitch_err = 0.0;

    // Map to joint velocities
    const double v_waist = clamp(k_yaw_ * yaw_err, -max_vel_waist_, max_vel_waist_);
    const double v_wrist    = clamp(k_pitch_ * pitch_err ,-max_vel_wrist_, max_vel_wrist_);
    RCLCPP_INFO(get_logger(), "yaw vel %.3f , pitch vel %.3f)", v_waist, v_wrist);

    // Publish jog
    control_msgs::msg::JointJog jog;
    jog.header.stamp = now();
    jog.header.frame_id = base_frame_;
    jog.joint_names = {joint_waist_, joint_wrist_};
    jog.velocities  = {v_waist,v_wrist};
    jog.duration = 1.0 / std::max(1.0, rate_hz_);  // seconds (float64)


    jog_pub_->publish(jog);

  }

  // TF
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  // Servo pub
  rclcpp::Publisher<control_msgs::msg::JointJog>::SharedPtr jog_pub_;

  // Service + timer
  rclcpp::Service<nav_search::srv::TrackTarget>::SharedPtr srv_;
  rclcpp::TimerBase::SharedPtr timer_;

  // State
  bool tracking_enabled_{false};
  bool have_target_{false};
  geometry_msgs::msg::PointStamped target_;

  // Params
  std::string base_frame_, ee_frame_;
  std::string joint_waist_, joint_wrist_;
  double rate_hz_{100.0};
  double deadband_rad_{0.01};
  double k_yaw_{1.5}, k_pitch_{1.5};
  double max_vel_waist_{0.8},  max_vel_wrist_{1.0};
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CameraServoTracker>());
  rclcpp::shutdown();
  return 0;
}
