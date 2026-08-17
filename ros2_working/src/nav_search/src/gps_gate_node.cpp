// map_odom_freeze_node.cpp
//
// Sole authority on the map->odom transform. Replaces robot_localization's own
// TF broadcast (set the map-frame EKF's publish_tf:=false).
//
//   LIVE  (service data=true, default): reconstruct map->odom the way r_l does
//         internally -- T_map_odom = T_map_base * inv(T_odom_base) -- where
//         T_map_base is the EKF's filtered odometry and T_odom_base is
//         Unitree's odometry (from /tf). Publish and cache.
//
//   FROZEN (service data=false): hold the last cached map->odom VALUE, but keep
//         republishing it every cycle. odom->base flows live from Unitree, so
//         map->base = frozen * live moves smoothly -- precise Nav2 / velocity
//         motion in a steady frame.
//
// STAMP FIX (the important part): map->odom is stamped at the SOURCE time of
// the odom->base it was composed against -- NOT this->now(). r_l stamps
// map->odom at the odometry source time, keeping the two links of the chain
// (map->odom and odom->base) aligned on one clock. Stamping at now() puts
// map->odom ahead of the latency-delayed odom->base, so controller lookups at
// ~now fall off the future edge (future-extrapolation) and costmap message
// filters drop odom-stamped sensor data as "earlier than all data in cache".
// Stamping at the odom->base source time makes the chain internally consistent
// at every instant a consumer queries it.
//
// Service: std_srvs/srv/SetBool, default "set_gps_gate" (drop-in for the
// existing arm_search cut_out_gps/restore_gps). data=false -> FREEZE,
// data=true -> LIVE.

#include <algorithm>
#include <atomic>
#include <memory>
#include <mutex>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

using namespace std::chrono_literals;

class MapOdomFreezeNode : public rclcpp::Node
{
public:
  MapOdomFreezeNode()
  : Node("gos_gate")
  {
    // --- Parameters ---
    filtered_odom_topic_ =
      this->declare_parameter<std::string>("filtered_odom_topic", "odometry/global");
    service_name_ =
      this->declare_parameter<std::string>("service_name", "set_gps_gate");

    map_frame_  = this->declare_parameter<std::string>("map_frame",  "map");
    odom_frame_ = this->declare_parameter<std::string>("odom_frame", "odom");
    base_frame_ = this->declare_parameter<std::string>("base_frame", "base_link");

    publish_rate_hz_ = this->declare_parameter<double>("publish_rate_hz", 100.0);
    // Optional small lead added to the SOURCE stamp (mirrors r_l's
    // transform_time_offset). Leave at 0.0 now that stamping is correct; only
    // add a few tens of ms if a residual future-extrapolation remains.
    transform_time_offset_ =
      this->declare_parameter<double>("transform_time_offset", 0.1);

    // Initial mode. true = LIVE (pass-through), false = FROZEN.
    pass_through_.store(this->declare_parameter<bool>("initial_live", true));

    // --- TF ---
    tf_buffer_      = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_    = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(*this);

    // --- I/O ---
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      filtered_odom_topic_, rclcpp::QoS(10),
      std::bind(&MapOdomFreezeNode::odomCallback, this, std::placeholders::_1));

    srv_ = this->create_service<std_srvs::srv::SetBool>(
      service_name_,
      std::bind(&MapOdomFreezeNode::freezeCallback, this,
                std::placeholders::_1, std::placeholders::_2));

    const auto period =
      std::chrono::duration<double>(1.0 / std::max(1.0, publish_rate_hz_));
    timer_ = this->create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      std::bind(&MapOdomFreezeNode::onTimer, this));

    RCLCPP_INFO(get_logger(),
      "map_odom_freeze up: filtered='%s' | %s->%s (child->%s) | %.0f Hz | service '%s' | initial %s",
      filtered_odom_topic_.c_str(), map_frame_.c_str(), odom_frame_.c_str(),
      base_frame_.c_str(), publish_rate_hz_, service_name_.c_str(),
      pass_through_.load() ? "LIVE" : "FROZEN");
    RCLCPP_WARN(get_logger(),
      "This node is the sole publisher of %s->%s. Ensure the EKF has "
      "publish_tf:=false, or the tree will have two authorities.",
      map_frame_.c_str(), odom_frame_.c_str());
  }

private:
  // Cache the latest filtered pose (map->base). odom->base is looked up fresh
  // in the timer, so nothing about it is stored here.
  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    tf2::Transform t;
    const auto & p = msg->pose.pose.position;
    const auto & o = msg->pose.pose.orientation;
    t.setOrigin(tf2::Vector3(p.x, p.y, p.z));
    tf2::Quaternion q(o.x, o.y, o.z, o.w);
    if (q.length2() < 1e-9) {   // guard an all-zero quaternion before init
      return;
    }
    q.normalize();
    t.setRotation(q);

    std::lock_guard<std::mutex> lock(mtx_);
    map_base_ = t;
    have_map_base_ = true;
  }

  void freezeCallback(
    const std::shared_ptr<std_srvs::srv::SetBool::Request> req,
    std::shared_ptr<std_srvs::srv::SetBool::Response> resp)
  {
    const bool live = req->data;   // true = LIVE, false = FREEZE
    pass_through_.store(live, std::memory_order_relaxed);

    bool have_cache;
    {
      std::lock_guard<std::mutex> lock(mtx_);
      have_cache = have_map_odom_;
    }

    // Per spec, success carries the resulting mode (LIVE=true / FROZEN=false).
    resp->success = live;
    if (live) {
      resp->message = "map->odom LIVE (recomputing from EKF)";
    } else if (have_cache) {
      resp->message = "map->odom FROZEN (holding last transform)";
    } else {
      resp->message = "map->odom FREEZE requested but no cached transform yet";
      RCLCPP_WARN(get_logger(), "%s", resp->message.c_str());
    }
    RCLCPP_INFO(get_logger(), "%s", resp->message.c_str());
  }

  void onTimer()
  {
    // Both modes need the latest odom->base: for its VALUE (live) and for its
    // STAMP (both). Stamping map->odom at this source time is the whole fix.
    geometry_msgs::msg::TransformStamped odom_base_msg;
    try {
      odom_base_msg = tf_buffer_->lookupTransform(
        odom_frame_, base_frame_, tf2::TimePointZero);
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "lookup %s->%s failed: %s",
        odom_frame_.c_str(), base_frame_.c_str(), ex.what());
      return;   // if odom->base is unavailable the whole chain is broken anyway
    }

    const rclcpp::Time source_stamp(odom_base_msg.header.stamp);
    const rclcpp::Time out_stamp =
      source_stamp + rclcpp::Duration::from_seconds(transform_time_offset_);

    // Suppress duplicate stamps (timer firing faster than odom->base updates)
    // to avoid TF_REPEATED_DATA spam. The have_last_pub_ guard also protects
    // the first comparison from a clock-source mismatch.
    if (have_last_pub_ && out_stamp == last_pub_stamp_) {
      return;
    }

    tf2::Transform map_odom;

    if (pass_through_.load(std::memory_order_relaxed)) {
      // LIVE: recompute from cached map->base and current odom->base.
      tf2::Transform map_base;
      {
        std::lock_guard<std::mutex> lock(mtx_);
        if (!have_map_base_) {
          RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
            "No filtered odom yet on '%s'; not publishing.",
            filtered_odom_topic_.c_str());
          return;
        }
        map_base = map_base_;
      }

      tf2::Transform odom_base;
      tf2::fromMsg(odom_base_msg.transform, odom_base);
      map_odom = map_base * odom_base.inverse();

      std::lock_guard<std::mutex> lock(mtx_);
      map_odom_ = map_odom;
      have_map_odom_ = true;
    } else {
      // FROZEN: hold the cached map->odom VALUE, restamped at the current
      // source time so the chain stays fresh and queryable.
      std::lock_guard<std::mutex> lock(mtx_);
      if (!have_map_odom_) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
          "Frozen but no cached transform to hold; not publishing.");
        return;
      }
      map_odom = map_odom_;
    }

    broadcast(map_odom, out_stamp);
    last_pub_stamp_ = out_stamp;
    have_last_pub_  = true;
  }

  void broadcast(const tf2::Transform & map_odom, const rclcpp::Time & stamp)
  {
    geometry_msgs::msg::TransformStamped out;
    out.header.stamp    = stamp;
    out.header.frame_id = map_frame_;    // parent
    out.child_frame_id  = odom_frame_;   // child
    out.transform       = tf2::toMsg(map_odom);
    tf_broadcaster_->sendTransform(out);
  }

  // --- Params ---
  std::string filtered_odom_topic_;
  std::string service_name_;
  std::string map_frame_, odom_frame_, base_frame_;
  double publish_rate_hz_{50.0};
  double transform_time_offset_{0.0};

  // --- Mode flag (LIVE=true / FROZEN=false) ---
  std::atomic<bool> pass_through_{true};

  // --- Cached transforms (guarded) ---
  std::mutex mtx_;
  tf2::Transform map_base_;
  tf2::Transform map_odom_;
  bool have_map_base_{false};
  bool have_map_odom_{false};

  // --- Duplicate-stamp suppression (timer thread only) ---
  rclcpp::Time last_pub_stamp_;
  bool have_last_pub_{false};

  // --- ROS handles ---
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr srv_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MapOdomFreezeNode>());
  rclcpp::shutdown();
  return 0;
}