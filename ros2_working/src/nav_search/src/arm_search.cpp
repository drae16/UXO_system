#include <chrono>
#include <memory>
#include <string>
#include <map>
#include <thread>
#include <future>
#include <cmath>
#include <mutex>
#include <algorithm>

#include <sensor_msgs/msg/joint_state.hpp>
#include <interbotix_xs_msgs/msg/joint_single_command.hpp>
#include <std_srvs/srv/set_bool.hpp>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <rclcpp/parameter_client.hpp>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <nav_msgs/msg/odometry.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Vector3.h>

#include "nav_search/action/detect_target.hpp"
#include "nav_search/action/scan_area.hpp"
#include "nav_search/srv/track_target.hpp"
#include "nav2_msgs/action/navigate_to_pose.hpp"

using namespace std::chrono_literals;

enum class ScanState {
  FULL_SCAN,
  EVALUATE_DISTANCE,
  NAV2_APPROACH,
  VELOCITY_APPROACH,
  LOCAL_RECOVERY,
  FINAL_INSPECTION,
  COMPLETE,
  ABORTED
};

struct ScanContext {
  int    sweep_count = 0;
  int    max_sweeps_effective = 0;

  double target_x_odom = 0.0;
  double target_y_odom = 0.0;
  double current_distance_to_target = 0.0;

  double x_base_last = 0.0;
  double y_base_last = 0.0;
};

class ArmSearchNode : public rclcpp::Node
{
public:
  using ScanArea       = nav_search::action::ScanArea;
  using ScanServer     = rclcpp_action::Server<ScanArea>;
  using ScanGoalHandle = rclcpp_action::ServerGoalHandle<ScanArea>;

  using DetectTarget   = nav_search::action::DetectTarget;
  using DetectClient   = rclcpp_action::Client<DetectTarget>;

  using TrackTarget = nav_search::srv::TrackTarget;
  using TrackClient = rclcpp::Client<TrackTarget>;

  using NavigateToPose = nav2_msgs::action::NavigateToPose;
  using Nav2Client     = rclcpp_action::Client<NavigateToPose>;

  ArmSearchNode()
  : Node("arm_search_tracking")
  {
    // --- Frames / joints ---
    planning_group_   = this->declare_parameter<std::string>("planning_group", "interbotix_arm");
    base_joint_name_  = this->declare_parameter<std::string>("base_joint_name", "waist");
    wrist_joint_name_ = this->declare_parameter<std::string>("wrist_joint_name", "wrist_angle");
    arm_base_frame_   = this->declare_parameter<std::string>("arm_base_frame", "vx300s/base_link");
    ee_link_          = this->declare_parameter<std::string>("ee_link", "vx300s/ee_gripper_link");
    cam_link_         = this->declare_parameter<std::string>("cam_link", "vx300s/camera_link");

    // --- Camera-circle geometry ---
    camera_height_        = this->declare_parameter<double>("camera_height",       0.4);
    radius_min_           = this->declare_parameter<double>("camera_radius_min",   0.20);
    radius_max_           = this->declare_parameter<double>("camera_radius_max",   0.65);
    radius_scale_factor_  = this->declare_parameter<double>("camera_radius_scale", 0.2);

    side_offset_       = this->declare_parameter<double>("side_offset", 0.3);
    yaw_tolerance_rad_ = this->declare_parameter<double>("yaw_tolerance_rad", 0.05);

    // --- Joint move polling ---
    joint_move_tolerance_ = this->declare_parameter<double>("joint_move_tolerance", 0.02);
    joint_move_timeout_   = this->declare_parameter<double>("joint_move_timeout", 3.0);
    joint_poll_rate_hz_   = this->declare_parameter<double>("joint_poll_rate_hz", 30.0);

    // --- FSM tunables ---
    max_sweeps_             = this->declare_parameter<int>("max_sweeps", 1);
    desired_final_distance_ = this->declare_parameter<double>("desired_final_distance", 0.40);
    nav2_switch_distance_   = this->declare_parameter<double>("nav2_switch_distance", 1.30);
    nav2_approach_margin_   = this->declare_parameter<double>("nav2_approach_margin", 0.3);

    local_recovery_waist_range_deg_ = this->declare_parameter<double>("local_recovery_waist_range_deg", 15.0);
    local_recovery_waist_steps_     = this->declare_parameter<int>("local_recovery_waist_steps", 5);
    local_recovery_wrist_delta_deg_ = this->declare_parameter<double>("local_recovery_wrist_delta_deg", 10.0);

    gps_gate_service_ = this->declare_parameter<std::string>("gps_gate_service", "set_gps_gate");

    // Stop the base rotation this many seconds' worth of travel early, so the
    // body's spin-down coast lands on target instead of overshooting it.
    rotation_stop_lead_s_ = this->declare_parameter<double>("rotation_stop_lead_s", 0.10);

    // --- Pub / sub ---
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "/odom", 10,
      std::bind(&ArmSearchNode::odom_callback, this, std::placeholders::_1));

    cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>(
      "/cmd_vel_out", 10);

    joint_single_pub_ = this->create_publisher<interbotix_xs_msgs::msg::JointSingleCommand>(
      "/vx300s/commands/joint_single", 10);

    joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
      "/vx300s/joint_states", 10,
      std::bind(&ArmSearchNode::joint_state_callback, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(), "Planning group: %s", planning_group_.c_str());
    RCLCPP_INFO(get_logger(), "Arm base frame: %s", arm_base_frame_.c_str());
    RCLCPP_INFO(get_logger(), "EE link: %s / Camera link: %s", ee_link_.c_str(), cam_link_.c_str());

    // --- TF ---
    tf_buffer_   = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // --- ScanArea action server ---
    scan_server_ = rclcpp_action::create_server<ScanArea>(
      this, "scan_area",
      std::bind(&ArmSearchNode::handle_scan_goal,     this, std::placeholders::_1, std::placeholders::_2),
      std::bind(&ArmSearchNode::handle_scan_cancel,   this, std::placeholders::_1),
      std::bind(&ArmSearchNode::handle_scan_accepted, this, std::placeholders::_1));
  }

private:
  // === Lazy init helpers ====================================================

  void init_move_group_if_needed()
  {
    if (move_group_) return;

    RCLCPP_INFO(get_logger(), "Fetching robot_description and SRDF from /move_group...");

    auto param_node = rclcpp::Node::make_shared("arm_search_param_client");
    const std::string move_group_node_name = "/move_group";

    auto param_client = std::make_shared<rclcpp::AsyncParametersClient>(
      param_node, move_group_node_name);

    while (!param_client->wait_for_service(1s)) {
      if (!rclcpp::ok()) {
        RCLCPP_ERROR(get_logger(), "Interrupted waiting for %s param service.",
                     move_group_node_name.c_str());
        return;
      }
      RCLCPP_INFO(get_logger(), "Waiting for %s parameter service...",
                  move_group_node_name.c_str());
    }

    auto future = param_client->get_parameters(
      {"robot_description", "robot_description_semantic"});

    rclcpp::executors::SingleThreadedExecutor exec;
    exec.add_node(param_node);
    auto ret = exec.spin_until_future_complete(future, 5s);
    exec.remove_node(param_node);

    if (ret != rclcpp::FutureReturnCode::SUCCESS) {
      RCLCPP_ERROR(get_logger(), "Timed out getting parameters from %s",
                   move_group_node_name.c_str());
      return;
    }

    auto params = future.get();
    if (!this->has_parameter("robot_description")) {
      this->declare_parameter<std::string>("robot_description", "");
    }
    if (!this->has_parameter("robot_description_semantic")) {
      this->declare_parameter<std::string>("robot_description_semantic", "");
    }
    this->set_parameters(params);

    RCLCPP_INFO(get_logger(), "Initializing MoveGroupInterface...");
    move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
      shared_from_this(), planning_group_);
    move_group_->setPlanningTime(3.0);

    RCLCPP_INFO(get_logger(), "MoveGroup planning frame: %s",
                move_group_->getPlanningFrame().c_str());
  }

  void init_detect_client_if_needed()
  {
    if (!detect_client_) {
      detect_client_ = rclcpp_action::create_client<DetectTarget>(
        shared_from_this(), "detect_target");
    }
  }

  void init_nav2_client_if_needed()
  {
    if (!nav2_client_) {
      nav2_client_ = rclcpp_action::create_client<NavigateToPose>(
        shared_from_this(), "navigate_to_pose");
    }
  }

  void init_track_client_if_needed()
  {
    if (!track_client_) {
      track_client_ = this->create_client<TrackTarget>("track_target");
    }
  }

  void init_gps_gate_client_if_needed()
  {
    if (!gps_gate_client_) {
      RCLCPP_INFO(get_logger(), "Creating GPS gate service client on '%s'...",
                  gps_gate_service_.c_str());
      gps_gate_client_ = this->create_client<std_srvs::srv::SetBool>(gps_gate_service_);
    }
  }

  // === GPS gate (map->odom freeze during a scan) ============================

  // pass_through = true  -> GPS flows to the global EKF (normal global nav)
  // pass_through = false -> GPS gated: map->odom held steady, motion odom-only
  bool call_gps_gate(bool pass_through)
  {
    init_gps_gate_client_if_needed();
    if (!gps_gate_client_) return false;

    if (!gps_gate_client_->wait_for_service(2s)) {
      RCLCPP_WARN(get_logger(),
                  "GPS gate service '%s' unavailable; leaving GPS fusion unchanged",
                  gps_gate_service_.c_str());
      return false;
    }

    auto request = std::make_shared<std_srvs::srv::SetBool::Request>();
    request->data = pass_through;

    auto fut = gps_gate_client_->async_send_request(request);
    if (fut.wait_for(2s) != std::future_status::ready) {
      RCLCPP_WARN(get_logger(), "GPS gate service call timed out");
      return false;
    }

    auto res = fut.get();
    RCLCPP_INFO(get_logger(), "GPS gate -> %s",
                (res && res->success) ? "PASS-THROUGH" : "GATED");
    return true;
  }

  void cut_out_gps() { RCLCPP_INFO(get_logger(), "[GPS] cutting out GPS (freezing map->odom)"); call_gps_gate(false); }
  void restore_gps() { RCLCPP_INFO(get_logger(), "[GPS] restoring GPS fusion");                call_gps_gate(true);  }

  // RAII: restores GPS on ANY exit from execute_scan (success, abort, cancel,
  // early return, or a thrown exception).
  struct GpsGateGuard {
    ArmSearchNode* node;
    explicit GpsGateGuard(ArmSearchNode* n) : node(n) {}
    ~GpsGateGuard() { if (node) node->restore_gps(); }
    GpsGateGuard(const GpsGateGuard&) = delete;
    GpsGateGuard& operator=(const GpsGateGuard&) = delete;
  };

  // === ScanArea action callbacks ============================================

  rclcpp_action::GoalResponse handle_scan_goal(
    const rclcpp_action::GoalUUID &,
    std::shared_ptr<const ScanArea::Goal> goal)
  {
    RCLCPP_INFO(get_logger(),
                "Received ScanArea goal: angles [%.2f, %.2f], steps=%d, min_conf=%.2f",
                goal->start_angle, goal->end_angle, goal->num_steps, goal->min_confidence);
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse handle_scan_cancel(const std::shared_ptr<ScanGoalHandle>)
  {
    RCLCPP_INFO(get_logger(), "ScanArea goal cancel requested");
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void handle_scan_accepted(const std::shared_ptr<ScanGoalHandle> goal_handle)
  {
    std::thread(&ArmSearchNode::execute_scan, this, goal_handle).detach();
  }

  // === Odometry cache (fresh yaw for base rotation) =========================

  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(odom_mutex_);
    odom_x_ = msg->pose.pose.position.x;
    odom_y_ = msg->pose.pose.position.y;
    const auto & q = msg->pose.pose.orientation;
    double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
    double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
    odom_yaw_ = std::atan2(siny_cosp, cosy_cosp);
    have_odom_ = true;
  }

  bool get_robot_pose_odom_fast(double &x, double &y, double &yaw)
  {
    std::lock_guard<std::mutex> lock(odom_mutex_);
    if (!have_odom_) return false;
    x = odom_x_; y = odom_y_; yaw = odom_yaw_;
    return true;
  }

  // === TF helpers ===========================================================

  bool get_robot_pose_odom(double &x, double &y, double &yaw)
  {
    if (!tf_buffer_) return false;

    geometry_msgs::msg::TransformStamped tf;
    try {
      tf = tf_buffer_->lookupTransform("odom", "base_link", tf2::TimePointZero);
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN(get_logger(), "Failed to get transform odom->base_link: %s", ex.what());
      return false;
    }

    x = tf.transform.translation.x;
    y = tf.transform.translation.y;
    const auto & q = tf.transform.rotation;
    double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
    double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
    yaw = std::atan2(siny_cosp, cosy_cosp);
    return true;
  }

  bool transform_arm_point_to_odom(double xb, double yb, double &x_odom, double &y_odom)
  {
    if (!tf_buffer_) return false;

    geometry_msgs::msg::PoseStamped p_arm;
    p_arm.header.frame_id = arm_base_frame_;
    p_arm.header.stamp    = this->now();
    p_arm.pose.position.x = xb;
    p_arm.pose.position.y = yb;
    p_arm.pose.orientation.w = 1.0;

    try {
      auto p_odom = tf_buffer_->transform(p_arm, "odom", tf2::durationFromSec(0.5));
      x_odom = p_odom.pose.position.x;
      y_odom = p_odom.pose.position.y;
      return true;
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN(get_logger(), "Transform %s->odom failed: %s", arm_base_frame_.c_str(), ex.what());
      return false;
    }
  }

  bool transform_arm_point_to_base_link(double xb, double yb, double &x_bl, double &y_bl)
  {
    if (!tf_buffer_) return false;

    geometry_msgs::msg::PoseStamped p_arm;
    p_arm.header.frame_id = arm_base_frame_;
    p_arm.header.stamp    = this->now();
    p_arm.pose.position.x = xb;
    p_arm.pose.position.y = yb;
    p_arm.pose.orientation.w = 1.0;

    try {
      auto p_bl = tf_buffer_->transform(p_arm, "base_link", tf2::durationFromSec(0.5));
      x_bl = p_bl.pose.position.x;
      y_bl = p_bl.pose.position.y;
      return true;
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN(get_logger(), "Transform %s->base_link failed: %s", arm_base_frame_.c_str(), ex.what());
      return false;
    }
  }

  // === Camera-circle pose computation =======================================

  geometry_msgs::msg::Pose compute_camera_pose_on_circle(const geometry_msgs::msg::Point &target_arm)
  {
    double X = target_arm.x, Y = target_arm.y, Z = target_arm.z;

    double r_xy = std::sqrt(X*X + Y*Y);
    if (r_xy < 1e-3) r_xy = 1e-3;

    double ux = X / r_xy, uy = Y / r_xy;

    double R_des = radius_scale_factor_ * r_xy;
    R_des = std::clamp(R_des, radius_min_, radius_max_);

    geometry_msgs::msg::Pose cam_pose;
    cam_pose.position.x = ux * R_des;
    cam_pose.position.y = uy * R_des;
    cam_pose.position.z = camera_height_;

    tf2::Vector3 cam_pos(cam_pose.position.x, cam_pose.position.y, cam_pose.position.z);
    tf2::Vector3 tgt_pos(X, Y, Z);

    tf2::Vector3 x_axis = tgt_pos - cam_pos;
    if (x_axis.length2() < 1e-6) x_axis = tf2::Vector3(1.0, 0.0, 0.0);
    x_axis.normalize();

    tf2::Vector3 world_up(0.0, 0.0, 1.0);
    tf2::Vector3 y_axis = x_axis.cross(-world_up);
    tf2::Vector3 z_axis = x_axis.cross(y_axis);

    if (y_axis.length2() < 1e-6) {
      z_axis = tf2::Vector3(0.0, 0.0, 1.0);
      y_axis = tf2::Vector3(0.0, 1.0, 0.0);
      x_axis = tf2::Vector3(1.0, 0.0, 0.0);
    }
    y_axis.normalize();
    z_axis.normalize();

    tf2::Matrix3x3 R(
      x_axis.x(), y_axis.x(), z_axis.x(),
      x_axis.y(), y_axis.y(), z_axis.y(),
      x_axis.z(), y_axis.z(), z_axis.z());

    tf2::Quaternion q;
    R.getRotation(q);
    q.normalize();
    cam_pose.orientation = tf2::toMsg(q);

    geometry_msgs::msg::Pose ee_pose;
    if (!cameraPoseToEePoseInBase(cam_pose, ee_pose)) {
      return cam_pose;
    }
    return ee_pose;
  }

  bool cameraPoseToEePoseInBase(const geometry_msgs::msg::Pose& cam_pose_in_base,
                                geometry_msgs::msg::Pose& ee_pose_in_base)
  {
    geometry_msgs::msg::TransformStamped ee_to_cam;
    try {
      ee_to_cam = tf_buffer_->lookupTransform(ee_link_, cam_link_, tf2::TimePointZero);
    } catch (const tf2::TransformException& ex) {
      RCLCPP_WARN(get_logger(), "TF lookup %s->%s failed: %s",
                  ee_link_.c_str(), cam_link_.c_str(), ex.what());
      return false;
    }

    tf2::Transform T_base_cam, T_ee_cam;
    tf2::fromMsg(cam_pose_in_base, T_base_cam);
    tf2::fromMsg(ee_to_cam.transform, T_ee_cam);

    tf2::Transform T_base_ee = T_base_cam * T_ee_cam.inverse();

    ee_pose_in_base.position.x = T_base_ee.getOrigin().x();
    ee_pose_in_base.position.y = T_base_ee.getOrigin().y();
    ee_pose_in_base.position.z = T_base_ee.getOrigin().z();
    ee_pose_in_base.orientation = tf2::toMsg(T_base_ee.getRotation());

    double roll, pitch, yaw;
    tf2::Matrix3x3(T_base_ee.getRotation()).getRPY(roll, pitch, yaw);
    RCLCPP_INFO(get_logger(),
      "EE pose in base: X=%.3f Y=%.3f Z=%.3f | RPY(deg) %.1f %.1f %.1f",
      ee_pose_in_base.position.x, ee_pose_in_base.position.y, ee_pose_in_base.position.z,
      roll * 180.0 / M_PI, pitch * 180.0 / M_PI, yaw * 180.0 / M_PI);
    return true;
  }

  bool move_arm_to_pose(const geometry_msgs::msg::Pose &pose)
  {
    if (!move_group_) return false;

    move_group_->setStartStateToCurrentState();
    move_group_->setPoseTarget(pose, ee_link_);

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    if (move_group_->plan(plan) != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_WARN(get_logger(), "No plan found for camera tracking pose");
      return false;
    }
    if (move_group_->execute(plan) != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_WARN(get_logger(), "Execution for camera tracking pose failed");
      return false;
    }
    return true;
  }

  // === Joint state tracking / single-joint moves ============================

  void joint_state_callback(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(joint_state_mutex_);
    for (size_t i = 0; i < msg->name.size(); ++i) {
      latest_joint_positions_[msg->name[i]] = msg->position[i];
    }
  }

  bool get_joint_position(const std::string &joint_name, double &position)
  {
    std::lock_guard<std::mutex> lock(joint_state_mutex_);
    auto it = latest_joint_positions_.find(joint_name);
    if (it == latest_joint_positions_.end()) return false;
    position = it->second;
    return true;
  }

  bool move_single_joint_fast(const std::string &joint_name, double target_angle)
  {
    if (!joint_single_pub_) {
      RCLCPP_ERROR(get_logger(), "joint_single_pub_ not initialized");
      return false;
    }

    double start_pos = 0.0;
    if (!get_joint_position(joint_name, start_pos)) {
      RCLCPP_WARN(get_logger(),
                  "No joint_states yet for '%s'; publishing command anyway", joint_name.c_str());
    }

    interbotix_xs_msgs::msg::JointSingleCommand cmd;
    cmd.name = joint_name;
    cmd.cmd  = target_angle;
    joint_single_pub_->publish(cmd);

    rclcpp::Rate rate(joint_poll_rate_hz_);
    auto start_time = this->now();
    int consecutive_ok = 0;

    while (rclcpp::ok()) {
      double elapsed = (this->now() - start_time).seconds();
      if (elapsed > joint_move_timeout_) {
        RCLCPP_WARN(get_logger(), "Timeout waiting for joint '%s' -> %.3f rad (%.2fs)",
                    joint_name.c_str(), target_angle, elapsed);
        return false;
      }

      double current_pos;
      if (get_joint_position(joint_name, current_pos)) {
        if (std::fabs(current_pos - target_angle) < joint_move_tolerance_) {
          if (++consecutive_ok >= 2) return true;
        } else {
          consecutive_ok = 0;
        }
      }
      rate.sleep();
    }
    return false;
  }

  bool move_base_to(double target_angle)  { return move_single_joint_fast(base_joint_name_, target_angle); }
  bool move_wrist_to(double target_angle) { return move_single_joint_fast(wrist_joint_name_, target_angle); }

  // === Full-arm joint state capture / restore ===============================

  std::map<std::string, double> capture_arm_joint_state()
  {
    std::map<std::string, double> state;
    if (!move_group_) return state;

    move_group_->setStartStateToCurrentState();
    std::vector<std::string> names  = move_group_->getJointNames();
    std::vector<double>      values = move_group_->getCurrentJointValues();
    for (size_t i = 0; i < names.size() && i < values.size(); ++i) {
      state[names[i]] = values[i];
    }
    return state;
  }

  bool restore_arm_joint_state(const std::map<std::string, double> &saved_state)
  {
    if (!move_group_ || saved_state.empty()) return false;

    move_group_->setStartStateToCurrentState();
    if (!move_group_->setJointValueTarget(saved_state)) {
      RCLCPP_ERROR(get_logger(), "Failed to set joint value target for state restore");
      return false;
    }

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    if (move_group_->plan(plan) != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_WARN(get_logger(), "No plan found to restore joint state");
      return false;
    }
    if (move_group_->execute(plan) != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_WARN(get_logger(), "Execution to restore joint state failed");
      return false;
    }
    return true;
  }

  // Track once, target given in ODOM: odom -> arm_base -> circle pose -> MoveIt
  bool track_target_once(double target_x_odom, double target_y_odom)
  {
    if (!tf_buffer_) return false;

    geometry_msgs::msg::PoseStamped tgt_odom;
    tgt_odom.header.frame_id = "odom";
    tgt_odom.header.stamp    = this->now();
    tgt_odom.pose.position.x = target_x_odom;
    tgt_odom.pose.position.y = target_y_odom;
    tgt_odom.pose.orientation.w = 1.0;

    geometry_msgs::msg::PoseStamped tgt_arm;
    try {
      tgt_arm = tf_buffer_->transform(tgt_odom, arm_base_frame_, tf2::durationFromSec(0.5));
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN(get_logger(), "Transform odom->%s failed: %s", arm_base_frame_.c_str(), ex.what());
      return false;
    }

    geometry_msgs::msg::Pose cam_pose = compute_camera_pose_on_circle(tgt_arm.pose.position);
    return move_arm_to_pose(cam_pose);
  }

  // === MoveIt named poses ===================================================

  bool move_arm_to_named(const char* name)
  {
    if (!move_group_) return false;

    RCLCPP_INFO(get_logger(), "Moving arm to named state '%s'", name);
    move_group_->setStartStateToCurrentState();
    if (!move_group_->setNamedTarget(name)) {
      RCLCPP_WARN(get_logger(), "Named target '%s' not found for group '%s'",
                  name, planning_group_.c_str());
      return false;
    }

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    if (move_group_->plan(plan) != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_WARN(get_logger(), "No plan found to '%s'", name);
      return false;
    }
    if (move_group_->execute(plan) != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_WARN(get_logger(), "Execution to '%s' failed", name);
      return false;
    }
    return true;
  }

  bool move_arm_to_search_pose() { return move_arm_to_named("Search"); }
  bool move_arm_to_stow_pose()   { return move_arm_to_named("Sleep");  }

  // === YOLO DetectTarget client =============================================

  bool call_detect_target(double min_conf, double &x, double &y)
  {
    double nx1, ny1, nx2, ny2;
    return call_detect_target(min_conf, x, y, nx1, ny1, nx2, ny2);
  }

  // Full overload: center + two nearest bbox corners (arm base frame).
  bool call_detect_target(double min_conf, double &x, double &y,
                          double &near_x1, double &near_y1,
                          double &near_x2, double &near_y2)
  {
    init_detect_client_if_needed();
    if (!detect_client_) return false;

    if (!detect_client_->wait_for_action_server(2s)) {
      RCLCPP_WARN(get_logger(), "detect_target action server not available");
      return false;
    }

    auto goal_msg = DetectTarget::Goal();
    goal_msg.min_confidence = static_cast<float>(min_conf);

    auto future_goal = detect_client_->async_send_goal(goal_msg, DetectClient::SendGoalOptions());
    if (future_goal.wait_for(5s) != std::future_status::ready) {
      RCLCPP_WARN(get_logger(), "Timeout waiting for detect_target goal response");
      return false;
    }
    auto goal_handle = future_goal.get();
    if (!goal_handle) {
      RCLCPP_WARN(get_logger(), "detect_target goal rejected");
      return false;
    }

    auto result_future = detect_client_->async_get_result(goal_handle);
    if (result_future.wait_for(10s) != std::future_status::ready) {
      RCLCPP_WARN(get_logger(), "Timeout waiting for detect_target result");
      return false;
    }

    auto result = result_future.get().result;
    if (!result || !result->found) {
      RCLCPP_INFO(get_logger(), "detect_target: no target found");
      return false;
    }

    x = result->x_base;   y = result->y_base;
    near_x1 = result->near_x1; near_y1 = result->near_y1;
    near_x2 = result->near_x2; near_y2 = result->near_y2;

    RCLCPP_INFO(get_logger(),
      "detect_target: (%.2f, %.2f) arm base, corners (%.2f,%.2f) (%.2f,%.2f)",
      x, y, near_x1, near_y1, near_x2, near_y2);
    return true;
  }

  // === Visual-servo track service (scaffolding for FINAL_POSITIONING) =======

  bool call_track_servo_start(double odom_x, double odom_y)
  {
    init_track_client_if_needed();
    if (!track_client_ || !track_client_->wait_for_service(2s)) {
      RCLCPP_WARN(get_logger(), "track_target service not available");
      return false;
    }

    auto request = std::make_shared<TrackTarget::Request>();
    request->enable = true;
    request->target.header.frame_id = "odom";
    request->target.header.stamp = this->now();
    request->target.point.x = odom_x;
    request->target.point.y = odom_y;

    auto fut = track_client_->async_send_request(request);
    if (fut.wait_for(1s) != std::future_status::ready) return false;
    auto res = fut.get();
    return res && res->accepted;
  }

  bool call_track_servo_stop()
  {
    init_track_client_if_needed();
    if (!track_client_ || !track_client_->wait_for_service(2s)) {
      RCLCPP_WARN(get_logger(), "track_target service not available");
      return false;
    }

    auto request = std::make_shared<TrackTarget::Request>();
    request->enable = false;
    request->target.header.frame_id = "odom";
    request->target.header.stamp = this->now();

    auto fut = track_client_->async_send_request(request);
    if (fut.wait_for(1s) != std::future_status::ready) return false;
    auto res = fut.get();
    return res && res->accepted;
  }

  // === Nav2 "step" toward target ============================================

  bool step_nav2_towards_target(double target_x_odom, double target_y_odom,
                                double desired_dist, double dist_tolerance)
  {
    init_nav2_client_if_needed();
    if (!nav2_client_) {
      RCLCPP_ERROR(get_logger(), "Nav2 client not initialized");
      return false;
    }

    double rx, ry, ryaw;
    if (!get_robot_pose_odom(rx, ry, ryaw)) return false;

    double dx = target_x_odom - rx;
    double dy = target_y_odom - ry;
    double dist = std::sqrt(dx*dx + dy*dy);
    RCLCPP_INFO(get_logger(), "Nav step: current dist to target = %.3f m", dist);

    if (dist <= desired_dist + dist_tolerance) {
      RCLCPP_INFO(get_logger(), "Already within %.2f +/- %.2f m, skipping Nav2 step",
                  desired_dist, dist_tolerance);
      return true;
    }

    double vx = dx / dist, vy = dy / dist;
    double goal_dist_from_robot = dist - desired_dist;
    RCLCPP_INFO(get_logger(), "[DEBUGG] goal dist from robot = %.2f",
                  goal_dist_from_robot);
    double gx = rx + vx * goal_dist_from_robot;
    double gy = ry + vy * goal_dist_from_robot;
    double goal_yaw = std::atan2(dy, dx);

    geometry_msgs::msg::PoseStamped goal_pose;
    goal_pose.header.frame_id = "odom";
    goal_pose.header.stamp = rclcpp::Time(0);   // latest transform each replan
    goal_pose.pose.position.x = gx;
    goal_pose.pose.position.y = gy;
    goal_pose.pose.orientation.w = std::cos(goal_yaw * 0.5);
    goal_pose.pose.orientation.z = std::sin(goal_yaw * 0.5);

    if (!nav2_client_->wait_for_action_server(5s)) {
      RCLCPP_WARN(get_logger(), "Nav2 NavigateToPose server not available");
      return false;
    }

    NavigateToPose::Goal goal_msg;
    goal_msg.pose = goal_pose;

    auto future_goal = nav2_client_->async_send_goal(goal_msg, Nav2Client::SendGoalOptions());
    auto goal_handle = future_goal.get();
    if (!goal_handle) {
      RCLCPP_WARN(get_logger(), "Nav2 goal rejected");
      return false;
    }

    auto code = nav2_client_->async_get_result(goal_handle).get().code;
    RCLCPP_INFO(get_logger(), "Nav2 NavigateToPose result code: %d", static_cast<int>(code));
    return code == rclcpp_action::ResultCode::SUCCEEDED;
  }

  // === Base rotation (closed-loop on /odom yaw) =============================

  static double normalize_angle(double a)
  {
    while (a >  M_PI) a -= 2.0 * M_PI;
    while (a < -M_PI) a += 2.0 * M_PI;
    return a;
  }

  // Rotate the base by a RELATIVE yaw, closed-loop on the /odom cache (fresh,
  // no TF buffer, no map-frame composition). A stop-lead cuts the command
  // early so the body's spin-down coast lands on target instead of overshooting.
  bool rotate_base_relative(double delta_rad, double ang_speed = 0.50)
  {
    if (!cmd_vel_pub_) return false;
    if (std::fabs(delta_rad) < 1e-3) return true;

    double x0, y0, yaw_start;
    if (!get_robot_pose_odom_fast(x0, y0, yaw_start)) {
      RCLCPP_WARN(get_logger(), "[Rotate] no /odom yaw yet; cannot rotate");
      return false;
    }

    const double yaw_goal   = normalize_angle(yaw_start + delta_rad);
    const double stop_lead   = rotation_stop_lead_s_ * ang_speed;   // rad
    const double timeout_s   = (std::fabs(delta_rad) / ang_speed) * 3.0 + 2.0;

    RCLCPP_INFO(get_logger(),
      "[Rotate] closed-loop %.1f deg (start=%.1f goal=%.1f) lead=%.3f rad",
      delta_rad * 180.0 / M_PI, yaw_start * 180.0 / M_PI, yaw_goal * 180.0 / M_PI, stop_lead);

    rclcpp::Rate rate(50.0);
    auto start_time = this->now();

    while (rclcpp::ok()) {
      if ((this->now() - start_time).seconds() > timeout_s) {
        RCLCPP_WARN(get_logger(), "[Rotate] timeout");
        break;
      }

      double rx, ry, ryaw;
      if (!get_robot_pose_odom_fast(rx, ry, ryaw)) { rate.sleep(); continue; }

      double err = normalize_angle(yaw_goal - ryaw);
      if (std::fabs(err) < yaw_tolerance_rad_ + stop_lead) break;

      geometry_msgs::msg::Twist cmd;
      cmd.angular.z = (err > 0.0) ? ang_speed : -ang_speed;
      cmd_vel_pub_->publish(cmd);
      rate.sleep();
    }

    cmd_vel_pub_->publish(geometry_msgs::msg::Twist{});   // stop
    rclcpp::sleep_for(400ms);                             // settle before read

    double x1, y1, yaw_after;
    if (get_robot_pose_odom_fast(x1, y1, yaw_after)) {
      RCLCPP_INFO(get_logger(),
        "[Rotate] commanded %.1f deg, achieved %.1f deg, residual %.1f deg",
        delta_rad * 180.0 / M_PI,
        normalize_angle(yaw_after - yaw_start) * 180.0 / M_PI,
        normalize_angle(yaw_goal - yaw_after) * 180.0 / M_PI);
    }
    return true;
  }

  // Compute the side-approach geometry from a base_link-relative target, rotate
  // to face the offset heading, and output the forward drive distance. Both
  // rotation and drive are widened by the target's near-edge half-width so the
  // robot clears the target's physical footprint, not just its center point.
  bool rotate_to_side_offset(
    double target_x_bl, double target_y_bl,
    double near1_x_bl, double near1_y_bl,
    double near2_x_bl, double near2_y_bl,
    double &drive_dist, double ang_speed = 0.50)
  {
    double remain_dist = std::sqrt(target_x_bl*target_x_bl + target_y_bl*target_y_bl);
    if (remain_dist < 1e-6) { drive_dist = 0.0; return true; }

    // Bearing to target relative to current heading (0 = straight ahead),
    // since the target is already expressed in base_link.
    double theta_rel = std::atan2(target_y_bl, target_x_bl);

    double edge_dx = near2_x_bl - near1_x_bl;
    double edge_dy = near2_y_bl - near1_y_bl;
    double edge_half_width = 0.5 * std::sqrt(edge_dx*edge_dx + edge_dy*edge_dy);

    const double s = side_offset_ + edge_half_width;
    double angle_to_goal = std::atan2(s, remain_dist);

    double s_sq_diff = remain_dist*remain_dist - s*s;
    if (s_sq_diff < 0.0) {
      RCLCPP_WARN(get_logger(),
        "[VelocityApproach] offset (%.3f) exceeds remaining distance (%.3f); clamping drive to 0",
        s, remain_dist);
      drive_dist = 0.0;
    } else {
      drive_dist = std::sqrt(s_sq_diff);
    }

    // Two candidate relative turns; take the smaller-magnitude one.
    double delta_left  = normalize_angle(theta_rel - angle_to_goal);
    double delta_right = normalize_angle(theta_rel + angle_to_goal);
    double delta = (std::fabs(delta_left) < std::fabs(delta_right)) ? delta_left : delta_right;

    RCLCPP_INFO(get_logger(),
      "[VelocityApproach] center_dist=%.3f edge_hw=%.3f offset=%.3f drive=%.3f turn=%.1f deg",
      remain_dist, edge_half_width, s, drive_dist, delta * 180.0 / M_PI);

    return rotate_base_relative(delta, ang_speed);
  }

  bool drive_forward_to_side_offset(double target_x_bl, double target_y_bl,
                                    double move_dist, double forward_speed = 0.3)
  {
    if (!cmd_vel_pub_) {
      RCLCPP_WARN(get_logger(), "cmd_vel publisher not initialized");
      return false;
    }

    double dist = std::sqrt(target_x_bl*target_x_bl + target_y_bl*target_y_bl);
    if (dist < side_offset_ + 0.05) {
      RCLCPP_INFO(get_logger(),
        "[VelocityApproach] already closer than side_offset (dist=%.3f side=%.3f)",
        dist, side_offset_);
      return true;
    }

    const double duration = move_dist * 0.60 / forward_speed;
    RCLCPP_INFO(get_logger(), "Driving relative: dist=%.2f m vx=%.2f dur=%.2f s",
                move_dist, forward_speed, duration);

    rclcpp::Rate rate(20.0);
    geometry_msgs::msg::Twist cmd;
    cmd.linear.x = forward_speed;

    auto start = this->now();
    while (rclcpp::ok() && (this->now() - start).seconds() < duration) {
      cmd_vel_pub_->publish(cmd);
      rate.sleep();
    }
    cmd_vel_pub_->publish(geometry_msgs::msg::Twist{});   // stop

    RCLCPP_INFO(get_logger(), "[VelocityApproach] estimated remaining distance ~ %.3f m",
                dist - move_dist);
    return true;
  }

  // === Full-scan sweep (detection phase) ====================================

  bool run_full_scan_sweep(
    const std::shared_ptr<ScanGoalHandle> &goal_handle,
    std::shared_ptr<const ScanArea::Goal> goal,
    double &target_x_odom, double &target_y_odom,
    double &x_base_det, double &y_base_det)
  {
    int steps = goal->num_steps;
    if (steps <= 1) steps = 2;

    double start_angle = goal->start_angle;
    double end_angle   = goal->end_angle;
    double step        = (end_angle - start_angle) / static_cast<double>(steps - 1);

    auto feedback = std::make_shared<ScanArea::Feedback>();

    const double search_wrist_angle = 1.75;
    const int    max_levels = 3;
    const double delta_rad  = 20.0 * M_PI / 180.0;
    const double offsets[max_levels] = { +delta_rad, 0.0, -delta_rad };

    int iteration = 0;
    for (int level = 0; level < max_levels; ++level) {
      double wrist_angle = search_wrist_angle + offsets[level];
      if (!move_wrist_to(wrist_angle)) {
        RCLCPP_WARN(get_logger(), "Could not move wrist to level %d; skipping", level);
        continue;
      }

      for (int i = 0; i < steps; ++i) {
        double angle = (iteration % 2 == 0)
          ? start_angle + step * static_cast<double>(i)
          : end_angle   - step * static_cast<double>(i);

        feedback->current_angle = static_cast<float>(angle);
        feedback->progress = static_cast<float>(
          100.0 * (static_cast<double>(i + level * steps) /
                   static_cast<double>(max_levels * steps - 1)));
        goal_handle->publish_feedback(feedback);

        if (goal_handle->is_canceling()) return false;

        if (!move_base_to(angle)) {
          RCLCPP_WARN(get_logger(), "Joint move failed for this step, skipping.");
          continue;
        }

        rclcpp::sleep_for(40ms);

        double xb, yb;
        if (!call_detect_target(goal->min_confidence, xb, yb)) continue;

        double cand_x_odom, cand_y_odom;
        if (!transform_arm_point_to_odom(xb, yb, cand_x_odom, cand_y_odom)) continue;

        // Aim the arm at the candidate, then re-detect to verify.
        auto pre_state = capture_arm_joint_state();
        track_target_once(cand_x_odom, cand_y_odom);

        double xb_verify, yb_verify;
        if (call_detect_target(goal->min_confidence, xb_verify, yb_verify)) {
          target_x_odom = cand_x_odom;
          target_y_odom = cand_y_odom;
          x_base_det = xb_verify;
          y_base_det = yb_verify;
          RCLCPP_INFO(get_logger(),
            "Sweep hit: odom (%.2f, %.2f), arm_base (%.2f, %.2f)",
            target_x_odom, target_y_odom, x_base_det, y_base_det);
          return true;
        }

        RCLCPP_INFO(get_logger(), "Verification failed; restoring pose and resuming sweep");
        restore_arm_joint_state(pre_state);
      }
      iteration += 1;
    }
    return false;
  }

  // === Local recovery search ================================================

  bool run_local_recovery_search(double min_confidence, double &xb_out, double &yb_out)
  {
    move_arm_to_stow_pose();

    if (!track_target_once(last_known_x_odom_, last_known_y_odom_)) {
      RCLCPP_WARN(get_logger(), "[LocalRecovery] failed to aim arm at last known position");
      return false;
    }
    RCLCPP_INFO(get_logger(), "[LocalRecovery] pointing at last known (%.3f, %.3f)",
                last_known_x_odom_, last_known_y_odom_);

    double center_waist, center_wrist;
    if (!get_joint_position(base_joint_name_, center_waist) ||
        !get_joint_position(wrist_joint_name_, center_wrist)) {
      RCLCPP_WARN(get_logger(), "[LocalRecovery] could not read waist/wrist state");
      return false;
    }

    const double wrist_delta = local_recovery_wrist_delta_deg_ * M_PI / 180.0;
    const double wrist_offsets[3] = { +wrist_delta, 0.0, -wrist_delta };

    const double waist_range = local_recovery_waist_range_deg_ * M_PI / 180.0;
    const int    waist_steps = std::max(2, local_recovery_waist_steps_);
    const double waist_step  = (2.0 * waist_range) / static_cast<double>(waist_steps - 1);

    for (int level = 0; level < 3; ++level) {
      double wrist_target = center_wrist + wrist_offsets[level];
      if (!move_wrist_to(wrist_target)) {
        RCLCPP_WARN(get_logger(), "[LocalRecovery] wrist move failed at level %d", level);
        continue;
      }

      for (int i = 0; i < waist_steps; ++i) {
        double waist_target = (center_waist - waist_range) + waist_step * static_cast<double>(i);
        if (!move_base_to(waist_target)) continue;

        rclcpp::sleep_for(40ms);

        double xb, yb;
        if (call_detect_target(min_confidence, xb, yb)) {
          RCLCPP_INFO(get_logger(),
            "[LocalRecovery] reacquired at wrist level %d, waist %.3f rad", level, waist_target);
          xb_out = xb;
          yb_out = yb;
          return true;
        }
      }
    }
    return false;
  }

  // === Core Scan implementation (FSM dispatch) ==============================

  void execute_scan(const std::shared_ptr<ScanGoalHandle> goal_handle)
  {
    auto goal = goal_handle->get_goal();
    auto result = std::make_shared<ScanArea::Result>();
    result->found = false;

    init_move_group_if_needed();
    init_detect_client_if_needed();
    init_nav2_client_if_needed();
    init_track_client_if_needed();
    init_gps_gate_client_if_needed();

    if (!move_group_) {
      RCLCPP_ERROR(get_logger(), "MoveGroupInterface not initialized; aborting scan.");
      goal_handle->abort(result);
      return;   // GPS never cut out here -- nothing to restore
    }

    cut_out_gps();
    GpsGateGuard gps_guard(this);   // restores GPS on every exit below

    ScanContext ctx;
    ctx.max_sweeps_effective = max_sweeps_;
    ScanState state = ScanState::FULL_SCAN;

    while (rclcpp::ok()) {
      if (goal_handle->is_canceling()) {
        move_arm_to_stow_pose();
        goal_handle->canceled(result);
        return;
      }

      switch (state) {

        case ScanState::FULL_SCAN: {
          ctx.sweep_count++;
          RCLCPP_INFO(get_logger(), "[FULL_SCAN] sweep #%d (max_effective=%d)",
                      ctx.sweep_count, ctx.max_sweeps_effective);

          if (!move_arm_to_search_pose()) {
            RCLCPP_WARN(get_logger(), "Could not move into search pose");
            state = ScanState::ABORTED;
            break;
          }

          double tx, ty, xb, yb;
          bool verified = run_full_scan_sweep(goal_handle, goal, tx, ty, xb, yb);

          if (goal_handle->is_canceling()) {
            move_arm_to_stow_pose();
            goal_handle->canceled(result);
            return;
          }

          if (verified) {
            ctx.target_x_odom = tx; ctx.target_y_odom = ty;
            ctx.x_base_last = xb;   ctx.y_base_last = yb;
            last_known_x_odom_ = tx; last_known_y_odom_ = ty;
            has_last_known_position_ = true;
            ctx.max_sweeps_effective = 5;

            result->found = true;
            result->x_base = static_cast<float>(xb);
            result->y_base = static_cast<float>(yb);
            state = ScanState::EVALUATE_DISTANCE;
          } else {
            state = (ctx.sweep_count < ctx.max_sweeps_effective)
                      ? ScanState::FULL_SCAN : ScanState::ABORTED;
          }
          break;
        }

        case ScanState::EVALUATE_DISTANCE: {
          double rx, ry, ryaw;
          if (!get_robot_pose_odom(rx, ry, ryaw)) {
            RCLCPP_WARN(get_logger(), "[EVALUATE_DISTANCE] failed to get robot pose");
            state = ScanState::ABORTED;
            break;
          }

          double dx = ctx.target_x_odom - rx;
          double dy = ctx.target_y_odom - ry;
          ctx.current_distance_to_target = std::sqrt(dx*dx + dy*dy);

          RCLCPP_INFO(get_logger(), "[EVALUATE_DISTANCE] dist=%.3f (nav2_switch=%.2f final=%.2f)",
                      ctx.current_distance_to_target, nav2_switch_distance_, desired_final_distance_);

          if (ctx.current_distance_to_target > nav2_switch_distance_ + nav2_approach_margin_) {
            state = ScanState::NAV2_APPROACH;
          } else if (ctx.current_distance_to_target > desired_final_distance_) {
            state = ScanState::VELOCITY_APPROACH;
          } else {
            state = ScanState::FINAL_INSPECTION;
          }
          break;
        }

        case ScanState::NAV2_APPROACH: {
          move_arm_to_stow_pose();

          double goal_dist = nav2_switch_distance_;
          if (!step_nav2_towards_target(ctx.target_x_odom, ctx.target_y_odom, goal_dist, 0.0)) {
            RCLCPP_WARN(get_logger(), "[NAV2_APPROACH] Nav2 step failed");
            state = ScanState::LOCAL_RECOVERY;
            break;
          }

          double xb, yb;
          if (track_target_once(ctx.target_x_odom, ctx.target_y_odom) &&
              call_detect_target(goal->min_confidence, xb, yb)) {
            ctx.x_base_last = xb; ctx.y_base_last = yb;
            result->x_base = static_cast<float>(xb);
            result->y_base = static_cast<float>(yb);
            state = ScanState::EVALUATE_DISTANCE;
          } else {
            state = ScanState::LOCAL_RECOVERY;
          }
          break;
        }

        case ScanState::VELOCITY_APPROACH: {
          double xb0, yb0, near1_xb, near1_yb, near2_xb, near2_yb;
          if (!call_detect_target(goal->min_confidence, xb0, yb0,
                                  near1_xb, near1_yb, near2_xb, near2_yb)) {
            RCLCPP_WARN(get_logger(), "[VELOCITY_APPROACH] no fresh detection with corners");
            state = ScanState::LOCAL_RECOVERY;
            break;
          }

          move_arm_to_stow_pose();

          double target_x_bl, target_y_bl, near1_x_bl, near1_y_bl, near2_x_bl, near2_y_bl;
          bool tf_ok =
            transform_arm_point_to_base_link(xb0, yb0, target_x_bl, target_y_bl) &&
            transform_arm_point_to_base_link(near1_xb, near1_yb, near1_x_bl, near1_y_bl) &&
            transform_arm_point_to_base_link(near2_xb, near2_yb, near2_x_bl, near2_y_bl);

          if (!tf_ok) {
            RCLCPP_WARN(get_logger(), "[VELOCITY_APPROACH] failed to transform target/corners to base_link");
            state = ScanState::LOCAL_RECOVERY;
            break;
          }

          double move_dist = 0.0;
          bool ok = rotate_to_side_offset(
                      target_x_bl, target_y_bl,
                      near1_x_bl, near1_y_bl,
                      near2_x_bl, near2_y_bl,
                      move_dist)
                 && drive_forward_to_side_offset(target_x_bl, target_y_bl, move_dist);

          if (!ok) {
            RCLCPP_WARN(get_logger(), "[VELOCITY_APPROACH] rotate/drive failed");
            state = ScanState::LOCAL_RECOVERY;
            break;
          }

          rclcpp::sleep_for(1s);   // let the estimator settle before re-aiming

          double xb, yb;
          if (track_target_once(ctx.target_x_odom, ctx.target_y_odom) &&
              call_detect_target(goal->min_confidence, xb, yb)) {
            ctx.x_base_last = xb; ctx.y_base_last = yb;
            result->x_base = static_cast<float>(xb);
            result->y_base = static_cast<float>(yb);
            state = ScanState::EVALUATE_DISTANCE;
          } else {
            state = ScanState::LOCAL_RECOVERY;
          }
          break;
        }

        case ScanState::LOCAL_RECOVERY: {
          double xb, yb;
          if (run_local_recovery_search(goal->min_confidence, xb, yb)) {
            double odom_x, odom_y;
            if (transform_arm_point_to_odom(xb, yb, odom_x, odom_y)) {
              ctx.target_x_odom = odom_x; ctx.target_y_odom = odom_y;
              last_known_x_odom_ = odom_x; last_known_y_odom_ = odom_y;
            }
            ctx.x_base_last = xb; ctx.y_base_last = yb;
            result->x_base = static_cast<float>(xb);
            result->y_base = static_cast<float>(yb);
            state = ScanState::EVALUATE_DISTANCE;
          } else {
            RCLCPP_WARN(get_logger(), "[LocalRecovery] exhausted, returning to FULL_SCAN");
            state = ScanState::FULL_SCAN;
          }
          break;
        }

        case ScanState::FINAL_INSPECTION: {
          track_target_once(ctx.target_x_odom, ctx.target_y_odom);

          double xb, yb;
          if (call_detect_target(goal->min_confidence, xb, yb)) {
            result->x_base = static_cast<float>(xb);
            result->y_base = static_cast<float>(yb);
          }
          result->found = true;
          state = ScanState::COMPLETE;
          break;
        }

        case ScanState::COMPLETE: {
          RCLCPP_INFO(get_logger(), "Scan complete: target at (%.3f, %.3f) odom",
                      ctx.target_x_odom, ctx.target_y_odom);
          rclcpp::sleep_for(200ms);
          move_arm_to_stow_pose();
          goal_handle->succeed(result);
          return;
        }

        case ScanState::ABORTED: {
          if (has_last_known_position_) {
            RCLCPP_WARN(get_logger(), "Scan aborted. Last known target: (%.3f, %.3f) odom",
                        last_known_x_odom_, last_known_y_odom_);
          } else {
            RCLCPP_WARN(get_logger(), "Scan aborted. No target ever detected.");
          }
          move_arm_to_stow_pose();
          goal_handle->abort(result);
          return;
        }
      }
    }
  }

  // === Members ==============================================================
  std::string planning_group_, base_joint_name_, wrist_joint_name_;
  std::string arm_base_frame_, ee_link_, cam_link_;

  std::mutex joint_state_mutex_;
  std::map<std::string, double> latest_joint_positions_;

  double camera_height_, radius_min_, radius_max_, radius_scale_factor_;
  double side_offset_;
  double yaw_tolerance_rad_{0.05};

  double joint_move_tolerance_, joint_move_timeout_, joint_poll_rate_hz_;

  // FSM tunables
  int    max_sweeps_;
  double desired_final_distance_, nav2_switch_distance_, nav2_approach_margin_;
  double local_recovery_waist_range_deg_;
  int    local_recovery_waist_steps_;
  double local_recovery_wrist_delta_deg_;

  double rotation_stop_lead_s_{0.10};

  // Last known target (odom frame), persists across sweeps/aborts
  double last_known_x_odom_ = 0.0;
  double last_known_y_odom_ = 0.0;
  bool   has_last_known_position_ = false;

  std::string gps_gate_service_;

  // /odom cache (fresh yaw for base rotation)
  std::mutex odom_mutex_;
  double odom_x_{0.0}, odom_y_{0.0}, odom_yaw_{0.0};
  bool   have_odom_{false};

  // ROS handles
  ScanServer::SharedPtr scan_server_;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
  std::shared_ptr<DetectClient> detect_client_;
  std::shared_ptr<Nav2Client>   nav2_client_;
  std::shared_ptr<TrackClient>  track_client_;
  std::shared_ptr<tf2_ros::Buffer>            tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
  rclcpp::Publisher<interbotix_xs_msgs::msg::JointSingleCommand>::SharedPtr joint_single_pub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
  rclcpp::Client<std_srvs::srv::SetBool>::SharedPtr gps_gate_client_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ArmSearchNode>();

  rclcpp::executors::MultiThreadedExecutor exec;
  exec.add_node(node);
  exec.spin();

  rclcpp::shutdown();
  return 0;
}