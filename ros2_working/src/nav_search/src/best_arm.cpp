#include <chrono>
#include <memory>
#include <string>
#include <map>
#include <thread>
#include <future>
#include <cmath>
#include <mutex>
#include <sensor_msgs/msg/joint_state.hpp>
#include <interbotix_xs_msgs/msg/joint_single_command.hpp>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <rclcpp/parameter_client.hpp>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <moveit/move_group_interface/move_group_interface.h>

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
    // --- Parameters ---
    planning_group_   = this->declare_parameter<std::string>("planning_group", "interbotix_arm");
    base_joint_name_  = this->declare_parameter<std::string>("base_joint_name", "waist");
    wrist_joint_name_ = this->declare_parameter<std::string>("wrist_joint_name", "wrist_angle");  // NEW
    arm_base_frame_   = this->declare_parameter<std::string>("arm_base_frame", "vx300s/base_link");
    ee_link_          = this->declare_parameter<std::string>("ee_link", "vx300s/ee_gripper_link");
    cam_link_          = this->declare_parameter<std::string>("cam_link", "vx300s/camera_link");

    start_angle_      = this->declare_parameter<double>("start_angle", -1.5);
    end_angle_        = this->declare_parameter<double>("end_angle",   1.5);
    num_steps_        = this->declare_parameter<int>("num_steps",      10);

    camera_height_        = this->declare_parameter<double>("camera_height",       0.5);  // z = b
    radius_min_           = this->declare_parameter<double>("camera_radius_min",   0.20);
    radius_max_           = this->declare_parameter<double>("camera_radius_max",   0.45);
    radius_scale_factor_  = this->declare_parameter<double>("camera_radius_scale", 0.2);

    desired_dist_         = this->declare_parameter<double>("desired_distance", 0.35);
    desired_dist_tol_     = this->declare_parameter<double>("desired_distance_tolerance", 0.15);
    
    yaw_tolerance_rad_ = this->declare_parameter<double>("yaw_tolerance_rad", 0.05);

    side_offset_ = this->declare_parameter<double>("side_offset", 0.40);

    joint_move_tolerance_ = this->declare_parameter<double>("joint_move_tolerance", 0.02);   // rad
    joint_move_timeout_   = this->declare_parameter<double>("joint_move_timeout", 3.0);       // sec
    joint_poll_rate_hz_   = this->declare_parameter<double>("joint_poll_rate_hz", 30.0);


    cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>(
      "/cmd_vel_out", 10);

    joint_single_pub_ = this->create_publisher<interbotix_xs_msgs::msg::JointSingleCommand>(
      "/vx300s/commands/joint_single", 10);

    joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
      "/vx300s/joint_states", 10,
      std::bind(&ArmSearchNode::joint_state_callback, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(), "Planning group: %s", planning_group_.c_str());
    RCLCPP_INFO(get_logger(), "Base joint: %s",     base_joint_name_.c_str());
    RCLCPP_INFO(get_logger(), "Arm base frame: %s", arm_base_frame_.c_str());
    RCLCPP_INFO(get_logger(), "EE link: %s",        ee_link_.c_str());
    RCLCPP_INFO(get_logger(), "Camera link: %s",    cam_link_.c_str());
    RCLCPP_INFO(get_logger(), "Wrist joint: %s",    wrist_joint_name_.c_str());

    // TF
    tf_buffer_   = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    

    // ScanArea action server
    scan_server_ = rclcpp_action::create_server<ScanArea>(
      this,
      "scan_area",
      std::bind(&ArmSearchNode::handle_scan_goal,     this, std::placeholders::_1, std::placeholders::_2),
      std::bind(&ArmSearchNode::handle_scan_cancel,   this, std::placeholders::_1),
      std::bind(&ArmSearchNode::handle_scan_accepted, this, std::placeholders::_1)
    
    );
  }

private:
  // === Init helpers =========================================================

  void init_move_group_if_needed()
  {
    if (move_group_) {
      return;
    }

    RCLCPP_INFO(get_logger(), "Fetching robot_description and SRDF from /move_group...");

    auto param_node = rclcpp::Node::make_shared("arm_search_param_client");
    const std::string move_group_node_name = "/move_group";

    auto param_client = std::make_shared<rclcpp::AsyncParametersClient>(
      param_node,
      move_group_node_name);

    while (!param_client->wait_for_service(1s)) {
      if (!rclcpp::ok()) {
        RCLCPP_ERROR(get_logger(),
                     "Interrupted while waiting for %s param service.",
                     move_group_node_name.c_str());
        return;
      }
      RCLCPP_INFO(get_logger(),
                  "Waiting for %s parameter service...",
                  move_group_node_name.c_str());
    }

    auto future = param_client->get_parameters(
      {"robot_description", "robot_description_semantic"});

    rclcpp::executors::SingleThreadedExecutor exec;
    exec.add_node(param_node);
    auto ret = exec.spin_until_future_complete(future, 5s);
    exec.remove_node(param_node);

    if (ret != rclcpp::FutureReturnCode::SUCCESS) {
      RCLCPP_ERROR(get_logger(),
                   "Timed out getting parameters from %s",
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
      shared_from_this(),
      planning_group_
    );

    move_group_->setPlanningTime(3.0);

    RCLCPP_INFO(get_logger(), "MoveGroup planning frame: %s",
                move_group_->getPlanningFrame().c_str());
  }

  void init_detect_client_if_needed()
  {
    if (!detect_client_) {
      RCLCPP_INFO(get_logger(), "Creating detect_target action client...");
      detect_client_ = rclcpp_action::create_client<DetectTarget>(
        shared_from_this(), "detect_target");
    }
  }

  void init_nav2_client_if_needed()
  {
    if (!nav2_client_) {
      RCLCPP_INFO(get_logger(), "Creating Nav2 NavigateToPose client...");
      nav2_client_ = rclcpp_action::create_client<NavigateToPose>(
        shared_from_this(), "navigate_to_pose");
    }
  }


  void init_track_client_if_needed()
  {
    if (!track_client_) {
      RCLCPP_INFO(get_logger(), "Creating track_target service client...");
      track_client_ = this->create_client<TrackTarget>("track_target");
    }
  }
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

  rclcpp_action::CancelResponse handle_scan_cancel(
    const std::shared_ptr<ScanGoalHandle> /*goal_handle*/)
  {
    RCLCPP_INFO(get_logger(), "ScanArea goal cancel requested");
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void handle_scan_accepted(const std::shared_ptr<ScanGoalHandle> goal_handle)
  {
    RCLCPP_INFO(get_logger(), "ScanArea goal accepted, starting scan thread...");
    std::thread(&ArmSearchNode::execute_scan, this, goal_handle).detach();
  }

  // === TF helpers ===========================================================

  bool get_robot_pose_map(double &x, double &y, double &yaw)
  {
    if (!tf_buffer_) {
      return false;
    }

    geometry_msgs::msg::TransformStamped tf;
    try {
      // map -> base_link (quadruped base)
      tf = tf_buffer_->lookupTransform("map", "base_link", tf2::TimePointZero);
    }
    catch (const tf2::TransformException & ex) {
      RCLCPP_WARN(get_logger(), "Failed to get transform map->base_link: %s", ex.what());
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

  // === Camera circle pose computation =======================================

  geometry_msgs::msg::Pose compute_camera_pose_on_circle(
    const geometry_msgs::msg::Point &target_arm)
  {
    // target in arm base frame
    double X = target_arm.x;
    double Y = target_arm.y;
    double Z = target_arm.z;

    // Horizontal distance
    double r_xy = std::sqrt(X*X + Y*Y);
    if (r_xy < 1e-3) {
      r_xy = 1e-3;
    }

    // Unit direction in XY toward target
    double ux = X / r_xy;
    double uy = Y / r_xy;

    // Desired radius band [radius_min_, radius_max_]
    double R_des = radius_scale_factor_ * r_xy;
    if (R_des < radius_min_) R_des = radius_min_;
    if (R_des > radius_max_) R_des = radius_max_;

    geometry_msgs::msg::Pose cam_pose;

    // Position on the circle at height camera_height_
    cam_pose.position.x = ux * R_des;
    cam_pose.position.y = uy * R_des;
    cam_pose.position.z = camera_height_;

    RCLCPP_INFO(get_logger(), "Z position should be %.2f", cam_pose.position.z);
    tf2::Vector3 cam_pos(
      cam_pose.position.x,
      cam_pose.position.y,
      cam_pose.position.z);

    tf2::Vector3 tgt_pos(X, Y, Z);

    // X axis: from camera to target
    tf2::Vector3 x_axis = tgt_pos - cam_pos;
    
    if (x_axis.length2() < 1e-6) {
      // Degenerate: default forward
      x_axis = tf2::Vector3(1.0, 0.0, 0.0);
    }
    RCLCPP_INFO(get_logger(), "target  location vector  %.2f , %.2f, %.2f wrt to arm frame", x_axis.x() , x_axis.y(), x_axis.z());

    x_axis.normalize();

    // Use "world up" to build a right-handed frame
    tf2::Vector3 world_up(0.0, 0.0, 1.0);

   //y axis is always parallel to the ground
    tf2::Vector3 y_axis = x_axis.cross(-world_up);

    // Y axis completes the right-handed frame
    tf2::Vector3 z_axis = x_axis.cross(y_axis);

    if (y_axis.length2() < 1e-6) {
      // x is almost parallel to world_up; pick arbitrary z
      z_axis = tf2::Vector3(0.0, 0.0, 1.0);
      y_axis = tf2::Vector3(0.0, 1.0, 0.0);
      x_axis = tf2::Vector3(1.0, 0.0, 0.0);
    }
    y_axis.normalize();
    z_axis.normalize();

    RCLCPP_INFO(get_logger(), "x_axis should be pointing %.2f,%.2f, %.2f", x_axis.x() , x_axis.y(), x_axis.z());

    // Rotation matrix with columns = (X, Y, Z) basis of the camera
    tf2::Matrix3x3 R(
      x_axis.x(), y_axis.x(), z_axis.x(),
      x_axis.y(), y_axis.y(), z_axis.y(),
      x_axis.z(), y_axis.z(), z_axis.z()
    );

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
    ee_to_cam = tf_buffer_->lookupTransform(
      ee_link_,   
      cam_link_, 
      tf2::TimePointZero);
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
  return true;
}

  bool move_arm_to_pose(const geometry_msgs::msg::Pose &pose)
  {
    if (!move_group_) {
      return false;
    }

    move_group_->setStartStateToCurrentState();
    move_group_->setPoseTarget(pose, ee_link_);

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    auto code = move_group_->plan(plan);
    if (code != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_WARN(get_logger(),
                  "No plan found for camera tracking pose (error %d)", code.val);
      return false;
    }

    auto exec_code = move_group_->execute(plan);
    if (exec_code != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_WARN(get_logger(),
                  "Execution for camera tracking pose failed (error %d)", exec_code.val);
      return false;
    }
    return true;
  }

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
    if (it == latest_joint_positions_.end()) {
      return false;
    }
    position = it->second;
    return true;
  }

  // Publishes a single-joint command and polls /joint_states until the joint
  // is within tolerance of the target (checked on 2 consecutive polls), or
  // until timeout elapses.
  bool move_single_joint_fast(const std::string &joint_name, double target_angle)
  {
    if (!joint_single_pub_) {
      RCLCPP_ERROR(get_logger(), "joint_single_pub_ not initialized");
      return false;
    }

    double start_pos = 0.0;
    if (!get_joint_position(joint_name, start_pos)) {
      RCLCPP_WARN(get_logger(),
                  "No joint_states received yet for '%s'; publishing command anyway",
                  joint_name.c_str());
    }

    interbotix_xs_msgs::msg::JointSingleCommand cmd;
    cmd.name = joint_name;
    cmd.cmd  = target_angle;
    joint_single_pub_->publish(cmd);

    RCLCPP_INFO(get_logger(), "Publishing joint_single: '%s' -> %.3f rad",
                joint_name.c_str(), target_angle);

    rclcpp::Rate rate(joint_poll_rate_hz_);
    auto start_time = this->now();
    int consecutive_ok = 0;

    while (rclcpp::ok()) {
      double elapsed = (this->now() - start_time).seconds();
      if (elapsed > joint_move_timeout_) {
        RCLCPP_WARN(get_logger(),
                    "Timeout waiting for joint '%s' to reach %.3f rad (elapsed %.2f s)",
                    joint_name.c_str(), target_angle, elapsed);
        return false;
      }

      double current_pos;
      if (get_joint_position(joint_name, current_pos)) {
        double err = std::fabs(current_pos - target_angle);
        if (err < joint_move_tolerance_) {
          consecutive_ok++;
          if (consecutive_ok >= 2) {
            RCLCPP_INFO(get_logger(),
                        "Joint '%s' reached target %.3f rad (current %.3f, err %.4f, %.2f s)",
                        joint_name.c_str(), target_angle, current_pos, err, elapsed);
            return true;
          }
        } else {
          consecutive_ok = 0;
        }
      }

      rate.sleep();
    }

    return false;  // rclcpp::ok() went false (shutdown)
  }


  // Track once given target in MAP frame: map -> arm_base -> circle pose -> MoveIt
  bool track_target_once(double target_x_map, double target_y_map)
  {
    if (!tf_buffer_) return false;

    geometry_msgs::msg::PoseStamped tgt_map;
    tgt_map.header.frame_id = "map";
    tgt_map.header.stamp    = this->now();
    tgt_map.pose.position.x = target_x_map;
    tgt_map.pose.position.y = target_y_map;
    tgt_map.pose.position.z = 0.0;
    tgt_map.pose.orientation.w = 1.0;

    geometry_msgs::msg::PoseStamped tgt_arm;
    try {
      tgt_arm = tf_buffer_->transform(tgt_map, arm_base_frame_, tf2::durationFromSec(0.5));
    }
    catch (const tf2::TransformException & ex) {
      RCLCPP_WARN(get_logger(), "Transform map->%s failed: %s",
                  arm_base_frame_.c_str(), ex.what());
      return false;
    }

    geometry_msgs::msg::Pose cam_pose = compute_camera_pose_on_circle(tgt_arm.pose.position);
    return move_arm_to_pose(cam_pose);
  }

  // === MoveIt named poses ===================================================

  bool move_arm_to_search_pose()
  {
    if (!move_group_) return false;

    RCLCPP_INFO(get_logger(), "Moving arm to named state 'Search'");
    move_group_->setStartStateToCurrentState();
    bool has_target = move_group_->setNamedTarget("Search");
    if (!has_target) {
      RCLCPP_WARN(get_logger(),
                  "Named target 'Search' not found for group '%s'",
                  planning_group_.c_str());
      return false;
    }

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    auto code = move_group_->plan(plan);
    if (code != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_WARN(get_logger(), "No plan found to 'Search' (error %d)", code.val);
      return false;
    }

    auto exec_code = move_group_->execute(plan);
    if (exec_code != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_WARN(get_logger(), "Execution to 'Search' failed (error %d)", exec_code.val);
      return false;
    }
    return true;
  }

  bool move_arm_to_stow_pose()
  {
    if (!move_group_) return false;

    RCLCPP_INFO(get_logger(), "Moving arm to named state 'Sleep'");
    move_group_->setStartStateToCurrentState();
    bool has_target = move_group_->setNamedTarget("Sleep");
    if (!has_target) {
      RCLCPP_WARN(get_logger(),
                  "Named target 'Sleep' not found for group '%s'",
                  planning_group_.c_str());
      return false;
    }

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    auto code = move_group_->plan(plan);
    if (code != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_WARN(get_logger(), "No plan found to 'Sleep' (error %d)", code.val);
      return false;
    }

    auto exec_code = move_group_->execute(plan);
    if (exec_code != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_WARN(get_logger(), "Execution to 'Sleep' failed (error %d)", exec_code.val);
      return false;
    }
    return true;
  }



  bool move_base_to(double target_angle)
  {
    return move_single_joint_fast(base_joint_name_, target_angle);
  }


    bool move_wrist_to(double target_angle)
  {
    return move_single_joint_fast(wrist_joint_name_, target_angle);
  }



  // === YOLO DetectTarget client =============================================

  bool call_detect_target(double min_conf, double &x, double &y)
  {
    if (!detect_client_) {
      init_detect_client_if_needed();
    }
    if (!detect_client_) return false;

    if (!detect_client_->wait_for_action_server(2s)) {
      RCLCPP_WARN(get_logger(), "detect_target action server not available");
      return false;
    }

    auto goal_msg = DetectTarget::Goal();
    goal_msg.min_confidence = static_cast<float>(min_conf);

    auto send_goal_options = DetectClient::SendGoalOptions();
    auto future_goal = detect_client_->async_send_goal(goal_msg, send_goal_options);

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

    auto wrapped_result = result_future.get();
    auto result = wrapped_result.result;

    if (!result || !result->found) {
      RCLCPP_INFO(get_logger(), "detect_target: no target found");
      return false;
    }

    x = result->x_base;
    y = result->y_base;
    RCLCPP_INFO(get_logger(), "detect_target: target at (%.2f, %.2f) in arm base frame", x, y);
    return true;
  }

// NEW!! Helper to begin tracking
  bool call_track_servo_start(double map_x, double map_y)
  {
    init_track_client_if_needed();
    if (!track_client_) return false;

    if (!track_client_->wait_for_service(2s)) {
      RCLCPP_WARN(get_logger(), "track_target service not available");
      return false;
    }

    auto request = std::make_shared<TrackTarget::Request>();
    request->enable = true;
    request->target.header.frame_id = "map";
    request->target.header.stamp = this->now();
    request->target.point.x = map_x;
    request->target.point.y = map_y;
    request->target.point.z = 0.0;

    auto fut = track_client_->async_send_request(request);
    if (fut.wait_for(1s) != std::future_status::ready) return false;
    auto res = fut.get();
    return res && res->accepted;
  }


  bool call_track_servo_stop()
  {
    init_track_client_if_needed();
    if (!track_client_) return false;

    if (!track_client_->wait_for_service(2s)) {
      RCLCPP_WARN(get_logger(), "track_target service not available");
      return false;
    }

    auto request = std::make_shared<TrackTarget::Request>();
    request->enable = false;
    // target ignored when enable=false, but fill frame anyway
    request->target.header.frame_id = "map";
    request->target.header.stamp = this->now();
    request->target.point.x = 0.0;
    request->target.point.y = 0.0;
    request->target.point.z = 0.0;

    auto fut = track_client_->async_send_request(request);
    if (fut.wait_for(1s) != std::future_status::ready) return false;
    auto res = fut.get();
    return res && res->accepted;
  }


  // === Nav2 "step" toward target with tracking during motion ================

  bool step_nav2_towards_target( double target_x_map, double target_y_map, double desired_dist, double dist_tolerance)
  {
    using namespace std::chrono_literals;

    if (!nav2_client_) {
      init_nav2_client_if_needed();
    }
    if (!nav2_client_) {
      RCLCPP_ERROR(get_logger(), "Nav2 client not initialized");
      return false;
    }

    // Current robot pose in map
    double rx, ry, ryaw;
    if (!get_robot_pose_map(rx, ry, ryaw)) {
      return false;
    }

    double dx = target_x_map - rx;
    double dy = target_y_map - ry;
    double dist = std::sqrt(dx*dx + dy*dy);

    RCLCPP_INFO(get_logger(),
                "Nav step: current dist to target = %.3f m", dist);

    if (dist <= desired_dist + dist_tolerance) {
      RCLCPP_INFO(get_logger(),
                  "Already within %.2f +/- %.2f m, skipping Nav2 step",
                  desired_dist, dist_tolerance);
      return true;
    }

    double vx = dx / dist;
    double vy = dy / dist;
    double goal_dist_from_robot = dist - desired_dist;

    double gx = rx + vx * goal_dist_from_robot;
    double gy = ry + vy * goal_dist_from_robot;

    double goal_yaw = std::atan2(dy, dx);

    geometry_msgs::msg::PoseStamped goal_pose;
    goal_pose.header.frame_id = "map";
    goal_pose.header.stamp    = this->now();
    goal_pose.pose.position.x = gx;
    goal_pose.pose.position.y = gy;
    goal_pose.pose.position.z = 0.0;
    goal_pose.pose.orientation.w = std::cos(goal_yaw * 0.5);
    goal_pose.pose.orientation.x = 0.0;
    goal_pose.pose.orientation.y = 0.0;
    goal_pose.pose.orientation.z = std::sin(goal_yaw * 0.5);

    if (!nav2_client_->wait_for_action_server(5s)) {
      RCLCPP_WARN(get_logger(), "Nav2 NavigateToPose server not available");
      return false;
    }

    NavigateToPose::Goal goal_msg;
    goal_msg.pose = goal_pose;

    auto send_goal_options = Nav2Client::SendGoalOptions();
    auto future_goal = nav2_client_->async_send_goal(goal_msg, send_goal_options);


    auto goal_handle = future_goal.get();
    if (!goal_handle) {
      RCLCPP_WARN(get_logger(), "Nav2 goal rejected");
      return false;
    }

    auto result_future = nav2_client_->async_get_result(goal_handle);

    auto wrapped_result = result_future.get();
    auto code = wrapped_result.code;
    RCLCPP_INFO(get_logger(), "Nav2 NavigateToPose result code: %d", static_cast<int>(code));
    return code == rclcpp_action::ResultCode::SUCCEEDED;
  }

  static double normalize_angle(double a)
  {
    while (a >  M_PI) a -= 2.0 * M_PI;
    while (a < -M_PI) a += 2.0 * M_PI;
    return a;
  }

  bool determine_angle_offset(double target_x_map, double target_y_map, double &calc_move, double ang_speed = 0.50)
  {

    double rx, ry, ryaw;
    if (!get_robot_pose_map(rx, ry, ryaw)) {
      return false;
    }

    // Direction from robot to target
    
    double dx = target_x_map - rx;
    double dy = target_y_map - ry;
    double remain_dist = std::sqrt(dx*dx + dy*dy);
    if (remain_dist < 1e-6) {
      calc_move = 0.0;
      return true;
    }

    double theta = 1.5*std::atan2(dy, dx);  // angle to target
    const double s = side_offset_;   
    double angle_to_goal = std::atan2(s, remain_dist);
    calc_move = std::sqrt(remain_dist*remain_dist - s*s);
    RCLCPP_INFO(get_logger(),
            "[Phase B] Distance to move = %.3f",
            calc_move);
    // Put target on the LEFT side of the dog (90° ccw from line-of-sight)
    double yaw_left = normalize_angle(theta - angle_to_goal);
    double yaw_right = normalize_angle(theta + angle_to_goal);

    double err_left  = normalize_angle(yaw_left  - ryaw);
    double err_right = normalize_angle(yaw_right - ryaw);

    double yaw_goal  = (std::fabs(err_left) < std::fabs(err_right)) ? yaw_left : yaw_right;

    RCLCPP_INFO(get_logger(),
                "[Phase B] rotate: theta_to_target=%.3f, yaw_goal=%.3f",
                theta, yaw_goal);

    rclcpp::Rate rate(20.0);  // 20 Hz

    while (rclcpp::ok()) {
      if (!get_robot_pose_map(rx, ry, ryaw)) {
        return false;
      }

      double err = normalize_angle(yaw_goal - ryaw);
      if (std::fabs(err) < yaw_tolerance_rad_) {
        RCLCPP_INFO(get_logger(), "[Phase B] rotation complete, yaw=%.3f", ryaw);
        break;
      }

      geometry_msgs::msg::Twist cmd;
      cmd.angular.z = (err > 0.0) ? ang_speed : -ang_speed;
      cmd_vel_pub_->publish(cmd);

      rate.sleep();
    }

    // Stop rotation
    geometry_msgs::msg::Twist stop;
    cmd_vel_pub_->publish(stop);
    return true;
  }











  bool drive_forward_to_side_offset(double target_x_map,double target_y_map, double move_dist, double forward_speed = 0.3)
  {
    if (!cmd_vel_pub_) {
      RCLCPP_WARN(get_logger(), "cmd_vel publisher not initialized for Phase B drive");
      return false;
    }

    double rx0, ry0, yaw0;
    if (!get_robot_pose_map(rx0, ry0, yaw0)) {
      return false;
    }

    // Current distance to target
    double dx = target_x_map - rx0;
    double dy = target_y_map - ry0;
    double dist = std::sqrt(dx*dx + dy*dy);
    if (dist < side_offset_ + 0.05) {
      RCLCPP_INFO(get_logger(),
                  "[Phase B] already closer than side_offset (dist=%.3f, side=%.3f)",
                  dist, side_offset_);
      return true;
    }

    auto start_time = this->now();
    const double duration = move_dist / forward_speed;
    const double rate_hz  = 20.0;


    RCLCPP_INFO(get_logger(),
                "Driving relative: dist=%.2f m, vx=%.2f, dur=%.2f s",
                move_dist, forward_speed, duration);

    rclcpp::Rate rate(rate_hz);
    geometry_msgs::msg::Twist cmd;
    cmd.linear.x = forward_speed;
    cmd.linear.y = 0.0;
    cmd.linear.z = 0.0;
    cmd.angular.z = 0.0;

    auto start = this->now();
    while (rclcpp::ok() &&
           (this->now() - start).seconds() < duration) {
      cmd_vel_pub_->publish(cmd);
      rate.sleep();
      RCLCPP_INFO(get_logger(),
            "Driving forward %.2f at vx=%.2f",
            move_dist, forward_speed);
    }

    // Stop at the end
    geometry_msgs::msg::Twist stop;
    cmd_vel_pub_->publish(stop);
  


    double rxf, ryf, yawf;
    if (get_robot_pose_map(rxf, ryf, yawf)) {
      double dxf = target_x_map - rxf;
      double dyf = target_y_map - ryf;
      double final_dist = std::sqrt(dxf*dxf + dyf*dyf);
      RCLCPP_INFO(get_logger(),
                  "[Phase B] final distance to target ~ %.3f m", final_dist);
    }

    return true;
  }

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

    RCLCPP_INFO(get_logger(), "Restoring arm to pre-track_target_once joint state");

    move_group_->setStartStateToCurrentState();
    if (!move_group_->setJointValueTarget(saved_state)) {
      RCLCPP_ERROR(get_logger(), "Failed to set joint value target for state restore");
      return false;
    }

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    auto code = move_group_->plan(plan);
    if (code != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_WARN(get_logger(), "No plan found to restore joint state (error %d)", code.val);
      return false;
    }

    auto exec_code = move_group_->execute(plan);
    if (exec_code != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_WARN(get_logger(), "Execution to restore joint state failed (error %d)", exec_code.val);
      return false;
    }
    return true;
  }





















  bool reposition_second_check(double &targetx, double &targety)
{
  if (!tf_buffer_ || !move_group_) {
    return false;
  }

  // How much to pitch down per step
  const double delta_rad = 10.0 * M_PI / 180.0;   // 5 degrees
  const int max_steps = 3;                       // 0,5,10,...,35 deg down (tune)

  // Get current camera pose in arm base frame
  geometry_msgs::msg::TransformStamped base_T_cam;
  try {
    base_T_cam = tf_buffer_->lookupTransform(
      arm_base_frame_,   // target frame
      cam_link_,         // source frame
      tf2::TimePointZero);
  } catch (const tf2::TransformException &ex) {
    RCLCPP_WARN(get_logger(), "TF lookup failed (%s -> %s): %s",
                arm_base_frame_.c_str(), cam_link_.c_str(), ex.what());
    return false;
  }

  // Current camera position
  geometry_msgs::msg::Pose cam_pose;
  cam_pose.position.x = base_T_cam.transform.translation.x;
  cam_pose.position.y = base_T_cam.transform.translation.y;
  cam_pose.position.z = base_T_cam.transform.translation.z;

  // Current camera orientation
  tf2::Quaternion q_cur;
  tf2::fromMsg(base_T_cam.transform.rotation, q_cur);
  q_cur.normalize();

  tf2::Matrix3x3 R_cur(q_cur);

  // In YOUR code you define camera X axis as "from camera to target", so treat +X as forward
  tf2::Vector3 fwd = R_cur * tf2::Vector3(1.0, 0.0, 0.0);

  // Compute yaw/pitch from forward vector in base frame (base: x forward, y left, z up)
  const double yaw   = std::atan2(fwd.y(), fwd.x());
  const double pitch = std::atan2(-fwd.z(), std::hypot(fwd.x(), fwd.y()));

  // Preserve roll from current orientation (optional but helps avoid “camera twist”)
  double roll_cur, pitch_cur, yaw_cur;
  R_cur.getRPY(roll_cur, pitch_cur, yaw_cur);

  // Try increasingly downward pitches
  for (int k = 0; k < max_steps; ++k) {
    const double pitch_new = pitch + k * delta_rad;  // more downward each step
    RCLCPP_WARN(get_logger(), "Entering Second check loop");
    tf2::Quaternion q_new;
    q_new.setRPY(roll_cur, pitch_new, yaw);
    q_new.normalize();
    cam_pose.orientation = tf2::toMsg(q_new);

    geometry_msgs::msg::Pose ee_pose;
    if (!cameraPoseToEePoseInBase(cam_pose, ee_pose)) {
      RCLCPP_WARN(get_logger(), "cameraPoseToEePoseInBase failed; cannot reposition");
      return false;
    }

    RCLCPP_INFO(get_logger(),
                "Reposition pitch-down step %d: pitch=%.3f rad (was %.3f), yaw=%.3f",
                k, pitch_new, pitch, yaw);

    if (!move_arm_to_pose(ee_pose)) {
      RCLCPP_WARN(get_logger(), "move_arm_to_pose failed at step %d", k);
      continue;
    }

    rclcpp::sleep_for(50ms);

    if (call_detect_target(0.6, targetx, targety)) {
      return true;
    }
  }

  return false;
}



















 // === Core Scan implementation =============================================
void execute_scan(const std::shared_ptr<ScanGoalHandle> goal_handle)
{
  auto goal = goal_handle->get_goal();
  auto result = std::make_shared<ScanArea::Result>();
  result->found  = false;
  result->x_base = 0.0f;
  result->y_base = 0.0f;

  init_move_group_if_needed();
  init_detect_client_if_needed();
  init_nav2_client_if_needed();
  init_track_client_if_needed();

  if (!move_group_) {
    RCLCPP_ERROR(get_logger(), "MoveGroupInterface not initialized; aborting scan.");
    goal_handle->abort(result);
    return;
  }

  int steps = goal->num_steps;
  if (steps <= 1) {
    RCLCPP_WARN(get_logger(), "num_steps <= 1; adjusting to 2");
    steps = 2;
  }
  bool complete_success = false;
  double start_angle = goal->start_angle;
  double end_angle   = goal->end_angle;
  double step        = (end_angle - start_angle) / static_cast<double>(steps - 1);

  auto  feedback     = std::make_shared<ScanArea::Feedback>();

  const int max_loops = 2;

  for (int loop = 0; loop < max_loops; ++loop){
    if (!move_arm_to_search_pose()) {
      RCLCPP_WARN(get_logger(), "Could not move into search pose");
      goal_handle->abort(result);
      return;
  }
    bool  initial_found = false;
    double x_base_det  = 0.0;
    double y_base_det  = 0.0;

    

    // === Initial scan with vertical wrist levels + horizontal sweep =========
    //
    // Approximate "Search" wrist angle – tune this as needed.
    double search_wrist_angle = 1.75;  

    const int    max_levels  = 3;
    const double delta_rad   = 20.0 * M_PI / 180.0;   // 10 degrees
    const double offsets[max_levels] = { +delta_rad, 0.0, -delta_rad };
    int iteration = 0;
    for (int level = 0; level < max_levels && !initial_found; ++level) {
      double wrist_angle = search_wrist_angle + offsets[level];

      RCLCPP_INFO(get_logger(),
                  "Initial scan: level %d, moving wrist '%s' to %.3f rad "
                  "(Search %.3f + offset %.3f)",
                  level, wrist_joint_name_.c_str(),
                  wrist_angle, search_wrist_angle, offsets[level]);

      if (!move_wrist_to(wrist_angle)) {
        RCLCPP_WARN(get_logger(),
                    "Could not move wrist to level %d; skipping this level",
                    level);
        continue;
      }

      // Horizontal sweep at this wrist angle
      

      RCLCPP_WARN(get_logger(),
            "iteration: %d",
             iteration);

      for (int i = 0; i < steps; ++i) {
        double angle = 0.0;
        if(iteration%2==0){
          angle = start_angle + step * static_cast<double>(i);
        }
        else{
          angle = end_angle - step * static_cast<double>(i);
        }
        feedback->current_angle = static_cast<float>(angle);


        // Progress across all levels
        feedback->progress = static_cast<float>(
          100.0 * (static_cast<double>(i + level * steps) /
                  static_cast<double>(max_levels * steps - 1)));
        goal_handle->publish_feedback(feedback);

        if (goal_handle->is_canceling()) {
          RCLCPP_INFO(get_logger(), "ScanArea goal canceled");
          move_arm_to_stow_pose();
          goal_handle->canceled(result);
          return;
        }

        RCLCPP_INFO(get_logger(),
                    "Initial scan: level %d, moving base joint '%s' to angle %.3f rad",
                    level, base_joint_name_.c_str(), angle);

        if (!move_base_to(angle)) {
          RCLCPP_WARN(get_logger(), "Planning failed for this step, skipping.");
          continue;
        }

        rclcpp::sleep_for(40ms);

        double xb, yb;
        bool found = call_detect_target(goal->min_confidence, xb, yb);
        if (found) {
          RCLCPP_INFO(get_logger(),
                      "Initial detection at level %d: (%.2f, %.2f) in arm base frame",
                      level, xb, yb);

              // Transform initial detection to MAP to get target world pose
          if (!tf_buffer_) {
            RCLCPP_ERROR(get_logger(), "TF buffer not initialized");
            move_arm_to_stow_pose();
            goal_handle->abort(result);
            return;
          }

          geometry_msgs::msg::PoseStamped det_arm;
          det_arm.header.frame_id = arm_base_frame_;
          det_arm.header.stamp    = this->now();
          det_arm.pose.position.x = xb;
          det_arm.pose.position.y = yb;
          det_arm.pose.position.z = 0.0;
          det_arm.pose.orientation.w = 1.0;

          geometry_msgs::msg::PoseStamped det_map;
          try {
            det_map = tf_buffer_->transform(det_arm, "map", tf2::durationFromSec(0.5));
          }
          catch (const tf2::TransformException & ex) {
            RCLCPP_WARN(get_logger(), "Transform %s->map failed: %s",
                        arm_base_frame_.c_str(), ex.what());
            move_arm_to_stow_pose();
            goal_handle->abort(result);
            return;
          }

          double target_x_map = det_map.pose.position.x;
          double target_y_map = det_map.pose.position.y;

          std::map<std::string, double> pre_track_joint_state = capture_arm_joint_state();

                      
          track_target_once(target_x_map, target_y_map);

          double x_base,y_base;
          if(!call_detect_target(goal->min_confidence, x_base, y_base)){
            restore_arm_joint_state(pre_track_joint_state);
            continue;
          }

          initial_found = true;
          x_base_det    = (xb);
          y_base_det    = (yb);
          break;  // break inner loop; outer sees initial_found and stops too
        };
      }
      iteration += 1;
    }

    // If nothing was found at any wrist + base angle
    if (!initial_found) {
      RCLCPP_INFO(get_logger(), "No target found in initial scan");
      move_arm_to_stow_pose();
      goal_handle->succeed(result);  // result->found stays false
      return;
    }

    // We have an initial detection in arm base frame
    result->found  = true;
    result->x_base = static_cast<float>(x_base_det);
    result->y_base = static_cast<float>(y_base_det);

    // Transform initial detection to MAP to get target world pose
    if (!tf_buffer_) {
      RCLCPP_ERROR(get_logger(), "TF buffer not initialized");
      move_arm_to_stow_pose();
      goal_handle->abort(result);
      return;
    }

    geometry_msgs::msg::PoseStamped det_arm;
    det_arm.header.frame_id = arm_base_frame_;
    det_arm.header.stamp    = this->now();
    det_arm.pose.position.x = x_base_det;
    det_arm.pose.position.y = y_base_det;
    det_arm.pose.position.z = 0.0;
    det_arm.pose.orientation.w = 1.0;

    geometry_msgs::msg::PoseStamped det_map;
    try {
      det_map = tf_buffer_->transform(det_arm, "map", tf2::durationFromSec(0.5));
    }
    catch (const tf2::TransformException & ex) {
      RCLCPP_WARN(get_logger(), "Transform %s->map failed: %s",
                  arm_base_frame_.c_str(), ex.what());
      move_arm_to_stow_pose();
      goal_handle->abort(result);
      return;
    }

    double target_x_map = det_map.pose.position.x;
    double target_y_map = det_map.pose.position.y;

    // === Phase A: Nav2 + arm tracking to ~desired_dist_ ======================
    // Track once before stepping

    while (initial_found == true) {
      double rx, ry, ryaw;
      if (!get_robot_pose_map(rx, ry, ryaw)) {
        break;
      }
      double dx = target_x_map - rx;
      double dy = target_y_map - ry;
      double dist = std::sqrt(dx*dx + dy*dy);
      

      RCLCPP_INFO(get_logger(),
                  "[Phase A] loop %d: dist to target = %.3f m", loop, dist);

      if (dist <= desired_dist_ + desired_dist_tol_) {
        RCLCPP_INFO(get_logger(),
                    "Reached ~%.2f m from target; Phase A complete", desired_dist_);
        break;
      }


      move_arm_to_stow_pose();
        
      // Step Nav2 toward the target; arm tracking runs inside step_nav2_towards_target
      if (!step_nav2_towards_target(target_x_map, target_y_map,
                                    desired_dist_, desired_dist_tol_)) {
        RCLCPP_WARN(get_logger(), "Nav2 step failed; stopping Phase A");
        break;
      }
      
      track_target_once(target_x_map,target_y_map);

      // Re-acquire target with YOLO; if lost, I should reattempt after some slight wrist and waist adjustments
      rclcpp::sleep_for(100ms);

      double xb_new, yb_new;
      bool found_again = call_detect_target(goal->min_confidence, xb_new, yb_new);
      if (!found_again) {

          if(!reposition_second_check(xb_new,yb_new)){
            RCLCPP_WARN(get_logger(),
              "Target lost after Nav2 move; ending scan action (found initially = true)");
            initial_found = false;
            continue;
        }
      }

      // Update result base-frame coordinates
      result->x_base = static_cast<float>(xb_new);
      result->y_base = static_cast<float>(yb_new);

      // Update target map pose using new detection (better accuracy)
      det_arm.header.stamp    = this->now();
      det_arm.pose.position.x = xb_new;
      det_arm.pose.position.y = yb_new;

      try {
        det_map = tf_buffer_->transform(det_arm, "map", tf2::durationFromSec(0.5));
        target_x_map = det_map.pose.position.x;
        target_y_map = det_map.pose.position.y;
      }
      catch (const tf2::TransformException & ex) {
        RCLCPP_WARN(get_logger(), "Transform %s->map (refine) failed: %s",
                    arm_base_frame_.c_str(), ex.what());
        // keep old target_x_map / target_y_map
      }
    }

    // === Phase B: side-on final approach with cmd_vel + tracking =============
    while (initial_found == true){
      double rx, ry, ryaw;

      if (!get_robot_pose_map(rx, ry, ryaw)) {
        RCLCPP_WARN(get_logger(), "[Phase B] failed to get robot pose; retrying");
        rclcpp::sleep_for(100ms);
        continue;  // or break; depending on what you want
      }


      double dx = target_x_map - rx;
      double dy = target_y_map - ry;
      double dist = std::sqrt(dx*dx + dy*dy);
    
      if(dist <= side_offset_){
        RCLCPP_INFO(get_logger(), " [Phase B] target is within desired distance. Arm search complete");
        complete_success = true;
        break;
      }
      double movement_distance = 0.0;
      if(!determine_angle_offset(target_x_map,target_y_map,movement_distance)){
        RCLCPP_INFO(get_logger(), " [Phase B] rotation failed restarting");
        break;
      }
      if(!drive_forward_to_side_offset(target_x_map,target_y_map,movement_distance)){
        RCLCPP_INFO(get_logger(), " [Phase B] forawrd failed restarting");
        break;
      }
      track_target_once(target_x_map,target_y_map);

      // 3) Optionally re-check with YOLO (if lost, caller can trigger a new scan)
      double xb_final, yb_final;
      if (!call_detect_target(goal->min_confidence, xb_final, yb_final)) {
        RCLCPP_WARN(get_logger(),
                    "[Phase B] target lost in final pose; restarting search");
        initial_found = false;
        continue;
      } else {
        result->x_base = static_cast<float>(xb_final);
        result->y_base = static_cast<float>(yb_final);
        RCLCPP_INFO(get_logger(),
                    "[Phase B] final YOLO detection at (%.2f, %.2f) in arm base frame",
                    xb_final, yb_final);
        complete_success = true;
      }
    }
    if(complete_success){
      break;
    }
  }
  // Stow the arm at the end of the whole procedure
  // move_arm_to_stow_pose();
  if(!complete_success){
    move_arm_to_stow_pose();
  }
  rclcpp::sleep_for(200ms);

  move_arm_to_stow_pose(); //remove this once picture taking implementation is in place!
  
  goal_handle->succeed(result);
}




  

  // === Members ===============================================================
  std::string wrist_joint_name_;
  std::string planning_group_;
  std::string base_joint_name_;
  std::string arm_base_frame_;
  std::string ee_link_;
  std::string cam_link_;
  std::mutex joint_state_mutex_;
  std::map<std::string, double> latest_joint_positions_;

  ScanServer::SharedPtr scan_server_;

  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
  std::shared_ptr<DetectClient> detect_client_;
  std::shared_ptr<Nav2Client>   nav2_client_;
  std::shared_ptr<TrackClient> track_client_;

  std::shared_ptr<tf2_ros::Buffer>           tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
  rclcpp::Publisher<interbotix_xs_msgs::msg::JointSingleCommand>::SharedPtr joint_single_pub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;

  double joint_move_tolerance_;   // rad
  double joint_move_timeout_;     // sec, max wait
  double joint_poll_rate_hz_;

  double start_angle_;
  double end_angle_;
  int    num_steps_;

  double camera_height_;
  double radius_min_;
  double radius_max_;
  double radius_scale_factor_;

  double desired_dist_;
  double desired_dist_tol_;
  double side_offset_;
  double yaw_tolerance_rad_{0.05};

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



