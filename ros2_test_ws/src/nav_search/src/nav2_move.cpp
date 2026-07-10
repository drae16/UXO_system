#include <chrono>
#include <memory>
#include <string>
#include <map>
#include <thread>
#include <future>
#include <cmath>

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
#include "nav_search/action/nav2_move.hpp"
#include "nav2_msgs/action/navigate_to_pose.hpp"

using namespace std::chrono_literals;

class Nav2Move : public rclcpp::Node
{
public:
  using NavMove      = nav_search::action::Nav2Move;
  using NavServer     = rclcpp_action::Server<NavMove>;
  using NavGoalHandle = rclcpp_action::ServerGoalHandle<NavMove>;


  using DetectTarget   = nav_search::action::DetectTarget;
  using DetectClient   = rclcpp_action::Client<DetectTarget>;

  using NavigateToPose = nav2_msgs::action::NavigateToPose;
  using Nav2Client     = rclcpp_action::Client<NavigateToPose>;



   Nav2Move()
  : Node("nav2_move")
  {
    // --- Parameters ---
    planning_group_   = this->declare_parameter<std::string>("planning_group", "interbotix_arm");
    base_joint_name_  = this->declare_parameter<std::string>("base_joint_name", "waist");
    wrist_joint_name_ = this->declare_parameter<std::string>("wrist_joint_name", "wrist_angle");  // NEW
    arm_base_frame_   = this->declare_parameter<std::string>("arm_base_frame", "vx300s/base_link");
    ee_link_          = this->declare_parameter<std::string>("ee_link", "vx300s/ee_gripper_link");
    cam_link_          = this->declare_parameter<std::string>("cam_link", "vx300s/camera_link");

    camera_height_        = this->declare_parameter<double>("camera_height",       0.40);  // z = b
    radius_min_           = this->declare_parameter<double>("camera_radius_min",   0.20);
    radius_max_           = this->declare_parameter<double>("camera_radius_max",   0.45);
    radius_scale_factor_  = this->declare_parameter<double>("camera_radius_scale", 0.2);
    side_offset_nav_ = this->declare_parameter<double>("side_offset_nav", 1.0);


    RCLCPP_INFO(get_logger(), "Planning group: %s", planning_group_.c_str());
    RCLCPP_INFO(get_logger(), "Base joint: %s",     base_joint_name_.c_str());
    RCLCPP_INFO(get_logger(), "Arm base frame: %s", arm_base_frame_.c_str());
    RCLCPP_INFO(get_logger(), "EE link: %s",        ee_link_.c_str());
    RCLCPP_INFO(get_logger(), "Camera link: %s",    cam_link_.c_str());
    RCLCPP_INFO(get_logger(), "Wrist joint: %s",    wrist_joint_name_.c_str());


    tf_buffer_   = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    nav_server_ = rclcpp_action::create_server<NavMove>(
      this,
      "nav2_move",
      std::bind(&Nav2Move::handle_goal,     this, std::placeholders::_1, std::placeholders::_2),
      std::bind(&Nav2Move::handle_cancel,   this, std::placeholders::_1),
      std::bind(&Nav2Move::handle_accepted, this, std::placeholders::_1)
    
    );
  }


  ~Nav2Move() {
    if (nav_thread_.joinable()) nav_thread_.join();
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


    void init_nav2_client_if_needed()
  {
    if (!nav2_client_) {
      RCLCPP_INFO(get_logger(), "Creating Nav2 NavigateToPose client...");
      nav2_client_ = rclcpp_action::create_client<NavigateToPose>(
        shared_from_this(), "navigate_to_pose");
    }
  }

  void init_detect_client_if_needed()
  {
    if (!detect_client_) {
      RCLCPP_INFO(get_logger(), "Creating detect_target action client...");
      detect_client_ = rclcpp_action::create_client<DetectTarget>(
        shared_from_this(), "detect_target");
    }
  }
    rclcpp_action::GoalResponse handle_goal(
    const rclcpp_action::GoalUUID &,
    std::shared_ptr<const NavMove::Goal> goal)
  {
    RCLCPP_INFO(get_logger(),
            "Received nav2 request: x=%.2f, y=%.2f, axis=%.2f",
            goal->x_map, goal->y_map, goal->axis);
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse handle_cancel(
    const std::shared_ptr<NavGoalHandle> /*goal_handle*/)
  {
    RCLCPP_INFO(get_logger(), "Nav Move goal cancel requested");
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void handle_accepted(const std::shared_ptr<NavGoalHandle> goal_handle)
  {
    if (nav_thread_.joinable()) nav_thread_.join();
    nav_thread_ = std::thread(&Nav2Move::step_nav2_towards_target, this, goal_handle);
  }


  bool call_detect_target(double min_conf, double &x, double &y, double &axis )
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
    axis = result->axis;
    RCLCPP_INFO(get_logger(), "detect_target: target at (%.2f, %.2f) in arm base frame", x, y);
    return true;
  }



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

  bool transform_arm_base_point_to_map(double x_base, double y_base,
                                     double &x_map, double &y_map)
    {
    if (!tf_buffer_) {
        RCLCPP_WARN(get_logger(), "TF buffer not initialized");
        return false;
    }

    geometry_msgs::msg::PoseStamped pt_arm;
    pt_arm.header.frame_id = arm_base_frame_;
    pt_arm.header.stamp = this->now();
    pt_arm.pose.position.x = x_base;
    pt_arm.pose.position.y = y_base;
    pt_arm.pose.position.z = 0.0;
    pt_arm.pose.orientation.w = 1.0;

    try {
        auto pt_map = tf_buffer_->transform(pt_arm, "map", tf2::durationFromSec(0.5));
        x_map = pt_map.pose.position.x;
        y_map = pt_map.pose.position.y;
        return true;
    } catch (const tf2::TransformException &ex) {
        RCLCPP_WARN(get_logger(), "Transform %s -> map failed: %s",
                    arm_base_frame_.c_str(), ex.what());
        return false;
    }
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


  void step_nav2_towards_target(const std::shared_ptr<NavGoalHandle> goal_handle)
{
  const auto goal = goal_handle->get_goal();
  auto feedback = std::make_shared<NavMove::Feedback>();
  auto result = std::make_shared<NavMove::Result>();

  const float target_x_map = goal->x_map;
  const float target_y_map = goal->y_map;
  const float axis = goal->axis;

  result->known = false;
  result->axis = axis;
  result->x_map_update = target_x_map;
  result->y_map_update = target_y_map;
  result->distance = 0.0f;

  if (goal_handle->is_canceling()) {
    goal_handle->canceled(result);
    RCLCPP_INFO(get_logger(), "Goal canceled before execution");
    return;
  }

  feedback->progress = 0.10f;
  goal_handle->publish_feedback(feedback);

  if (!nav2_client_) {
    init_nav2_client_if_needed();
  }
  if (!nav2_client_) {
    RCLCPP_ERROR(get_logger(), "Nav2 client not initialized");
    goal_handle->abort(result);
    return;
  }

  init_move_group_if_needed();
  if (!move_group_) {
    RCLCPP_ERROR(get_logger(), "Failed to initialize move group");
    goal_handle->abort(result);
    return;
  }

  double rx, ry, ryaw;
  if (!get_robot_pose_map(rx, ry, ryaw)) {
    RCLCPP_WARN(get_logger(), "Failed to get robot pose");
    goal_handle->abort(result);
    return;
  }

  // 1 meter standoff to either side of target axis
  
  double point_one_x = target_x_map + side_offset_nav_ * std::cos(axis + M_PI / 2.0);
  double point_one_y = target_y_map + side_offset_nav_ * std::sin(axis + M_PI / 2.0);
  double point_two_x = target_x_map + side_offset_nav_ * std::cos(axis - M_PI / 2.0);
  double point_two_y = target_y_map + side_offset_nav_ * std::sin(axis - M_PI / 2.0);

  double dist_one = std::hypot(point_one_x - rx, point_one_y - ry);
  double dist_two = std::hypot(point_two_x - rx, point_two_y - ry);

  double chosen_x, chosen_y;
  if (dist_one <= dist_two) {
    chosen_x = point_one_x;
    chosen_y = point_one_y;
  } else {
    chosen_x = point_two_x;
    chosen_y = point_two_y;
  }

  feedback->progress = 0.25f;
  goal_handle->publish_feedback(feedback);

  double goal_yaw = std::atan2(target_y_map - chosen_y,
                               target_x_map - chosen_x);

  geometry_msgs::msg::PoseStamped goal_pose;
  goal_pose.header.frame_id = "map";
  goal_pose.header.stamp = this->now();
  goal_pose.pose.position.x = chosen_x;
  goal_pose.pose.position.y = chosen_y;
  goal_pose.pose.position.z = 0.0;
  goal_pose.pose.orientation.w = std::cos(goal_yaw * 0.5);
  goal_pose.pose.orientation.x = 0.0;
  goal_pose.pose.orientation.y = 0.0;
  goal_pose.pose.orientation.z = std::sin(goal_yaw * 0.5);

  if (!nav2_client_->wait_for_action_server(5s)) {
    RCLCPP_WARN(get_logger(), "Nav2 NavigateToPose server not available");
    goal_handle->abort(result);
    return;
  }

  NavigateToPose::Goal goal_msg;
  goal_msg.pose = goal_pose;

  feedback->progress = 0.50f;
  goal_handle->publish_feedback(feedback);

  auto send_goal_options = Nav2Client::SendGoalOptions();
  auto future_goal = nav2_client_->async_send_goal(goal_msg, send_goal_options);

  if (future_goal.wait_for(5s) != std::future_status::ready) {
    RCLCPP_WARN(get_logger(), "Timed out waiting for Nav2 goal response");
    goal_handle->abort(result);
    return;
  }

  auto nav2_goal_handle = future_goal.get();
  if (!nav2_goal_handle) {
    RCLCPP_WARN(get_logger(), "Nav2 goal rejected");
    goal_handle->abort(result);
    return;
  }

  auto result_future = nav2_client_->async_get_result(nav2_goal_handle);
  auto wrapped_result = result_future.get();
  auto code = wrapped_result.code;

  RCLCPP_INFO(get_logger(), "Nav2 NavigateToPose result code: %d",
              static_cast<int>(code));

  if (code != rclcpp_action::ResultCode::SUCCEEDED) {
    RCLCPP_WARN(get_logger(), "Nav2 failed");
    goal_handle->abort(result);
    return;
  }

  feedback->progress = 0.75f;
  goal_handle->publish_feedback(feedback);

  // Aim arm/camera toward original map target
  if (!track_target_once(target_x_map, target_y_map)) {
    RCLCPP_WARN(get_logger(), "track_target_once failed");
  }

  double x_base_update = 0.0;
  double y_base_update = 0.0;
  double axis_update = axis;

  if (!call_detect_target(0.7, x_base_update, y_base_update, axis_update)) {
    RCLCPP_WARN(get_logger(), "Re-detect failed, returning original target");
    if (get_robot_pose_map(rx, ry, ryaw)) {
      result->distance = std::hypot(target_x_map - rx, target_y_map - ry);
    }
    move_arm_to_stow_pose();
    goal_handle->succeed(result);
    return;
  }

  double x_map_update = 0.0;
  double y_map_update = 0.0;

  if (!transform_arm_base_point_to_map(x_base_update, y_base_update,
                                       x_map_update, y_map_update)) {
    RCLCPP_WARN(get_logger(), "Detected point could not be transformed to map");
    if (get_robot_pose_map(rx, ry, ryaw)) {
      result->distance = std::hypot(target_x_map - rx, target_y_map - ry);
    }
    move_arm_to_stow_pose();
    goal_handle->succeed(result);
    return;
  }

  result->known = true;
  result->x_map_update = static_cast<float>(x_map_update);
  result->y_map_update = static_cast<float>(y_map_update);

  feedback->progress = 0.90f;
  goal_handle->publish_feedback(feedback);


  result->axis = static_cast<float>(axis_update);


  if (!get_robot_pose_map(rx, ry, ryaw)) {
    RCLCPP_WARN(get_logger(), "Failed to get final robot pose");
    goal_handle->abort(result);
    return;
  }

  result->distance = static_cast<float>(
      std::hypot(x_map_update - rx, y_map_update - ry));

  move_arm_to_stow_pose();

  feedback->progress = 1.00f;
  goal_handle->publish_feedback(feedback);

  goal_handle->succeed(result);




  
}
  std::string wrist_joint_name_;
  std::string planning_group_;
  std::string base_joint_name_;
  std::string arm_base_frame_;
  std::string ee_link_;
  std::string cam_link_;
  std::thread nav_thread_;

  double camera_height_;
  double radius_min_;
  double radius_max_;
  double radius_scale_factor_;
  double side_offset_nav_;

  NavServer::SharedPtr nav_server_;

  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
  std::shared_ptr<DetectClient> detect_client_;
  std::shared_ptr<Nav2Client> nav2_client_;


  std::shared_ptr<tf2_ros::Buffer>           tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<Nav2Move>();

  rclcpp::executors::MultiThreadedExecutor exec;
  exec.add_node(node);
  exec.spin();

  rclcpp::shutdown();
  return 0;
}
