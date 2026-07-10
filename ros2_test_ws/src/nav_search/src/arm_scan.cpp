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

#include "nav_search/action/scan_area.hpp"
#include "nav_search/action/detect_target.hpp"

using namespace std::chrono_literals;

class ArmSearchNode : public rclcpp::Node
{
public:
  using ScanArea       = nav_search::action::ScanArea;
  using ScanServer     = rclcpp_action::Server<ScanArea>;
  using ScanGoalHandle = rclcpp_action::ServerGoalHandle<ScanArea>;


  using DetectTarget   = nav_search::action::DetectTarget;
  using DetectClient   = rclcpp_action::Client<DetectTarget>;



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

    camera_height_        = this->declare_parameter<double>("camera_height",       0.40);  // z = b
    radius_min_           = this->declare_parameter<double>("camera_radius_min",   0.20);
    radius_max_           = this->declare_parameter<double>("camera_radius_max",   0.45);
    radius_scale_factor_  = this->declare_parameter<double>("camera_radius_scale", 0.2);
    
    detect_cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

    RCLCPP_INFO(get_logger(), "Planning group: %s", planning_group_.c_str());
    RCLCPP_INFO(get_logger(), "Base joint: %s",     base_joint_name_.c_str());
    RCLCPP_INFO(get_logger(), "Arm base frame: %s", arm_base_frame_.c_str());
    RCLCPP_INFO(get_logger(), "EE link: %s",        ee_link_.c_str());
    RCLCPP_INFO(get_logger(), "Camera link: %s",    cam_link_.c_str());
    RCLCPP_INFO(get_logger(), "Wrist joint: %s",    wrist_joint_name_.c_str());


    tf_buffer_   = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    scan_server_ = rclcpp_action::create_server<ScanArea>(
      this,
      "scan_area",
      std::bind(&ArmSearchNode::handle_scan_goal,     this, std::placeholders::_1, std::placeholders::_2),
      std::bind(&ArmSearchNode::handle_scan_cancel,   this, std::placeholders::_1),
      std::bind(&ArmSearchNode::handle_scan_accepted, this, std::placeholders::_1)
    
    );
  }

  ~ArmSearchNode() {
  if (scan_thread_.joinable()) scan_thread_.join();
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
      shared_from_this(),
        "detect_target",
        detect_cb_group_);  
    }
  }
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
    if (scan_thread_.joinable()) scan_thread_.join();
      scan_thread_ = std::thread(&ArmSearchNode::execute_scan, this, goal_handle);
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
    if (!move_group_) return false;

    std::map<std::string, double> target;
    target[base_joint_name_] = target_angle;

    move_group_->setStartStateToCurrentState();
    if (!move_group_->setJointValueTarget(target)) {
      RCLCPP_ERROR(get_logger(),
                   "Failed to set joint value target for joint '%s' in group '%s'",
                   base_joint_name_.c_str(), planning_group_.c_str());
      return false;
    }

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    auto code = move_group_->plan(plan);
    if (code != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_WARN(get_logger(), "No plan found for base joint move (error %d)", code.val);
      return false;
    }

    auto exec_code = move_group_->execute(plan);
    if (exec_code != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_WARN(get_logger(),
                  "Execution for base joint move failed (error %d)", exec_code.val);
      return false;
    }
    return true;
  }

  bool move_wrist_to(double target_angle)
{
  if (!move_group_) return false;

  std::map<std::string, double> target;
  target[wrist_joint_name_] = target_angle;

  move_group_->setStartStateToCurrentState();
  if (!move_group_->setJointValueTarget(target)) {
    RCLCPP_ERROR(get_logger(),
                 "Failed to set joint value target for wrist '%s' in group '%s'",
                 wrist_joint_name_.c_str(), planning_group_.c_str());
    return false;
  }

  moveit::planning_interface::MoveGroupInterface::Plan plan;
  auto code = move_group_->plan(plan);
  if (code != moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_WARN(get_logger(),
                "No plan found for wrist move (error %d)", code.val);
    return false;
  }

  auto exec_code = move_group_->execute(plan);
  if (exec_code != moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_WARN(get_logger(),
                "Execution for wrist move failed (error %d)", exec_code.val);
    return false;
  }

  return true;
}


  // === YOLO DetectTarget client =============================================

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


bool transform_to_map(geometry_msgs::msg::PoseStamped &det_map, double target_x_detect, double target_y_detect){
        // Transform initial detection to MAP to get target world pose
    if (!tf_buffer_) {
      RCLCPP_ERROR(get_logger(), "TF buffer not initialized");
      return false;
    }

    geometry_msgs::msg::PoseStamped det_arm;
    det_arm.header.frame_id = arm_base_frame_;
    det_arm.header.stamp    = this->now();
    det_arm.pose.position.x = target_x_detect;
    det_arm.pose.position.y = target_y_detect;
    det_arm.pose.position.z = 0.0;
    det_arm.pose.orientation.w = 1.0;

  
    try {
      det_map = tf_buffer_->transform(det_arm, "map", tf2::durationFromSec(0.5));
    }
    catch (const tf2::TransformException & ex) {
      RCLCPP_WARN(get_logger(), "Transform %s->map failed: %s",
                  arm_base_frame_.c_str(), ex.what());
      return false;
    }

    return true;
    
  }


  /// core scan implementation

void execute_scan(const std::shared_ptr<ScanGoalHandle> goal_handle)
{
  auto goal = goal_handle->get_goal();
  auto scan_result = std::make_shared<ScanArea::Result>();
  scan_result->found  = false;
  scan_result->known = false;
  scan_result->x_map = 0.0f;
  scan_result->y_map = 0.0f;
  scan_result->axis = 0.0f;
  scan_result->distance = 0.0f;

  init_move_group_if_needed();
  init_detect_client_if_needed();

  if (!move_group_) {
    RCLCPP_ERROR(get_logger(), "MoveGroupInterface not initialized; aborting scan.");
    goal_handle->abort(scan_result);
    return;
  }

  int steps = goal->num_steps;
  if (steps <= 1) {
    RCLCPP_WARN(get_logger(), "num_steps <= 1; adjusting to 2");
    steps = 2;
  }

  double start_angle = goal->start_angle;
  double end_angle   = goal->end_angle;
  double step        = (end_angle - start_angle) / static_cast<double>(steps - 1);
  auto  feedback     = std::make_shared<ScanArea::Feedback>();




  if (!move_arm_to_search_pose()) {
    RCLCPP_WARN(get_logger(), "Could not move into search pose");
    goal_handle->abort(scan_result);
    return;
  }
  bool  initial_found = false;
  double x_base_det  = 0.0;
  double y_base_det  = 0.0;
  double target_axis = 0.0;
  double target_x_map = 0.0;  // ← ADD
  double target_y_map = 0.0;  // ← ADD

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
        goal_handle->canceled(scan_result);
        return;
      }

      RCLCPP_INFO(get_logger(),
                  "Initial scan: level %d, moving base joint '%s' to angle %.3f rad",
                  level, base_joint_name_.c_str(), angle);

      if (!move_base_to(angle)) {
        RCLCPP_WARN(get_logger(), "Planning failed for this step, skipping.");
        continue;
      }

      rclcpp::sleep_for(100ms);

      double xb, yb, axis;
      bool found = call_detect_target(goal->min_confidence, xb, yb, axis);
      if (found) {
        RCLCPP_INFO(get_logger(),
                    "Initial detection at level %d: (%.2f, %.2f) in arm base frame",
                    level, xb, yb);

        

        double x_second, y_second, axis_second;
        geometry_msgs::msg::PoseStamped det_map;

        if(!transform_to_map(det_map, xb, yb )){
          RCLCPP_WARN(get_logger(), "failed to transform to map.");
          move_arm_to_stow_pose();
          goal_handle->abort(scan_result);
          return;
        }

        target_x_map = det_map.pose.position.x;
        target_y_map = det_map.pose.position.y;

        if(!track_target_once(target_x_map, target_y_map)){
          RCLCPP_WARN(get_logger(), "Arm failed to move to verification position. Continuing with detection.");
        }

            

        bool double_check = call_detect_target(goal->min_confidence, x_second, y_second, axis_second);



        if (double_check) {
          RCLCPP_INFO(get_logger(),
                      "Verified detection at: (%.2f, %.2f) in arm base frame",
                       x_second, y_second);
          initial_found = true;
          x_base_det  = xb;
          y_base_det  = yb;
          target_axis = axis;

          break; } // break inner loop; outer sees initial_found and stops too
      }
    }
    iteration += 1;
  }
  
  // If nothing was found at any wrist + base angle
    if (!initial_found) {
      RCLCPP_INFO(get_logger(), "No target found in initial scan");
      move_arm_to_stow_pose();
      goal_handle->succeed(scan_result);  // result->found stays false
      return;
    }

    // We have an initial detection in arm base frame
    scan_result->found  = true;
    scan_result->known = true;

    
    scan_result->axis = static_cast<float>(target_axis);
    scan_result->x_map = static_cast<float>(target_x_map);
    scan_result->y_map = static_cast<float>(target_y_map);

    double rx, ry, ryaw;
    if (!get_robot_pose_map(rx, ry, ryaw)) {
      RCLCPP_WARN(get_logger(), "Failed to get robot pose. Aborting");
      move_arm_to_stow_pose();
      goal_handle->abort(scan_result);
      return;
    }
    double dx = target_x_map - rx;
    double dy = target_y_map - ry;
    double dist = std::sqrt(dx*dx + dy*dy);

    scan_result->distance = static_cast<float>(dist);

    RCLCPP_INFO(get_logger(), "Blackboard params = axis: %.2f , x: %.2f, y: %.2f, distance %.2f, found = %d ",target_axis, target_x_map,target_y_map,dist,scan_result->found);

    move_arm_to_stow_pose();

    goal_handle->succeed(scan_result);

  }


  // === Members ===============================================================
  std::string wrist_joint_name_;
  std::string planning_group_;
  std::string base_joint_name_;
  std::string arm_base_frame_;
  std::string ee_link_;
  std::string cam_link_;
  std::thread scan_thread_;
  rclcpp::CallbackGroup::SharedPtr detect_cb_group_;

  double camera_height_;
  double radius_min_;
  double radius_max_;
  double radius_scale_factor_;

  ScanServer::SharedPtr scan_server_;

  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
  std::shared_ptr<DetectClient> detect_client_;


  std::shared_ptr<tf2_ros::Buffer>           tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

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



          
