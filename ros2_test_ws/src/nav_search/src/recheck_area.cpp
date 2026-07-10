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
#include "nav_search/action/search_near.hpp"

using namespace std::chrono_literals;

class CheckNearby : public rclcpp::Node{
public:

    using SearchNear = nav_search::action::SearchNear;
    using SearchServer = rclcpp_action::Server<SearchNear>;
    using SearchGoalHandle = rclcpp_action::ServerGoalHandle<SearchNear>;

    using DetectTarget = nav_search::action::DetectTarget;
    using DetectClient = rclcpp_action::Client<DetectTarget>;

    CheckNearby()
    : Node("search_nearby")
    {
        planning_group_   = this->declare_parameter<std::string>("planning_group", "interbotix_arm");
        base_joint_name_  = this->declare_parameter<std::string>("base_joint_name", "waist");
        wrist_joint_name_ = this->declare_parameter<std::string>("wrist_joint_name", "wrist_angle");  // NEW
        arm_base_frame_   = this->declare_parameter<std::string>("arm_base_frame", "vx300s/base_link");
        ee_link_          = this->declare_parameter<std::string>("ee_link", "vx300s/ee_gripper_link");
        cam_link_         = this->declare_parameter<std::string>("cam_link", "vx300s/camera_link");

        tf_buffer_   = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        search_server_ = rclcpp_action::create_server<SearchNear>(
          this,
          "search_nearby",
          std::bind(&CheckNearby::handle_scan_goal,     this, std::placeholders::_1, std::placeholders::_2),
          std::bind(&CheckNearby::handle_scan_cancel,   this, std::placeholders::_1),
          std::bind(&CheckNearby::handle_scan_accepted, this, std::placeholders::_1)
          
        );
  }

    ~CheckNearby() {
      if (search_thread_.joinable()) search_thread_.join();
    }
    private:

    void init_move_group_if_needed()
    {
    if (move_group_) {
      return;
    }

    RCLCPP_INFO(get_logger(), "Fetching robot_description and SRDF from /move_group...");

    auto param_node = rclcpp::Node::make_shared("search_near_param_client");
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
        rclcpp_action::GoalResponse handle_scan_goal(
        const rclcpp_action::GoalUUID &,
        std::shared_ptr<const SearchNear::Goal> goal)
    {
        RCLCPP_INFO(get_logger(),
                    "Performing Second check in nearby area");
        return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
    }

    rclcpp_action::CancelResponse handle_scan_cancel(
        const std::shared_ptr<SearchGoalHandle> /*goal_handle*/)
    {
        RCLCPP_INFO(get_logger(), "SearchNearby goal cancel requested");
        return rclcpp_action::CancelResponse::ACCEPT;
    }

    void handle_scan_accepted(const std::shared_ptr<SearchGoalHandle> goal_handle)
  {
   if (search_thread_.joinable()) search_thread_.join();
    search_thread_ = std::thread(&CheckNearby::reposition_second_check, this, goal_handle);
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


   void reposition_second_check(const std::shared_ptr<SearchGoalHandle> goal_handle)
  {
    auto goal     = goal_handle->get_goal();
    auto result   = std::make_shared<SearchNear::Result>();
    auto feedback = std::make_shared<SearchNear::Feedback>();

    result->x_map_update = goal->x_map;
    result->y_map_update = goal->y_map;
    result->distance     = 0.0f;
    result->known        = false;

    if (goal_handle->is_canceling()) {
      goal_handle->canceled(result);
      RCLCPP_INFO(get_logger(), "Search canceled");
      return;
    }

    init_move_group_if_needed();
    init_detect_client_if_needed();

    if (!tf_buffer_ || !move_group_) {
      goal_handle->abort(result);
      return;
    }

    // --- Sweep parameters ---
    const double pitch_delta_rad   = 10.0 * M_PI / 180.0;  // 10 deg per pitch step
    const double lateral_delta_rad = 15.0 * M_PI / 180.0;  // ±15 deg lateral sweep
    const int    max_pitch_steps   = 3;

    // Lateral offsets: -15°, 0°, +15°
    const double lateral_offsets[] = { -lateral_delta_rad, 0.0, lateral_delta_rad };
    const int    num_lateral       = 3;

    // --- Get current camera pose in arm base frame ---
    geometry_msgs::msg::TransformStamped base_T_cam;
    try {
      base_T_cam = tf_buffer_->lookupTransform(
        arm_base_frame_, cam_link_, tf2::TimePointZero);
    } catch (const tf2::TransformException &ex) {
      RCLCPP_WARN(get_logger(), "TF lookup failed (%s -> %s): %s",
                  arm_base_frame_.c_str(), cam_link_.c_str(), ex.what());
      goal_handle->abort(result);
      return;
    }

    // Current camera position (held constant — only orientation changes)
    geometry_msgs::msg::Pose cam_pose;
    cam_pose.position.x = base_T_cam.transform.translation.x;
    cam_pose.position.y = base_T_cam.transform.translation.y;
    cam_pose.position.z = base_T_cam.transform.translation.z;

    // Decompose current camera orientation
    tf2::Quaternion q_cur;
    tf2::fromMsg(base_T_cam.transform.rotation, q_cur);
    q_cur.normalize();

    tf2::Matrix3x3 R_cur(q_cur);
    tf2::Vector3 fwd = R_cur * tf2::Vector3(1.0, 0.0, 0.0);

    const double base_yaw   = std::atan2(fwd.y(), fwd.x());
    const double base_pitch = std::atan2(-fwd.z(), std::hypot(fwd.x(), fwd.y()));

    double roll_cur, pitch_cur, yaw_cur;
    R_cur.getRPY(roll_cur, pitch_cur, yaw_cur);

    // Get current base joint angle for lateral offsets
    const std::vector<double>& joint_values = move_group_->getCurrentJointValues();
    const std::vector<std::string>& joint_names = move_group_->getJointNames();

    double base_joint_angle = 0.0;
    for (size_t j = 0; j < joint_names.size(); ++j) {
      if (joint_names[j] == base_joint_name_) {
        base_joint_angle = joint_values[j];
        break;
      }
    }

    RCLCPP_INFO(get_logger(),
                "RecheckArea: base_yaw=%.3f, base_pitch=%.3f, base_joint=%.3f",
                base_yaw, base_pitch, base_joint_angle);

    // --- Outer loop: pitch levels ---
    for (int k = 0; k < max_pitch_steps; ++k) {

      const double pitch_new = base_pitch + (k + 1) * pitch_delta_rad;

      RCLCPP_INFO(get_logger(),
                  "Pitch level %d: pitching wrist to %.3f rad", k, pitch_new);

      // Set wrist orientation for this pitch level
      tf2::Quaternion q_new;
      q_new.setRPY(roll_cur, pitch_new, base_yaw);
      q_new.normalize();
      cam_pose.orientation = tf2::toMsg(q_new);

      geometry_msgs::msg::Pose ee_pose;
      if (!cameraPoseToEePoseInBase(cam_pose, ee_pose)) {
        RCLCPP_WARN(get_logger(),
                    "cameraPoseToEePoseInBase failed at pitch level %d; skipping", k);
        continue;
      }

      if (!move_arm_to_pose(ee_pose)) {
        RCLCPP_WARN(get_logger(), "Wrist move failed at pitch level %d; skipping", k);
        continue;
      }

      // --- Inner loop: lateral sweep ---
      for (int l = 0; l < num_lateral; ++l) {

        // Overall progress across both loops
        feedback->progress = static_cast<float>(k * num_lateral + l + 1) /
                            static_cast<float>(max_pitch_steps * num_lateral);
        goal_handle->publish_feedback(feedback);

        if (goal_handle->is_canceling()) {
          RCLCPP_INFO(get_logger(), "RecheckArea canceled during lateral sweep");
          goal_handle->canceled(result);
          return;
        }

        double lateral_angle = base_joint_angle + lateral_offsets[l];
        RCLCPP_INFO(get_logger(),
                    "Pitch level %d, lateral step %d: base joint to %.3f rad",
                    k, l, lateral_angle);

        if (!move_base_to(lateral_angle)) {
          RCLCPP_WARN(get_logger(),
                      "Base joint move failed at pitch %d lateral %d; skipping", k, l);
          continue;
        }

        rclcpp::sleep_for(150ms);

        double targetx_base, targety_base, targetx, targety;

        if (call_detect_target(0.7, targetx_base, targety_base)) {
          if (transform_arm_base_point_to_map(targetx_base, targety_base,
                                              targetx, targety)) {
            double rx, ry, ryaw;
            if (!get_robot_pose_map(rx, ry, ryaw)) {
              RCLCPP_WARN(get_logger(), "Failed to get robot pose; aborting");
              move_arm_to_stow_pose();
              goal_handle->abort(result);
              return;
            }

            double del_x = targetx - rx;
            double del_y = targety - ry;

            result->x_map_update = targetx;
            result->y_map_update = targety;
            result->distance     = std::sqrt(del_x * del_x + del_y * del_y);
            result->known        = true;

            RCLCPP_INFO(get_logger(),
                        "Target found at pitch level %d, lateral step %d: "
                        "(%.2f, %.2f) map frame, dist=%.2f",
                        k, l, targetx, targety, result->distance);
            move_arm_to_stow_pose();
            goal_handle->succeed(result);
            return;
          }
        }
      }

      // Return base joint to center after each pitch level's lateral sweep
      if (!move_base_to(base_joint_angle)) {
        RCLCPP_WARN(get_logger(), "Failed to re-center base joint after pitch level %d", k);
      }
    }
    move_arm_to_stow_pose();
    // Exhausted all pitch + lateral combinations
    RCLCPP_INFO(get_logger(), "RecheckArea: no target found after full sweep");
    goal_handle->succeed(result);  // result->found stays false

    
  }

  std::string wrist_joint_name_;
  std::string planning_group_;
  std::string base_joint_name_;
  std::string arm_base_frame_;
  std::string ee_link_;
  std::string cam_link_;
  std::thread search_thread_;
  
  SearchServer::SharedPtr search_server_;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
  std::shared_ptr<DetectClient> detect_client_;

  std::shared_ptr<tf2_ros::Buffer>           tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<CheckNearby>();

  rclcpp::executors::MultiThreadedExecutor exec;
  exec.add_node(node);
  exec.spin();

  rclcpp::shutdown();
  return 0;
}