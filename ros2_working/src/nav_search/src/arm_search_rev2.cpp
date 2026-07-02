#include <chrono>
#include <memory>
#include <string>
#include <map>
#include <thread>
#include <future>
#include <cmath>  // for std::sqrt

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <rclcpp/parameter_client.hpp>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>  // NEW: for cmd_vel_out
#include <moveit/move_group_interface/move_group_interface.h>

#include "nav_search/action/detect_target.hpp"
#include "nav_search/action/scan_area.hpp"

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
  : Node("arm_search")
  {
    planning_group_ = this->declare_parameter<std::string>("planning_group", "interbotix_arm");
    base_joint_name_ = this->declare_parameter<std::string>("base_joint_name", "waist");
    wrist_joint_name_ = this->declare_parameter<std::string>("wrist_joint_name", "wrist_angle");  
    start_angle_     = this->declare_parameter<double>("start_angle", -1.14);
    end_angle_       = this->declare_parameter<double>("end_angle",   1.14);
    num_steps_       = this->declare_parameter<int>("num_steps",      10);

    RCLCPP_INFO(get_logger(), "Planning group: %s", planning_group_.c_str());
    RCLCPP_INFO(get_logger(), "Base joint: %s", base_joint_name_.c_str());
    RCLCPP_INFO(get_logger(), "Wrist joint: %s",    wrist_joint_name_.c_str());  // NEW

    // NEW: cmd_vel_out publisher for direct base-frame motion
    cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>(
      "/cmd_vel_out", 10);

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
  // === Small init helpers ====================================================

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

    using namespace std::chrono_literals;

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

    move_group_->setPlanningTime(5.0);
  }

  void init_detect_client_if_needed()
  {
    if (!detect_client_) {
      RCLCPP_INFO(get_logger(), "Creating detect_target action client...");
      detect_client_ = rclcpp_action::create_client<DetectTarget>(
        shared_from_this(), "detect_target");
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

  // === Direct cmd_vel_out "micro-nav" helper =================================

  void drive_relative_xy(double x_base, double y_base)
  {
    if (!cmd_vel_pub_) {
      RCLCPP_WARN(get_logger(), "cmd_vel_out publisher not initialized");
      return;
    }

    double dist = std::sqrt(x_base * x_base + y_base * y_base);

    if (dist < 0.05) {
      RCLCPP_INFO(get_logger(), "Target within 5cm; skipping micro-move");
      return;
    }

    // Cap max distance for safety (e.g. 2 m)
    const double max_dist = 2.0;
    if (dist > max_dist) {
      x_base *= max_dist / dist;
      y_base *= max_dist / dist;
      dist = max_dist;
    }

    // Speed: 0.5 m/s as requested
    const double speed    = 0.3;
    const double duration = dist / speed;
    const double rate_hz  = 20.0;

    double vx = (x_base / dist) * speed;
    double vy = (y_base / dist) * speed;

    RCLCPP_INFO(get_logger(),
                "Driving relative: dist=%.2f m, vx=%.2f, vy=%.2f, dur=%.2f s",
                dist, vx, vy, duration);

    rclcpp::Rate rate(rate_hz);
    geometry_msgs::msg::Twist cmd;
    cmd.linear.x = vx;
    cmd.linear.y = vy;
    cmd.linear.z = 0.0;
    cmd.angular.z = 0.0;

    auto start = this->now();
    while (rclcpp::ok() &&
           (this->now() - start).seconds() < duration) {
      cmd_vel_pub_->publish(cmd);
      rate.sleep();
    }

    // Stop at the end
    geometry_msgs::msg::Twist stop;
    cmd_vel_pub_->publish(stop);
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

  if (!move_group_) {
    RCLCPP_ERROR(get_logger(), "MoveGroupInterface not initialized; aborting scan.");
    goal_handle->abort(result);
    return;
  }

  // 1) Move to Search pose
  if (!move_arm_to_search_pose()) {
    RCLCPP_WARN(get_logger(), "Could not move into search pose");
    goal_handle->abort(result);
    return;
  }

  // 2) Get wrist angle at Search
  double search_wrist_angle = 1.970;


    // 3) Horizontal sweep parameters
  int steps = goal->num_steps;
  if (steps <= 1) {
    RCLCPP_WARN(get_logger(), "num_steps <= 1; adjusting to 2");
    steps = 2;
  }

  double start_angle = goal->start_angle;
  double end_angle   = goal->end_angle;
  double step        = (end_angle - start_angle) / static_cast<double>(steps - 1);

  auto feedback   = std::make_shared<ScanArea::Feedback>();
  bool arm_stowed = false;
  bool target_found = false;

  // 4) Vertical sweep: Search -10°, Search, Search +10°
  const int    max_levels  = 3;
  const double delta_rad   = 6.0 * M_PI / 180.0;  // 10 degrees in radians
  const double offsets[3]  = { -delta_rad, 0.0, 2.5*delta_rad };

  for (int level = 0; level < max_levels && !target_found; ++level) {
    double wrist_angle = search_wrist_angle + offsets[level];

    RCLCPP_INFO(get_logger(),
                "Vertical level %d: moving wrist '%s' to %.3f rad "
                "(Search %.3f + offset %.3f)",
                level, wrist_joint_name_.c_str(),
                wrist_angle, search_wrist_angle, offsets[level]);

    if (!move_wrist_to(wrist_angle)) {
      RCLCPP_WARN(get_logger(),
                  "Could not move wrist to level %d; skipping this level",
                  level);
      continue;
    }

    // --- horizontal sweep at this wrist angle ---
    for (int i = 0; i < steps; ++i) {
      double angle = start_angle + step * static_cast<double>(i);
      feedback->current_angle = static_cast<float>(angle);

      // Progress across all vertical levels
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
                  "Level %d: moving base joint '%s' to angle %.3f rad",
                  level, base_joint_name_.c_str(), angle);

      if (!move_base_to(angle)) {
        RCLCPP_WARN(get_logger(), "Planning failed for this step, skipping.");
        continue;
      }

      rclcpp::sleep_for(200ms);

      double x_base = 0.0, y_base = 0.0;
      bool found = call_detect_target(goal->min_confidence, x_base, y_base);
      if (found) {
        RCLCPP_INFO(get_logger(),
                    "Target found at (%.2f, %.2f) in base frame at level %d",
                    x_base, y_base, level);
        result->found  = true;
        result->x_base = static_cast<float>(x_base);
        result->y_base = static_cast<float>(y_base);

        // Stow arm BEFORE moving the dog
        if (!move_arm_to_stow_pose()) {
          RCLCPP_WARN(get_logger(), "Failed to stow arm before driving");
        } else {
          arm_stowed = true;
        }

        // Direct micro-move via cmd_vel_out
        //drive_relative_xy(x_base, y_base);
        //rotate_side_on_to_target(x_base, y_base);

        target_found = true;
        break;  // break horizontal sweep
      }
    }
  }

  // 5) If we never stowed explicitly, stow now
  if (!arm_stowed) {
    move_arm_to_stow_pose();
  }

  // 6) Finish the action
  goal_handle->succeed(result);
}

  // === MoveIt helpers ========================================================

  bool move_arm_to_search_pose()
  {
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
      RCLCPP_WARN(get_logger(), "No plan found to 'Search' (error code %d)", code.val);
      return false;
    }

    auto exec_code = move_group_->execute(plan);
    if (exec_code != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_WARN(get_logger(), "Execution to 'Search' failed (error code %d)", exec_code.val);
      return false;
    }
    return true;
  }

  bool move_arm_to_stow_pose()
  {
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
      RCLCPP_WARN(get_logger(), "No plan found to 'Sleep' (error code %d)", code.val);
      return false;
    }

    auto exec_code = move_group_->execute(plan);
    if (exec_code != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_WARN(get_logger(), "Execution to 'Sleep' failed (error code %d)", exec_code.val);
      return false;
    }
    return true;
  }

  bool move_base_to(double target_angle)
  {
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
      RCLCPP_WARN(get_logger(), "No plan found for base joint move (error code %d)", code.val);
      return false;
    }

    auto exec_code = move_group_->execute(plan);
    if (exec_code != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_WARN(get_logger(),
                  "Execution for base joint move failed (error code %d)", exec_code.val);
      return false;
    }

    return true;
  }

  // === YOLO DetectTarget action client =======================================

  bool call_detect_target(double min_conf, double &x, double &y)
  {
    using namespace std::chrono_literals;

    if (!detect_client_) {
      init_detect_client_if_needed();
    }

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
    RCLCPP_INFO(get_logger(), "detect_target: target at (%.2f, %.2f)", x, y);
    return true;
  }
  bool move_wrist_to(double target_angle)
{
  // Get current state so other joints stay where they are
  std::map<std::string, double> target;
  target[wrist_joint_name_] = target_angle;

  move_group_->setStartStateToCurrentState();
  if (!move_group_->setJointValueTarget(target)) {
    RCLCPP_ERROR(get_logger(),
                 "Failed to set joint value target for joint '%s' in group '%s'",
                 wrist_joint_name_.c_str(), planning_group_.c_str());
    return false;
  }

  moveit::planning_interface::MoveGroupInterface::Plan plan;
  auto code = move_group_->plan(plan);
  if (code != moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_WARN(get_logger(),
                "No plan found for wrist move (error code %d)", code.val);
    return false;
  }

  auto exec_code = move_group_->execute(plan);
  if (exec_code != moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_WARN(get_logger(),
                "Execution for wrist move failed (error code %d)", exec_code.val);
    return false;
  }

  return true;
}
void rotate_side_on_to_target(double x_base, double y_base)
{
  if (!cmd_vel_pub_) {
    RCLCPP_WARN(get_logger(), "cmd_vel_out publisher not initialized (rotate)");
    return;
  }

  double dist = std::sqrt(x_base * x_base + y_base * y_base);
  if (dist < 1e-3) {
    RCLCPP_INFO(get_logger(), "Target very close; skipping final rotation");
    return;
  }

  // Angle to target from +x axis
  double theta = std::atan2(y_base, x_base);

  // Two options: target on left side (+y) or right side (-y)
  double phi_left  = theta - M_PI_2;   // make target lie along +y
  double phi_right = theta + M_PI_2;   // make target lie along -y

  // Normalize to [-pi, pi]
  auto norm = [](double a) {
    while (a >  M_PI) a -= 2.0 * M_PI;
    while (a < -M_PI) a += 2.0 * M_PI;
    return a;
  };

  phi_left  = norm(phi_left);
  phi_right = norm(phi_right);

  // Choose smaller absolute rotation
  double rot = (std::fabs(phi_left) < std::fabs(phi_right)) ? phi_left : phi_right;

  RCLCPP_INFO(get_logger(),
              "Final side-on rotation: theta=%.3f rad, rot=%.3f rad",
              theta, rot);

  if (std::fabs(rot) < 1e-2) {
    RCLCPP_INFO(get_logger(), "Rotation small; skipping");
    return;
  }

  // Execute rotation with constant angular speed
  const double ang_speed = 0.5;  // rad/s, adjust as needed
  double duration = std::fabs(rot) / ang_speed;

  rclcpp::Rate rate(20.0);  // 20 Hz
  geometry_msgs::msg::Twist cmd;
  cmd.linear.x  = 0.0;
  cmd.linear.y  = 0.0;
  cmd.linear.z  = 0.0;

  // Sign of rotation: CCW positive
  cmd.angular.z = (rot > 0.0) ? ang_speed : -ang_speed;

  auto start = this->now();
  while (rclcpp::ok() &&
         (this->now() - start).seconds() < duration) {
    cmd_vel_pub_->publish(cmd);
    rate.sleep();
  }

  // Stop
  geometry_msgs::msg::Twist stop;
  cmd_vel_pub_->publish(stop);
}


  // === Members ===============================================================

  std::string planning_group_;
  std::string base_joint_name_;
  std::string wrist_joint_name_; 
  double start_angle_;
  double end_angle_;
  int    num_steps_;

  ScanServer::SharedPtr scan_server_;

  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
  std::shared_ptr<DetectClient> detect_client_;

  // NEW: cmd_vel_out publisher
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ArmSearchNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}



































































































































































