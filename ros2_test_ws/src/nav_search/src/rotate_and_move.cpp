#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <future>
#include <cmath>
#include <map>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <rclcpp/parameter_client.hpp>

#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <moveit/move_group_interface/move_group_interface.h>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Vector3.h>

#include "nav_search/action/detect_target.hpp"
#include "nav_search/action/rotate_and_move_short_distance.hpp"

using namespace std::chrono_literals;

class RotateAndMoveShortDistanceNode : public rclcpp::Node
{
public:
  using FinalMove = nav_search::action::RotateAndMoveShortDistance;
  using FinalServer = rclcpp_action::Server<FinalMove>;
  using FinalGoalHandle = rclcpp_action::ServerGoalHandle<FinalMove>;

  using DetectTarget = nav_search::action::DetectTarget;
  using DetectClient = rclcpp_action::Client<DetectTarget>;

  RotateAndMoveShortDistanceNode()
  : Node("rotate_and_move_short_distance")
  {
    planning_group_   = this->declare_parameter<std::string>("planning_group", "interbotix_arm");
    arm_base_frame_   = this->declare_parameter<std::string>("arm_base_frame", "vx300s/base_link");
    base_frame_       = this->declare_parameter<std::string>("base_frame", "base_link");
    ee_link_          = this->declare_parameter<std::string>("ee_link", "vx300s/ee_gripper_link");
    cam_link_         = this->declare_parameter<std::string>("cam_link", "vx300s/camera_link");
    cmd_vel_topic_    = this->declare_parameter<std::string>("cmd_vel_topic", "/cmd_vel");

    camera_height_       = this->declare_parameter<double>("camera_height", 0.40);
    radius_min_          = this->declare_parameter<double>("camera_radius_min", 0.20);
    radius_max_          = this->declare_parameter<double>("camera_radius_max", 0.45);
    radius_scale_factor_ = this->declare_parameter<double>("camera_radius_scale", 0.2);

    min_confidence_         = this->declare_parameter<double>("min_confidence", 0.7);
    final_distance_         = this->declare_parameter<double>("final_distance", 0.40);
    heading_tol_deg_        = this->declare_parameter<double>("heading_tolerance_deg", 6.0);
    x_center_tol_           = this->declare_parameter<double>("x_center_tolerance", 0.08);

    rotate_speed_           = this->declare_parameter<double>("rotate_speed", 1.0);
    forward_speed_          = this->declare_parameter<double>("forward_speed", 0.4);
    lateral_speed_          = this->declare_parameter<double>("lateral_speed", 0.3);

    rotate_burst_s_         = this->declare_parameter<double>("rotate_burst_s", 0.25);
    translate_burst_s_      = this->declare_parameter<double>("translate_burst_s", 0.25);
    settle_s_               = this->declare_parameter<double>("settle_s", 0.15);

    max_align_iters_        = this->declare_parameter<int>("max_align_iters", 12);
    max_center_iters_       = this->declare_parameter<int>("max_center_iters", 12);
    max_approach_iters_     = this->declare_parameter<int>("max_approach_iters", 20);

    tf_buffer_   = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    cmd_pub_ = this->create_publisher<geometry_msgs::msg::Twist>(cmd_vel_topic_, 10);

    action_server_ = rclcpp_action::create_server<FinalMove>(
      this,
      "rotate_and_move_short_distance",
      std::bind(&RotateAndMoveShortDistanceNode::handle_goal, this, std::placeholders::_1, std::placeholders::_2),
      std::bind(&RotateAndMoveShortDistanceNode::handle_cancel, this, std::placeholders::_1),
      std::bind(&RotateAndMoveShortDistanceNode::handle_accepted, this, std::placeholders::_1)
    );
  }

  ~RotateAndMoveShortDistanceNode() {
    if (exec_thread_.joinable()) exec_thread_.join();
  }

private:
  struct TargetState
  {
    bool known{false};

    double x_base{0.0};
    double y_base{0.0};

    double x_map{0.0};
    double y_map{0.0};

    double axis_map{0.0};
    double distance{0.0};
  };

  rclcpp_action::GoalResponse handle_goal(
    const rclcpp_action::GoalUUID &,
    std::shared_ptr<const FinalMove::Goal> goal)
  {
    RCLCPP_INFO(
      get_logger(),
      "Final local move request: x_map=%.2f y_map=%.2f axis_map=%.2f",
      goal->x_map, goal->y_map, goal->axis);
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse handle_cancel(const std::shared_ptr<FinalGoalHandle>)
  {
    stop_robot();
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void handle_accepted(const std::shared_ptr<FinalGoalHandle> goal_handle)
  {
    if (exec_thread_.joinable()) exec_thread_.join();
    exec_thread_ = std::thread(&RotateAndMoveShortDistanceNode::execute, this, goal_handle);
  }

  void execute(const std::shared_ptr<FinalGoalHandle> goal_handle)
  {
    auto goal = goal_handle->get_goal();
    auto result = std::make_shared<FinalMove::Result>();
    auto feedback = std::make_shared<FinalMove::Feedback>();

    result->known = false;
    result->axis = goal->axis;
    result->x_map_update = goal->x_map;
    result->y_map_update = goal->y_map;
    result->distance = 0.0f;

    init_move_group_if_needed();
    init_detect_client_if_needed();

    if (!move_group_ || !detect_client_ || !tf_buffer_) {
      goal_handle->abort(result);
      return;
    }

    TargetState state;
    state.axis_map = goal->axis;
    state.x_map = goal->x_map;
    state.y_map = goal->y_map;



    if (!transform_map_point_to_arm_base(goal->x_map, goal->y_map,
                                     state.x_base, state.y_base)) {
      RCLCPP_WARN(get_logger(), "Could not transform initial map position to arm frame, "
                                "arm will use fallback position");
                                     }

    if (!raise_arm_and_redetect(state)) {
      stop_robot();
      fill_result_from_state(result, state);
      move_arm_to_stow_pose();
      goal_handle->succeed(result);
      return;
    }

    publish_distance_feedback(goal_handle, feedback, state.distance);

    if (!align_heading(goal_handle, feedback,result, state)) {
      stop_robot();
      move_arm_to_stow_pose();
      fill_result_from_state(result, state);
      goal_handle->succeed(result);
      return;
    }

    if (!center_target_x(goal_handle, feedback,result, state)) {
      stop_robot();
      move_arm_to_stow_pose();
      fill_result_from_state(result, state);
      goal_handle->succeed(result);
      return;
    }

    if (!approach_sideways(goal_handle, feedback,result, state)) {
      stop_robot();
      move_arm_to_stow_pose();
      fill_result_from_state(result, state);
      goal_handle->succeed(result);
      return;
    }

    stop_robot();
    move_arm_to_stow_pose();
    fill_result_from_state(result, state);
    goal_handle->succeed(result);
  }

  void init_move_group_if_needed()
  {
    if (move_group_) {
      return;
    }

    auto param_node = rclcpp::Node::make_shared("final_move_param_client");
    auto param_client = std::make_shared<rclcpp::AsyncParametersClient>(param_node, "/move_group");

    while (!param_client->wait_for_service(1s)) {
      if (!rclcpp::ok()) {
        return;
      }
    }

    auto future = param_client->get_parameters({"robot_description", "robot_description_semantic"});
    rclcpp::executors::SingleThreadedExecutor exec;
    exec.add_node(param_node);

    auto ret = exec.spin_until_future_complete(future, 5s);
    exec.remove_node(param_node);
    if (ret != rclcpp::FutureReturnCode::SUCCESS) {
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

    move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
      shared_from_this(), planning_group_);

    move_group_->setPlanningTime(3.0);
  }

  void init_detect_client_if_needed()
  {
    if (!detect_client_) {
      detect_client_ = rclcpp_action::create_client<DetectTarget>(shared_from_this(), "detect_target");
    }
  }

  bool get_robot_pose_map(double &x, double &y, double &yaw)
  {
    try {
      auto tf = tf_buffer_->lookupTransform("map", base_frame_, tf2::TimePointZero);

      x = tf.transform.translation.x;
      y = tf.transform.translation.y;

      const auto &q = tf.transform.rotation;
      double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
      double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
      yaw = std::atan2(siny_cosp, cosy_cosp);
      return true;
    }
    catch (const tf2::TransformException &ex) {
      RCLCPP_WARN(get_logger(), "Failed map->%s lookup: %s", base_frame_.c_str(), ex.what());
      return false;
    }
  }


  bool transform_map_point_to_arm_base(double x_map, double y_map,
                                      double &x_base, double &y_base)
  {
    geometry_msgs::msg::PoseStamped pt_map;
    pt_map.header.frame_id = "map";
    pt_map.header.stamp = this->now();
    pt_map.pose.position.x = x_map;
    pt_map.pose.position.y = y_map;
    pt_map.pose.position.z = 0.0;
    pt_map.pose.orientation.w = 1.0;

    try {
      auto pt_arm = tf_buffer_->transform(pt_map, arm_base_frame_,
                                          tf2::durationFromSec(0.5));
      x_base = pt_arm.pose.position.x;
      y_base = pt_arm.pose.position.y;
      return true;
    }
    catch (const tf2::TransformException &ex) {
      RCLCPP_WARN(get_logger(), "Transform map->%s failed: %s",
                  arm_base_frame_.c_str(), ex.what());
      return false;
    }
  }

  bool transform_arm_base_point_to_map(double x_base, double y_base, double &x_map, double &y_map)
  {
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
    }
    catch (const tf2::TransformException &ex) {
      RCLCPP_WARN(get_logger(), "Transform %s->map failed: %s", arm_base_frame_.c_str(), ex.what());
      return false;
    }
  }

  bool call_detect_target(double min_conf, double &x_base, double &y_base, double &axis_map)
  {
    if (!detect_client_) return false;

    if (!detect_client_->wait_for_action_server(2s)) {
      RCLCPP_WARN(get_logger(), "detect_target not available");
      return false;
    }

    DetectTarget::Goal goal_msg;
    goal_msg.min_confidence = static_cast<float>(min_conf);

    auto future_goal = detect_client_->async_send_goal(goal_msg);
    if (future_goal.wait_for(5s) != std::future_status::ready) {
      return false;
    }

    auto goal_handle = future_goal.get();
    if (!goal_handle) {
      return false;
    }

    auto result_future = detect_client_->async_get_result(goal_handle);
    if (result_future.wait_for(10s) != std::future_status::ready) {
      return false;
    }

    auto wrapped = result_future.get();
    auto result = wrapped.result;
    if (!result || !result->found) {
      return false;
    }

    x_base = result->x_base;
    y_base = result->y_base;
    axis_map = result->axis;
    return true;
  }

  static double normalize_angle(double a)
  {
    while (a > M_PI) a -= 2.0 * M_PI;
    while (a < -M_PI) a += 2.0 * M_PI;
    return a;
  }

  double best_axis_heading_error(double axis_map, double robot_yaw)
  {
    RCLCPP_INFO(get_logger(),"current yaw = %.2f", robot_yaw);
    double e1 = normalize_angle(axis_map - robot_yaw);
    double e2 = normalize_angle(axis_map + M_PI - robot_yaw);
    return (std::abs(e1) < std::abs(e2)) ? e1 : e2;
  }

  void publish_distance_feedback(
    const std::shared_ptr<FinalGoalHandle> &goal_handle,
    const std::shared_ptr<FinalMove::Feedback> &feedback,
    double distance)
  {
    feedback->progress = static_cast<float>(distance);
    goal_handle->publish_feedback(feedback);
  }

  bool check_cancel(const std::shared_ptr<FinalGoalHandle> &goal_handle,
                    const std::shared_ptr<FinalMove::Result> &result)
  {
    if (goal_handle->is_canceling()) {
      stop_robot();
      goal_handle->canceled(result);
      return true;
    }
    return false;
  }

  void stop_robot()
  {
    geometry_msgs::msg::Twist cmd;
    for (int i = 0; i < 5; ++i) {
      cmd_pub_->publish(cmd);
      rclcpp::sleep_for(30ms);
    }
  }

  void publish_cmd_for_duration(double vx, double vy, double wz, double seconds)
  {
    geometry_msgs::msg::Twist cmd;
    cmd.linear.x = vx;
    cmd.linear.y = vy;
    cmd.angular.z = wz;

    rclcpp::Rate rate(30.0);
    auto start = this->now();
    while ((this->now() - start).seconds() < seconds && rclcpp::ok()) {
      cmd_pub_->publish(cmd);
      rate.sleep();
    }
  }

  bool move_arm_to_stow_pose()
  {
    if (!move_group_) return false;

    move_group_->setStartStateToCurrentState();
    if (!move_group_->setNamedTarget("Sleep")) {
      return false;
    }

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    auto code = move_group_->plan(plan);
    if (code != moveit::core::MoveItErrorCode::SUCCESS) {
      return false;
    }

    auto exec_code = move_group_->execute(plan);
    return exec_code == moveit::core::MoveItErrorCode::SUCCESS;
  }

  geometry_msgs::msg::Pose compute_camera_pose_on_circle_base(double target_x_base, double target_y_base)
  {
    double X = target_x_base;
    double Y = target_y_base;
    double Z = 0.0;

    double r_xy = std::sqrt(X * X + Y * Y);
    if (r_xy < 1e-3) r_xy = 1e-3;

    double ux = X / r_xy;
    double uy = Y / r_xy;

    double R_des = radius_scale_factor_ * r_xy;
    if (R_des < radius_min_) R_des = radius_min_;
    if (R_des > radius_max_) R_des = radius_max_;

    geometry_msgs::msg::Pose cam_pose;
    cam_pose.position.x = ux * R_des;
    cam_pose.position.y = uy * R_des;
    cam_pose.position.z = camera_height_;

    tf2::Vector3 cam_pos(cam_pose.position.x, cam_pose.position.y, cam_pose.position.z);
    tf2::Vector3 tgt_pos(X, Y, Z);

    tf2::Vector3 x_axis = tgt_pos - cam_pos;
    if (x_axis.length2() < 1e-6) {
      x_axis = tf2::Vector3(1.0, 0.0, 0.0);
    }
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
      x_axis.z(), y_axis.z(), z_axis.z()
    );

    tf2::Quaternion q;
    R.getRotation(q);
    q.normalize();
    cam_pose.orientation = tf2::toMsg(q);

    return cam_pose;
  }

  bool cameraPoseToEePoseInBase(const geometry_msgs::msg::Pose &cam_pose_in_base,
                                geometry_msgs::msg::Pose &ee_pose_in_base)
  {
    try {
      auto ee_to_cam = tf_buffer_->lookupTransform(ee_link_, cam_link_, tf2::TimePointZero);

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
    catch (const tf2::TransformException &ex) {
      RCLCPP_WARN(get_logger(), "TF lookup %s->%s failed: %s", ee_link_.c_str(), cam_link_.c_str(), ex.what());
      return false;
    }
  }

  bool move_arm_to_pose(const geometry_msgs::msg::Pose &pose)
  {
    if (!move_group_) return false;

    move_group_->setStartStateToCurrentState();
    move_group_->setPoseTarget(pose, ee_link_);

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    auto code = move_group_->plan(plan);
    if (code != moveit::core::MoveItErrorCode::SUCCESS) {
      return false;
    }

    auto exec_code = move_group_->execute(plan);
    return exec_code == moveit::core::MoveItErrorCode::SUCCESS;
  }

  bool track_target_once_base(double target_x_base, double target_y_base)
  {
    geometry_msgs::msg::Pose cam_pose = compute_camera_pose_on_circle_base(target_x_base, target_y_base);

    geometry_msgs::msg::Pose ee_pose;
    if (!cameraPoseToEePoseInBase(cam_pose, ee_pose)) {
      return false;
    }

    return move_arm_to_pose(ee_pose);
  }

  void update_target_estimate_base(TargetState &state, double dx, double dy, double dtheta)
  {
    double c = std::cos(dtheta);
    double s = std::sin(dtheta);

    double x1 = state.x_base - dx;
    double y1 = state.y_base - dy;

    double x2 = c * x1 + s * y1;
    double y2 = -s * x1 + c * y1;

    RCLCPP_INFO(get_logger(),"Old target estimate in base- x: %.2f, y: %.2f", state.x_base ,state.y_base);

    state.x_base = x2;
    state.y_base = y2;
    RCLCPP_INFO(get_logger(),"New target estimate in base- x: %.2f, y: %.2f", x2,y2);
    state.distance = std::hypot(state.x_base, state.y_base);
  }

  bool raise_arm_and_redetect(TargetState &state)
  { 
    RCLCPP_INFO(get_logger(),"Checking target at x: %.2f, y: %.2f", state.x_base,state.y_base);
    if (!track_target_once_base(state.x_base == 0.0 && state.y_base == 0.0 ? 0.3 : state.x_base,
                                state.x_base == 0.0 && state.y_base == 0.0 ? 0.3 : state.y_base)) {
      return false;
    }

    rclcpp::sleep_for(std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::duration<double>(settle_s_)));

    double xb, yb, axis_map;
    if (!call_detect_target(min_confidence_, xb, yb, axis_map)) {
      state.known= false;
      return false;
    }

    state.known = true;
    state.x_base = xb;
    state.y_base = yb;
    state.axis_map = axis_map;
    state.distance = std::hypot(xb, yb);

    transform_arm_base_point_to_map(xb, yb, state.x_map, state.y_map);
    return true;
  }

  bool do_motion_and_recheck(TargetState &state, double vx, double vy, double wz, double duration_s)
  {
    if (!move_arm_to_stow_pose()) {
      return false;
    }

    publish_cmd_for_duration(vx, vy, wz, duration_s);
    stop_robot();
    rclcpp::sleep_for(std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::duration<double>(settle_s_)));

    double dx = vx * duration_s;
    double dy = vy * duration_s;
    double dtheta = wz * duration_s;

    update_target_estimate_base(state, dx, dy, dtheta);

    if (!raise_arm_and_redetect(state)) {
      state.known = false;
      return false;
    }

    return true;
  }

  bool align_heading(const std::shared_ptr<FinalGoalHandle> &goal_handle,
                     const std::shared_ptr<FinalMove::Feedback> &feedback,
                     std::shared_ptr<FinalMove::Result> &result, 
                     TargetState &state)
  {
    double tol = heading_tol_deg_ * M_PI / 180.0;

    for (int i = 0; i < max_align_iters_; ++i) {

      fill_result_from_state(result, state);
      if (check_cancel(goal_handle, result)){
        return false;
      }

      double rx, ry, ryaw;
      if (!get_robot_pose_map(rx, ry, ryaw)) {
        return false;
      }

      publish_distance_feedback(goal_handle, feedback, state.distance);

      double err = best_axis_heading_error(state.axis_map, ryaw);
      if (std::abs(err) <= tol) {
        RCLCPP_INFO(get_logger(),"alignment is complete");
        return true;
      }


      RCLCPP_INFO(get_logger(),"aligning to theta: %.2f",err);



      double wz = (err > 0.0) ? rotate_speed_ : -rotate_speed_;
      if (!do_motion_and_recheck(state, 0.0, 0.0, wz, rotate_burst_s_)) {
        return false;
      }
    }

    return false;
  }

  bool center_target_x(const std::shared_ptr<FinalGoalHandle> &goal_handle,
                       const std::shared_ptr<FinalMove::Feedback> &feedback,
                       std::shared_ptr<FinalMove::Result> &result, 
                       TargetState &state)
  {
    for (int i = 0; i < max_center_iters_; ++i) {
      fill_result_from_state(result, state);
      if (check_cancel(goal_handle, result)){ 
        return false;}

      publish_distance_feedback(goal_handle, feedback, state.distance);

      if (std::abs(state.x_base) <= x_center_tol_) {
        return true;
      }

      double vx = (state.x_base > 0.0) ? forward_speed_ : -forward_speed_;
      if (!do_motion_and_recheck(state, vx, 0.0, 0.0, translate_burst_s_)) {
        return false;
      }
    }

    return false;
  }

  bool approach_sideways(const std::shared_ptr<FinalGoalHandle> &goal_handle,
                         const std::shared_ptr<FinalMove::Feedback> &feedback,
                         std::shared_ptr<FinalMove::Result> &result, 
                         TargetState &state)
  {
    double tol = heading_tol_deg_ * M_PI / 180.0;

    for (int i = 0; i < max_approach_iters_; ++i) {
      fill_result_from_state(result, state);
      if (check_cancel(goal_handle, result)) {
         return false;
      }

      publish_distance_feedback(goal_handle, feedback, state.distance);

      if (state.distance <= final_distance_) {
        return true;
      }

      double rx, ry, ryaw;
      if (!get_robot_pose_map(rx, ry, ryaw)) {
        return false;
      }

      double heading_err = best_axis_heading_error(state.axis_map, ryaw);
      if (std::abs(heading_err) > tol) {
        if (!align_heading(goal_handle, feedback,result, state)) {
          return false;
        }
      }

      if (std::abs(state.y_base) < 1e-6) {
        return false;
      }

      double vy = (state.y_base > 0.0) ? lateral_speed_ : -lateral_speed_;
      if (!do_motion_and_recheck(state, 0.0, vy, 0.0, translate_burst_s_)) {
        return false;
      }
    }

    return false;
  }

  void fill_result_from_state(std::shared_ptr<FinalMove::Result> &result, const TargetState &state)
  {
    result->known = state.known;
    result->axis = static_cast<float>(state.axis_map);
    result->x_map_update = static_cast<float>(state.x_map);
    result->y_map_update = static_cast<float>(state.y_map);
    result->distance = static_cast<float>(state.distance);
  }

  std::string planning_group_;
  std::string arm_base_frame_;
  std::string base_frame_;
  std::string ee_link_;
  std::string cam_link_;
  std::string cmd_vel_topic_;
  std::thread exec_thread_;

  double camera_height_;
  double radius_min_;
  double radius_max_;
  double radius_scale_factor_;

  double min_confidence_;
  double final_distance_;
  double heading_tol_deg_;
  double x_center_tol_;

  double rotate_speed_;
  double forward_speed_;
  double lateral_speed_;
  double rotate_burst_s_;
  double translate_burst_s_;
  double settle_s_;

  int max_align_iters_;
  int max_center_iters_;
  int max_approach_iters_;

  FinalServer::SharedPtr action_server_;
  std::shared_ptr<DetectClient> detect_client_;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<RotateAndMoveShortDistanceNode>();

  rclcpp::executors::MultiThreadedExecutor exec;
  exec.add_node(node);
  exec.spin();

  rclcpp::shutdown();
  return 0;
}