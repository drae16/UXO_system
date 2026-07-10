#pragma once

#include <string>
#include <memory>

#include <behaviortree_cpp/bt_factory.h>
#include <behaviortree_ros2/bt_action_node.hpp>
#include <rclcpp/rclcpp.hpp>

#include "nav_search/action/scan_area.hpp"
#include "nav_search/action/search_near.hpp"
#include "nav_search/action/nav2_move.hpp"
#include "nav_search/action/rotate_and_move_short_distance.hpp"

using namespace BT;

//
// ========================= CONDITIONS =========================
//

class InitialFound : public BT::ConditionNode
{
public:
  InitialFound(const std::string& name, const BT::NodeConfig& config)
    : BT::ConditionNode(name, config) {}

  static BT::PortsList providedPorts()
  {
    return { InputPort<bool>("found") };
  }

  BT::NodeStatus tick() override
  {
    bool found = false;
    if (!getInput("found", found)) {
      return NodeStatus::FAILURE;
    }
    return found ? NodeStatus::SUCCESS : NodeStatus::FAILURE;
  }
};

class PositionKnown : public BT::ConditionNode
{
public:
  PositionKnown(const std::string& name, const BT::NodeConfig& config)
    : BT::ConditionNode(name, config) {}

  static BT::PortsList providedPorts()
  {
    return { InputPort<bool>("known") };
  }

  BT::NodeStatus tick() override
  {
    bool known = false;
    if (!getInput("known", known)) {
      return NodeStatus::FAILURE;
    }
    return known ? NodeStatus::SUCCESS : NodeStatus::FAILURE;
  }
};

class AtStandoff : public BT::ConditionNode
{
public:
  AtStandoff(const std::string& name, const BT::NodeConfig& config)
    : BT::ConditionNode(name, config) {}

  static BT::PortsList providedPorts()
  {
    return { InputPort<float>("distance") };
  }

  BT::NodeStatus tick() override
  {
    float distance = 0.0f;
    if (!getInput("distance", distance)) {
      return NodeStatus::FAILURE;
    }

    // Adjust if you want a different standoff threshold.
    return (distance <= 1.0f) ? NodeStatus::SUCCESS : NodeStatus::FAILURE;
  }
};

class AtFinalPose : public BT::ConditionNode
{
public:
  AtFinalPose(const std::string& name, const BT::NodeConfig& config)
    : BT::ConditionNode(name, config) {}

  static BT::PortsList providedPorts()
  {
    return { InputPort<float>("distance") };
  }

  BT::NodeStatus tick() override
  {
    float distance = 0.0f;
    if (!getInput("distance", distance)) {
      return NodeStatus::FAILURE;
    }

    // Adjust if you want a different final threshold.
    return (distance <= 0.40f) ? NodeStatus::SUCCESS : NodeStatus::FAILURE;
  }
};


class TargetReached : public BT::ConditionNode
{
public:
  TargetReached(const std::string& name, const BT::NodeConfig& config)
    : BT::ConditionNode(name, config) {}

  static BT::PortsList providedPorts()
  {
    return {
      InputPort<bool>("found"),
      InputPort<bool>("known"),
      InputPort<float>("distance")
    };
  }

  BT::NodeStatus tick() override
  {
    bool found = false;
    bool known = false;
    float distance = 999.0f;

    if (!getInput("found", found))    return NodeStatus::FAILURE;
    if (!getInput("known", known))    return NodeStatus::FAILURE;
    if (!getInput("distance", distance)) return NodeStatus::FAILURE;

    if (!found || !known) {
      RCLCPP_INFO(rclcpp::get_logger("bt_tree"), "TargetReached: target lost, restarting search");
      return NodeStatus::FAILURE;
    }

    return (distance <= 0.40f) ? NodeStatus::SUCCESS : NodeStatus::FAILURE;
  }
};

//
// ========================= ACTION WRAPPERS =========================
//

class ArmScan : public BT::RosActionNode<nav_search::action::ScanArea>
{
public:
  using ActionT = nav_search::action::ScanArea;
  using Base = BT::RosActionNode<ActionT>;
  using Goal = ActionT::Goal;
  using WrappedResult = typename rclcpp_action::ClientGoalHandle<ActionT>::WrappedResult;

  ArmScan(const std::string& name,
          const BT::NodeConfig& conf,
          const BT::RosNodeParams& params)
    : Base(name, conf, params) {}

  static PortsList providedPorts()
  {
    return providedBasicPorts({
      InputPort<float>("start_angle"),
      InputPort<float>("end_angle"),
      InputPort<int>("num_steps"),
      InputPort<float>("min_confidence"),

      OutputPort<bool>("found"),
      OutputPort<bool>("known"),
      OutputPort<float>("x_map"),
      OutputPort<float>("y_map"),
      OutputPort<float>("distance"),
      OutputPort<float>("axis")
    });
  }

  bool setGoal(Goal& goal) override
  {
    int num_steps = 0;

    if (!getInput("start_angle", goal.start_angle)) return false;
    if (!getInput("end_angle", goal.end_angle)) return false;
    if (!getInput("num_steps", num_steps)) return false;
    if (!getInput("min_confidence", goal.min_confidence)) return false;

    goal.num_steps = num_steps;
    return true;
  }

  NodeStatus onResultReceived(const WrappedResult& wr) override
  {
    if (!wr.result) {
      return NodeStatus::FAILURE;
    }

    setOutput("found", wr.result->found);
    setOutput("known", wr.result->known);
    setOutput("x_map", wr.result->x_map);
    setOutput("y_map", wr.result->y_map);
    setOutput("distance", wr.result->distance);
    setOutput("axis", wr.result->axis);

    return (wr.code == rclcpp_action::ResultCode::SUCCEEDED)
             ? NodeStatus::SUCCESS
             : NodeStatus::FAILURE;
  }

  NodeStatus onFailure(BT::ActionNodeErrorCode) override
  {
    return NodeStatus::FAILURE;
  }

  NodeStatus onFeedback(const std::shared_ptr<const ActionT::Feedback>)
  {
    return NodeStatus::RUNNING;
  }
};

class RecheckArea : public BT::RosActionNode<nav_search::action::SearchNear>
{
public:
  using ActionT = nav_search::action::SearchNear;
  using Base = BT::RosActionNode<ActionT>;
  using Goal = ActionT::Goal;
  using WrappedResult = typename rclcpp_action::ClientGoalHandle<ActionT>::WrappedResult;

  RecheckArea(const std::string& name,
              const BT::NodeConfig& conf,
              const BT::RosNodeParams& params)
    : Base(name, conf, params) {}

  static PortsList providedPorts()
  {
    return providedBasicPorts({
      InputPort<float>("x_map"),
      InputPort<float>("y_map"),

      OutputPort<bool>("known"),
      OutputPort<float>("x_map_update"),
      OutputPort<float>("y_map_update"),
      OutputPort<float>("distance")
    });
  }

  bool setGoal(Goal& goal) override
  {
    if (!getInput("x_map", goal.x_map)) return false;
    if (!getInput("y_map", goal.y_map)) return false;
    return true;
  }

  NodeStatus onResultReceived(const WrappedResult& wr) override
  {
    if (!wr.result) {
      return NodeStatus::FAILURE;
    }

    setOutput("known", wr.result->known);
    setOutput("x_map_update", wr.result->x_map_update);
    setOutput("y_map_update", wr.result->y_map_update);
    setOutput("distance", wr.result->distance);

    return (wr.code == rclcpp_action::ResultCode::SUCCEEDED)
             ? NodeStatus::SUCCESS
             : NodeStatus::FAILURE;
  }

  NodeStatus onFailure(BT::ActionNodeErrorCode) override
  {
    return NodeStatus::FAILURE;
  }

  NodeStatus onFeedback(const std::shared_ptr<const ActionT::Feedback>)
  {
    return NodeStatus::RUNNING;
  }
};

class Nav2Move : public BT::RosActionNode<nav_search::action::Nav2Move>
{
public:
  using ActionT = nav_search::action::Nav2Move;
  using Base = BT::RosActionNode<ActionT>;
  using Goal = ActionT::Goal;
  using WrappedResult = typename rclcpp_action::ClientGoalHandle<ActionT>::WrappedResult;

  Nav2Move(const std::string& name,
           const BT::NodeConfig& conf,
           const BT::RosNodeParams& params)
    : Base(name, conf, params) {}

  static PortsList providedPorts()
  {
    return providedBasicPorts({
      InputPort<float>("x_map"),
      InputPort<float>("y_map"),
     

      OutputPort<bool>("known"),
      OutputPort<float>("x_map_update"),
      OutputPort<float>("y_map_update"),
      OutputPort<float>("distance"),
      
      BidirectionalPort<float>("axis")
    });
  }

  bool setGoal(Goal& goal) override
  {
    if (!getInput("x_map", goal.x_map)) return false;
    if (!getInput("y_map", goal.y_map)) return false;
    if (!getInput("axis", goal.axis)) return false;
    return true;
  }

  NodeStatus onResultReceived(const WrappedResult& wr) override
  {
    if (!wr.result) {
      return NodeStatus::FAILURE;
    }

    setOutput("known", wr.result->known);
    setOutput("x_map_update", wr.result->x_map_update);
    setOutput("y_map_update", wr.result->y_map_update);
    setOutput("distance", wr.result->distance);
    setOutput("axis", wr.result->axis);

    return (wr.code == rclcpp_action::ResultCode::SUCCEEDED)
             ? NodeStatus::SUCCESS
             : NodeStatus::FAILURE;
  }

  NodeStatus onFailure(BT::ActionNodeErrorCode) override
  {
    return NodeStatus::FAILURE;
  }

  NodeStatus onFeedback(const std::shared_ptr<const ActionT::Feedback>)
  {
    return NodeStatus::RUNNING;
  }
};

class RotateAndMoveShortDistance
  : public BT::RosActionNode<nav_search::action::RotateAndMoveShortDistance>
{
public:
  using ActionT = nav_search::action::RotateAndMoveShortDistance;
  using Base = BT::RosActionNode<ActionT>;
  using Goal = ActionT::Goal;
  using WrappedResult = typename rclcpp_action::ClientGoalHandle<ActionT>::WrappedResult;

  RotateAndMoveShortDistance(const std::string& name,
                             const BT::NodeConfig& conf,
                             const BT::RosNodeParams& params)
    : Base(name, conf, params) {}

  static PortsList providedPorts()
  {
    return providedBasicPorts({
      InputPort<float>("x_map"),
      InputPort<float>("y_map"),

      OutputPort<bool>("known"),
      OutputPort<float>("x_map_update"),
      OutputPort<float>("y_map_update"),
      OutputPort<float>("distance"),

      BidirectionalPort<float>("axis")
    });
  }

  bool setGoal(Goal& goal) override
  {
    if (!getInput("x_map", goal.x_map)) return false;
    if (!getInput("y_map", goal.y_map)) return false;
    if (!getInput("axis", goal.axis)) return false;
    return true;
  }

NodeStatus onResultReceived(const WrappedResult& wr) override
{
  if (!wr.result) {
    return NodeStatus::FAILURE;
  }

  setOutput("known", wr.result->known);
  setOutput("x_map_update", wr.result->x_map_update);
  setOutput("y_map_update", wr.result->y_map_update);
  setOutput("distance", wr.result->distance);
  setOutput("axis", wr.result->axis);

  // Lost target = FAILURE so ReactiveSequence re-evaluates
  if (!wr.result->known) {
    return NodeStatus::FAILURE;
  }

  return (wr.code == rclcpp_action::ResultCode::SUCCEEDED)
           ? NodeStatus::SUCCESS
           : NodeStatus::FAILURE;
}
  NodeStatus onFailure(BT::ActionNodeErrorCode) override
  {
    return NodeStatus::FAILURE;
  }

  NodeStatus onFeedback(const std::shared_ptr<const ActionT::Feedback>)
  {
    return NodeStatus::RUNNING;
  }
};

//
// ========================= REGISTRATION =========================
//

inline void RegisterSearchTreeNodes(BT::BehaviorTreeFactory& factory,
                                    const BT::RosNodeParams& params)
{
  factory.registerNodeType<InitialFound>("InitialFound");
  factory.registerNodeType<PositionKnown>("PositionKnown");
  factory.registerNodeType<AtStandoff>("AtStandoff");
  factory.registerNodeType<AtFinalPose>("AtFinalPose");
  factory.registerNodeType<TargetReached>("TargetReached");

  factory.registerNodeType<ArmScan>("ArmScan", params);
  factory.registerNodeType<RecheckArea>("RecheckArea", params);
  factory.registerNodeType<Nav2Move>("Nav2Move", params);
  factory.registerNodeType<RotateAndMoveShortDistance>("RotateAndMoveShortDistance", params);
  
}

