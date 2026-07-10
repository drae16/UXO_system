#include "nav_search/search_and_approach.hpp"
#include "nav_search/action/run_search_tree.hpp"
#include <ament_index_cpp/get_package_share_directory.hpp>

class SearchBTRunner : public rclcpp::Node
{
public:
  using RunSearchTree = nav_search::action::RunSearchTree;
  using BTServer      = rclcpp_action::Server<RunSearchTree>;
  using BTGoalHandle  = rclcpp_action::ServerGoalHandle<RunSearchTree>;

  SearchBTRunner() : Node("search_bt_runner")
  {
    bt_server_ = rclcpp_action::create_server<RunSearchTree>(
      this, "run_search_tree",
      std::bind(&SearchBTRunner::handle_goal,     this, std::placeholders::_1, std::placeholders::_2),
      std::bind(&SearchBTRunner::handle_cancel,   this, std::placeholders::_1),
      std::bind(&SearchBTRunner::handle_accepted, this, std::placeholders::_1)
    );
  }

  ~SearchBTRunner() {
    if (bt_thread_.joinable()) bt_thread_.join();
  }

private:
  rclcpp_action::GoalResponse handle_goal(
    const rclcpp_action::GoalUUID &,
    std::shared_ptr<const RunSearchTree::Goal> /*goal*/)
  {
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse handle_cancel(
    const std::shared_ptr<BTGoalHandle> /*goal_handle*/)
  {
    cancel_requested_ = true;
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void handle_accepted(const std::shared_ptr<BTGoalHandle> goal_handle)
  {
    if (bt_thread_.joinable()) bt_thread_.join();
    bt_thread_ = std::thread(&SearchBTRunner::run_tree, this, goal_handle);
  }

  void run_tree(const std::shared_ptr<BTGoalHandle> goal_handle)
{
  cancel_requested_ = false;
  const auto goal = goal_handle->get_goal();
  auto result     = std::make_shared<RunSearchTree::Result>();
  auto feedback   = std::make_shared<RunSearchTree::Feedback>();

  BT::BehaviorTreeFactory factory;
  BT::RosNodeParams params;
  params.nh             = shared_from_this();
  params.server_timeout = std::chrono::milliseconds(2000);
  RegisterSearchTreeNodes(factory, params);

  auto blackboard = BT::Blackboard::create();
  blackboard->set("found",            false);
  blackboard->set("known",            false);
  blackboard->set("target_x_map",     0.0f);
  blackboard->set("target_y_map",     0.0f);
  blackboard->set("target_distance",  999.0f);
  blackboard->set("target_axis",      0.0f);
  blackboard->set("scan_start_angle", goal->scan_start_angle);
  blackboard->set("scan_end_angle",   goal->scan_end_angle);
  blackboard->set("scan_steps",       goal->scan_steps);
  blackboard->set("min_confidence",   goal->min_confidence);

  std::string tree_path =
    ament_index_cpp::get_package_share_directory("nav_search")
    + "/trees/search_and_approach.xml";

  auto tree = factory.createTreeFromFile(tree_path, blackboard);

  // --- Node status logger ---
  std::vector<BT::TreeNode::StatusChangeSubscriber> subscribers;
  for (auto& subtree : tree.subtrees) {
    for (auto& node : subtree->nodes) {
      auto node_name = node->name();
      subscribers.push_back(
        node->subscribeToStatusChange(
          [node_name](BT::TimePoint,
                      const BT::TreeNode&,
                      BT::NodeStatus prev,
                      BT::NodeStatus curr) {
            if (prev != curr) {
              RCLCPP_INFO(rclcpp::get_logger("bt_tree"),
                          "[%-35s] %s -> %s",
                          node_name.c_str(),
                          BT::toStr(prev).c_str(),
                          BT::toStr(curr).c_str());
            }
          }));
    }
  }

  // --- Tick loop ---
  BT::NodeStatus status = BT::NodeStatus::RUNNING;
  int restart_count = 0;
  const int max_restarts = 1;

  while (rclcpp::ok())
  {
    if (cancel_requested_) {
      tree.haltTree();
      goal_handle->canceled(result);
      return;
    }

    status = tree.tickOnce();

    feedback->progress = blackboard->get<float>("target_distance");
    goal_handle->publish_feedback(feedback);

    if (status == BT::NodeStatus::SUCCESS) {
      break;
    }

    if (status == BT::NodeStatus::FAILURE) {
      if (restart_count >= max_restarts) {
        RCLCPP_WARN(get_logger(), "Tree failed after %d restart(s), giving up",
                    restart_count);
        break;
      }
      RCLCPP_INFO(get_logger(), "Tree failed, restarting search (attempt %d/%d)...",
                  restart_count + 1, max_restarts);
      blackboard->set("found",           false);
      blackboard->set("known",           false);
      blackboard->set("target_distance", 999.0f);
      tree.haltTree();
      status = BT::NodeStatus::RUNNING;
      ++restart_count;
    }

    tree.sleep(std::chrono::milliseconds(100));
  }

  // --- Fill result from blackboard ---
  float x_map = 0.0f, y_map = 0.0f, distance = 999.0f;

  if (!blackboard->get("target_x_map",    x_map))    x_map    = 0.0f;
  if (!blackboard->get("target_y_map",    y_map))    y_map    = 0.0f;
  if (!blackboard->get("target_distance", distance)) distance = 999.0f;

  result->succeeded      = (status == BT::NodeStatus::SUCCESS);
  result->final_x_map    = x_map;
  result->final_y_map    = y_map;
  result->final_distance = distance;

  if (result->succeeded) {
    goal_handle->succeed(result);
  } else {
    goal_handle->abort(result);
  }
}

  BTServer::SharedPtr bt_server_;
  std::thread         bt_thread_;
  std::atomic<bool>   cancel_requested_{false};
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<SearchBTRunner>();

  rclcpp::executors::MultiThreadedExecutor exec;
  exec.add_node(node);
  exec.spin();

  rclcpp::shutdown();
  return 0;
}