// am_shim.h -- drop-in replacement for the parts of AutoModality's am_utils
// that the y3space_driver actually uses. Plain rclcpp only.
#pragma once

#include <rclcpp/rclcpp.hpp>
#include <chrono>
#include <cstdint>
#include <string>

namespace am
{

// Global node handle (am_utils exposed the same thing as am::Node::node).
// Defined once in yost_main.cpp.
struct Node
{
  static rclcpp::Node::SharedPtr node;
};

inline rclcpp::Logger shimLogger()
{
  return Node::node ? Node::node->get_logger() : rclcpp::get_logger("y3space_driver");
}

inline rclcpp::Time ClockNow()
{
  return Node::node->now();
}

inline std::chrono::nanoseconds toDuration(double seconds)
{
  return std::chrono::nanoseconds(static_cast<int64_t>(seconds * 1e9));
}

// am::getParam<T>(name, out, default): declare (if needed) then read.
template <typename T>
bool getParam(const std::string &name, T &value, const T &default_value)
{
  if (!Node::node->has_parameter(name))
  {
    Node::node->declare_parameter<T>(name, default_value);
  }
  return Node::node->get_parameter(name, value);
}

}  // namespace am

// ---- ROS1-style logging macros mapped onto rclcpp -------------------------
// THROTTLE periods are in seconds (ROS1 convention); rclcpp wants ms.
#define AM_SHIM_CLOCK (*am::Node::node->get_clock())
#define AM_SHIM_MS(sec) static_cast<int>((sec) * 1000.0)

#define ROS_DEBUG(...) RCLCPP_DEBUG(am::shimLogger(), __VA_ARGS__)
#define ROS_INFO(...)  RCLCPP_INFO(am::shimLogger(), __VA_ARGS__)
#define ROS_WARN(...)  RCLCPP_WARN(am::shimLogger(), __VA_ARGS__)
#define ROS_ERROR(...) RCLCPP_ERROR(am::shimLogger(), __VA_ARGS__)

#define ROS_DEBUG_STREAM(x) RCLCPP_DEBUG_STREAM(am::shimLogger(), x)
#define ROS_INFO_STREAM(x)  RCLCPP_INFO_STREAM(am::shimLogger(), x)
#define ROS_WARN_STREAM(x)  RCLCPP_WARN_STREAM(am::shimLogger(), x)
#define ROS_ERROR_STREAM(x) RCLCPP_ERROR_STREAM(am::shimLogger(), x)

#define ROS_INFO_THROTTLE(sec, ...)  RCLCPP_INFO_THROTTLE(am::shimLogger(), AM_SHIM_CLOCK, AM_SHIM_MS(sec), __VA_ARGS__)
#define ROS_WARN_THROTTLE(sec, ...)  RCLCPP_WARN_THROTTLE(am::shimLogger(), AM_SHIM_CLOCK, AM_SHIM_MS(sec), __VA_ARGS__)
#define ROS_ERROR_THROTTLE(sec, ...) RCLCPP_ERROR_THROTTLE(am::shimLogger(), AM_SHIM_CLOCK, AM_SHIM_MS(sec), __VA_ARGS__)

#define ROS_INFO_STREAM_THROTTLE(sec, x)  RCLCPP_INFO_STREAM_THROTTLE(am::shimLogger(), AM_SHIM_CLOCK, AM_SHIM_MS(sec), x)
#define ROS_WARN_STREAM_THROTTLE(sec, x)  RCLCPP_WARN_STREAM_THROTTLE(am::shimLogger(), AM_SHIM_CLOCK, AM_SHIM_MS(sec), x)
#define ROS_ERROR_STREAM_THROTTLE(sec, x) RCLCPP_ERROR_STREAM_THROTTLE(am::shimLogger(), AM_SHIM_CLOCK, AM_SHIM_MS(sec), x)
