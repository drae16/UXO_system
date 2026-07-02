cmake_minimum_required(VERSION 3.5)
project(nav_search)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rosidl_default_runtime REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(control_msgs REQUIRED)
#find_package(go2_robot_sdk REQUIRED)


# ---- Actions ----
rosidl_generate_interfaces(${PROJECT_NAME}
  "action/DetectTarget.action"
  "action/ScanArea.action"
  "srv/TrackTarget.srv"
  DEPENDENCIES geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime geometry_msgs)

# ---- C++ node ----
add_executable(arm_search src/arm_search.cpp)
ament_target_dependencies(arm_search
  rclcpp
  rclcpp_action
  moveit_ros_planning_interface
  nav2_msgs
  tf2_ros
  tf2_geometry_msgs
)

rosidl_target_interfaces(arm_search
  ${PROJECT_NAME} "rosidl_typesupport_cpp")

add_executable(track_target src/track_target.cpp)
ament_target_dependencies(track_target
  rclcpp
  geometry_msgs
  control_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)

rosidl_target_interfaces(track_target
  ${PROJECT_NAME} "rosidl_typesupport_cpp")
  
install(TARGETS 
  arm_search
  track_target
  DESTINATION lib/${PROJECT_NAME})
 
 install(
  DIRECTORY models/
  DESTINATION share/${PROJECT_NAME}/models
)

# ---- Python node (just a script) ----
install(PROGRAMS
  scripts/yolo_node.py
  DESTINATION lib/${PROJECT_NAME})

# Export action interfaces
ament_export_dependencies(rosidl_default_runtime)

ament_package()
