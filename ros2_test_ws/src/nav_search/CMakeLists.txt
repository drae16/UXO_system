cmake_minimum_required(VERSION 3.8)
project(nav_search)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ── Dependencies ────────────────────────────────────────────────────────────
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(behaviortree_cpp REQUIRED)
find_package(behaviortree_ros2 REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# ── Action interfaces ────────────────────────────────────────────────────────
rosidl_generate_interfaces(${PROJECT_NAME}
  "action/ScanArea.action"
  "action/SearchNear.action"
  "action/Nav2Move.action"
  "action/RotateAndMoveShortDistance.action"
  "action/DetectTarget.action"
  "action/RunSearchTree.action" 
  DEPENDENCIES geometry_msgs
)

# Needed so nodes can use the generated interfaces from the same package
rosidl_get_typesupport_target(cpp_typesupport_target
  ${PROJECT_NAME} "rosidl_typesupport_cpp")

# ── Common dependency list ───────────────────────────────────────────────────
set(NODE_DEPS
  rclcpp
  rclcpp_action
  geometry_msgs
  nav2_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  moveit_ros_planning_interface
  ament_index_cpp
)

# ── Executables ──────────────────────────────────────────────────────────────

# BT runner
add_executable(search_bt_runner
  src/search_bt_runner.cpp
)
ament_target_dependencies(search_bt_runner ${NODE_DEPS})
target_include_directories(search_bt_runner PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(search_bt_runner
  ${cpp_typesupport_target}
  behaviortree_cpp::behaviortree_cpp
  behaviortree_ros2::behaviortree_ros2
)

# Arm scan server
add_executable(arm_scan
  src/arm_scan.cpp
)
ament_target_dependencies(arm_scan ${NODE_DEPS})
target_link_libraries(arm_scan ${cpp_typesupport_target})

# Recheck area server
add_executable(recheck_area
  src/recheck_area.cpp
)
ament_target_dependencies(recheck_area ${NODE_DEPS})
target_link_libraries(recheck_area ${cpp_typesupport_target})

# Nav2 move server
add_executable(nav2_move
  src/nav2_move.cpp
)
ament_target_dependencies(nav2_move ${NODE_DEPS})
target_link_libraries(nav2_move ${cpp_typesupport_target})

# Rotate and move server
add_executable(rotate_and_move
  src/rotate_and_move.cpp
)
ament_target_dependencies(rotate_and_move ${NODE_DEPS})
target_link_libraries(rotate_and_move ${cpp_typesupport_target})

# ── Install ──────────────────────────────────────────────────────────────────
install(TARGETS
  search_bt_runner
  arm_scan
  recheck_area
  nav2_move
  rotate_and_move
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY trees/
  DESTINATION share/${PROJECT_NAME}/trees
)

 install(
  DIRECTORY models/
  DESTINATION share/${PROJECT_NAME}/models
)

install(PROGRAMS
  scripts/yolo_node.py
  DESTINATION lib/${PROJECT_NAME}
)


ament_package()
