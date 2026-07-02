cmake_minimum_required(VERSION 3.8)
project(d1_550_description)

find_package(ament_cmake REQUIRED)

# Install common description-package resources into share/<pkg>/
install(
  DIRECTORY config launch meshes urdf
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)

ament_package()

