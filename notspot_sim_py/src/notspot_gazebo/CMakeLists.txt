cmake_minimum_required(VERSION 3.0.2)
project(notspot_gazebo)

find_package(catkin REQUIRED)

catkin_package()

# Install model file
install(
    DIRECTORY models/
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/models
)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  gazebo_ros
  sensor_msgs
  std_msgs
  gazebo_model_collision_plugin
  # your_custom_msgs_package, if you have custom messages or services
)

# Declare a catkin package
catkin_package(
  # INCLUDE_DIRS include
  # LIBRARIES your_package_name
  CATKIN_DEPENDS roscpp gazebo_ros sensor_msgs std_msgs gazebo_model_collision_plugin # your_custom_msgs_package
  # DEPENDS system_lib
)
