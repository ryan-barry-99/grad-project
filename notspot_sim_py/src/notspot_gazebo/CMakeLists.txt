cmake_minimum_required(VERSION 3.0.2)
project(notspot_gazebo)

find_package(catkin REQUIRED)

catkin_package()

# Install model file
install(
    DIRECTORY models/
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/models
)

