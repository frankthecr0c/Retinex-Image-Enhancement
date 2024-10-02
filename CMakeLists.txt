cmake_minimum_required(VERSION 3.0.2)
project(rretinex)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  cv_bridge
  image_transport
  dynamic_reconfigure
)

generate_dynamic_reconfigure_options(
  cfg/MSRetinex.cfg
  cfg/SSRetinex.cfg
)

catkin_package(
  CATKIN_DEPENDS
    rospy
    sensor_msgs
    cv_bridge
    image_transport
    dynamic_reconfigure
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)