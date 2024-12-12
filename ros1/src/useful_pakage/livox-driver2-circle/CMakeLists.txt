cmake_minimum_required(VERSION 3.0.2)
project(circle)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  pcl_ros
  sensor_msgs
  geometry_msgs
  OpenCV REQUIRED
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_compile_options(-std=c++14)

add_executable(circle src/circle.cpp)

target_link_libraries(circle
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)