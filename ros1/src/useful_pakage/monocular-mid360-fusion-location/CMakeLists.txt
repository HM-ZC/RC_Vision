cmake_minimum_required(VERSION 3.0.2)
project(yolo_lidar_fusion)

## 查找catkin和系统依赖项
find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  std_msgs
  tf
  cv_bridge
)
## 声明catkin包
catkin_package(
  CATKIN_DEPENDS rospy sensor_msgs std_msgs tf cv_bridge
)

## 指定包含目录
include_directories(
  ${catkin_INCLUDE_DIRS}
)

## 安装Python脚本
catkin_install_python(PROGRAMS
  scripts/yolo_detector.py
  scripts/point_cloud_processor.py
  scripts/sensor_fusion_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## 安装配置文件和启动文件
install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

