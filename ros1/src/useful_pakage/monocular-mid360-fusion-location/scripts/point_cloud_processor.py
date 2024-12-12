#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import std_msgs.msg  # 导入标准消息模块
import numpy as np  # 导入 NumPy

class PointCloudProcessor:
    def __init__(self):
        self.pointcloud_sub = rospy.Subscriber("/livox/lidar", PointCloud2, self.pointcloud_callback)
        self.filtered_pub = rospy.Publisher("/filtered_points", PointCloud2, queue_size=1)  # 发布过滤后的点云
        self.points = []
        self.x_range = (-float('inf'), float('inf'))  # 默认范围为无穷大
        self.y_range = (-float('inf'), float('inf'))  # 默认范围为无穷大
        self.z_range = (-float('inf'), float('inf'))  # 默认范围为无穷大

    def set_xyz_range(self, x_range, y_range, z_range):
        """设置XYZ范围"""
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range

    def pointcloud_callback(self, msg):
        self.points = []
        for point in pc2.read_points(msg, skip_nans=True):
            x, y, z = point[:3]
            # 只添加在设定范围内的点
            if (self.x_range[0] <= x <= self.x_range[1] and
                self.y_range[0] <= y <= self.y_range[1] and
                self.z_range[0] <= z <= self.z_range[1]):
                self.points.append(point)

        # 发布过滤后的点云
        self.publish_filtered_pointcloud()

    def publish_filtered_pointcloud(self):
        """将过滤后的点云发布到ROS话题"""
        if not self.points:
            return  # 如果没有点，直接返回

        # 创建点云的结构
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "livox_frame"  # 请根据您的坐标系命名

        # 转换过滤后的点为 PointCloud2 格式
        filtered_cloud = pc2.create_cloud_xyz32(header, self.points)
        self.filtered_pub.publish(filtered_cloud)

    def get_points(self):
        return self.points

if __name__ == "__main__":
    rospy.init_node("point_cloud_processor")
    processor = PointCloudProcessor()

    # 示例：设置XYZ范围
    processor.set_xyz_range((-1, 1), (-1, 1), (0, 1))  # 设置XYZ范围
    rospy.spin()
