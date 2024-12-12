#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
import std_msgs.msg
import numpy as np
import cv2
import yaml
import threading
from ultralytics import YOLO
from visualization_msgs.msg import Marker  # 导入Marker消息类型

class SensorFusion:
    def __init__(self):
        rospy.init_node("sensor_fusion")
        rospy.loginfo("Initializing Sensor Fusion Node")

        self.bridge = CvBridge()
        self.model = YOLO("/root/orb_ws/src/yolo_lidar_fusion/best.pt")

        # 预加载YOLO模型来避免第一次推理卡顿
        self.warmup_yolo()

        # 创建图像和点云的订阅者
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.image_callback)
        self.pointcloud_sub = rospy.Subscriber("/livox/lidar", PointCloud2, self.pointcloud_callback)

        # 发布检测后的图像和过滤后的点云
        self.detection_pub = rospy.Publisher("/yolo/detections", Image, queue_size=1)
        self.filtered_pub = rospy.Publisher("/filtered_points", PointCloud2, queue_size=1)
        
        # 发布3D标记的Publisher
        self.marker_pub = rospy.Publisher("/visualization_marker", Marker, queue_size=10)

        # 加载标定文件
        self.load_calibration("/root/orb_ws/src/yolo_lidar_fusion/config/calibration.yaml")
        
        # 初始化点云存储和范围
        self.points = []
        self.x_range = (-float('inf'), float('inf'))
        self.y_range = (-float('inf'), float('inf'))
        self.z_range = (-float('inf'), float('inf'))

        # 初始化存储的图像和点云消息
        self.last_image_msg = None
        self.last_pcl_msg = None
        self.sync_slop = rospy.Duration(0.1)  # 设置时间同步的阈值，例如50ms

    def warmup_yolo(self):
        """提前加载YOLO模型，通过一次空白推理来缓冲"""
        rospy.loginfo("Warming up YOLO model...")
        blank_image = np.zeros((640, 480, 3), dtype=np.uint8)  # 创建一张空白图像
        self.model(blank_image)  # 运行一次空白推理
        rospy.loginfo("YOLO model warmup complete.")

    def load_calibration(self, file_path):
        rospy.loginfo(f"Loading calibration from {file_path}")
        with open(file_path, "r") as file:
            calib_data = yaml.safe_load(file)

        # 提取相机内参
        camera_matrix = calib_data['camera_matrix']['data']
        self.fx = camera_matrix[0]
        self.fy = camera_matrix[4]
        self.cx = camera_matrix[2]
        self.cy = camera_matrix[5]
        rospy.loginfo("Camera calibration loaded successfully")

    def image_callback(self, img_msg):
        rospy.loginfo("Image message received")
        self.last_image_msg = img_msg
        self.check_and_process_data()  # 检查是否有同步的点云消息

    def pointcloud_callback(self, pcl_msg):
        rospy.loginfo("PointCloud message received")
        self.last_pcl_msg = pcl_msg
        self.check_and_process_data()  # 检查是否有同步的图像消息

    def check_and_process_data(self):
        """检查图像和点云消息的时间戳是否接近，若接近则处理数据"""
        if self.last_image_msg is None or self.last_pcl_msg is None:
            return  # 如果缺少图像或点云消息，退出

        # 检查图像和点云消息的时间差是否在允许的同步范围内
        time_diff = abs(self.last_image_msg.header.stamp - self.last_pcl_msg.header.stamp)
        if time_diff < self.sync_slop:
            # 启动一个线程来异步处理数据
            processing_thread = threading.Thread(target=self.process_data, args=(self.last_image_msg, self.last_pcl_msg))
            processing_thread.start()

            # 清除已处理的消息
            self.last_image_msg = None
            self.last_pcl_msg = None

    def process_data(self, img_msg, pcl_msg):
        """异步处理图像和点云数据"""
        # 处理图像
        rospy.loginfo("Processing image in async thread")
        self.image_processing(img_msg)

        # 处理点云
        rospy.loginfo("Processing point cloud in async thread")
        self.pointcloud_processing(pcl_msg)

    def image_processing(self, msg):
        rospy.loginfo("Processing image")
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        results = self.model(cv_image)

        # 遍历检测结果并绘制边界框
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 计算物体中心坐标
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                
                # 调用map_to_3d方法进行三维映射
                self.map_to_3d(cx, cy)

        # 发布检测后的图像
        self.detection_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))

        rospy.loginfo("Image processed and published")

    def pointcloud_processing(self, msg):
        rospy.loginfo("Processing point cloud")
        self.points = []
        for point in pc2.read_points(msg, skip_nans=True):
            # 只添加在设定范围内的点
            x, y, z = point[:3]
            if (self.x_range[0] <= x <= self.x_range[1] and
                self.y_range[0] <= y <= self.y_range[1] and
                self.z_range[0] <= z <= self.z_range[1]):
                self.points.append((x, y, z))  # 确保只添加 (x, y, z)

        rospy.loginfo(f"Filtered points: {len(self.points)}")
        # 发布过滤后的点云
        self.publish_filtered_pointcloud()

    def publish_filtered_pointcloud(self):
        """将过滤后的点云发布到ROS话题"""
        if not self.points:
            rospy.logwarn("No points to publish")
            return  # 如果没有点，直接返回

        # 创建点云的结构
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "livox_frame"  # 请根据您的坐标系命名

        # 转换过滤后的点为 PointCloud2 格式
        filtered_cloud = pc2.create_cloud_xyz32(header, self.points)
        self.filtered_pub.publish(filtered_cloud)
        rospy.loginfo("Filtered point cloud published")

    def map_to_3d(self, cx, cy):
        # 使用加载的内参（K矩阵）将2D像素坐标转换为相机坐标系的3D点
        K = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])
        
        # 从过滤后的点云中获取深度
        depth = self.get_depth_from_pointcloud(cx, cy)
        if depth is not None:
            u, v = cx, cy
            x = (u - K[0][2]) * depth / K[0][0]
            y = (v - K[1][2]) * depth / K[1][1]
            z = depth
            rospy.loginfo(f"3D Position: {x}, {y}, {z}")

            # 调用方法发布到RViz
            self.publish_marker(x, y, z)
        else:
            rospy.logwarn("Depth not found for point")

    def get_depth_from_pointcloud(self, cx, cy):
        closest_point = None
        min_distance = float('inf')

        for point in self.points:
            x, y, z = point[:3]
            # 计算坐标点的欧几里得距离
            distance = np.sqrt((cx - y) ** 2 + (cy - z) ** 2)

            if distance < min_distance:
                min_distance = distance
                closest_point = x  # 获取深度值

        return closest_point

    def publish_marker(self, x, y, z):
        marker = Marker()
        marker.header.frame_id = "livox_frame"
        marker.header.stamp = rospy.Time.now()

        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        # 设置球体的位置
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z

        # 设置球体的缩放和大小
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1

        # 设置颜色
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        # 发布marker
        self.marker_pub.publish(marker)
        rospy.loginfo("Marker published in RViz")

    def set_xyz_range(self, x_range, y_range, z_range):
        """设置XYZ范围"""
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range

if __name__ == "__main__":
    processor = SensorFusion()

    # 示例：设置XYZ范围
    processor.set_xyz_range((-1, 1), (-1, 1), (0, 1))
    rospy.spin()