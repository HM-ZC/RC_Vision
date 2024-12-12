#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
import std_msgs.msg
import numpy as np
import cv2
import yaml
import message_filters
from ultralytics import YOLO
from visualization_msgs.msg import Marker

class KalmanFilter:
    def __init__(self, process_noise=1e-2, measurement_noise=1e-1, estimation_error=1.0):
        self.A = 1  # 状态转移模型
        self.H = 1  # 观测模型
        self.Q = process_noise  # 过程噪声协方差
        self.R = measurement_noise  # 测量噪声协方差
        self.P = estimation_error  # 初始估计误差协方差
        self.x = 0  # 初始状态

    def update(self, measurement):
        # 预测
        self.P = self.A * self.P * self.A + self.Q
        # 更新
        K = self.P * self.H / (self.H * self.P * self.H + self.R)
        self.x = self.x + K * (measurement - self.H * self.x)
        self.P = (1 - K * self.H) * self.P
        return self.x

class SensorFusion:
    def __init__(self):
        rospy.init_node("sensor_fusion")

        self.bridge = CvBridge()
        self.model = YOLO("/root/orb_ws/src/yolo_lidar_fusion/model/yolo11n.pt")

        # 预加载YOLO模型来避免第一次推理卡顿
        self.warmup_yolo()

        # 创建消息过滤器以进行时间同步
        image_sub = message_filters.Subscriber("/usb_cam/image_raw", Image)
        pointcloud_sub = message_filters.Subscriber("/livox/lidar", PointCloud2)
        self.ts = message_filters.ApproximateTimeSynchronizer([image_sub, pointcloud_sub], queue_size=5, slop=0.05)
        self.ts.registerCallback(self.data_callback)

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

        # 初始化卡尔曼滤波器
        self.kf_x = KalmanFilter()
        self.kf_y = KalmanFilter()
        self.kf_z = KalmanFilter()

    def warmup_yolo(self):
        blank_image = np.zeros((640, 480, 3), dtype=np.uint8)
        self.model(blank_image)

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

        # 构造相机内参矩阵
        self.camera_matrix = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])

        # 提取畸变系数
        self.distortion_coeffs = np.array(calib_data['distortion_coefficients']['data'])

        # 加载相机到雷达的外参（旋转和平移）
        rotation = calib_data['extrinsic']['rotation']
        translation = calib_data['extrinsic']['translation']
        self.rotation_matrix = np.array(rotation).reshape(3, 3)
        self.translation_vector = np.array(translation).reshape(3, 1)

    def data_callback(self, img_msg, pcl_msg):
        try:
            self.process_data(img_msg, pcl_msg)
        except Exception as e:
            rospy.logerr(f"Error in data_callback: {e}")

    def process_data(self, img_msg, pcl_msg):
        self.image_callback(img_msg)
        self.pointcloud_callback(pcl_msg)

    def image_callback(self, msg):
        # 将 ROS 图像消息转换为 OpenCV 图像
        undistorted_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # 畸变矫正
        # undistorted_image = cv2.undistort(undistorted_image, self.camera_matrix, self.distortion_coeffs)

        # 用矫正后的图像进行YOLO检测
        results = self.model(undistorted_image)

        known_diameter = 0.3  # 设定球的实际直径
        best_box = None
        best_confidence = 0

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]  # 获取置信度
                cv2.rectangle(undistorted_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 在框内绘制置信度
                cv2.putText(undistorted_image, f"{confidence:.2f}", (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 检查置信度并更新最佳框
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_box = (x1, y1, x2, y2)

        # 处理最佳框
        if best_box is not None:
            x1, y1, x2, y2 = best_box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            self.map_to_3d(cx, cy, w, h, known_diameter)

        # 发布检测后的图像
        self.detection_pub.publish(self.bridge.cv2_to_imgmsg(undistorted_image, "bgr8"))

    def pointcloud_callback(self, msg):
        self.points = []
        for point in pc2.read_points(msg, skip_nans=True):
            x, y, z = point[:3]
            if (self.x_range[0] <= x <= self.x_range[1] and
                self.y_range[0] <= y <= self.y_range[1] and
                self.z_range[0] <= z <= self.z_range[1]):
                self.points.append((x, y, z))

        self.publish_filtered_pointcloud()

    def publish_filtered_pointcloud(self):
        if not self.points:
            return
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "livox_frame"
        filtered_cloud = pc2.create_cloud_xyz32(header, self.points)
        self.filtered_pub.publish(filtered_cloud)

    def map_to_3d(self, cx, cy, w, h, known_diameter):
        # 使用相机内参估算深度
        focal_length = (self.fx + self.fy) / 2  # 使用 fx 和 fy 的平均值作为焦距
        estimated_depth = focal_length * known_diameter / ((w + h) / 2)
        rospy.loginfo(f"相机深度: {estimated_depth}")
        # 将像素坐标转换为相机坐标
        u = cx
        v = cy
        rospy.loginfo(f"像素坐标: {u}, {v}")
        # 计算相机坐标系下的 3D 坐标
        x_camera = (u - self.cx) * estimated_depth / self.fx
        y_camera = (self.cy - v) * estimated_depth / self.fy
        rospy.loginfo(f"相机坐标系下的3D坐标: {x_camera}, {y_camera}")
        # 将相机坐标系下的 3D 点转换为雷达坐标系
        x_lidar, y_lidar, z_lidar = self.transform_to_lidar_frame(estimated_depth, x_camera, y_camera)
        rospy.loginfo(f"雷达坐标系的相机3D点: {x_lidar}, {y_lidar}, {z_lidar}")
        # 在点云中寻找匹配的深度和对应的点
        matched_depth = self.get_depth_from_pointcloud(x_lidar, y_lidar, z_lidar)
        rospy.loginfo(f"匹配深度: {matched_depth}")
        # 使用平均深度来计算 3D 坐标
        if matched_depth is not None:
            # 计算x和y的平均值
            y_values = []
            z_values = []

            for point in self.points:
                x, y, z = point[:3]
                if abs(x - matched_depth) <= 0.2 and np.sqrt((y_lidar - y) ** 2 + (z_lidar - z) ** 2) <= 0.2:  # 容差范围
                    y_values.append(y)
                    z_values.append(z)

            # 计算平均值
            if y_values and z_values:
                avg_x = matched_depth
                avg_y = np.mean(y_values)
                avg_z = np.mean(z_values)

                # 使用卡尔曼滤波器平滑位置
                avg_x = self.kf_x.update(avg_x)
                avg_y = self.kf_y.update(avg_y)
                avg_z = self.kf_z.update(avg_z)

                rospy.loginfo(f"Matched 3D Position in Lidar Frame: {avg_x}, {avg_y}, {avg_z}")

                # 发布到 RViz
                self.publish_marker(avg_x, avg_y, avg_z)
            else:
                rospy.logwarn("No matching x or y values found in point cloud for estimated depth.")
        else:
            rospy.logwarn("No matching depth found in point cloud for estimated depth.")


    def get_depth_from_pointcloud(self, cx, cy, estimated_depth, tolerance=0.2):
        matching_depths = []

        for point in self.points:
            x, y, z = point[:3]

            if abs(x - estimated_depth) <= tolerance and np.sqrt((cx - y) ** 2 + (cy - z) ** 2) <= tolerance:
                matching_depths.append(x)

        if matching_depths:
            return np.mean(matching_depths)
        
        return None

    def transform_to_lidar_frame(self, x, y, z):
        """
        将相机坐标系下的 3D 点 (x, y, z) 转换到雷达坐标系
        """
        camera_point = np.array([[x], [y], [z]])
        lidar_point = self.rotation_matrix @ camera_point + self.translation_vector
        return lidar_point.flatten()  # 转换为 1D 数组 [x, y, z]

    def publish_marker(self, x, y, z):
        marker = Marker()
        marker.header.frame_id = "livox_frame"  # 雷达坐标系的 frame_id
        marker.header.stamp = rospy.Time.now()
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        marker.scale.x = marker.scale.y = marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.r = 1.0
        self.marker_pub.publish(marker)

    def set_xyz_range(self, x_range, y_range, z_range):
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range

if __name__ == "__main__":
    processor = SensorFusion()
    processor.set_xyz_range((-5, 5), (-5, 5), (0, 5))
    rospy.spin()

