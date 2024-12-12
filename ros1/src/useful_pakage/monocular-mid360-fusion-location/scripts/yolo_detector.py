#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImageDisplay:
    def __init__(self):
        rospy.init_node("image_display")
        #rospy.loginfo("Initializing Image Display Node")
        
        self.bridge = CvBridge()
        rospy.Subscriber("/yolo/detections", Image, self.display_callback)

    def display_callback(self, msg):
        try:
            # 将 ROS 图像消息转换为 OpenCV 格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # 显示图像
            cv2.imshow("YOLO Detection Display", cv_image)
            cv2.waitKey(1)  # 等待1毫秒，确保窗口刷新
        except Exception as e:
            rospy.logerr(f"Error displaying image: {e}")

if __name__ == "__main__":
    display_node = ImageDisplay()
    rospy.spin()