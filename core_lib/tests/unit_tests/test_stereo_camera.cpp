#include "rc_vision/core/camera.hpp"
#include "rc_vision/core/stereo_camera.hpp"
#include "rc_vision/core/logger.hpp"
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

using namespace rc_vision::core;

// 继承测试类以便在测试前后进行初始化
class StereoCameraTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logger::getInstance().setLogLevel(Logger::LogLevel::DEBUG);
    }

    Camera left_cam;
    Camera right_cam;
    Eigen::Matrix4d extrinsic;
    StereoCamera stereo_cam;
};

TEST_F(StereoCameraTest, RectifyImagesTest) {
// 初始化左相机和右相机
Eigen::Matrix3d intrinsic;
intrinsic << 800, 0, 320,
0, 800, 240,
0,   0,   1;
Eigen::Vector<double, 5> distortion;
distortion << 0.1, -0.05, 0, 0, 0;

left_cam.setIntrinsic(intrinsic);
left_cam.setDistortion(distortion);
right_cam.setIntrinsic(intrinsic);
right_cam.setDistortion(distortion);

// 设置外参
extrinsic = Eigen::Matrix4d::Identity();
extrinsic(0, 3) = 0.1; // 右相机相对于左相机在 x 方向平移 0.1 米
stereo_cam = StereoCamera(left_cam, right_cam, extrinsic);

// 创建两张简单的测试图像（黑色图像）
cv::Mat left_image = cv::Mat::zeros(480, 640, CV_8UC3);
cv::Mat right_image = cv::Mat::zeros(480, 640, CV_8UC3);

cv::Mat rectified_left, rectified_right;
bool success = stereo_cam.rectifyImages(left_image, right_image, rectified_left, rectified_right);
EXPECT_TRUE(success);
EXPECT_FALSE(rectified_left.empty());
EXPECT_FALSE(rectified_right.empty());
}

TEST_F(StereoCameraTest, ComputeDisparityTest) {
// 创建两张简单的测试图像（灰度图）
cv::Mat rectified_left = cv::Mat::zeros(480, 640, CV_8UC1);
cv::Mat rectified_right = cv::Mat::zeros(480, 640, CV_8UC1);

// 在左图像中绘制一个白色矩形
cv::rectangle(rectified_left, cv::Point(100, 100), cv::Point(200, 200), cv::Scalar(255), -1);

// 在右图像中绘制相同的矩形，向右平移 10 像素
cv::rectangle(rectified_right, cv::Point(110, 100), cv::Point(210, 200), cv::Scalar(255), -1);

// 初始化相机和立体相机
Eigen::Matrix3d intrinsic;
intrinsic << 800, 0, 320,
0, 800, 240,
0,   0,   1;
Eigen::Vector<double, 5> distortion;
distortion << 0.0, 0.0, 0, 0, 0;

left_cam.setIntrinsic(intrinsic);
left_cam.setDistortion(distortion);
right_cam.setIntrinsic(intrinsic);
right_cam.setDistortion(distortion);

extrinsic = Eigen::Matrix4d::Identity();
extrinsic(0, 3) = 0.1; // 右相机平移
stereo_cam = StereoCamera(left_cam, right_cam, extrinsic);

// 计算视差图
cv::Mat disparity;
bool success = stereo_cam.computeDisparity(rectified_left, rectified_right, disparity);
EXPECT_TRUE(success);
EXPECT_FALSE(disparity.empty());

// 检查视差值
// 由于矩形平移了10像素，期望视差值接近10
double disparity_sum = 0.0;
int count = 0;
for(int y = 100; y < 200; y++) {
for(int x = 100; x < 200; x++) {
double d = disparity.at<short>(y, x);
if(d > 0) { // 只考虑有效视差
disparity_sum += d;
count++;
}
}
}

if(count > 0) {
double avg_disparity = disparity_sum / count;
EXPECT_NEAR(avg_disparity, 10.0 * 16, 5.0); // OpenCV StereoSGBM 视差乘以16存储
}
}