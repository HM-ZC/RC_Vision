#include "rc_vision/core/camera.hpp"
#include "rc_vision/core/logger.hpp"
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

using namespace rc_vision::core;

// 继承测试类以便在测试前后进行初始化
class CameraTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logger::getInstance().setLogLevel(Logger::LogLevel::DEBUG);
    }

    Camera cam;
};

TEST_F(CameraTest, ProjectionTest) {
Eigen::Matrix3d intrinsic;
intrinsic << 800, 0, 320,
0, 800, 240,
0,   0,   1;
Eigen::Vector<double, 5> distortion;
distortion << 0.1, -0.05, 0, 0, 0;

cam.setIntrinsic(intrinsic);
cam.setDistortion(distortion);

// 设置外参为单位矩阵
Eigen::Matrix4d extrinsic = Eigen::Matrix4d::Identity();
cam.setExtrinsic(extrinsic);

Eigen::Vector3d point_3d(1.0, 2.0, 5.0);
Eigen::Vector2d projected = cam.project(point_3d);

// 预期投影点
// 投影公式: (800 * (1/5) + 320, 800 * (2/5) + 240) = (480, 560)
EXPECT_NEAR(projected[0], 480.0, 1e-5);
EXPECT_NEAR(projected[1], 560.0, 1e-5);
}

TEST_F(CameraTest, UndistortPointTest) {
Eigen::Matrix3d intrinsic;
intrinsic << 800, 0, 320,
0, 800, 240,
0,   0,   1;
Eigen::Vector<double, 5> distortion;
distortion << 0.1, -0.05, 0, 0, 0;

cam.setIntrinsic(intrinsic);
cam.setDistortion(distortion);

// 畸变点
Eigen::Vector2d distorted_point(400, 300);
Eigen::Vector2d undistorted = cam.undistort(distorted_point);

// 因为畸变系数较小，预计去畸变后的点接近原点
EXPECT_NEAR(undistorted[0], 400.0, 10.0);
EXPECT_NEAR(undistorted[1], 300.0, 10.0);
}

TEST_F(CameraTest, SaveAndLoadParametersTest) {
Eigen::Matrix3d intrinsic;
intrinsic << 800, 0, 320,
0, 800, 240,
0,   0,   1;
Eigen::Vector<double, 5> distortion;
distortion << 0.1, -0.05, 0, 0, 0;
Eigen::Matrix4d extrinsic = Eigen::Matrix4d::Identity();
extrinsic(0, 3) = 0.5; // 平移

cam.setIntrinsic(intrinsic);
cam.setDistortion(distortion);
cam.setExtrinsic(extrinsic);

// 保存参数
std::string file_path = "test_camera_params.bin";
ASSERT_TRUE(cam.saveParameters(file_path));

// 创建一个新的 Camera 对象并加载参数
Camera loaded_cam;
ASSERT_TRUE(loaded_cam.loadParameters(file_path));

// 比较内参
EXPECT_TRUE(loaded_cam.getIntrinsic().isApprox(cam.getIntrinsic(), 1e-6));

// 比较畸变系数
EXPECT_TRUE(loaded_cam.getDistortion().isApprox(cam.getDistortion(), 1e-6));

// 比较外参
EXPECT_TRUE(loaded_cam.getExtrinsic().isApprox(cam.getExtrinsic(), 1e-6));

// 删除测试文件
std::remove(file_path.c_str());
}