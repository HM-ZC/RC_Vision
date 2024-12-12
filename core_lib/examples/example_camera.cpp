#include "rc_vision/core/image.hpp"
#include "rc_vision/core/camera.hpp"
#include "rc_vision/core/logger.hpp"

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

using namespace rc_vision::core;

int main(int argc, char** argv) {
    // 初始化日志级别
    Logger::getInstance().setLogLevel(Logger::LogLevel::DEBUG);

    Logger::getInstance().log(Logger::LogLevel::INFO, "程序开始运行");

    // 创建 Camera 对象并设置内参、畸变系数
    Eigen::Matrix3d intrinsic;
    intrinsic << 800, 0, 320,
            0, 800, 240,
            0,   0,   1;
    Eigen::Vector<double, 5> distortion;
    distortion << 0.1, -0.05, 0, 0, 0;

    Camera cam(intrinsic, distortion);

    // 保存相机参数到文件
    if(!cam.saveParameters("camera_params.bin")) {
        Logger::getInstance().log(Logger::LogLevel::ERROR, "保存相机参数失败");
    }

    // 加载相机参数从文件
    Camera loaded_cam;
    if(!loaded_cam.loadParameters("camera_params.bin")) {
        Logger::getInstance().log(Logger::LogLevel::ERROR, "加载相机参数失败");
    }

    // 创建一个 3D 点并投影到 2D 图像平面
    Eigen::Vector3d point_3d(1.0, 2.0, 5.0);
    Eigen::Vector2d projected = loaded_cam.project(point_3d);
    Logger::getInstance().log(Logger::LogLevel::INFO, "投影后的2D点: (" + std::to_string(projected[0]) + ", " + std::to_string(projected[1]) + ")");

    // 去畸变一个 2D 点
    Eigen::Vector2d distorted_point(400, 300);
    Eigen::Vector2d undistorted = loaded_cam.undistort(distorted_point);
    Logger::getInstance().log(Logger::LogLevel::INFO, "去畸变后的2D点: (" + std::to_string(undistorted[0]) + ", " + std::to_string(undistorted[1]) + ")");

    // 加载一张图像并去畸变
    Image img;
    if(!img.load("path/to/distorted_image.jpg")) {
        Logger::getInstance().log(Logger::LogLevel::ERROR, "图像加载失败");
    } else {
        cv::Mat undistorted_image = loaded_cam.undistortImage(img.getMat());
        if(!undistorted_image.empty()) {
            cv::imwrite("undistorted_image.jpg", undistorted_image);
            Logger::getInstance().log(Logger::LogLevel::INFO, "去畸变后的图像已保存");
        }
    }

    // 使用 StereoCamera 进行立体校正和视差计算
    // 假设已经有两个相机的内参和外参
    Camera left_cam = cam;
    Camera right_cam = cam; // 这里为了示例，使用相同的相机参数
    Eigen::Matrix4d extrinsic = Eigen::Matrix4d::Identity();
    extrinsic(0, 3) = 0.1; // 假设右相机相对于左相机在 x 方向平移 0.1 米

    StereoCamera stereo_cam(left_cam, right_cam, extrinsic);

    // 加载左右图像
    Image left_img, right_img;
    if(!left_img.load("path/to/left_image.jpg") || !right_img.load("path/to/right_image.jpg")) {
        Logger::getInstance().log(Logger::LogLevel::ERROR, "左右图像加载失败");
    } else {
        cv::Mat rectified_left, rectified_right;
        if(stereo_cam.rectifyImages(left_img.getMat(), right_img.getMat(), rectified_left, rectified_right)) {
            cv::imwrite("rectified_left.jpg", rectified_left);
            cv::imwrite("rectified_right.jpg", rectified_right);
            Logger::getInstance().log(Logger::LogLevel::INFO, "左右图像已校正并保存");
        }

        // 计算视差图
        cv::Mat disparity;
        if(stereo_cam.computeDisparity(rectified_left, rectified_right, disparity)) {
            // 归一化视差图以便显示
            cv::Mat disparity_display;
            disparity.convertTo(disparity_display, CV_8U, 255/(16*5));
            cv::imwrite("disparity_map.jpg", disparity_display);
            Logger::getInstance().log(Logger::LogLevel::INFO, "视差图已计算并保存");
        }
    }

    Logger::getInstance().log(Logger::LogLevel::INFO, "程序结束");
    return 0;
}