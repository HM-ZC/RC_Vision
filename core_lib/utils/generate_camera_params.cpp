#include "rc_vision/core/camera.hpp"
#include "rc_vision/core/logger.hpp"

#include <Eigen/Dense>

using namespace rc_vision::core;

int main() {
    Logger::getInstance().setLogLevel(Logger::LogLevel::DEBUG);

    // 设置相机内参
    Eigen::Matrix3d intrinsic;
    intrinsic << 800, 0, 320,
            0, 800, 240,
            0,   0,   1;

    // 设置畸变系数 (k1, k2, p1, p2, k3)
    Eigen::Vector<double, 5> distortion;
    distortion << 0.1, -0.05, 0.001, 0.001, 0.0;

    // 设置外参（假设为单位矩阵，即相机与世界坐标系重合）
    Eigen::Matrix4d extrinsic = Eigen::Matrix4d::Identity();

    Camera cam(intrinsic, distortion);
    cam.setExtrinsic(extrinsic);

    // 保存参数
    if(cam.saveParameters("camera_params.bin")) {
        Logger::getInstance().log(Logger::LogLevel::INFO, "相机参数已生成并保存到 camera_params.bin");
    } else {
        Logger::getInstance().log(Logger::LogLevel::ERROR, "相机参数生成失败");
    }

    return 0;
}