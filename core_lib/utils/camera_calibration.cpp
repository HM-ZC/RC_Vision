#include "rc_vision/core/camera.hpp"
#include "rc_vision/core/logger.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <string>
#include <iostream>

using namespace rc_vision::core;

int main(int argc, char** argv) {
    // 检查命令行参数
    if(argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <pattern_size_width> <pattern_size_height> <calibration_images_dir>" << std::endl;
        return -1;
    }

    int pattern_size_width = std::stoi(argv[1]);
    int pattern_size_height = std::stoi(argv[2]);
    std::string calib_images_dir = argv[3];

    cv::Size pattern_size(pattern_size_width, pattern_size_height);
    float square_size = 1.0f; // 根据你的棋盘格实际大小调整

    // 准备物体点
    std::vector<std::vector<cv::Point3f>> object_points;
    std::vector<std::vector<cv::Point2f>> image_points;
    std::vector<cv::String> filenames;

    cv::glob(calib_images_dir + "/*.jpg", filenames, false);

    for(const auto& fname : filenames) {
        cv::Mat img = cv::imread(fname);
        if(img.empty()) {
            Logger::getInstance().log(Logger::LogLevel::WARN, "无法读取图像: " + fname);
            continue;
        }

        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(gray, pattern_size, corners,
                                               cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);

        if(found) {
            cv::cornerSubPix(gray, corners, cv::Size(11,11), cv::Size(-1,-1),
                             cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.001));

            // 绘制并显示角点
            cv::drawChessboardCorners(img, pattern_size, corners, found);
            cv::imshow("Chessboard", img);
            cv::waitKey(100);

            // 添加物体点
            std::vector<cv::Point3f> objp;
            for(int i=0; i<pattern_size_height; ++i) {
                for(int j=0; j<pattern_size_width; ++j) {
                    objp.emplace_back(j*square_size, i*square_size, 0.0f);
                }
            }
            object_points.emplace_back(objp);
            image_points.emplace_back(corners);
        } else {
            Logger::getInstance().log(Logger::LogLevel::WARN, "在图像中未找到棋盘格: " + fname);
        }
    }

    cv::destroyAllWindows();

    if(object_points.empty() || image_points.empty()) {
        Logger::getInstance().log(Logger::LogLevel::ERROR, "未找到有效的标定图像.");
        return -1;
    }

    // 执行标定
    Eigen::Matrix3d intrinsic;
    Eigen::Vector<double, 5> distortion;

    bool calib_success = Camera::calibrateCamera(object_points, image_points, cv::Size(filenames[0].size(), filenames[0].size()),
                                                 intrinsic, distortion);

    if(!calib_success) {
        Logger::getInstance().log(Logger::LogLevel::ERROR, "相机标定失败.");
        return -1;
    }

    // 创建相机实例并设置参数
    Camera cam;
    cam.setIntrinsic(intrinsic);
    cam.setDistortion(distortion);
    // 设置外参为单位矩阵，或根据需要调整

    // 保存标定参数
    if(cam.saveParameters("calibrated_camera_params.bin")) {
        Logger::getInstance().log(Logger::LogLevel::INFO, "相机参数已保存到 calibrated_camera_params.bin");
    } else {
        Logger::getInstance().log(Logger::LogLevel::ERROR, "保存相机参数失败.");
        return -1;
    }

    return 0;
}