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
    if(argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <pattern_size_width> <pattern_size_height> <left_calib_images_dir> <right_calib_images_dir>" << std::endl;
        return -1;
    }

    int pattern_size_width = std::stoi(argv[1]);
    int pattern_size_height = std::stoi(argv[2]);
    std::string calib_images_dir_left = argv[3];
    std::string calib_images_dir_right = argv[4];

    cv::Size pattern_size(pattern_size_width, pattern_size_height);
    float square_size = 1.0f; // 根据你的棋盘格实际大小调整

    // 准备物体点
    std::vector<std::vector<cv::Point3f>> object_points;
    std::vector<std::vector<cv::Point2f>> image_points_left;
    std::vector<std::vector<cv::Point2f>> image_points_right;
    std::vector<cv::String> filenames_left, filenames_right;

    cv::glob(calib_images_dir_left + "/*.jpg", filenames_left, false);
    cv::glob(calib_images_dir_right + "/*.jpg", filenames_right, false);

    if(filenames_left.size() != filenames_right.size()) {
        Logger::getInstance().log(Logger::LogLevel::ERROR, "左右相机标定图像数量不一致.");
        return -1;
    }

    for(size_t i = 0; i < filenames_left.size(); ++i) {
        cv::Mat img_left = cv::imread(filenames_left[i]);
        cv::Mat img_right = cv::imread(filenames_right[i]);

        if(img_left.empty() || img_right.empty()) {
            Logger::getInstance().log(Logger::LogLevel::WARN, "无法读取图像: " + filenames_left[i] + " 或 " + filenames_right[i]);
            continue;
        }

        cv::Mat gray_left, gray_right;
        cv::cvtColor(img_left, gray_left, cv::COLOR_BGR2GRAY);
        cv::cvtColor(img_right, gray_right, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> corners_left, corners_right;
        bool found_left = cv::findChessboardCorners(gray_left, pattern_size, corners_left,
                                                    cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);
        bool found_right = cv::findChessboardCorners(gray_right, pattern_size, corners_right,
                                                     cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);

        if(found_left && found_right) {
            cv::cornerSubPix(gray_left, corners_left, cv::Size(11,11), cv::Size(-1,-1),
                             cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.001));
            cv::cornerSubPix(gray_right, corners_right, cv::Size(11,11), cv::Size(-1,-1),
                             cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.001));

            // 绘制并显示角点
            cv::drawChessboardCorners(img_left, pattern_size, corners_left, found_left);
            cv::drawChessboardCorners(img_right, pattern_size, corners_right, found_right);
            cv::imshow("Left Chessboard", img_left);
            cv::imshow("Right Chessboard", img_right);
            cv::waitKey(100);

            // 添加物体点
            std::vector<cv::Point3f> objp;
            for(int j=0; j<pattern_size_height; ++j) {
                for(int k=0; k<pattern_size_width; ++k) {
                    objp.emplace_back(k*square_size, j*square_size, 0.0f);
                }
            }
            object_points.emplace_back(objp);
            image_points_left.emplace_back(corners_left);
            image_points_right.emplace_back(corners_right);
        } else {
            Logger::getInstance().log(Logger::LogLevel::WARN, "在图像中未找到棋盘格: " + filenames_left[i] + " 或 " + filenames_right[i]);
        }
    }

    cv::destroyAllWindows();

    if(object_points.empty() || image_points_left.empty() || image_points_right.empty()) {
        Logger::getInstance().log(Logger::LogLevel::ERROR, "未找到有效的立体标定图像.");
        return -1;
    }

    // 初始化相机参数
    Camera left_camera;
    Camera right_camera;

    // 进行立体相机标定
    Eigen::Matrix4d extrinsic;
    bool calib_success = StereoCamera::calibrateStereoCamera(object_points, image_points_left, image_points_right,
                                                             cv::Size(filenames_left[0].size(), filenames_left[0].size()),
                                                             left_camera, right_camera, extrinsic);

    if(!calib_success) {
        Logger::getInstance().log(Logger::LogLevel::ERROR, "立体相机标定失败.");
        return -1;
    }

    // 创建立体相机实例并设置参数
    StereoCamera stereo_cam(left_camera, right_camera, extrinsic);

    // 保存立体相机参数
    if(left_camera.saveParameters("calibrated_left_camera_params.bin") &&
       right_camera.saveParameters("calibrated_right_camera_params.bin")) {
        Logger::getInstance().log(Logger::LogLevel::INFO, "立体相机参数已保存.");
    } else {
        Logger::getInstance().log(Logger::LogLevel::ERROR, "保存立体相机参数失败.");
        return -1;
    }

    // 保存外参
    std::ofstream ofs("stereo_extrinsic.bin", std::ios::binary);
    if(ofs.is_open()) {
        ofs.write(reinterpret_cast<const char*>(extrinsic.data()), extrinsic.size() * sizeof(double));
        ofs.close();
        Logger::getInstance().log(Logger::LogLevel::INFO, "立体相机外参已保存到 stereo_extrinsic.bin");
    } else {
        Logger::getInstance().log(Logger::LogLevel::ERROR, "无法保存立体相机外参.");
        return -1;
    }

    return 0;
}