#include "rc_vision/core/image.hpp"
#include "rc_vision/core/logger.hpp"

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace rc_vision::core;

int main() {
    // 初始化日志
    Logger::getInstance().setLogLevel(Logger::LogLevel::DEBUG);
    Logger::getInstance().log(Logger::LogLevel::INFO, "示例程序开始运行");

    // 1. 加载源图像
    Image source_img("assets/source_image.jpg");
    if(source_img.getMat().empty()) {
        Logger::getInstance().log(Logger::LogLevel::ERROR, "源图像加载失败");
        return -1;
    }
    Logger::getInstance().log(Logger::LogLevel::INFO, "源图像加载成功");

    // 2. 转换颜色空间（BGR 转灰度）
    source_img.convertColor(cv::COLOR_BGR2GRAY);
    Logger::getInstance().log(Logger::LogLevel::INFO, "颜色空间转换为灰度图");

    // 3. 调整对比度和亮度
    source_img.adjustContrast(1.5, 50); // new_image = 1.5 * image + 50
    Logger::getInstance().log(Logger::LogLevel::INFO, "调整对比度和亮度");

    // 4. 应用高斯模糊
    source_img.gaussianBlur(5, 1.5);
    Logger::getInstance().log(Logger::LogLevel::INFO, "应用高斯模糊");

    // 5. 应用边缘检测
    source_img.edgeDetection(100, 200);
    Logger::getInstance().log(Logger::LogLevel::INFO, "应用边缘检测");

    // 6. 计算图像均值和标准差
    double mean_val = source_img.mean();
    double stddev_val = source_img.stddev();
    Logger::getInstance().log(Logger::LogLevel::INFO, "图像均值: " + std::to_string(mean_val));
    Logger::getInstance().log(Logger::LogLevel::INFO, "图像标准差: " + std::to_string(stddev_val));

    // 7. 检测特征点并计算描述子
    std::vector<cv::KeyPoint> keypoints = source_img.detectKeypoints("ORB");
    cv::Mat descriptors = source_img.computeDescriptors("ORB");
    Logger::getInstance().log(Logger::LogLevel::INFO, "检测并计算特征点");

    // 8. 保存处理后的图像
    source_img.save("output/processed_image.jpg");
    Logger::getInstance().log(Logger::LogLevel::INFO, "处理后的图像已保存到 output/processed_image.jpg");

    // 9. 加载模板图像
    Image template_img("assets/template_image.jpg");
    if(template_img.getMat().empty()) {
        Logger::getInstance().log(Logger::LogLevel::ERROR, "模板图像加载失败");
        return -1;
    }
    Logger::getInstance().log(Logger::LogLevel::INFO, "模板图像加载成功");

    // 10. 执行模板匹配
    cv::Mat match_result = source_img.matchTemplate(template_img, cv::TM_CCOEFF_NORMED);
    if(match_result.empty()) {
        Logger::getInstance().log(Logger::LogLevel::ERROR, "模板匹配失败");
        return -1;
    }

    // 11. 查找最佳匹配位置
    double min_val, max_val;
    cv::Point best_match = source_img.findBestMatch(template_img, cv::TM_CCOEFF_NORMED, &min_val);

    // 12. 输出匹配结果
    Logger::getInstance().log(Logger::LogLevel::INFO, "匹配方法: TM_CCOEFF_NORMED");
    Logger::getInstance().log(Logger::LogLevel::INFO, "最佳匹配位置: (" + std::to_string(best_match.x) + ", " + std::to_string(best_match.y) + ")");
    Logger::getInstance().log(Logger::LogLevel::INFO, "匹配值: " + std::to_string(max_val));

    // 13. 在源图像上绘制匹配结果
    source_img.drawMatch(template_img, best_match, cv::Scalar(0, 0, 255), 2);
    source_img.save("output/matched_image.jpg");
    Logger::getInstance().log(Logger::LogLevel::INFO, "匹配结果已保存到 output/matched_image.jpg");

    // 14. 图像算术操作示例
    Image img_add = source_img + template_img;
    img_add.save("output/image_add.jpg");
    Logger::getInstance().log(Logger::LogLevel::INFO, "图像加法结果已保存到 output/image_add.jpg");

    Image img_sub = source_img - template_img;
    img_sub.save("output/image_sub.jpg");
    Logger::getInstance().log(Logger::LogLevel::INFO, "图像减法结果已保存到 output/image_sub.jpg");

    // 15. ROI 管理示例
    cv::Rect roi(50, 50, 200, 200);
    source_img.setROI(roi);
    Image roi_img = source_img.getROIImage();
    if(!roi_img.getMat().empty()) {
        roi_img.save("output/roi_image.jpg");
        Logger::getInstance().log(Logger::LogLevel::INFO, "ROI 图像已保存到 output/roi_image.jpg");
    }

    Logger::getInstance().log(Logger::LogLevel::INFO, "示例程序运行结束");
    return 0;
}