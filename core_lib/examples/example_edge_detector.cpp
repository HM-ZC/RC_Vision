#include "rc_vision/core/edge_detector.hpp"
#include "rc_vision/core/image.hpp"
#include "rc_vision/core/logger.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace rc_vision::core;

int main(int argc, char** argv) {
    // 初始化日志
    Logger::getInstance().setLogLevel(Logger::LogLevel::DEBUG);
    Logger::getInstance().setLogFile("rc_vision.log");

    LOG_INFO("开始边缘检测示例程序");

    // 检查命令行参数
    if (argc < 2) {
        LOG_ERROR("请提供图像路径作为参数");
        return -1;
    }

    std::string image_path = argv[1];

    // 加载图像
    Image img(image_path);
    if(img.getMat().empty()) {
        LOG_ERROR("图像加载失败: " + image_path);
        return -1;
    }
    LOG_INFO("图像加载成功: " + image_path);

    // 创建 EdgeDetector 实例
    EdgeDetector detector;

    // 加载配置文件（可选）
    std::string config_file = "edge_detector_config.yaml";
    detector.loadConfig(config_file);

    // 执行边缘检测
    EdgeDetectionResult result = detector.detectEdges(img.getMat());

    if(result.edges.empty()) {
        LOG_ERROR("边缘检测失败或未检测到边缘");
        return -1;
    }

    LOG_INFO("边缘检测完成");

    // 可视化边缘类型
    cv::Mat color_edges;
    cv::cvtColor(result.edges, color_edges, cv::COLOR_GRAY2BGR);
    for (size_t i = 0; i < result.edge_points.size(); ++i) {
        const cv::Point& pt = result.edge_points[i];
        const EdgeType& type = result.edge_types[i];
        cv::Scalar color;
        switch(type) {
            case EdgeType::GENERAL:
                color = cv::Scalar(255, 255, 255); // 白色
                break;
            case EdgeType::LINE:
                color = cv::Scalar(0, 0, 255); // 红色
                break;
            case EdgeType::CORNER:
                color = cv::Scalar(0, 255, 0); // 绿色
                break;
            case EdgeType::CIRCLE:
                color = cv::Scalar(255, 0, 0); // 蓝色
                break;
            case EdgeType::POLYGON:
                color = cv::Scalar(0, 255, 255); // 黄色
                break;
            default:
                color = cv::Scalar(255, 255, 255); // 白色
                break;
        }
        cv::circle(color_edges, pt, 1, color, -1);
    }

    // 显示结果
    cv::imshow("Original Image", img.getMat());
    cv::imshow("Edge Detection with Types", color_edges);
    cv::waitKey(0);

    LOG_INFO("示例程序结束");
    return 0;
}