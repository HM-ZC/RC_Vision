/*
 * 提供边缘检测功能
 */
#include "rc_vision/core/edge_detector.hpp"

namespace rc_vision {
    namespace core {

        EdgeDetector::EdgeDetector()
                : current_method_(EdgeDetectionMethod::CANNY), parameters_{100.0, 200.0, 3},
                  detect_edge_types_{EdgeType::GENERAL} { // 默认检测 GENERAL 类型边缘
            Logger::getInstance().log(Logger::LogLevel::INFO, "EdgeDetector 初始化完成，使用 CANNY 算法");
        }

        void EdgeDetector::setMethod(EdgeDetectionMethod method) {
            current_method_ = method;
            Logger::getInstance().log(Logger::LogLevel::INFO, "边缘检测方法设置为 " + std::to_string(static_cast<int>(method)));
        }

        void EdgeDetector::setParameters(const std::vector<double>& params) {
            parameters_ = params;
            Logger::getInstance().log(Logger::LogLevel::INFO, "边缘检测参数已更新");
        }

        bool EdgeDetector::loadConfig(const std::string& config_file) {
            try {
                YAML::Node config = YAML::LoadFile(config_file);
                if (config["method"]) {
                    std::string method_str = config["method"].as<std::string>();
                    if (method_str == "CANNY") {
                        setMethod(EdgeDetectionMethod::CANNY);
                    } else if (method_str == "SOBEL") {
                        setMethod(EdgeDetectionMethod::SOBEL);
                    } else if (method_str == "LAPLACIAN") {
                        setMethod(EdgeDetectionMethod::LAPLACIAN);
                    } else {
                        Logger::getInstance().log(Logger::LogLevel::WARN, "未知的边缘检测方法，使用默认方法 CANNY");
                    }
                }

                if (config["parameters"]) {
                    std::vector<double> params;
                    for (const auto& param : config["parameters"]) {
                        if (param.IsMap()) {
                            for (const auto& kv : param) {
                                params.push_back(kv.second.as<double>());
                            }
                        } else if (param.IsScalar()) {
                            params.push_back(param.as<double>());
                        }
                    }
                    if (!params.empty()) {
                        setParameters(params);
                    }
                }

                if (config["detect_edge_types"]) {
                    std::vector<EdgeType> types;
                    for (const auto& type_str : config["detect_edge_types"]) {
                        std::string type = type_str.as<std::string>();
                        if (type == "GENERAL") {
                            types.push_back(EdgeType::GENERAL);
                        } else if (type == "LINE") {
                            types.push_back(EdgeType::LINE);
                        } else if (type == "CORNER") {
                            types.push_back(EdgeType::CORNER);
                        } else if (type == "CIRCLE") {  // 新增
                            types.push_back(EdgeType::CIRCLE);
                        } else if (type == "POLYGON") { // 新增
                            types.push_back(EdgeType::POLYGON);
                        } else {
                            Logger::getInstance().log(Logger::LogLevel::WARN, "未知的边缘类型: " + type);
                        }
                    }
                    if (!types.empty()) {
                        setDetectEdgeTypes(types);
                    }
                }

                Logger::getInstance().log(Logger::LogLevel::INFO, "边缘检测配置加载完成");
                return true;
            } catch (const YAML::Exception& e) {
                Logger::getInstance().log(Logger::LogLevel::ERROR, std::string("加载边缘检测配置失败: ") + e.what());
                return false;
            }
        }

        void EdgeDetector::setDetectEdgeTypes(const std::vector<EdgeType>& types) {
            detect_edge_types_ = types;
            std::string types_str;
            for (const auto& type : types) {
                switch(type) {
                    case EdgeType::GENERAL: types_str += "GENERAL, "; break;
                    case EdgeType::LINE: types_str += "LINE, "; break;
                    case EdgeType::CORNER: types_str += "CORNER, "; break;
                    case EdgeType::CIRCLE: types_str += "CIRCLE, "; break; // 新增
                    case EdgeType::POLYGON: types_str += "POLYGON, "; break; // 新增
                        // 添加更多类型
                    default: types_str += "UNKNOWN, "; break;
                }
            }
            if (!types_str.empty()) {
                types_str = types_str.substr(0, types_str.size() - 2); // 移除最后的 ", "
            }
            Logger::getInstance().log(Logger::LogLevel::INFO, "设置检测的边缘类型为: " + types_str);
        }

        EdgeDetectionResult EdgeDetector::detectEdges(const cv::Mat& image) {
            if (image.empty()) {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "输入图像为空，无法执行边缘检测");
                return EdgeDetectionResult{};
            }

            EdgeDetectionResult result;

            switch (current_method_) {
                case EdgeDetectionMethod::CANNY:
                    result = detectCanny(image);
                    break;
                case EdgeDetectionMethod::SOBEL:
                    result = detectSobel(image);
                    break;
                case EdgeDetectionMethod::LAPLACIAN:
                    result = detectLaplacian(image);
                    break;
                default:
                    Logger::getInstance().log(Logger::LogLevel::ERROR, "未知的边缘检测方法");
                    return EdgeDetectionResult{};
            }

            // 提取边缘点
            cv::findNonZero(result.edges, result.edge_points);

            // 初始化边缘类型为 GENERAL
            result.edge_types.assign(result.edge_points.size(), EdgeType::GENERAL);

            // 分类边缘类型
            classifyEdgeTypes(result.edges, result);

            return result;
        }

        EdgeDetectionMethod EdgeDetector::getMethod() const {
            return current_method_;
        }

        EdgeDetectionResult EdgeDetector::detectCanny(const cv::Mat& image) {
            cv::Mat gray, edges;
            if (image.channels() == 3) {
                cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
            } else {
                gray = image;
            }

            double threshold1 = parameters_.size() > 0 ? parameters_[0] : 100.0;
            double threshold2 = parameters_.size() > 1 ? parameters_[1] : 200.0;
            int apertureSize = parameters_.size() > 2 ? static_cast<int>(parameters_[2]) : 3;

            cv::Canny(gray, edges, threshold1, threshold2, apertureSize);
            Logger::getInstance().log(Logger::LogLevel::DEBUG, "执行 Canny 边缘检测");
            return EdgeDetectionResult{edges, {}, {}};
        }

        EdgeDetectionResult EdgeDetector::detectSobel(const cv::Mat& image) {
            cv::Mat gray, grad_x, grad_y, abs_grad_x, abs_grad_y, edges;
            if (image.channels() == 3) {
                cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
            } else {
                gray = image;
            }

            int ddepth = parameters_.size() > 0 ? static_cast<int>(parameters_[0]) : CV_16S;
            int kernel_size = parameters_.size() > 1 ? static_cast<int>(parameters_[1]) : 3;

            cv::Sobel(gray, grad_x, ddepth, 1, 0, kernel_size);
            cv::Sobel(gray, grad_y, ddepth, 0, 1, kernel_size);

            cv::convertScaleAbs(grad_x, abs_grad_x);
            cv::convertScaleAbs(grad_y, abs_grad_y);

            cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, edges);
            Logger::getInstance().log(Logger::LogLevel::DEBUG, "执行 Sobel 边缘检测");
            return EdgeDetectionResult{edges, {}, {}};
        }

        EdgeDetectionResult EdgeDetector::detectLaplacian(const cv::Mat& image) {
            cv::Mat gray, edges;
            if (image.channels() == 3) {
                cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
            } else {
                gray = image;
            }

            int ddepth = parameters_.size() > 0 ? static_cast<int>(parameters_[0]) : CV_16S;
            int kernel_size = parameters_.size() > 1 ? static_cast<int>(parameters_[1]) : 3;

            cv::Laplacian(gray, edges, ddepth, kernel_size);
            cv::convertScaleAbs(edges, edges);
            Logger::getInstance().log(Logger::LogLevel::DEBUG, "执行 Laplacian 边缘检测");
            return EdgeDetectionResult{edges, {}, {}};
        }

        void EdgeDetector::classifyEdgeTypes(const cv::Mat& edges, EdgeDetectionResult& result) {
            // 检测直线边缘
            if (std::find(detect_edge_types_.begin(), detect_edge_types_.end(), EdgeType::LINE) != detect_edge_types_.end()) {
                detectLines(edges, result);
            }

            // 检测角点边缘
            if (std::find(detect_edge_types_.begin(), detect_edge_types_.end(), EdgeType::CORNER) != detect_edge_types_.end()) {
                detectCorners(edges, result);
            }

            // 检测圆形边缘
            if (std::find(detect_edge_types_.begin(), detect_edge_types_.end(), EdgeType::CIRCLE) != detect_edge_types_.end()) {
                detectCircles(edges, result);
            }

            // 检测多边形边缘
            if (std::find(detect_edge_types_.begin(), detect_edge_types_.end(), EdgeType::POLYGON) != detect_edge_types_.end()) {
                detectPolygons(edges, result);
            }

            // 可以添加更多边缘类型的检测
        }

        void EdgeDetector::detectLines(const cv::Mat& edges, EdgeDetectionResult& result) {
            // 使用 HoughLinesP 检测直线
            std::vector<cv::Vec4i> lines;
            cv::HoughLinesP(edges, lines, 1, CV_PI/180, 50, 50, 10);
            Logger::getInstance().log(Logger::LogLevel::DEBUG, "检测到 " + std::to_string(lines.size()) + " 条直线");

            for (const auto& line : lines) {
                cv::Point p1(line[0], line[1]);
                cv::Point p2(line[2], line[3]);
                // 遍历边缘点，标记在直线附近的点为 LINE 类型
                for (size_t i = 0; i < result.edge_points.size(); ++i) {
                    const cv::Point& pt = result.edge_points[i];
                    double distance = cv::pointPolygonTest(std::vector<cv::Point>{p1, p2}, pt, true);
                    if (std::abs(distance) < 2.0) { // 距离阈值，可调整
                        result.edge_types[i] = EdgeType::LINE;
                    }
                }
            }
        }

        void EdgeDetector::detectCorners(const cv::Mat& edges, EdgeDetectionResult& result) {
            // 使用 Harris 角点检测
            cv::Mat gray;
            if (edges.channels() == 3) {
                cv::cvtColor(edges, gray, cv::COLOR_BGR2GRAY);
            } else {
                gray = edges;
            }

            cv::Mat dst, dst_norm, dst_norm_scaled;
            dst = cv::Mat::zeros(gray.size(), CV_32FC1);

            cv::cornerHarris(gray, dst, 2, 3, 0.04);
            cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
            cv::convertScaleAbs(dst_norm, dst_norm_scaled);

            // 标记角点
            for(int j = 0; j < dst_norm.rows ; j++) {
                for(int i = 0; i < dst_norm.cols; i++) {
                    if((int) dst_norm.at<float>(j,i) > 200) { // 阈值，可调整
                        cv::Point corner(i, j);
                        // 遍历边缘点，标记在角点附近的点为 CORNER 类型
                        for (size_t k = 0; k < result.edge_points.size(); ++k) {
                            const cv::Point& pt = result.edge_points[k];
                            double distance = cv::norm(corner - pt);
                            if (distance < 3.0) { // 距离阈值，可调整
                                result.edge_types[k] = EdgeType::CORNER;
                            }
                        }
                    }
                }
            }
        }

        void EdgeDetector::detectCircles(const cv::Mat& edges, EdgeDetectionResult& result) {
            // 使用 HoughCircles 检测圆形
            cv::Mat gray;
            if (edges.channels() == 3) {
                cv::cvtColor(edges, gray, cv::COLOR_BGR2GRAY);
            } else {
                gray = edges;
            }

            std::vector<cv::Vec3f> circles;
            double dp = parameters_.size() > 0 ? parameters_[0] : 1.0;
            double minDist = parameters_.size() > 1 ? parameters_[1] : gray.rows / 8;
            double param1 = parameters_.size() > 2 ? parameters_[2] : 100.0;
            double param2 = parameters_.size() > 3 ? parameters_[3] : 30.0;
            int minRadius = parameters_.size() > 4 ? static_cast<int>(parameters_[4]) : 0;
            int maxRadius = parameters_.size() > 5 ? static_cast<int>(parameters_[5]) : 0;

            cv::HoughCircles(gray, circles, cv::HOUGH_GRADIENT, dp, minDist, param1, param2, minRadius, maxRadius);
            Logger::getInstance().log(Logger::LogLevel::DEBUG, "检测到 " + std::to_string(circles.size()) + " 个圆形边缘");

            for (const auto& circle : circles) {
                cv::Point center(cvRound(circle[0]), cvRound(circle[1]));
                int radius = cvRound(circle[2]);

                // 遍历边缘点，标记在圆形边缘附近的点为 CIRCLE 类型
                for (size_t i = 0; i < result.edge_points.size(); ++i) {
                    const cv::Point& pt = result.edge_points[i];
                    double distance = cv::norm(center - pt) - radius;
                    if (std::abs(distance) < 2.0) { // 距离阈值，可调整
                        result.edge_types[i] = EdgeType::CIRCLE;
                    }
                }
            }
        }

        void EdgeDetector::detectPolygons(const cv::Mat& edges, EdgeDetectionResult& result) {
            // 查找轮廓
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            Logger::getInstance().log(Logger::LogLevel::DEBUG, "检测到 " + std::to_string(contours.size()) + " 个轮廓");

            for (const auto& contour : contours) {
                // 多边形逼近
                std::vector<cv::Point> approx;
                double epsilon = 0.02 * cv::arcLength(contour, true);
                cv::approxPolyDP(contour, approx, epsilon, true);

                // 判断多边形类型
                if (approx.size() >= 3 && approx.size() <= 10) { // 简单过滤
                    // 遍历边缘点，标记在多边形边缘附近的点为 POLYGON 类型
                    for (size_t i = 0; i < result.edge_points.size(); ++i) {
                        const cv::Point& pt = result.edge_points[i];
                        double distance = cv::pointPolygonTest(approx, pt, true);
                        if (std::abs(distance) < 2.0) { // 距离阈值，可调整
                            result.edge_types[i] = EdgeType::POLYGON;
                        }
                    }
                }
            }
        }

    } // namespace core
} // namespace rc_vision