#include <gtest/gtest.h>
#include "rc_vision/core/edge_detector.hpp"
#include "rc_vision/core/image.hpp"
#include "rc_vision/core/logger.hpp"
#include <opencv2/opencv.hpp>

using namespace rc_vision::core;

// 边缘类型辅助函数
bool hasEdgeType(const std::vector<EdgeType>& types, EdgeType target) {
    for (const auto& type : types) {
        if (type == target) {
            return true;
        }
    }
    return false;
}

TEST(EdgeDetectorTest, CannyEdgeDetectionGeneral) {
// 创建 EdgeDetector 实例
EdgeDetector detector;
detector.setMethod(EdgeDetectionMethod::CANNY);
detector.setParameters({50.0, 150.0, 3});
detector.setDetectEdgeTypes({EdgeType::GENERAL});

// 创建测试图像（简单的黑白图像）
cv::Mat test_image = cv::Mat::zeros(100, 100, CV_8UC1);
cv::rectangle(test_image, cv::Point(20, 20), cv::Point(80, 80), cv::Scalar(255), -1);

// 执行边缘检测
EdgeDetectionResult result = detector.detectEdges(test_image);

// 检查结果
ASSERT_FALSE(result.edges.empty());
// 检查是否检测到边缘
int edge_pixels = cv::countNonZero(result.edges);
EXPECT_GT(edge_pixels, 0);
// 检查边缘类型数量是否匹配
EXPECT_EQ(result.edge_points.size(), result.edge_types.size());
}

TEST(EdgeDetectorTest, CannyEdgeDetectionLine) {
EdgeDetector detector;
detector.setMethod(EdgeDetectionMethod::CANNY);
detector.setParameters({50.0, 150.0, 3});
detector.setDetectEdgeTypes({EdgeType::LINE});

cv::Mat test_image = cv::Mat::zeros(100, 100, CV_8UC1);
cv::line(test_image, cv::Point(10, 50), cv::Point(90, 50), cv::Scalar(255), 2);

EdgeDetectionResult result = detector.detectEdges(test_image);

ASSERT_FALSE(result.edges.empty());
int edge_pixels = cv::countNonZero(result.edges);
EXPECT_GT(edge_pixels, 0);

// 检查是否有 LINE 类型边缘
bool has_line = false;
for (const auto& type : result.edge_types) {
if (type == EdgeType::LINE) {
has_line = true;
break;
}
}
EXPECT_TRUE(has_line);
}

TEST(EdgeDetectorTest, CannyEdgeDetectionCorner) {
EdgeDetector detector;
detector.setMethod(EdgeDetectionMethod::CANNY);
detector.setParameters({50.0, 150.0, 3});
detector.setDetectEdgeTypes({EdgeType::CORNER});

cv::Mat test_image = cv::Mat::zeros(100, 100, CV_8UC1);
cv::rectangle(test_image, cv::Point(20, 20), cv::Point(80, 80), cv::Scalar(255), 2); // 绘制矩形边缘

EdgeDetectionResult result = detector.detectEdges(test_image);

ASSERT_FALSE(result.edges.empty());
int edge_pixels = cv::countNonZero(result.edges);
EXPECT_GT(edge_pixels, 0);

// 检查是否有 CORNER 类型边缘
bool has_corner = false;
for (const auto& type : result.edge_types) {
if (type == EdgeType::CORNER) {
has_corner = true;
break;
}
}
EXPECT_TRUE(has_corner);
}

TEST(EdgeDetectorTest, CannyEdgeDetectionCircle) {
EdgeDetector detector;
detector.setMethod(EdgeDetectionMethod::CANNY);
// 设置 HoughCircles 参数
detector.setParameters({1.0, 100.0, 100.0, 30.0, 0, 0});
detector.setDetectEdgeTypes({EdgeType::CIRCLE});

cv::Mat test_image = cv::Mat::zeros(200, 200, CV_8UC1);
cv::circle(test_image, cv::Point(100, 100), 50, cv::Scalar(255), 2);

EdgeDetectionResult result = detector.detectEdges(test_image);

ASSERT_FALSE(result.edges.empty());
int edge_pixels = cv::countNonZero(result.edges);
EXPECT_GT(edge_pixels, 0);

// 检查是否有 CIRCLE 类型边缘
bool has_circle = false;
for (const auto& type : result.edge_types) {
if (type == EdgeType::CIRCLE) {
has_circle = true;
break;
}
}
EXPECT_TRUE(has_circle);
}

TEST(EdgeDetectorTest, CannyEdgeDetectionPolygon) {
EdgeDetector detector;
detector.setMethod(EdgeDetectionMethod::CANNY);
detector.setParameters({50.0, 150.0, 3});
detector.setDetectEdgeTypes({EdgeType::POLYGON});

cv::Mat test_image = cv::Mat::zeros(200, 200, CV_8UC1);
std::vector<cv::Point> polygon = {cv::Point(50, 50), cv::Point(150, 50), cv::Point(100, 150)};
cv::polylines(test_image, polygon, true, cv::Scalar(255), 2);

EdgeDetectionResult result = detector.detectEdges(test_image);

ASSERT_FALSE(result.edges.empty());
int edge_pixels = cv::countNonZero(result.edges);
EXPECT_GT(edge_pixels, 0);

// 检查是否有 POLYGON 类型边缘
bool has_polygon = false;
for (const auto& type : result.edge_types) {
if (type == EdgeType::POLYGON) {
has_polygon = true;
break;
}
}
EXPECT_TRUE(has_polygon);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}