#include <gtest/gtest.h>
#include "rc_vision/core/image.hpp"
#include "rc_vision/core/logger.hpp"

#include <opencv2/opencv.hpp>

using namespace rc_vision::core;

// 定义测试类
class ImageTest : public ::testing::Test {
protected:
    virtual void SetUp() {
        // 初始化日志为 ERROR 级别，减少测试输出
        Logger::getInstance().setLogLevel(Logger::LogLevel::ERROR);

        // 加载测试图像
        source_img = Image("assets/test_source.jpg");
        template_img = Image("assets/test_template.jpg");

        // 创建一个空图像
        empty_img = Image();
    }

    Image source_img;
    Image template_img;
    Image empty_img;
};

// 测试图像加载
TEST_F(ImageTest, LoadImage) {
EXPECT_FALSE(source_img.getMat().empty()) << "源图像加载失败";
EXPECT_FALSE(template_img.getMat().empty()) << "模板图像加载失败";
EXPECT_TRUE(empty_img.getMat().empty()) << "空图像应为空";
}

// 测试图像保存
TEST_F(ImageTest, SaveImage) {
bool save_success = source_img.save("output/test_save.jpg");
EXPECT_TRUE(save_success) << "图像保存失败";

// 验证图像是否成功保存
cv::Mat saved_img = cv::imread("output/test_save.jpg", cv::IMREAD_COLOR);
EXPECT_FALSE(saved_img.empty()) << "保存的图像为空";
}

// 测试颜色空间转换
TEST_F(ImageTest, ConvertColor) {
Image img_copy = source_img;
img_copy.convertColor(cv::COLOR_BGR2GRAY);
EXPECT_EQ(img_copy.getMat().channels(), 1) << "颜色空间转换失败，图像通道数不正确";
}

// 测试图像缩放
TEST_F(ImageTest, ResizeImage) {
Image img_copy = source_img;
img_copy.resize(100, 100);
EXPECT_EQ(img_copy.getMat().cols, 100);
EXPECT_EQ(img_copy.getMat().rows, 100);
}

// 测试图像裁剪
TEST_F(ImageTest, CropImage) {
Image cropped = source_img.crop(10, 10, 50, 50);
EXPECT_EQ(cropped.getMat().cols, 50);
EXPECT_EQ(cropped.getMat().rows, 50);
}

// 测试图像旋转
TEST_F(ImageTest, RotateImage) {
Image img_copy = source_img;
img_copy.rotate(45.0);
EXPECT_FALSE(img_copy.getMat().empty()) << "图像旋转后为空";
}

// 测试图像翻转
TEST_F(ImageTest, FlipImage) {
Image img_copy = source_img;
img_copy.flip(1); // 水平翻转
EXPECT_FALSE(img_copy.getMat().empty()) << "图像翻转后为空";
}

// 测试高斯模糊
TEST_F(ImageTest, GaussianBlur) {
Image img_copy = source_img;
img_copy.gaussianBlur(5, 1.5);
EXPECT_FALSE(img_copy.getMat().empty()) << "高斯模糊后图像为空";
}

// 测试边缘检测
TEST_F(ImageTest, EdgeDetection) {
Image img_copy = source_img;
img_copy.edgeDetection(100, 200);
EXPECT_FALSE(img_copy.getMat().empty()) << "边缘检测后图像为空";
}

// 测试直方图均衡化
TEST_F(ImageTest, EqualizeHist) {
Image img_copy = source_img;
img_copy.equalizeHist();
EXPECT_FALSE(img_copy.getMat().empty()) << "直方图均衡化后图像为空";
}

// 测试对比度与亮度调整
TEST_F(ImageTest, AdjustContrast) {
Image img_copy = source_img;
img_copy.adjustContrast(1.5, 50);
EXPECT_FALSE(img_copy.getMat().empty()) << "对比度与亮度调整后图像为空";
}

// 测试特征检测与描述
TEST_F(ImageTest, FeatureDetectionAndDescription) {
std::vector<cv::KeyPoint> keypoints = source_img.detectKeypoints("ORB");
cv::Mat descriptors = source_img.computeDescriptors("ORB");
EXPECT_GT(keypoints.size(), 0) << "特征点检测失败，没有检测到关键点";
EXPECT_FALSE(descriptors.empty()) << "描述子计算失败，描述子为空";
}

// 测试模板匹配
TEST_F(ImageTest, TemplateMatching) {
// 执行模板匹配
cv::Mat match_result = source_img.matchTemplate(template_img, cv::TM_CCOEFF_NORMED);
EXPECT_FALSE(match_result.empty()) << "模板匹配结果为空";

// 查找最佳匹配位置
double min_val, max_val;
cv::Point best_match = source_img.findBestMatch(template_img, cv::TM_CCOEFF_NORMED, &min_val);
EXPECT_GE(max_val, 0.0) << "匹配值异常";

// 绘制匹配结果
Image img_copy = source_img;
img_copy.drawMatch(template_img, best_match, cv::Scalar(0, 0, 255), 2);
EXPECT_FALSE(img_copy.getMat().empty()) << "绘制匹配结果后图像为空";
}

// 测试图像算术操作
TEST_F(ImageTest, ArithmeticOperations) {
// 加法
Image img_add = source_img + template_img;
EXPECT_FALSE(img_add.getMat().empty()) << "图像加法结果为空";

// 减法
Image img_sub = source_img - template_img;
EXPECT_FALSE(img_sub.getMat().empty()) << "图像减法结果为空";

// 乘法
Image img_mul = source_img * template_img;
EXPECT_FALSE(img_mul.getMat().empty()) << "图像乘法结果为空";

// 除法
Image img_div = source_img / template_img;
EXPECT_FALSE(img_div.getMat().empty()) << "图像除法结果为空";
}

// 测试 ROI 管理
TEST_F(ImageTest, ROIMangement) {
cv::Rect roi(10, 10, 50, 50);
source_img.setROI(roi);
cv::Rect current_roi = source_img.getROI();
EXPECT_EQ(current_roi.x, 10);
EXPECT_EQ(current_roi.y, 10);
EXPECT_EQ(current_roi.width, 50);
EXPECT_EQ(current_roi.height, 50);

Image roi_img = source_img.getROIImage();
EXPECT_EQ(roi_img.getMat().cols, 50);
EXPECT_EQ(roi_img.getMat().rows, 50);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}