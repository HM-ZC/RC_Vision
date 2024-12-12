#ifndef RC_VISION_CORE_IMAGE_HPP
#define RC_VISION_CORE_IMAGE_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>
#include <Eigen/Dense>

namespace rc_vision {
    namespace core {

        /**
         * @brief 图像处理类，封装了多种常用的图像操作和分析功能。
         *
         * 该类提供了图像的加载、保存、转换、滤波、特征检测与描述、统计分析等多种功能，
         * 以及图像的算术操作和模板匹配等高级操作。
         */
        class Image {
        public:
            /**
             * @brief 默认构造函数，创建一个空的图像对象。
             */
            Image();

            /**
             * @brief 从文件路径加载图像。
             *
             * @param file_path 要加载的图像文件路径。
             * @return true 如果图像成功加载。
             * @return false 如果加载失败。
             */
            Image(const std::string& file_path);

            /**
             * @brief 使用给定的 cv::Mat 对象初始化图像。
             *
             * @param mat 要使用的 OpenCV 图像矩阵。
             */
            Image(const cv::Mat& mat);

            /**
             * @brief 从文件路径加载图像。
             *
             * @param file_path 要加载的图像文件路径。
             * @return true 如果图像成功加载。
             * @return false 如果加载失败。
             */
            bool load(const std::string& file_path);

            /**
             * @brief 将图像保存到指定的文件路径。
             *
             * @param file_path 要保存的图像文件路径。
             * @return true 如果图像成功保存。
             * @return false 如果保存失败。
             */
            bool save(const std::string& file_path) const;

            /**
             * @brief 获取当前图像的 cv::Mat 对象。
             *
             * @return cv::Mat 当前图像的数据矩阵。
             */
            cv::Mat getMat() const;

            /**
             * @brief 设置当前图像的数据矩阵。
             *
             * @param mat 要设置的 OpenCV 图像矩阵。
             */
            void setMat(const cv::Mat& mat);

            /**
             * @brief 将图像转换为指定的颜色空间。
             *
             * @param code 颜色转换代码，例如 cv::COLOR_BGR2GRAY。
             */
            void convertColor(int code);

            /**
             * @brief 调整图像的尺寸。
             *
             * @param width 目标宽度。
             * @param height 目标高度。
             * @param interpolation 插值方法，默认为 cv::INTER_LINEAR。
             */
            void resize(int width, int height, int interpolation = cv::INTER_LINEAR);

            /**
             * @brief 裁剪图像的一部分。
             *
             * @param x 裁剪区域左上角的 x 坐标。
             * @param y 裁剪区域左上角的 y 坐标。
             * @param width 裁剪区域的宽度。
             * @param height 裁剪区域的高度。
             * @return Image 裁剪后的新图像对象。
             */
            Image crop(int x, int y, int width, int height) const;

            /**
             * @brief 旋转图像。
             *
             * @param angle 旋转角度，以度为单位。
             * @param center 旋转中心点，默认为 (-1, -1) 表示图像中心。
             * @param scale 缩放因子，默认为 1.0。
             */
            void rotate(double angle, const cv::Point2f& center = cv::Point2f(-1,-1), double scale = 1.0);

            /**
             * @brief 翻转图像。
             *
             * @param flip_code 翻转代码：
             *                  0 - 垂直翻转，
             *                  1 - 水平翻转，
             *                  -1 - 同时水平和垂直翻转。
             */
            void flip(int flip_code);

            /**
             * @brief 对图像应用高斯模糊滤波。
             *
             * @param ksize 核大小，默认为 5。
             * @param sigmaX 高斯核在 X 方向的标准差，默认为 1.0。
             */
            void gaussianBlur(int ksize = 5, double sigmaX = 1.0);

            /**
             * @brief 对图像应用中值滤波。
             *
             * @param ksize 核大小，默认为 5。
             */
            void medianBlur(int ksize = 5);

            /**
             * @brief 对图像应用边缘检测。
             *
             * 使用 Canny 边缘检测算法。
             *
             * @param threshold1 第一个阈值，默认为 100。
             * @param threshold2 第二个阈值，默认为 200。
             */
            void edgeDetection(int threshold1 = 100, int threshold2 = 200);

            /**
             * @brief 对图像应用直方图均衡化。
             *
             * 仅适用于单通道图像。
             */
            void equalizeHist();

            /**
             * @brief 调整图像的对比度和亮度。
             *
             * 新图像 = alpha * 原图像 + beta
             *
             * @param alpha 对比度控制（1.0-3.0）。
             * @param beta 亮度控制（0-100）。
             */
            void adjustContrast(double alpha, double beta); // new_image = alpha*image + beta

            /**
             * @brief 检测图像中的关键点。
             *
             * 支持多种关键点检测器，例如 ORB、SIFT、SURF 等。
             *
             * @param detector_type 关键点检测器类型，默认为 "ORB"。
             * @return std::vector<cv::KeyPoint> 检测到的关键点列表。
             */
            std::vector<cv::KeyPoint> detectKeypoints(const std::string& detector_type = "ORB") const;

            /**
             * @brief 计算图像中的描述子。
             *
             * 支持多种描述子计算器，例如 ORB、SIFT、SURF 等。
             *
             * @param descriptor_type 描述子类型，默认为 "ORB"。
             * @return cv::Mat 计算得到的描述子矩阵。
             */
            cv::Mat computeDescriptors(const std::string& descriptor_type = "ORB") const;

            /**
             * @brief 计算图像的平均值。
             *
             * @return double 图像像素的平均值。
             */
            double mean() const;

            /**
             * @brief 计算图像的标准差。
             *
             * @return double 图像像素的标准差。
             */
            double stddev() const;

            /**
             * @brief 计算图像的直方图。
             *
             * @param histSize 直方图大小，默认为 256。
             * @param range 直方图范围，默认为 {0, 256}。
             * @return cv::Mat 计算得到的直方图矩阵。
             */
            cv::Mat histogram(int histSize = 256, float range[] = {0, 256}) const;

            /**
             * @brief 图像的加法运算。
             *
             * @param other 另一个图像对象。
             * @return Image 结果图像。
             */
            Image operator+(const Image& other) const;

            /**
             * @brief 图像的减法运算。
             *
             * @param other 另一个图像对象。
             * @return Image 结果图像。
             */
            Image operator-(const Image& other) const;

            /**
             * @brief 图像的乘法运算。
             *
             * @param other 另一个图像对象。
             * @return Image 结果图像。
             */
            Image operator*(const Image& other) const;

            /**
             * @brief 图像的除法运算。
             *
             * @param other 另一个图像对象。
             * @return Image 结果图像。
             */
            Image operator/(const Image& other) const;

            /**
             * @brief 设置图像的感兴趣区域（ROI）。
             *
             * @param roi 要设置的 ROI 区域。
             */
            void setROI(const cv::Rect& roi);

            /**
             * @brief 获取当前图像的感兴趣区域（ROI）。
             *
             * @return cv::Rect 当前的 ROI 区域。
             */
            cv::Rect getROI() const;

            /**
             * @brief 获取当前 ROI 区域内的图像。
             *
             * @return Image ROI 区域内的新图像对象。
             */
            Image getROIImage() const;

            /**
             * @brief 执行模板匹配。
             *
             * 使用 OpenCV 的 matchTemplate 函数进行模板匹配。
             *
             * @param template_img 模板图像对象。
             * @param method 匹配方法，如 cv::TM_CCOEFF, cv::TM_SQDIFF 等，默认为 cv::TM_CCOEFF。
             * @return cv::Mat 匹配结果矩阵。
             */
            cv::Mat matchTemplate(const Image& template_img, int method = cv::TM_CCOEFF) const;

            /**
             * @brief 查找模板在源图像中的最佳匹配位置。
             *
             * 使用 OpenCV 的 minMaxLoc 函数查找最佳匹配位置。
             *
             * @param template_img 模板图像对象。
             * @param method 匹配方法，如 cv::TM_CCOEFF, cv::TM_SQDIFF 等，默认为 cv::TM_CCOEFF。
             * @param min_max_vals 返回匹配结果的最小值和最大值，可为 nullptr。
             * @return cv::Point 最佳匹配位置的左上角坐标。
             */
            cv::Point findBestMatch(const Image& template_img, int method = cv::TM_CCOEFF, double* min_max_vals = nullptr) const;

            /**
             * @brief 在源图像上绘制模板匹配结果。
             *
             * 绘制一个矩形框标记出模板匹配的位置。
             *
             * @param template_img 模板图像对象。
             * @param top_left 匹配区域的左上角坐标。
             * @param color 绘制矩形的颜色，默认为红色 (0, 0, 255)。
             * @param thickness 矩形边框的厚度，默认为 2。
             */
            void drawMatch(const Image& template_img, const cv::Point& top_left, const cv::Scalar& color = cv::Scalar(0, 0, 255), int thickness = 2);

        private:
            cv::Mat mat_; /**< 存储图像数据的 OpenCV 矩阵。 */
            cv::Rect roi_; /**< 当前图像的感兴趣区域（ROI）。 */

            /**
             * @brief 应用当前设置的 ROI 到图像数据。
             *
             * 如果 ROI 被设置，则裁剪图像到 ROI 区域。
             */
            void applyROI();
        };

    } // namespace core
} // namespace rc_vision

#endif // RC_VISION_CORE_IMAGE_HPP
