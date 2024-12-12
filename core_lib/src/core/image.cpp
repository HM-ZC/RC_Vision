#include "rc_vision/core/image.hpp"
#include "rc_vision/core/logger.hpp"

namespace rc_vision {
    namespace core {

// 构造函数实现
        Image::Image() {}

        Image::Image(const std::string& file_path) {
            load(file_path);
        }

        Image::Image(const cv::Mat& mat) : mat_(mat.clone()) {}

// 加载图像
        bool Image::load(const std::string& file_path) {
            mat_ = cv::imread(file_path, cv::IMREAD_COLOR);
            if(mat_.empty()) {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "Failed to load image: " + file_path);
                return false;
            }
            Logger::getInstance().log(Logger::LogLevel::INFO, "Image loaded: " + file_path);
            return true;
        }

// 保存图像
        bool Image::save(const std::string& file_path) const {
            if(mat_.empty()) {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "No image data to save.");
                return false;
            }
            bool success = cv::imwrite(file_path, mat_);
            if(success) {
                Logger::getInstance().log(Logger::LogLevel::INFO, "Image saved: " + file_path);
            } else {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "Failed to save image: " + file_path);
            }
            return success;
        }

// 获取图像数据
        cv::Mat Image::getMat() const {
            return mat_;
        }

// 设置图像数据
        void Image::setMat(const cv::Mat& mat) {
            mat_ = mat.clone();
        }

// 颜色空间转换
        void Image::convertColor(int code) {
            if(mat_.empty()) {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "No image data to convert color.");
                return;
            }
            cv::Mat converted;
            cv::cvtColor(mat_, converted, code);
            mat_ = converted;
            Logger::getInstance().log(Logger::LogLevel::INFO, "Color space converted.");
        }

// 图像缩放
        void Image::resize(int width, int height, int interpolation) {
            if(mat_.empty()) {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "No image data to resize.");
                return;
            }
            cv::Mat resized;
            cv::resize(mat_, resized, cv::Size(width, height), 0, 0, interpolation);
            mat_ = resized;
            Logger::getInstance().log(Logger::LogLevel::INFO, "Image resized to " + std::to_string(width) + "x" + std::to_string(height));
        }

// 图像裁剪
        Image Image::crop(int x, int y, int width, int height) const {
            if(mat_.empty()) {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "No image data to crop.");
                return Image();
            }
            cv::Rect roi(x, y, width, height);
            // 确保 ROI 在图像范围内
            roi &= cv::Rect(0, 0, mat_.cols, mat_.rows);
            cv::Mat cropped = mat_(roi).clone();
            Logger::getInstance().log(Logger::LogLevel::INFO, "Image cropped.");
            return Image(cropped);
        }

// 图像旋转
        void Image::rotate(double angle, const cv::Point2f& center, double scale) {
            if(mat_.empty()) {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "No image data to rotate.");
                return;
            }
            cv::Point2f rot_center = (center.x >= 0 && center.y >= 0) ? center : cv::Point2f(mat_.cols / 2.0f, mat_.rows / 2.0f);
            cv::Mat rot_mat = cv::getRotationMatrix2D(rot_center, angle, scale);
            cv::Mat rotated;
            cv::warpAffine(mat_, rotated, rot_mat, mat_.size());
            mat_ = rotated;
            Logger::getInstance().log(Logger::LogLevel::INFO, "Image rotated by " + std::to_string(angle) + " degrees.");
        }

// 图像翻转
        void Image::flip(int flip_code) {
            if(mat_.empty()) {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "No image data to flip.");
                return;
            }
            cv::Mat flipped;
            cv::flip(mat_, flipped, flip_code);
            mat_ = flipped;
            Logger::getInstance().log(Logger::LogLevel::INFO, "Image flipped with flip code: " + std::to_string(flip_code));
        }

// 高斯模糊
        void Image::gaussianBlur(int ksize, double sigmaX) {
            if(mat_.empty()) {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "No image data to apply Gaussian blur.");
                return;
            }
            cv::GaussianBlur(mat_, mat_, cv::Size(ksize, ksize), sigmaX);
            Logger::getInstance().log(Logger::LogLevel::INFO, "Gaussian blur applied with kernel size: " + std::to_string(ksize));
        }

// 中值滤波
        void Image::medianBlur(int ksize) {
            if(mat_.empty()) {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "No image data to apply median blur.");
                return;
            }
            cv::medianBlur(mat_, mat_, ksize);
            Logger::getInstance().log(Logger::LogLevel::INFO, "Median blur applied with kernel size: " + std::to_string(ksize));
        }

// 边缘检测
        void Image::edgeDetection(int threshold1, int threshold2) {
            if(mat_.empty()) {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "No image data to perform edge detection.");
                return;
            }
            cv::Mat gray;
            if(mat_.channels() == 3) {
                cv::cvtColor(mat_, gray, cv::COLOR_BGR2GRAY);
            } else {
                gray = mat_;
            }
            cv::Mat edges;
            cv::Canny(gray, edges, threshold1, threshold2);
            mat_ = edges;
            Logger::getInstance().log(Logger::LogLevel::INFO, "Edge detection performed with thresholds: " +
                                                              std::to_string(threshold1) + ", " + std::to_string(threshold2));
        }

// 直方图均衡化
        void Image::equalizeHist() {
            if(mat_.empty()) {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "No image data to perform histogram equalization.");
                return;
            }
            cv::Mat gray;
            if(mat_.channels() == 3) {
                cv::cvtColor(mat_, gray, cv::COLOR_BGR2GRAY);
            } else {
                gray = mat_;
            }
            cv::Mat equalized;
            cv::equalizeHist(gray, equalized);
            mat_ = equalized;
            Logger::getInstance().log(Logger::LogLevel::INFO, "Histogram equalization performed.");
        }

// 对比度和亮度调整
        void Image::adjustContrast(double alpha, double beta) {
            if(mat_.empty()) {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "No image data to adjust contrast and brightness.");
                return;
            }
            cv::Mat adjusted;
            mat_.convertTo(adjusted, -1, alpha, beta);
            mat_ = adjusted;
            Logger::getInstance().log(Logger::LogLevel::INFO, "Contrast and brightness adjusted with alpha: " +
                                                              std::to_string(alpha) + ", beta: " + std::to_string(beta));
        }

// 特征点检测
        std::vector<cv::KeyPoint> Image::detectKeypoints(const std::string& detector_type) const {
            std::vector<cv::KeyPoint> keypoints;
            if(mat_.empty()) {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "No image data to detect keypoints.");
                return keypoints;
            }

            cv::Mat gray;
            if(mat_.channels() == 3) {
                cv::cvtColor(mat_, gray, cv::COLOR_BGR2GRAY);
            } else {
                gray = mat_;
            }

            if(detector_type == "SIFT") {
                cv::Ptr<cv::SIFT> detector = cv::SIFT::create();
                detector->detect(gray, keypoints);
            }
            else if(detector_type == "SURF") {
                cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create();
                detector->detect(gray, keypoints);
            }
            else if(detector_type == "ORB") {
                cv::Ptr<cv::ORB> detector = cv::ORB::create();
                detector->detect(gray, keypoints);
            }
            else {
                Logger::getInstance().log(Logger::LogLevel::WARN, "Unsupported detector type: " + detector_type + ". Using ORB by default.");
                cv::Ptr<cv::ORB> detector = cv::ORB::create();
                detector->detect(gray, keypoints);
            }

            Logger::getInstance().log(Logger::LogLevel::INFO, "Detected " + std::to_string(keypoints.size()) + " keypoints using " + detector_type + " detector.");
            return keypoints;
        }

// 特征描述子计算
        cv::Mat Image::computeDescriptors(const std::string& descriptor_type) const {
            cv::Mat descriptors;
            if(mat_.empty()) {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "No image data to compute descriptors.");
                return descriptors;
            }

            cv::Mat gray;
            if(mat_.channels() == 3) {
                cv::cvtColor(mat_, gray, cv::COLOR_BGR2GRAY);
            } else {
                gray = mat_;
            }

            std::vector<cv::KeyPoint> keypoints;
            if(descriptor_type == "SIFT" || descriptor_type == "SURF" || descriptor_type == "ORB") {
                keypoints = detectKeypoints(descriptor_type);
            } else {
                Logger::getInstance().log(Logger::LogLevel::WARN, "Unsupported descriptor type: " + descriptor_type + ". Using ORB by default.");
                keypoints = detectKeypoints("ORB");
            }

            if(descriptor_type == "SIFT") {
                cv::Ptr<cv::SIFT> extractor = cv::SIFT::create();
                extractor->compute(gray, keypoints, descriptors);
            }
            else if(descriptor_type == "SURF") {
                cv::Ptr<cv::xfeatures2d::SURF> extractor = cv::xfeatures2d::SURF::create();
                extractor->compute(gray, keypoints, descriptors);
            }
            else if(descriptor_type == "ORB") {
                cv::Ptr<cv::ORB> extractor = cv::ORB::create();
                extractor->compute(gray, keypoints, descriptors);
            }

            Logger::getInstance().log(Logger::LogLevel::INFO, "Computed " + std::to_string(descriptors.rows) + " descriptors using " + descriptor_type + " extractor.");
            return descriptors;
        }

// 计算图像均值
        double Image::mean() const {
            if(mat_.empty()) {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "No image data to compute mean.");
                return 0.0;
            }
            cv::Scalar mean_scalar = cv::mean(mat_);
            return (mat_.channels() == 1) ? mean_scalar[0] : (mean_scalar[0] + mean_scalar[1] + mean_scalar[2]) / 3.0;
        }

// 计算图像标准差
        double Image::stddev() const {
            if(mat_.empty()) {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "No image data to compute standard deviation.");
                return 0.0;
            }
            cv::Scalar mean_scalar, stddev_scalar;
            cv::meanStdDev(mat_, mean_scalar, stddev_scalar);
            return (mat_.channels() == 1) ? stddev_scalar[0] : (stddev_scalar[0] + stddev_scalar[1] + stddev_scalar[2]) / 3.0;
        }

// 计算直方图
        cv::Mat Image::histogram(int histSize, float range[]) const {
            if(mat_.empty()) {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "No image data to compute histogram.");
                return cv::Mat();
            }
            cv::Mat hist;
            int channels = mat_.channels();
            cv::calcHist(&mat_, 1, nullptr, cv::Mat(), hist, 1, &histSize, &range, true, false);
            return hist;
        }

// 图像加法
        Image Image::operator+(const Image& other) const {
            Image result;
            if(mat_.empty() || other.mat_.empty()) {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "One of the images is empty for addition.");
                return result;
            }
            cv::add(mat_, other.mat_, result.mat_);
            return result;
        }

// 图像减法
        Image Image::operator-(const Image& other) const {
            Image result;
            if(mat_.empty() || other.mat_.empty()) {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "One of the images is empty for subtraction.");
                return result;
            }
            cv::subtract(mat_, other.mat_, result.mat_);
            return result;
        }

// 图像乘法
        Image Image::operator*(const Image& other) const {
            Image result;
            if(mat_.empty() || other.mat_.empty()) {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "One of the images is empty for multiplication.");
                return result;
            }
            cv::multiply(mat_, other.mat_, result.mat_);
            return result;
        }

// 图像除法
        Image Image::operator/(const Image& other) const {
            Image result;
            if(mat_.empty() || other.mat_.empty()) {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "One of the images is empty for division.");
                return result;
            }
            cv::divide(mat_, other.mat_, result.mat_);
            return result;
        }

// 设置 ROI
        void Image::setROI(const cv::Rect& roi) {
            if(mat_.empty()) {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "No image data to set ROI.");
                return;
            }
            roi_ = roi & cv::Rect(0, 0, mat_.cols, mat_.rows);
            Logger::getInstance().log(Logger::LogLevel::INFO, "ROI set to x: " + std::to_string(roi_.x) +
                                                              ", y: " + std::to_string(roi_.y) +
                                                              ", width: " + std::to_string(roi_.width) +
                                                              ", height: " + std::to_string(roi_.height));
        }

// 获取 ROI
        cv::Rect Image::getROI() const {
            return roi_;
        }

// 获取 ROI 图像
        Image Image::getROIImage() const {
            if(mat_.empty()) {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "No image data to get ROI image.");
                return Image();
            }
            cv::Rect valid_roi = roi_ & cv::Rect(0, 0, mat_.cols, mat_.rows);
            if(valid_roi.area() <= 0) {
                Logger::getInstance().log(Logger::LogLevel::WARN, "Invalid ROI area. Returning empty image.");
                return Image();
            }
            cv::Mat roi_image = mat_(valid_roi).clone();
            Logger::getInstance().log(Logger::LogLevel::INFO, "ROI image extracted.");
            return Image(roi_image);
        }

// 模板匹配
        cv::Mat Image::matchTemplate(const Image& template_img, int method) const {
            cv::Mat result;
            if(mat_.empty() || template_img.getMat().empty()) {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "Source or template image is empty for template matching.");
                return result;
            }

            // 确保源图像和模板图像是灰度图
            cv::Mat src_gray, tmpl_gray;
            if(mat_.channels() == 3) {
                cv::cvtColor(mat_, src_gray, cv::COLOR_BGR2GRAY);
            } else {
                src_gray = mat_;
            }

            if(template_img.getMat().channels() == 3) {
                cv::cvtColor(template_img.getMat(), tmpl_gray, cv::COLOR_BGR2GRAY);
            } else {
                tmpl_gray = template_img.getMat();
            }

            // 执行模板匹配
            cv::matchTemplate(src_gray, tmpl_gray, result, method);
            Logger::getInstance().log(Logger::LogLevel::INFO, "Template matching performed with method: " + std::to_string(method));
            return result;
        }

        cv::Point Image::findBestMatch(const Image& template_img, int method, double* min_max_vals) const {
            cv::Mat result = matchTemplate(template_img, method);
            if(result.empty()) {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "Template matching result is empty.");
                return cv::Point(-1, -1);
            }

            double min_val, max_val;
            cv::Point min_loc, max_loc;
            cv::minMaxLoc(result, &min_val, &max_val, &min_loc, &max_loc, cv::Mat());

            cv::Point best_match;
            if(method == cv::TM_SQDIFF || method == cv::TM_SQDIFF_NORMED) {
                best_match = min_loc;
                if(min_max_vals) {
                    min_max_vals[0] = min_val;
                    min_max_vals[1] = max_val;
                }
            }
            else {
                best_match = max_loc;
                if(min_max_vals) {
                    min_max_vals[0] = min_val;
                    min_max_vals[1] = max_val;
                }
            }

            Logger::getInstance().log(Logger::LogLevel::INFO, "Best match found at (" + std::to_string(best_match.x) + ", " + std::to_string(best_match.y) + ").");
            return best_match;
        }

        void Image::drawMatch(const Image& template_img, const cv::Point& top_left, const cv::Scalar& color, int thickness) {
            if(mat_.empty() || template_img.getMat().empty()) {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "Source or template image is empty for drawing match.");
                return;
            }

            cv::Mat result_image = mat_.clone();
            cv::rectangle(result_image, top_left,
                          cv::Point(top_left.x + template_img.getMat().cols, top_left.y + template_img.getMat().rows),
                          color, thickness);

            // 更新内部图像
            mat_ = result_image;
            Logger::getInstance().log(Logger::LogLevel::INFO, "Match rectangle drawn on the image.");
        }

    } // namespace core
} // namespace rc_vision