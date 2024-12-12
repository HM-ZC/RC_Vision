/*
 * 定义相机模型，包括内参、外参、畸变系数等，用于处理相机相关的计算。
 */
#include "rc_vision/core/camera.hpp"
#include "rc_vision/core/logger.hpp"

namespace rc_vision {
    namespace core {

// Camera Class Implementations

        Camera::Camera()
                : intrinsic_(Eigen::Matrix3d::Identity()),
                  distortion_(Eigen::Vector<double, 5>::Zero()),
                  extrinsic_(Eigen::Matrix4d::Identity()) {
            updateOpenCVCameraParams();
        }

        Camera::Camera(const Eigen::Matrix3d& intrinsic, const Eigen::Vector<double, 5>& distortion)
                : intrinsic_(intrinsic),
                  distortion_(distortion),
                  extrinsic_(Eigen::Matrix4d::Identity()) {
            updateOpenCVCameraParams();
        }

        const Eigen::Matrix3d& Camera::getIntrinsic() const {
            return intrinsic_;
        }

        const Eigen::Vector<double, 5>& Camera::getDistortion() const {
            return distortion_;
        }

        const Eigen::Matrix4d& Camera::getExtrinsic() const {
            return extrinsic_;
        }

        void Camera::setIntrinsic(const Eigen::Matrix3d& intrinsic) {
            intrinsic_ = intrinsic;
            updateOpenCVCameraParams();
        }

        void Camera::setDistortion(const Eigen::Vector<double, 5>& distortion) {
            distortion_ = distortion;
            updateOpenCVCameraParams();
        }

        void Camera::setExtrinsic(const Eigen::Matrix4d& extrinsic) {
            extrinsic_ = extrinsic;
        }

        void Camera::updateOpenCVCameraParams() {
            cv_intrinsic_ = cv::Mat::zeros(3, 3, CV_64F);
            for(int i=0;i<3;i++) {
                for(int j=0;j<3;j++) {
                    cv_intrinsic_.at<double>(i,j) = intrinsic_(i,j);
                }
            }

            cv_distortion_ = cv::Mat::zeros(1, 5, CV_64F);
            for(int i=0;i<5;i++) {
                cv_distortion_.at<double>(0,i) = distortion_(i);
            }
        }

        Eigen::Vector2d Camera::project(const Eigen::Vector3d& point_3d) const {
            if (point_3d[2] == 0) {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "Z coordinate is zero in project method.");
                return Eigen::Vector2d::Zero();
            }

            // Transform to camera coordinate system
            Eigen::Vector4d point_homo(point_3d[0], point_3d[1], point_3d[2], 1.0);
            Eigen::Vector4d point_cam_homo = extrinsic_ * point_homo;
            Eigen::Vector3d point_cam = point_cam_homo.head<3>() / point_cam_homo[3];

            // Project to 2D image plane
            Eigen::Vector3d pixel = intrinsic_ * (point_cam / point_cam[2]);

            return Eigen::Vector2d(pixel[0], pixel[1]);
        }

        std::vector<Eigen::Vector2d> Camera::project(const std::vector<Eigen::Vector3d>& points_3d) const {
            std::vector<Eigen::Vector2d> projected_points;
            projected_points.reserve(points_3d.size());
            for(const auto& point : points_3d) {
                projected_points.emplace_back(project(point));
            }
            return projected_points;
        }

        Eigen::Vector2d Camera::undistort(const Eigen::Vector2d& distorted_point) const {
            // Convert Eigen::Vector2d to OpenCV format
            std::vector<cv::Point2d> distorted = { cv::Point2d(distorted_point[0], distorted_point[1]) };
            std::vector<cv::Point2d> undistorted;

            // Use OpenCV's undistortPoints
            cv::undistortPoints(distorted, undistorted, cv_intrinsic_, cv_distortion_, cv::noArray(), cv_intrinsic_);

            if (undistorted.empty()) {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "Undistortion failed.");
                return Eigen::Vector2d::Zero();
            }

            return Eigen::Vector2d(undistorted[0].x, undistorted[0].y);
        }

        std::vector<Eigen::Vector2d> Camera::undistort(const std::vector<Eigen::Vector2d>& distorted_points) const {
            std::vector<Eigen::Vector2d> undistorted_points;
            undistorted_points.reserve(distorted_points.size());

            std::vector<cv::Point2d> distorted_cv;
            distorted_cv.reserve(distorted_points.size());
            for(const auto& p : distorted_points) {
                distorted_cv.emplace_back(cv::Point2d(p[0], p[1]));
            }

            std::vector<cv::Point2d> undistorted_cv;
            cv::undistortPoints(distorted_cv, undistorted_cv, cv_intrinsic_, cv_distortion_, cv::noArray(), cv_intrinsic_);

            for(const auto& p : undistorted_cv) {
                undistorted_points.emplace_back(Eigen::Vector2d(p.x, p.y));
            }

            return undistorted_points;
        }

        cv::Mat Camera::undistortImage(const cv::Mat& distorted_image) const {
            if(distorted_image.empty()) {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "Input image for undistortion is empty.");
                return cv::Mat();
            }

            cv::Mat undistorted;
            cv::undistort(distorted_image, undistorted, cv_intrinsic_, cv_distortion_);
            return undistorted;
        }

        cv::Mat Camera::correctDistortion(const cv::Mat& distorted_image) const {
            return undistortImage(distorted_image);
        }

        bool Camera::saveParameters(const std::string& file_path) const {
            std::ofstream ofs(file_path, std::ios::binary);
            if(!ofs.is_open()) {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "无法打开文件保存相机参数: " + file_path);
                return false;
            }

            // Save intrinsic
            ofs.write(reinterpret_cast<const char*>(intrinsic_.data()), intrinsic_.size() * sizeof(double));

            // Save distortion coefficients
            ofs.write(reinterpret_cast<const char*>(distortion_.data()), distortion_.size() * sizeof(double));

            // Save extrinsic
            ofs.write(reinterpret_cast<const char*>(extrinsic_.data()), extrinsic_.size() * sizeof(double));

            ofs.close();
            Logger::getInstance().log(Logger::LogLevel::INFO, "相机参数已保存到 " + file_path);
            return true;
        }

        bool Camera::loadParameters(const std::string& file_path) {
            std::ifstream ifs(file_path, std::ios::binary);
            if(!ifs.is_open()) {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "无法打开文件加载相机参数: " + file_path);
                return false;
            }

            // Load intrinsic
            ifs.read(reinterpret_cast<char*>(intrinsic_.data()), intrinsic_.size() * sizeof(double));

            // Load distortion coefficients
            ifs.read(reinterpret_cast<char*>(distortion_.data()), distortion_.size() * sizeof(double));

            // Load extrinsic
            ifs.read(reinterpret_cast<char*>(extrinsic_.data()), extrinsic_.size() * sizeof(double));

            if(ifs.fail()) {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "相机参数加载失败: " + file_path);
                return false;
            }

            updateOpenCVCameraParams();
            ifs.close();
            Logger::getInstance().log(Logger::LogLevel::INFO, "相机参数已从 " + file_path + " 加载");
            return true;
        }

        Eigen::Vector3d Camera::worldToCamera(const Eigen::Vector3d& point_world) const {
            Eigen::Vector4d point_homo(point_world[0], point_world[1], point_world[2], 1.0);
            Eigen::Vector4d point_cam_homo = extrinsic_ * point_homo;
            return point_cam_homo.head<3>() / point_cam_homo[3];
        }

        Eigen::Vector3d Camera::cameraToWorld(const Eigen::Vector3d& point_camera) const {
            Eigen::Matrix4d extrinsic_inv = extrinsic_.inverse();
            Eigen::Vector4d point_cam_homo(point_camera[0], point_camera[1], point_camera[2], 1.0);
            Eigen::Vector4d point_world_homo = extrinsic_inv * point_cam_homo;
            return point_world_homo.head<3>() / point_world_homo[3];
        }

        bool Camera::calibrateCamera(const std::vector<std::vector<cv::Point3f>>& object_points,
                                     const std::vector<std::vector<cv::Point2f>>& image_points,
                                     const cv::Size& image_size,
                                     Eigen::Matrix3d& out_intrinsic,
                                     Eigen::Vector<double, 5>& out_distortion) {
            if(object_points.empty() || image_points.empty() || object_points.size() != image_points.size()) {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "Invalid input for camera calibration.");
                return false;
            }

            cv::Mat camera_matrix = cv::Mat::eye(3, 3, CV_64F);
            cv::Mat dist_coeffs = cv::Mat::zeros(1, 5, CV_64F);
            std::vector<cv::Mat> rvecs, tvecs;

            int flags = 0;
            // flags |= cv::CALIB_FIX_K4;
            // flags |= cv::CALIB_FIX_K5;
            // ... set flags as needed

            Logger::getInstance().log(Logger::LogLevel::INFO, "Starting camera calibration...");

            double rms = cv::calibrateCamera(object_points, image_points, image_size,
                                             camera_matrix, dist_coeffs, rvecs, tvecs, flags);

            Logger::getInstance().log(Logger::LogLevel::INFO, "Camera calibration completed with RMS error: " + std::to_string(rms));

            // Convert to Eigen
            for(int i=0;i<3;i++) {
                for(int j=0;j<3;j++) {
                    out_intrinsic(i,j) = camera_matrix.at<double>(i,j);
                }
            }

            for(int i=0;i<5;i++) {
                out_distortion(i) = dist_coeffs.at<double>(0,i);
            }

            return true;
        }

// StereoCamera Class Implementations

        StereoCamera::StereoCamera()
                : left_camera_(),
                  right_camera_(),
                  extrinsic_(Eigen::Matrix4d::Identity()) {}

        StereoCamera::StereoCamera(const Camera& left_camera, const Camera& right_camera, const Eigen::Matrix4d& extrinsic)
                : left_camera_(left_camera),
                  right_camera_(right_camera),
                  extrinsic_(extrinsic) {}

        const Camera& StereoCamera::getLeftCamera() const {
            return left_camera_;
        }

        const Camera& StereoCamera::getRightCamera() const {
            return right_camera_;
        }

        const Eigen::Matrix4d& StereoCamera::getExtrinsic() const {
            return extrinsic_;
        }

        void StereoCamera::setLeftCamera(const Camera& left_camera) {
            left_camera_ = left_camera;
        }

        void StereoCamera::setRightCamera(const Camera& right_camera) {
            right_camera_ = right_camera;
        }

        void StereoCamera::setExtrinsic(const Eigen::Matrix4d& extrinsic) {
            extrinsic_ = extrinsic;
        }

        bool StereoCamera::rectifyImages(const cv::Mat& left_image, const cv::Mat& right_image,
                                         cv::Mat& rectified_left, cv::Mat& rectified_right) const {
            if(left_image.empty() || right_image.empty()) {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "Empty images provided for rectification.");
                return false;
            }

            // Get intrinsic and distortion parameters from both cameras
            cv::Mat M1 = left_camera_.getIntrinsic().cast<double>();
            cv::Mat D1 = left_camera_.getDistortion().cast<double>();
            cv::Mat M2 = right_camera_.getIntrinsic().cast<double>();
            cv::Mat D2 = right_camera_.getDistortion().cast<double>();

            // Compute rotation and translation between cameras
            // Assuming extrinsic_ is the transformation from left to right
            Eigen::Matrix3d R_eigen = extrinsic_.block<3,3>(0,0);
            Eigen::Vector3d T_eigen = extrinsic_.block<3,1>(0,3);

            cv::Mat R_cv, T_cv;
            cv::eigen2cv(R_eigen, R_cv);
            cv::eigen2cv(T_eigen, T_cv);

            // Compute rectification transforms
            cv::Mat R1, R2, P1, P2, Q;
            cv::stereoRectify(M1, D1, M2, D2, left_image.size(),
                              R_cv, T_cv, R1, R2, P1, P2, Q);

            // Compute rectification maps
            cv::Mat map1x, map1y, map2x, map2y;
            cv::initUndistortRectifyMap(M1, D1, R1, P1, left_image.size(), CV_32FC1, map1x, map1y);
            cv::initUndistortRectifyMap(M2, D2, R2, P2, right_image.size(), CV_32FC1, map2x, map2y);

            // Apply rectification
            cv::remap(left_image, rectified_left, map1x, map1y, cv::INTER_LINEAR);
            cv::remap(right_image, rectified_right, map2x, map2y, cv::INTER_LINEAR);

            Logger::getInstance().log(Logger::LogLevel::INFO, "Stereo images rectified successfully.");
            return true;
        }

        bool StereoCamera::computeDisparity(const cv::Mat& rectified_left, const cv::Mat& rectified_right,
                                            cv::Mat& disparity_map) const {
            if(rectified_left.empty() || rectified_right.empty()) {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "Empty rectified images provided for disparity computation.");
                return false;
            }

            // Create StereoBM or StereoSGBM object
            int numDisparities = 16*5; // must be divisible by 16
            int blockSize = 21;

            cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create(0, numDisparities, blockSize);
            stereo->compute(rectified_left, rectified_right, disparity_map);

            Logger::getInstance().log(Logger::LogLevel::INFO, "Disparity map computed successfully.");
            return true;
        }

        bool StereoCamera::calibrateStereoCamera(const std::vector<std::vector<cv::Point3f>>& object_points,
                                                 const std::vector<std::vector<cv::Point2f>>& image_points_left,
                                                 const std::vector<std::vector<cv::Point2f>>& image_points_right,
                                                 const cv::Size& image_size,
                                                 Camera& left_camera,
                                                 Camera& right_camera,
                                                 Eigen::Matrix4d& out_extrinsic) {
            if(object_points.empty() || image_points_left.empty() || image_points_right.empty() ||
               image_points_left.size() != image_points_right.size() ||
               object_points.size() != image_points_left.size()) {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "Invalid input for stereo camera calibration.");
                return false;
            }

            // Initial guesses for camera matrices
            cv::Mat M1 = left_camera.getIntrinsic().cast<double>();
            cv::Mat D1 = left_camera.getDistortion().cast<double>();
            cv::Mat M2 = right_camera.getIntrinsic().cast<double>();
            cv::Mat D2 = right_camera.getDistortion().cast<double>();

            // Output rotation, translation
            cv::Mat R, T, E, F;

            Logger::getInstance().log(Logger::LogLevel::INFO, "Starting stereo camera calibration...");

            double rms = cv::stereoCalibrate(object_points, image_points_left, image_points_right,
                                             M1, D1, M2, D2, image_size,
                                             R, T, E, F,
                                             cv::CALIB_FIX_INTRINSIC,
                                             cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-5));

            Logger::getInstance().log(Logger::LogLevel::INFO, "Stereo camera calibration completed with RMS error: " + std::to_string(rms));

            // Convert R and T to extrinsic transformation matrix
            Eigen::Matrix3d R_eigen;
            cv::cv2eigen(R, R_eigen);
            Eigen::Vector3d T_eigen;
            cv::cv2eigen(T, T_eigen);

            out_extrinsic = Eigen::Matrix4d::Identity();
            out_extrinsic.block<3,3>(0,0) = R_eigen;
            out_extrinsic.block<3,1>(0,3) = T_eigen;

            // Update right camera's extrinsic
            right_camera.setExtrinsic(out_extrinsic);

            return true;
        }

// Static Camera Calibration Method

        bool Camera::calibrateCamera(const std::vector<std::vector<cv::Point3f>>& object_points,
                                     const std::vector<std::vector<cv::Point2f>>& image_points,
                                     const cv::Size& image_size,
                                     Eigen::Matrix3d& out_intrinsic,
                                     Eigen::Vector<double, 5>& out_distortion) {
            if(object_points.empty() || image_points.empty() || object_points.size() != image_points.size()) {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "Invalid input for camera calibration.");
                return false;
            }

            cv::Mat camera_matrix = cv::Mat::eye(3, 3, CV_64F);
            cv::Mat dist_coeffs = cv::Mat::zeros(1, 5, CV_64F);
            std::vector<cv::Mat> rvecs, tvecs;

            int flags = 0;
            // flags |= cv::CALIB_FIX_K4;
            // flags |= cv::CALIB_FIX_K5;
            // ... set flags as needed

            Logger::getInstance().log(Logger::LogLevel::INFO, "Starting camera calibration...");

            double rms = cv::calibrateCamera(object_points, image_points, image_size,
                                             camera_matrix, dist_coeffs, rvecs, tvecs, flags);

            Logger::getInstance().log(Logger::LogLevel::INFO, "Camera calibration completed with RMS error: " + std::to_string(rms));

            // Convert to Eigen
            for(int i=0;i<3;i++) {
                for(int j=0;j<3;j++) {
                    out_intrinsic(i,j) = camera_matrix.at<double>(i,j);
                }
            }

            for(int i=0;i<5;i++) {
                out_distortion(i) = dist_coeffs.at<double>(0,i);
            }

            return true;
        }

// StereoCamera Class

        StereoCamera::StereoCamera()
                : left_camera_(),
                  right_camera_(),
                  extrinsic_(Eigen::Matrix4d::Identity()) {}

        StereoCamera::StereoCamera(const Camera& left_camera, const Camera& right_camera, const Eigen::Matrix4d& extrinsic)
                : left_camera_(left_camera),
                  right_camera_(right_camera),
                  extrinsic_(extrinsic) {}

        const Camera& StereoCamera::getLeftCamera() const {
            return left_camera_;
        }

        const Camera& StereoCamera::getRightCamera() const {
            return right_camera_;
        }

        const Eigen::Matrix4d& StereoCamera::getExtrinsic() const {
            return extrinsic_;
        }

        void StereoCamera::setLeftCamera(const Camera& left_camera) {
            left_camera_ = left_camera;
        }

        void StereoCamera::setRightCamera(const Camera& right_camera) {
            right_camera_ = right_camera;
        }

        void StereoCamera::setExtrinsic(const Eigen::Matrix4d& extrinsic) {
            extrinsic_ = extrinsic;
        }

        bool StereoCamera::rectifyImages(const cv::Mat& left_image, const cv::Mat& right_image,
                                         cv::Mat& rectified_left, cv::Mat& rectified_right) const {
            if(left_image.empty() || right_image.empty()) {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "Empty images provided for rectification.");
                return false;
            }

            // Get intrinsic and distortion parameters from both cameras
            cv::Mat M1 = left_camera_.getIntrinsic().cast<double>();
            cv::Mat D1 = left_camera_.getDistortion().cast<double>();
            cv::Mat M2 = right_camera_.getIntrinsic().cast<double>();
            cv::Mat D2 = right_camera_.getDistortion().cast<double>();

            // Compute rotation and translation between cameras
            // Assuming extrinsic_ is the transformation from left to right
            Eigen::Matrix3d R_eigen = extrinsic_.block<3,3>(0,0);
            Eigen::Vector3d T_eigen = extrinsic_.block<3,1>(0,3);

            cv::Mat R_cv, T_cv;
            cv::eigen2cv(R_eigen, R_cv);
            cv::eigen2cv(T_eigen, T_cv);

            // Compute rectification transforms
            cv::Mat R1, R2, P1, P2, Q;
            cv::stereoRectify(M1, D1, M2, D2, left_image.size(),
                              R_cv, T_cv, R1, R2, P1, P2, Q);

            // Compute rectification maps
            cv::Mat map1x, map1y, map2x, map2y;
            cv::initUndistortRectifyMap(M1, D1, R1, P1, left_image.size(), CV_32FC1, map1x, map1y);
            cv::initUndistortRectifyMap(M2, D2, R2, P2, right_image.size(), CV_32FC1, map2x, map2y);

            // Apply rectification
            cv::remap(left_image, rectified_left, map1x, map1y, cv::INTER_LINEAR);
            cv::remap(right_image, rectified_right, map2x, map2y, cv::INTER_LINEAR);

            Logger::getInstance().log(Logger::LogLevel::INFO, "Stereo images rectified successfully.");
            return true;
        }

        bool StereoCamera::computeDisparity(const cv::Mat& rectified_left, const cv::Mat& rectified_right,
                                            cv::Mat& disparity_map) const {
            if(rectified_left.empty() || rectified_right.empty()) {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "Empty rectified images provided for disparity computation.");
                return false;
            }

            // Create StereoBM or StereoSGBM object
            int numDisparities = 16*5; // must be divisible by 16
            int blockSize = 21;

            cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create(0, numDisparities, blockSize);
            stereo->compute(rectified_left, rectified_right, disparity_map);

            Logger::getInstance().log(Logger::LogLevel::INFO, "Disparity map computed successfully.");
            return true;
        }

        bool StereoCamera::calibrateStereoCamera(const std::vector<std::vector<cv::Point3f>>& object_points,
                                                 const std::vector<std::vector<cv::Point2f>>& image_points_left,
                                                 const std::vector<std::vector<cv::Point2f>>& image_points_right,
                                                 const cv::Size& image_size,
                                                 Camera& left_camera,
                                                 Camera& right_camera,
                                                 Eigen::Matrix4d& out_extrinsic) {
            if(object_points.empty() || image_points_left.empty() || image_points_right.empty() ||
               image_points_left.size() != image_points_right.size() ||
               object_points.size() != image_points_left.size()) {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "Invalid input for stereo camera calibration.");
                return false;
            }

            // Initial guesses for camera matrices
            cv::Mat M1 = left_camera.getIntrinsic().cast<double>();
            cv::Mat D1 = left_camera.getDistortion().cast<double>();
            cv::Mat M2 = right_camera.getIntrinsic().cast<double>();
            cv::Mat D2 = right_camera.getDistortion().cast<double>();

            // Output rotation, translation
            cv::Mat R, T, E, F;

            Logger::getInstance().log(Logger::LogLevel::INFO, "Starting stereo camera calibration...");

            double rms = cv::stereoCalibrate(object_points, image_points_left, image_points_right,
                                             M1, D1, M2, D2, image_size,
                                             R, T, E, F,
                                             cv::CALIB_FIX_INTRINSIC,
                                             cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-5));

            Logger::getInstance().log(Logger::LogLevel::INFO, "Stereo camera calibration completed with RMS error: " + std::to_string(rms));

            // Convert R and T to extrinsic transformation matrix
            Eigen::Matrix3d R_eigen;
            cv::cv2eigen(R, R_eigen);
            Eigen::Vector3d T_eigen;
            cv::cv2eigen(T, T_eigen);

            out_extrinsic = Eigen::Matrix4d::Identity();
            out_extrinsic.block<3,3>(0,0) = R_eigen;
            out_extrinsic.block<3,1>(0,3) = T_eigen;

            // Update right camera's extrinsic
            right_camera.setExtrinsic(out_extrinsic);

            return true;
        }

// Static Camera Calibration Method

        bool Camera::calibrateCamera(const std::vector<std::vector<cv::Point3f>>& object_points,
                                     const std::vector<std::vector<cv::Point2f>>& image_points,
                                     const cv::Size& image_size,
                                     Eigen::Matrix3d& out_intrinsic,
                                     Eigen::Vector<double, 5>& out_distortion) {
            if(object_points.empty() || image_points.empty() || object_points.size() != image_points.size()) {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "Invalid input for camera calibration.");
                return false;
            }

            cv::Mat camera_matrix = cv::Mat::eye(3, 3, CV_64F);
            cv::Mat dist_coeffs = cv::Mat::zeros(1, 5, CV_64F);
            std::vector<cv::Mat> rvecs, tvecs;

            int flags = 0;
            // flags |= cv::CALIB_FIX_K4;
            // flags |= cv::CALIB_FIX_K5;
            // ... set flags as needed

            Logger::getInstance().log(Logger::LogLevel::INFO, "Starting camera calibration...");

            double rms = cv::calibrateCamera(object_points, image_points, image_size,
                                             camera_matrix, dist_coeffs, rvecs, tvecs, flags);

            Logger::getInstance().log(Logger::LogLevel::INFO, "Camera calibration completed with RMS error: " + std::to_string(rms));

            // Convert to Eigen
            for(int i=0;i<3;i++) {
                for(int j=0;j<3;j++) {
                    out_intrinsic(i,j) = camera_matrix.at<double>(i,j);
                }
            }

            for(int i=0;i<5;i++) {
                out_distortion(i) = dist_coeffs.at<double>(0,i);
            }

            return true;
        }

// StereoCamera Class

        StereoCamera::StereoCamera()
                : left_camera_(),
                  right_camera_(),
                  extrinsic_(Eigen::Matrix4d::Identity()) {}

        StereoCamera::StereoCamera(const Camera& left_camera, const Camera& right_camera, const Eigen::Matrix4d& extrinsic)
                : left_camera_(left_camera),
                  right_camera_(right_camera),
                  extrinsic_(extrinsic) {}

        const Camera& StereoCamera::getLeftCamera() const {
            return left_camera_;
        }

        const Camera& StereoCamera::getRightCamera() const {
            return right_camera_;
        }

        const Eigen::Matrix4d& StereoCamera::getExtrinsic() const {
            return extrinsic_;
        }

        void StereoCamera::setLeftCamera(const Camera& left_camera) {
            left_camera_ = left_camera;
        }

        void StereoCamera::setRightCamera(const Camera& right_camera) {
            right_camera_ = right_camera;
        }

        void StereoCamera::setExtrinsic(const Eigen::Matrix4d& extrinsic) {
            extrinsic_ = extrinsic;
        }

        bool StereoCamera::rectifyImages(const cv::Mat& left_image, const cv::Mat& right_image,
                                         cv::Mat& rectified_left, cv::Mat& rectified_right) const {
            if(left_image.empty() || right_image.empty()) {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "Empty images provided for rectification.");
                return false;
            }

            // Get intrinsic and distortion parameters from both cameras
            cv::Mat M1 = left_camera_.getIntrinsic().cast<double>();
            cv::Mat D1 = left_camera_.getDistortion().cast<double>();
            cv::Mat M2 = right_camera_.getIntrinsic().cast<double>();
            cv::Mat D2 = right_camera_.getDistortion().cast<double>();

            // Compute rotation and translation between cameras
            // Assuming extrinsic_ is the transformation from left to right
            Eigen::Matrix3d R_eigen = extrinsic_.block<3,3>(0,0);
            Eigen::Vector3d T_eigen = extrinsic_.block<3,1>(0,3);

            cv::Mat R_cv, T_cv;
            cv::eigen2cv(R_eigen, R_cv);
            cv::eigen2cv(T_eigen, T_cv);

            // Compute rectification transforms
            cv::Mat R1, R2, P1, P2, Q;
            cv::stereoRectify(M1, D1, M2, D2, left_image.size(),
                              R_cv, T_cv, R1, R2, P1, P2, Q);

            // Compute rectification maps
            cv::Mat map1x, map1y, map2x, map2y;
            cv::initUndistortRectifyMap(M1, D1, R1, P1, left_image.size(), CV_32FC1, map1x, map1y);
            cv::initUndistortRectifyMap(M2, D2, R2, P2, right_image.size(), CV_32FC1, map2x, map2y);

            // Apply rectification
            cv::remap(left_image, rectified_left, map1x, map1y, cv::INTER_LINEAR);
            cv::remap(right_image, rectified_right, map2x, map2y, cv::INTER_LINEAR);

            Logger::getInstance().log(Logger::LogLevel::INFO, "Stereo images rectified successfully.");
            return true;
        }

        bool StereoCamera::computeDisparity(const cv::Mat& rectified_left, const cv::Mat& rectified_right,
                                            cv::Mat& disparity_map) const {
            if(rectified_left.empty() || rectified_right.empty()) {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "Empty rectified images provided for disparity computation.");
                return false;
            }

            // Create StereoBM or StereoSGBM object
            int numDisparities = 16*5; // must be divisible by 16
            int blockSize = 21;

            cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create(0, numDisparities, blockSize);
            stereo->compute(rectified_left, rectified_right, disparity_map);

            Logger::getInstance().log(Logger::LogLevel::INFO, "Disparity map computed successfully.");
            return true;
        }

        bool StereoCamera::calibrateStereoCamera(const std::vector<std::vector<cv::Point3f>>& object_points,
                                                 const std::vector<std::vector<cv::Point2f>>& image_points_left,
                                                 const std::vector<std::vector<cv::Point2f>>& image_points_right,
                                                 const cv::Size& image_size,
                                                 Camera& left_camera,
                                                 Camera& right_camera,
                                                 Eigen::Matrix4d& out_extrinsic) {
            if(object_points.empty() || image_points_left.empty() || image_points_right.empty() ||
               image_points_left.size() != image_points_right.size() ||
               object_points.size() != image_points_left.size()) {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "Invalid input for stereo camera calibration.");
                return false;
            }

            // Initial guesses for camera matrices
            cv::Mat M1 = left_camera.getIntrinsic().cast<double>();
            cv::Mat D1 = left_camera.getDistortion().cast<double>();
            cv::Mat M2 = right_camera.getIntrinsic().cast<double>();
            cv::Mat D2 = right_camera.getDistortion().cast<double>();

            // Output rotation, translation
            cv::Mat R, T, E, F;

            Logger::getInstance().log(Logger::LogLevel::INFO, "Starting stereo camera calibration...");

            double rms = cv::stereoCalibrate(object_points, image_points_left, image_points_right,
                                             M1, D1, M2, D2, image_size,
                                             R, T, E, F,
                                             cv::CALIB_FIX_INTRINSIC,
                                             cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-5));

            Logger::getInstance().log(Logger::LogLevel::INFO, "Stereo camera calibration completed with RMS error: " + std::to_string(rms));

            // Convert R and T to extrinsic transformation matrix
            Eigen::Matrix3d R_eigen;
            cv::cv2eigen(R, R_eigen);
            Eigen::Vector3d T_eigen;
            cv::cv2eigen(T, T_eigen);

            out_extrinsic = Eigen::Matrix4d::Identity();
            out_extrinsic.block<3,3>(0,0) = R_eigen;
            out_extrinsic.block<3,1>(0,3) = T_eigen;

            // Update right camera's extrinsic
            right_camera.setExtrinsic(out_extrinsic);

            return true;
        }

    } // namespace core
} // namespace rc_vision