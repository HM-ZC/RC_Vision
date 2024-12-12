#ifndef RC_VISION_CORE_CAMERA_HPP
#define RC_VISION_CORE_CAMERA_HPP

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <fstream>

namespace rc_vision {
    namespace core {

        /**
         * @brief 相机模型类，包含内参、畸变系数和外参，并提供投影、去畸变等功能。
         */
        class Camera {
        public:
            /**
             * @brief 默认构造函数，初始化内参为单位矩阵，畸变系数和外参为零。
             */
            Camera();

            /**
             * @brief 带参数的构造函数，初始化内参、畸变系数和外参。
             * @param intrinsic 相机内参矩阵（3x3）。
             * @param distortion 相机畸变系数向量（5维）。
             */
            Camera(const Eigen::Matrix3d& intrinsic, const Eigen::Vector<double, 5>& distortion);

            // =====================
            // Getters
            // =====================

            /**
             * @brief 获取相机内参矩阵。
             * @return 3x3 的相机内参矩阵。
             */
            const Eigen::Matrix3d& getIntrinsic() const;

            /**
             * @brief 获取相机畸变系数向量。
             * @return 5维的畸变系数向量。
             */
            const Eigen::Vector<double, 5>& getDistortion() const;

            /**
             * @brief 获取相机外参矩阵。
             * @return 4x4 的外参矩阵（世界坐标系到相机坐标系的变换）。
             */
            const Eigen::Matrix4d& getExtrinsic() const;

            // =====================
            // Setters
            // =====================

            /**
             * @brief 设置相机内参矩阵。
             * @param intrinsic 新的3x3相机内参矩阵。
             */
            void setIntrinsic(const Eigen::Matrix3d& intrinsic);

            /**
             * @brief 设置相机畸变系数向量。
             * @param distortion 新的5维畸变系数向量。
             */
            void setDistortion(const Eigen::Vector<double, 5>& distortion);

            /**
             * @brief 设置相机外参矩阵。
             * @param extrinsic 新的4x4外参矩阵。
             */
            void setExtrinsic(const Eigen::Matrix4d& extrinsic);

            // =====================
            // Projection
            // =====================

            /**
             * @brief 将3D点投影到2D图像平面。
             * @param point_3d 需要投影的3D点。
             * @return 投影后的2D点。
             */
            Eigen::Vector2d project(const Eigen::Vector3d& point_3d) const;

            /**
             * @brief 将多个3D点投影到2D图像平面。
             * @param points_3d 需要投影的3D点集合。
             * @return 投影后的2D点集合。
             */
            std::vector<Eigen::Vector2d> project(const std::vector<Eigen::Vector3d>& points_3d) const;

            // =====================
            // Undistortion
            // =====================

            /**
             * @brief 去畸变一个2D点。
             * @param distorted_point 畸变的2D点。
             * @return 去畸变后的2D点。
             */
            Eigen::Vector2d undistort(const Eigen::Vector2d& distorted_point) const;

            /**
             * @brief 去畸变多个2D点。
             * @param distorted_points 畸变的2D点集合。
             * @return 去畸变后的2D点集合。
             */
            std::vector<Eigen::Vector2d> undistort(const std::vector<Eigen::Vector2d>& distorted_points) const;

            /**
             * @brief 去畸变一张图像。
             * @param distorted_image 畸变的输入图像。
             * @return 去畸变后的输出图像。
             */
            cv::Mat undistortImage(const cv::Mat& distorted_image) const;

            // =====================
            // Distortion Correction
            // =====================

            /**
             * @brief 校正图像畸变。
             * @param distorted_image 畸变的输入图像。
             * @return 校正后的输出图像。
             */
            cv::Mat correctDistortion(const cv::Mat& distorted_image) const;

            // =====================
            // Serialization
            // =====================

            /**
             * @brief 保存相机参数到文件。
             * @param file_path 要保存的文件路径。
             * @return 如果保存成功返回 true，否则返回 false。
             */
            bool saveParameters(const std::string& file_path) const;

            /**
             * @brief 从文件加载相机参数。
             * @param file_path 要加载的文件路径。
             * @return 如果加载成功返回 true，否则返回 false。
             */
            bool loadParameters(const std::string& file_path);

            // =====================
            // Coordinate Transformation
            // =====================

            /**
             * @brief 将世界坐标系下的3D点转换到相机坐标系。
             * @param point_world 世界坐标系下的3D点。
             * @return 相机坐标系下的3D点。
             */
            Eigen::Vector3d worldToCamera(const Eigen::Vector3d& point_world) const;

            /**
             * @brief 将相机坐标系下的3D点转换到世界坐标系。
             * @param point_camera 相机坐标系下的3D点。
             * @return 世界坐标系下的3D点。
             */
            Eigen::Vector3d cameraToWorld(const Eigen::Vector3d& point_camera) const;

            // =====================
            // Camera Calibration Tool
            // =====================

            /**
             * @brief 使用图像点和物体点进行相机标定。
             * @param object_points 物体点集合，每个视图对应一组3D点。
             * @param image_points 图像点集合，每个视图对应一组2D点。
             * @param image_size 图像尺寸。
             * @param out_intrinsic 输出的相机内参矩阵。
             * @param out_distortion 输出的相机畸变系数向量。
             * @return 如果标定成功返回 true，否则返回 false。
             */
            static bool calibrateCamera(const std::vector<std::vector<cv::Point3f>>& object_points,
                                        const std::vector<std::vector<cv::Point2f>>& image_points,
                                        const cv::Size& image_size,
                                        Eigen::Matrix3d& out_intrinsic,
                                        Eigen::Vector<double, 5>& out_distortion);

        private:
            Eigen::Matrix3d intrinsic_; /**< 相机内参矩阵（3x3）。 */
            Eigen::Vector<double, 5> distortion_; /**< 相机畸变系数向量（k1, k2, p1, p2, k3）。 */
            Eigen::Matrix4d extrinsic_; /**< 相机外参矩阵（世界坐标系到相机坐标系的4x4变换矩阵）。 */

            // OpenCV camera parameters
            cv::Mat cv_intrinsic_; /**< OpenCV 格式的相机内参矩阵。 */
            cv::Mat cv_distortion_; /**< OpenCV 格式的相机畸变系数。 */

            /**
             * @brief 更新 OpenCV 格式的相机参数矩阵。
             */
            void updateOpenCVCameraParams();
        };

        /**
         * @brief 立体相机模型类，包含左右相机及其相对外参，并提供图像校正和视差计算功能。
         */
        class StereoCamera {
        public:
            /**
             * @brief 默认构造函数，初始化左右相机和外参。
             */
            StereoCamera();

            /**
             * @brief 带参数的构造函数，初始化左右相机及其相对外参。
             * @param left_camera 左相机模型。
             * @param right_camera 右相机模型。
             * @param extrinsic 左右相机之间的4x4外参矩阵。
             */
            StereoCamera(const Camera& left_camera, const Camera& right_camera, const Eigen::Matrix4d& extrinsic);

            // =====================
            // Getters
            // =====================

            /**
             * @brief 获取左相机模型。
             * @return 左相机模型的引用。
             */
            const Camera& getLeftCamera() const;

            /**
             * @brief 获取右相机模型。
             * @return 右相机模型的引用。
             */
            const Camera& getRightCamera() const;

            /**
             * @brief 获取左右相机之间的外参矩阵。
             * @return 左右相机之间的4x4外参矩阵。
             */
            const Eigen::Matrix4d& getExtrinsic() const;

            // =====================
            // Setters
            // =====================

            /**
             * @brief 设置左相机模型。
             * @param left_camera 新的左相机模型。
             */
            void setLeftCamera(const Camera& left_camera);

            /**
             * @brief 设置右相机模型。
             * @param right_camera 新的右相机模型。
             */
            void setRightCamera(const Camera& right_camera);

            /**
             * @brief 设置左右相机之间的外参矩阵。
             * @param extrinsic 新的4x4外参矩阵。
             */
            void setExtrinsic(const Eigen::Matrix4d& extrinsic);

            // =====================
            // Rectify Images
            // =====================

            /**
             * @brief 对左右图像进行校正。
             * @param left_image 左图像。
             * @param right_image 右图像。
             * @param rectified_left 校正后的左图像。
             * @param rectified_right 校正后的右图像。
             * @return 如果校正成功返回 true，否则返回 false。
             */
            bool rectifyImages(const cv::Mat& left_image, const cv::Mat& right_image,
                               cv::Mat& rectified_left, cv::Mat& rectified_right) const;

            // =====================
            // Compute Disparity
            // =====================

            /**
             * @brief 计算视差图。
             * @param rectified_left 校正后的左图像。
             * @param rectified_right 校正后的右图像。
             * @param disparity_map 输出的视差图。
             * @return 如果计算成功返回 true，否则返回 false。
             */
            bool computeDisparity(const cv::Mat& rectified_left, const cv::Mat& rectified_right,
                                  cv::Mat& disparity_map) const;

            // =====================
            // Stereo Calibration Tool
            // =====================

            /**
             * @brief 使用图像点和物体点进行立体相机标定。
             * @param object_points 物体点集合，每个视图对应一组3D点。
             * @param image_points_left 左图像点集合，每个视图对应一组2D点。
             * @param image_points_right 右图像点集合，每个视图对应一组2D点。
             * @param image_size 图像尺寸。
             * @param left_camera 输出的左相机模型。
             * @param right_camera 输出的右相机模型。
             * @param out_extrinsic 输出的左右相机之间的4x4外参矩阵。
             * @return 如果标定成功返回 true，否则返回 false。
             */
            static bool calibrateStereoCamera(const std::vector<std::vector<cv::Point3f>>& object_points,
                                              const std::vector<std::vector<cv::Point2f>>& image_points_left,
                                              const std::vector<std::vector<cv::Point2f>>& image_points_right,
                                              const cv::Size& image_size,
                                              Camera& left_camera,
                                              Camera& right_camera,
                                              Eigen::Matrix4d& out_extrinsic);

        private:
            Camera left_camera_; /**< 左相机模型。 */
            Camera right_camera_; /**< 右相机模型。 */
            Eigen::Matrix4d extrinsic_; /**< 左右相机之间的外参矩阵（4x4）。 */
        };

    } // namespace core
} // namespace rc_vision

#endif // RC_VISION_CORE_CAMERA_HPP
