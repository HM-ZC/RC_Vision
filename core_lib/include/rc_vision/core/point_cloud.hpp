#ifndef RC_VISION_CORE_POINT_CLOUD_HPP
#define RC_VISION_CORE_POINT_CLOUD_HPP

#include <vector>
#include <Eigen/Dense>
#include <optional>
#include <string>
#include <unordered_map>
#include <opencv2/opencv.hpp>

namespace rc_vision {
    namespace core {

        /**
         * @brief 点结构体，表示三维空间中的一个点。
         *
         * 支持可选的颜色（RGB）、法向量和强度信息。
         */
        struct Point {
            double x; /**< 点的X坐标。 */
            double y; /**< 点的Y坐标。 */
            double z; /**< 点的Z坐标。 */
            std::optional<Eigen::Vector3d> color;      /**< 可选的RGB颜色信息。 */
            std::optional<Eigen::Vector3d> normal;     /**< 可选的法向量信息。 */
            std::optional<double> intensity;           /**< 可选的强度信息。 */

            /**
             * @brief 默认构造函数，仅初始化坐标。
             *
             * @param x_ 点的X坐标。
             * @param y_ 点的Y坐标。
             * @param z_ 点的Z坐标。
             */
            Point(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

            /**
             * @brief 带颜色的构造函数，初始化坐标和颜色。
             *
             * @param x_ 点的X坐标。
             * @param y_ 点的Y坐标。
             * @param z_ 点的Z坐标。
             * @param color_ 点的RGB颜色。
             */
            Point(double x_, double y_, double z_, const Eigen::Vector3d& color_)
                    : x(x_), y(y_), z(z_), color(color_) {}

            // 可根据需要添加更多构造函数
        };

        /**
         * @brief 点云统计信息结构体。
         *
         * 包含点云在各个轴上的最小值、最大值、均值和方差。
         */
        struct PointCloudStats {
            double min_x, max_x; /**< 点云在X轴的最小值和最大值。 */
            double min_y, max_y; /**< 点云在Y轴的最小值和最大值。 */
            double min_z, max_z; /**< 点云在Z轴的最小值和最大值。 */
            double mean_x, mean_y, mean_z; /**< 点云在各轴上的均值。 */
            double variance_x, variance_y, variance_z; /**< 点云在各轴上的方差。 */
        };

        /**
         * @brief 点云类，管理和处理三维点云数据。
         *
         * 提供点的添加、变换、滤波、法向量估计、配准、分割、合并以及统计信息计算等功能。
         */
        class PointCloud {
        public:
            /**
             * @brief 默认构造函数。
             */
            PointCloud() = default;

            /**
             * @brief 添加一个点到点云。
             *
             * @param point 需要添加的点。
             */
            void addPoint(const Point& point);

            /**
             * @brief 添加一个点到点云。
             *
             * @param x 点的X坐标。
             * @param y 点的Y坐标。
             * @param z 点的Z坐标。
             */
            void addPoint(double x, double y, double z);

            /**
             * @brief 添加一个带颜色的点到点云。
             *
             * @param x 点的X坐标。
             * @param y 点的Y坐标。
             * @param z 点的Z坐标。
             * @param color 点的RGB颜色。
             */
            void addPoint(double x, double y, double z, const Eigen::Vector3d& color);

            /**
             * @brief 获取点云中的点数。
             *
             * @return 点云中的点数。
             */
            size_t size() const;

            /**
             * @brief 获取所有点的集合。
             *
             * @return 常量引用，指向点云中的所有点。
             */
            const std::vector<Point>& getPoints() const;

            /**
             * @brief 计算点云的质心。
             *
             * @return 点云的质心坐标。
             */
            Eigen::Vector3d computeCentroid() const;

            /**
             * @brief 清空点云中的所有点。
             */
            void clear();

            /**
             * @brief 使用4x4变换矩阵对点云进行变换。
             *
             * @param transformation 4x4的变换矩阵。
             */
            void transform(const Eigen::Matrix4d& transformation);

            /**
             * @brief 对点云进行旋转。
             *
             * @param rotation_matrix 3x3的旋转矩阵。
             */
            void rotate(const Eigen::Matrix3d& rotation_matrix);

            /**
             * @brief 对点云进行平移。
             *
             * @param translation_vector 3维的平移向量。
             */
            void translate(const Eigen::Vector3d& translation_vector);

            /**
             * @brief 对点云进行下采样。
             *
             * 使用体素网格滤波器进行下采样。
             *
             * @param voxel_size 体素大小。
             */
            void downsample(double voxel_size);

            /**
             * @brief 统计异常点移除。
             *
             * 移除点云中的统计异常点。
             *
             * @param mean_k 邻域内点的数量。
             * @param std_dev_mul_thresh 标准差倍数阈值。
             */
            void statisticalOutlierRemoval(int mean_k = 50, double std_dev_mul_thresh = 1.0);

            /**
             * @brief 半径异常点移除。
             *
             * 移除点云中邻居点少于指定数量的点。
             *
             * @param radius 搜索半径。
             * @param min_neighbors 最小邻居数量。
             */
            void radiusOutlierRemoval(double radius, int min_neighbors);

            /**
             * @brief 计算点云中每个点的法向量。
             *
             * 使用K最近邻方法估计法向量。
             *
             * @param k 用于法向量估计的邻居点数量。
             */
            void computeNormals(int k = 10);

            /**
             * @brief 使用ICP算法进行点云配准。
             *
             * 将当前点云与目标点云对齐。
             *
             * @param target 目标点云。
             * @param max_iterations 最大迭代次数。
             * @param tolerance 收敛容差。
             * @return 最优的4x4变换矩阵。
             */
            Eigen::Matrix4d icp(const PointCloud& target, int max_iterations = 50, double tolerance = 1e-6);

            /**
             * @brief 使用欧式聚类对点云进行分割。
             *
             * 将点云分割为多个独立的聚类。
             *
             * @param cluster_tolerance 聚类容差半径。
             * @param min_size 每个聚类的最小点数。
             * @param max_size 每个聚类的最大点数。
             * @return 分割后的点云集合。
             */
            std::vector<PointCloud> euclideanClustering(double cluster_tolerance, int min_size, int max_size) const;

            /**
             * @brief 合并另一个点云到当前点云中。
             *
             * @param other 需要合并的点云。
             */
            void merge(const PointCloud& other);

            /**
             * @brief 合并另一个点云到当前点云中，并应用变换。
             *
             * @param other 需要合并的点云。
             * @param transformation 需要应用的4x4变换矩阵。
             */
            void mergeWith(const PointCloud& other, const Eigen::Matrix4d& transformation);

            /**
             * @brief 计算点云的统计信息。
             *
             * @return 点云的统计信息结构体。
             */
            PointCloudStats computeStats() const;

            /**
             * @brief 从PLY文件加载点云数据。
             *
             * @param file_path PLY文件路径。
             * @return 成功返回true，失败返回false。
             */
            bool loadFromPLY(const std::string& file_path);

            /**
             * @brief 将点云数据保存到PLY文件。
             *
             * @param file_path PLY文件路径。
             * @return 成功返回true，失败返回false。
             */
            bool saveToPLY(const std::string& file_path) const;

            /**
             * @brief 从PCD文件加载点云数据。
             *
             * @param file_path PCD文件路径。
             * @return 成功返回true，失败返回false。
             */
            bool loadFromPCD(const std::string& file_path);

            /**
             * @brief 将点云数据保存到PCD文件。
             *
             * @param file_path PCD文件路径。
             * @return 成功返回true，失败返回false。
             */
            bool saveToPCD(const std::string& file_path) const;

        private:
            std::vector<Point> points_; /**< 存储点云中的所有点。 */

            /**
             * @brief 应用4x4变换矩阵到点云。
             *
             * @param transformation 4x4的变换矩阵。
             */
            void applyTransformation(const Eigen::Matrix4d& transformation);
        };

    } // namespace core
} // namespace rc_vision

#endif // RC_VISION_CORE_POINT_CLOUD_HPP
