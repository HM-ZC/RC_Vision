/*
 * 用于处理和表示三维点云数据，支持常见的点云操作，如滤波、配准和可视化。
 */
#include "rc_vision/core/point_cloud.hpp"
#include "rc_vision/core/logger.hpp" // 用于日志记录
#include <fstream>
#include <sstream>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/features/normal_3d.h>
#include <pcl/registration/icp.h>
#include <pcl/segmentation/extract_clusters.h>

namespace rc_vision {
    namespace core {

        void PointCloud::addPoint(const Point& point) {
            points_.push_back(point);
        }

        void PointCloud::addPoint(double x, double y, double z) {
            points_.emplace_back(x, y, z);
        }

        void PointCloud::addPoint(double x, double y, double z, const Eigen::Vector3d& color) {
            points_.emplace_back(x, y, z, color);
        }

        size_t PointCloud::size() const {
            return points_.size();
        }

        const std::vector<Point>& PointCloud::getPoints() const {
            return points_;
        }

        Eigen::Vector3d PointCloud::computeCentroid() const {
            Eigen::Vector3d centroid(0, 0, 0);
            for (const auto& point : points_) {
                centroid += Eigen::Vector3d(point.x, point.y, point.z);
            }
            if (!points_.empty()) {
                centroid /= static_cast<double>(points_.size());
            }
            return centroid;
        }

        void PointCloud::clear() {
            points_.clear();
        }

        void PointCloud::applyTransformation(const Eigen::Matrix4d& transformation) {
            pcl::PointCloud<pcl::PointXYZRGB> pcl_cloud;
            // 转换到 PCL 点云
            for (const auto& point : points_) {
                pcl::PointXYZRGB pcl_point;
                pcl_point.x = static_cast<float>(point.x);
                pcl_point.y = static_cast<float>(point.y);
                pcl_point.z = static_cast<float>(point.z);
                if (point.color.has_value()) {
                    pcl_point.r = static_cast<uint8_t>(point.color.value()[0]);
                    pcl_point.g = static_cast<uint8_t>(point.color.value()[1]);
                    pcl_point.b = static_cast<uint8_t>(point.color.value()[2]);
                }
                pcl_cloud.points.emplace_back(pcl_point);
            }

            // 应用变换
            pcl::PointCloud<pcl::PointXYZRGB> transformed_cloud;
            pcl::transformPointCloud(pcl_cloud, transformed_cloud, transformation);

            // 更新点云数据
            points_.clear();
            for (const auto& pcl_point : transformed_cloud.points) {
                Point point(pcl_point.x, pcl_point.y, pcl_point.z);
                if (pcl_point.r || pcl_point.g || pcl_point.b) {
                    Eigen::Vector3d color(pcl_point.r, pcl_point.g, pcl_point.b);
                    point.color = color;
                }
                points_.push_back(point);
            }

            Logger::getInstance().log(Logger::LogLevel::INFO, "Applied transformation to point cloud.");
        }

        void PointCloud::transform(const Eigen::Matrix4d& transformation) {
            applyTransformation(transformation);
        }

        void PointCloud::rotate(const Eigen::Matrix3d& rotation_matrix) {
            Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity();
            transformation.block<3,3>(0,0) = rotation_matrix;
            applyTransformation(transformation);
            Logger::getInstance().log(Logger::LogLevel::INFO, "Rotated point cloud.");
        }

        void PointCloud::translate(const Eigen::Vector3d& translation_vector) {
            Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity();
            transformation.block<3,1>(0,3) = translation_vector;
            applyTransformation(transformation);
            Logger::getInstance().log(Logger::LogLevel::INFO, "Translated point cloud.");
        }

        void PointCloud::downsample(double voxel_size) {
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
            // 转换到 PCL 点云
            for (const auto& point : points_) {
                pcl::PointXYZRGB pcl_point;
                pcl_point.x = static_cast<float>(point.x);
                pcl_point.y = static_cast<float>(point.y);
                pcl_point.z = static_cast<float>(point.z);
                if (point.color.has_value()) {
                    pcl_point.r = static_cast<uint8_t>(point.color.value()[0]);
                    pcl_point.g = static_cast<uint8_t>(point.color.value()[1]);
                    pcl_point.b = static_cast<uint8_t>(point.color.value()[2]);
                }
                pcl_cloud->points.emplace_back(pcl_point);
            }
            pcl_cloud->width = points_.size();
            pcl_cloud->height = 1;
            pcl_cloud->is_dense = true;

            // 体素滤波
            pcl::VoxelGrid<pcl::PointXYZRGB> sor;
            sor.setInputCloud(pcl_cloud);
            sor.setLeafSize(static_cast<float>(voxel_size), static_cast<float>(voxel_size), static_cast<float>(voxel_size));
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
            sor.filter(*cloud_filtered);

            // 更新点云数据
            points_.clear();
            for (const auto& pcl_point : cloud_filtered->points) {
                Point point(pcl_point.x, pcl_point.y, pcl_point.z);
                if (pcl_point.r || pcl_point.g || pcl_point.b) {
                    Eigen::Vector3d color(pcl_point.r, pcl_point.g, pcl_point.b);
                    point.color = color;
                }
                points_.push_back(point);
            }

            Logger::getInstance().log(Logger::LogLevel::INFO, "Downsampled point cloud with voxel size: " + std::to_string(voxel_size));
        }

        void PointCloud::statisticalOutlierRemoval(int mean_k, double std_dev_mul_thresh) {
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
            // 转换到 PCL 点云
            for (const auto& point : points_) {
                pcl::PointXYZRGB pcl_point;
                pcl_point.x = static_cast<float>(point.x);
                pcl_point.y = static_cast<float>(point.y);
                pcl_point.z = static_cast<float>(point.z);
                if (point.color.has_value()) {
                    pcl_point.r = static_cast<uint8_t>(point.color.value()[0]);
                    pcl_point.g = static_cast<uint8_t>(point.color.value()[1]);
                    pcl_point.b = static_cast<uint8_t>(point.color.value()[2]);
                }
                pcl_cloud->points.emplace_back(pcl_point);
            }
            pcl_cloud->width = points_.size();
            pcl_cloud->height = 1;
            pcl_cloud->is_dense = true;

            // 统计滤波
            pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
            sor.setInputCloud(pcl_cloud);
            sor.setMeanK(mean_k);
            sor.setStddevMulThresh(static_cast<float>(std_dev_mul_thresh));
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
            sor.filter(*cloud_filtered);

            // 更新点云数据
            points_.clear();
            for (const auto& pcl_point : cloud_filtered->points) {
                Point point(pcl_point.x, pcl_point.y, pcl_point.z);
                if (pcl_point.r || pcl_point.g || pcl_point.b) {
                    Eigen::Vector3d color(pcl_point.r, pcl_point.g, pcl_point.b);
                    point.color = color;
                }
                points_.push_back(point);
            }

            Logger::getInstance().log(Logger::LogLevel::INFO, "Applied Statistical Outlier Removal with mean_k: " + std::to_string(mean_k) + ", std_dev_mul_thresh: " + std::to_string(std_dev_mul_thresh));
        }

        void PointCloud::radiusOutlierRemoval(double radius, int min_neighbors) {
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
            // 转换到 PCL 点云
            for (const auto& point : points_) {
                pcl::PointXYZRGB pcl_point;
                pcl_point.x = static_cast<float>(point.x);
                pcl_point.y = static_cast<float>(point.y);
                pcl_point.z = static_cast<float>(point.z);
                if (point.color.has_value()) {
                    pcl_point.r = static_cast<uint8_t>(point.color.value()[0]);
                    pcl_point.g = static_cast<uint8_t>(point.color.value()[1]);
                    pcl_point.b = static_cast<uint8_t>(point.color.value()[2]);
                }
                pcl_cloud->points.emplace_back(pcl_point);
            }
            pcl_cloud->width = points_.size();
            pcl_cloud->height = 1;
            pcl_cloud->is_dense = true;

            // 半径滤波
            pcl::RadiusOutlierRemoval<pcl::PointXYZRGB> ror;
            ror.setInputCloud(pcl_cloud);
            ror.setRadiusSearch(static_cast<float>(radius));
            ror.setMinNeighborsInRadius(min_neighbors);
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
            ror.filter(*cloud_filtered);

            // 更新点云数据
            points_.clear();
            for (const auto& pcl_point : cloud_filtered->points) {
                Point point(pcl_point.x, pcl_point.y, pcl_point.z);
                if (pcl_point.r || pcl_point.g || pcl_point.b) {
                    Eigen::Vector3d color(pcl_point.r, pcl_point.g, pcl_point.b);
                    point.color = color;
                }
                points_.push_back(point);
            }

            Logger::getInstance().log(Logger::LogLevel::INFO, "Applied Radius Outlier Removal with radius: " + std::to_string(radius) + ", min_neighbors: " + std::to_string(min_neighbors));
        }

        void PointCloud::computeNormals(int k) {
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
            // 转换到 PCL 点云
            for (const auto& point : points_) {
                pcl::PointXYZRGB pcl_point;
                pcl_point.x = static_cast<float>(point.x);
                pcl_point.y = static_cast<float>(point.y);
                pcl_point.z = static_cast<float>(point.z);
                if (point.color.has_value()) {
                    pcl_point.r = static_cast<uint8_t>(point.color.value()[0]);
                    pcl_point.g = static_cast<uint8_t>(point.color.value()[1]);
                    pcl_point.b = static_cast<uint8_t>(point.color.value()[2]);
                }
                pcl_cloud->points.emplace_back(pcl_point);
            }
            pcl_cloud->width = points_.size();
            pcl_cloud->height = 1;
            pcl_cloud->is_dense = true;

            // 法向量估计
            pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
            ne.setInputCloud(pcl_cloud);
            pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());
            ne.setSearchMethod(tree);
            ne.setKSearch(k);
            pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
            ne.compute(*normals);

            // 更新点云数据中的法向量
            for (size_t i = 0; i < points_.size() && i < normals->points.size(); ++i) {
                const auto& pcl_normal = normals->points[i];
                if (!std::isnan(pcl_normal.normal_x) && !std::isnan(pcl_normal.normal_y) && !std::isnan(pcl_normal.normal_z)) {
                    Eigen::Vector3d normal(pcl_normal.normal_x, pcl_normal.normal_y, pcl_normal.normal_z);
                    points_[i].normal = normal.normalized();
                }
            }

            Logger::getInstance().log(Logger::LogLevel::INFO, "Computed normals for point cloud with k: " + std::to_string(k));
        }

        Eigen::Matrix4d PointCloud::icp(const PointCloud& target, int max_iterations, double tolerance) {
            // 转换到 PCL 点云
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr source_pcl(new pcl::PointCloud<pcl::PointXYZRGB>);
            for (const auto& point : points_) {
                pcl::PointXYZRGB pcl_point;
                pcl_point.x = static_cast<float>(point.x);
                pcl_point.y = static_cast<float>(point.y);
                pcl_point.z = static_cast<float>(point.z);
                if (point.color.has_value()) {
                    pcl_point.r = static_cast<uint8_t>(point.color.value()[0]);
                    pcl_point.g = static_cast<uint8_t>(point.color.value()[1]);
                    pcl_point.b = static_cast<uint8_t>(point.color.value()[2]);
                }
                source_pcl->points.emplace_back(pcl_point);
            }
            source_pcl->width = points_.size();
            source_pcl->height = 1;
            source_pcl->is_dense = true;

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr target_pcl(new pcl::PointCloud<pcl::PointXYZRGB>);
            for (const auto& point : target.getPoints()) {
                pcl::PointXYZRGB pcl_point;
                pcl_point.x = static_cast<float>(point.x);
                pcl_point.y = static_cast<float>(point.y);
                pcl_point.z = static_cast<float>(point.z);
                if (point.color.has_value()) {
                    pcl_point.r = static_cast<uint8_t>(point.color.value()[0]);
                    pcl_point.g = static_cast<uint8_t>(point.color.value()[1]);
                    pcl_point.b = static_cast<uint8_t>(point.color.value()[2]);
                }
                target_pcl->points.emplace_back(pcl_point);
            }
            target_pcl->width = target.size();
            target_pcl->height = 1;
            target_pcl->is_dense = true;

            // ICP 配准
            pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp;
            icp.setInputSource(source_pcl);
            icp.setInputTarget(target_pcl);
            icp.setMaximumIterations(max_iterations);
            icp.setTransformationEpsilon(tolerance);

            pcl::PointCloud<pcl::PointXYZRGB> Final;
            icp.align(Final);

            if (icp.hasConverged()) {
                Logger::getInstance().log(Logger::LogLevel::INFO, "ICP converged with score: " + std::to_string(icp.getFitnessScore()));
                Eigen::Matrix4f transformation = icp.getFinalTransformation();
                return transformation.cast<double>();
            } else {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "ICP did not converge.");
                return Eigen::Matrix4d::Identity();
            }
        }

        std::vector<PointCloud> PointCloud::euclideanClustering(double cluster_tolerance, int min_size, int max_size) const {
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
            // 转换到 PCL 点云
            for (const auto& point : points_) {
                pcl::PointXYZRGB pcl_point;
                pcl_point.x = static_cast<float>(point.x);
                pcl_point.y = static_cast<float>(point.y);
                pcl_point.z = static_cast<float>(point.z);
                if (point.color.has_value()) {
                    pcl_point.r = static_cast<uint8_t>(point.color.value()[0]);
                    pcl_point.g = static_cast<uint8_t>(point.color.value()[1]);
                    pcl_point.b = static_cast<uint8_t>(point.color.value()[2]);
                }
                pcl_cloud->points.emplace_back(pcl_point);
            }
            pcl_cloud->width = points_.size();
            pcl_cloud->height = 1;
            pcl_cloud->is_dense = true;

            // 创建 KdTree
            pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
            tree->setInputCloud(pcl_cloud);

            // 设置聚类参数
            std::vector<pcl::PointIndices> cluster_indices;
            pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
            ec.setClusterTolerance(static_cast<float>(cluster_tolerance)); // 距离容差
            ec.setMinClusterSize(min_size);
            ec.setMaxClusterSize(max_size);
            ec.setSearchMethod(tree);
            ec.setInputCloud(pcl_cloud);
            ec.extract(cluster_indices);

            std::vector<PointCloud> clusters;
            for (const auto& indices : cluster_indices) {
                PointCloud cluster;
                for (const auto& idx : indices.indices) {
                    const auto& pcl_point = pcl_cloud->points[idx];
                    Point point(pcl_point.x, pcl_point.y, pcl_point.z);
                    if (pcl_point.r || pcl_point.g || pcl_point.b) {
                        Eigen::Vector3d color(pcl_point.r, pcl_point.g, pcl_point.b);
                        point.color = color;
                    }
                    cluster.addPoint(point);
                }
                clusters.push_back(cluster);
            }

            Logger::getInstance().log(Logger::LogLevel::INFO, "Euclidean Clustering found " + std::to_string(clusters.size()) + " clusters.");
            return clusters;
        }

        void PointCloud::merge(const PointCloud& other) {
            const auto& other_points = other.getPoints();
            for (const auto& point : other_points) {
                points_.push_back(point);
            }
            Logger::getInstance().log(Logger::LogLevel::INFO, "Merged another point cloud. Total points: " + std::to_string(points_.size()));
        }

        void PointCloud::mergeWith(const PointCloud& other, const Eigen::Matrix4d& transformation) {
            PointCloud transformed_other = other;
            transformed_other.transform(transformation);

            merge(transformed_other);
            Logger::getInstance().log(Logger::LogLevel::INFO, "Merged point cloud with transformation.");
        }

        PointCloudStats PointCloud::computeStats() const {
            PointCloudStats stats;
            if (points_.empty()) {
                return stats;
            }

            stats.min_x = stats.max_x = points_[0].x;
            stats.min_y = stats.max_y = points_[0].y;
            stats.min_z = stats.max_z = points_[0].z;

            double sum_x = 0, sum_y = 0, sum_z = 0;
            for (const auto& point : points_) {
                if (point.x < stats.min_x) stats.min_x = point.x;
                if (point.x > stats.max_x) stats.max_x = point.x;
                if (point.y < stats.min_y) stats.min_y = point.y;
                if (point.y > stats.max_y) stats.max_y = point.y;
                if (point.z < stats.min_z) stats.min_z = point.z;
                if (point.z > stats.max_z) stats.max_z = point.z;

                sum_x += point.x;
                sum_y += point.y;
                sum_z += point.z;
            }

            size_t n = points_.size();
            stats.mean_x = sum_x / static_cast<double>(n);
            stats.mean_y = sum_y / static_cast<double>(n);
            stats.mean_z = sum_z / static_cast<double>(n);

            double var_x = 0, var_y = 0, var_z = 0;
            for (const auto& point : points_) {
                var_x += (point.x - stats.mean_x) * (point.x - stats.mean_x);
                var_y += (point.y - stats.mean_y) * (point.y - stats.mean_y);
                var_z += (point.z - stats.mean_z) * (point.z - stats.mean_z);
            }

            stats.variance_x = var_x / static_cast<double>(n);
            stats.variance_y = var_y / static_cast<double>(n);
            stats.variance_z = var_z / static_cast<double>(n);

            return stats;
        }

        bool PointCloud::loadFromPLY(const std::string& file_path) {
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
            if (pcl::io::loadPLYFile<pcl::PointXYZRGB>(file_path, *pcl_cloud) == -1) {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "Couldn't read PLY file: " + file_path);
                return false;
            }

            // 转换到自定义点云
            points_.clear();
            for (const auto& pcl_point : pcl_cloud->points) {
                Point point(pcl_point.x, pcl_point.y, pcl_point.z);
                if (pcl_point.r || pcl_point.g || pcl_point.b) {
                    Eigen::Vector3d color(pcl_point.r, pcl_point.g, pcl_point.b);
                    point.color = color;
                }
                points_.push_back(point);
            }

            Logger::getInstance().log(Logger::LogLevel::INFO, "Loaded " + std::to_string(points_.size()) + " points from PLY file: " + file_path);
            return true;
        }

        bool PointCloud::saveToPLY(const std::string& file_path) const {
            pcl::PointCloud<pcl::PointXYZRGB> pcl_cloud;
            // 转换到 PCL 点云
            for (const auto& point : points_) {
                pcl::PointXYZRGB pcl_point;
                pcl_point.x = static_cast<float>(point.x);
                pcl_point.y = static_cast<float>(point.y);
                pcl_point.z = static_cast<float>(point.z);
                if (point.color.has_value()) {
                    pcl_point.r = static_cast<uint8_t>(point.color.value()[0]);
                    pcl_point.g = static_cast<uint8_t>(point.color.value()[1]);
                    pcl_point.b = static_cast<uint8_t>(point.color.value()[2]);
                }
                pcl_cloud.points.emplace_back(pcl_point);
            }
            pcl_cloud.width = points_.size();
            pcl_cloud.height = 1;
            pcl_cloud.is_dense = true;

            if (pcl::io::savePLYFileASCII(file_path, pcl_cloud) == -1) {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "Couldn't write PLY file: " + file_path);
                return false;
            }

            Logger::getInstance().log(Logger::LogLevel::INFO, "Saved " + std::to_string(points_.size()) + " points to PLY file: " + file_path);
            return true;
        }

        bool PointCloud::loadFromPCD(const std::string& file_path) {
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
            if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(file_path, *pcl_cloud) == -1) {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "Couldn't read PCD file: " + file_path);
                return false;
            }

            // 转换到自定义点云
            points_.clear();
            for (const auto& pcl_point : pcl_cloud->points) {
                Point point(pcl_point.x, pcl_point.y, pcl_point.z);
                if (pcl_point.r || pcl_point.g || pcl_point.b) {
                    Eigen::Vector3d color(pcl_point.r, pcl_point.g, pcl_point.b);
                    point.color = color;
                }
                points_.push_back(point);
            }

            Logger::getInstance().log(Logger::LogLevel::INFO, "Loaded " + std::to_string(points_.size()) + " points from PCD file: " + file_path);
            return true;
        }

        bool PointCloud::saveToPCD(const std::string& file_path) const {
            pcl::PointCloud<pcl::PointXYZRGB> pcl_cloud;
            // 转换到 PCL 点云
            for (const auto& point : points_) {
                pcl::PointXYZRGB pcl_point;
                pcl_point.x = static_cast<float>(point.x);
                pcl_point.y = static_cast<float>(point.y);
                pcl_point.z = static_cast<float>(point.z);
                if (point.color.has_value()) {
                    pcl_point.r = static_cast<uint8_t>(point.color.value()[0]);
                    pcl_point.g = static_cast<uint8_t>(point.color.value()[1]);
                    pcl_point.b = static_cast<uint8_t>(point.color.value()[2]);
                }
                pcl_cloud.points.emplace_back(pcl_point);
            }
            pcl_cloud.width = points_.size();
            pcl_cloud.height = 1;
            pcl_cloud.is_dense = true;

            if (pcl::io::savePCDFileASCII(file_path, pcl_cloud) == -1) {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "Couldn't write PCD file: " + file_path);
                return false;
            }

            Logger::getInstance().log(Logger::LogLevel::INFO, "Saved " + std::to_string(points_.size()) + " points to PCD file: " + file_path);
            return true;
        }

        Eigen::Matrix4d PointCloud::icp(const PointCloud& target, int max_iterations, double tolerance) {
            // 转换到 PCL 点云
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr source_pcl(new pcl::PointCloud<pcl::PointXYZRGB>);
            for (const auto& point : points_) {
                pcl::PointXYZRGB pcl_point;
                pcl_point.x = static_cast<float>(point.x);
                pcl_point.y = static_cast<float>(point.y);
                pcl_point.z = static_cast<float>(point.z);
                if (point.color.has_value()) {
                    pcl_point.r = static_cast<uint8_t>(point.color.value()[0]);
                    pcl_point.g = static_cast<uint8_t>(point.color.value()[1]);
                    pcl_point.b = static_cast<uint8_t>(point.color.value()[2]);
                }
                source_pcl->points.emplace_back(pcl_point);
            }
            source_pcl->width = points_.size();
            source_pcl->height = 1;
            source_pcl->is_dense = true;

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr target_pcl(new pcl::PointCloud<pcl::PointXYZRGB>);
            for (const auto& point : target.getPoints()) {
                pcl::PointXYZRGB pcl_point;
                pcl_point.x = static_cast<float>(point.x);
                pcl_point.y = static_cast<float>(point.y);
                pcl_point.z = static_cast<float>(point.z);
                if (point.color.has_value()) {
                    pcl_point.r = static_cast<uint8_t>(point.color.value()[0]);
                    pcl_point.g = static_cast<uint8_t>(point.color.value()[1]);
                    pcl_point.b = static_cast<uint8_t>(point.color.value()[2]);
                }
                target_pcl->points.emplace_back(pcl_point);
            }
            target_pcl->width = target.size();
            target_pcl->height = 1;
            target_pcl->is_dense = true;

            // ICP 配准
            pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp;
            icp.setInputSource(source_pcl);
            icp.setInputTarget(target_pcl);
            icp.setMaximumIterations(max_iterations);
            icp.setTransformationEpsilon(tolerance);

            pcl::PointCloud<pcl::PointXYZRGB> Final;
            icp.align(Final);

            if (icp.hasConverged()) {
                Logger::getInstance().log(Logger::LogLevel::INFO, "ICP converged with score: " + std::to_string(icp.getFitnessScore()));
                Eigen::Matrix4f transformation = icp.getFinalTransformation();
                return transformation.cast<double>();
            } else {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "ICP did not converge.");
                return Eigen::Matrix4d::Identity();
            }
        }

        std::vector<PointCloud> PointCloud::euclideanClustering(double cluster_tolerance, int min_size, int max_size) const {
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
            // 转换到 PCL 点云
            for (const auto& point : points_) {
                pcl::PointXYZRGB pcl_point;
                pcl_point.x = static_cast<float>(point.x);
                pcl_point.y = static_cast<float>(point.y);
                pcl_point.z = static_cast<float>(point.z);
                if (point.color.has_value()) {
                    pcl_point.r = static_cast<uint8_t>(point.color.value()[0]);
                    pcl_point.g = static_cast<uint8_t>(point.color.value()[1]);
                    pcl_point.b = static_cast<uint8_t>(point.color.value()[2]);
                }
                pcl_cloud->points.emplace_back(pcl_point);
            }
            pcl_cloud->width = points_.size();
            pcl_cloud->height = 1;
            pcl_cloud->is_dense = true;

            // 创建 KdTree
            pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
            tree->setInputCloud(pcl_cloud);

            // 设置聚类参数
            std::vector<pcl::PointIndices> cluster_indices;
            pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
            ec.setClusterTolerance(static_cast<float>(cluster_tolerance)); // 距离容差
            ec.setMinClusterSize(min_size);
            ec.setMaxClusterSize(max_size);
            ec.setSearchMethod(tree);
            ec.setInputCloud(pcl_cloud);
            ec.extract(cluster_indices);

            std::vector<PointCloud> clusters;
            for (const auto& indices : cluster_indices) {
                PointCloud cluster;
                for (const auto& idx : indices.indices) {
                    const auto& pcl_point = pcl_cloud->points[idx];
                    Point point(pcl_point.x, pcl_point.y, pcl_point.z);
                    if (pcl_point.r || pcl_point.g || pcl_point.b) {
                        Eigen::Vector3d color(pcl_point.r, pcl_point.g, pcl_point.b);
                        point.color = color;
                    }
                    cluster.addPoint(point);
                }
                clusters.push_back(cluster);
            }

            Logger::getInstance().log(Logger::LogLevel::INFO, "Euclidean Clustering found " + std::to_string(clusters.size()) + " clusters.");
            return clusters;
        }

        void PointCloud::merge(const PointCloud& other) {
            const auto& other_points = other.getPoints();
            for (const auto& point : other_points) {
                points_.push_back(point);
            }
            Logger::getInstance().log(Logger::LogLevel::INFO, "Merged another point cloud. Total points: " + std::to_string(points_.size()));
        }

        void PointCloud::mergeWith(const PointCloud& other, const Eigen::Matrix4d& transformation) {
            PointCloud transformed_other = other;
            transformed_other.transform(transformation);

            merge(transformed_other);
            Logger::getInstance().log(Logger::LogLevel::INFO, "Merged point cloud with transformation.");
        }

        PointCloudStats PointCloud::computeStats() const {
            PointCloudStats stats;
            if (points_.empty()) {
                return stats;
            }

            stats.min_x = stats.max_x = points_[0].x;
            stats.min_y = stats.max_y = points_[0].y;
            stats.min_z = stats.max_z = points_[0].z;

            double sum_x = 0, sum_y = 0, sum_z = 0;
            for (const auto& point : points_) {
                if (point.x < stats.min_x) stats.min_x = point.x;
                if (point.x > stats.max_x) stats.max_x = point.x;
                if (point.y < stats.min_y) stats.min_y = point.y;
                if (point.y > stats.max_y) stats.max_y = point.y;
                if (point.z < stats.min_z) stats.min_z = point.z;
                if (point.z > stats.max_z) stats.max_z = point.z;

                sum_x += point.x;
                sum_y += point.y;
                sum_z += point.z;
            }

            size_t n = points_.size();
            stats.mean_x = sum_x / static_cast<double>(n);
            stats.mean_y = sum_y / static_cast<double>(n);
            stats.mean_z = sum_z / static_cast<double>(n);

            double var_x = 0, var_y = 0, var_z = 0;
            for (const auto& point : points_) {
                var_x += (point.x - stats.mean_x) * (point.x - stats.mean_x);
                var_y += (point.y - stats.mean_y) * (point.y - stats.mean_y);
                var_z += (point.z - stats.mean_z) * (point.z - stats.mean_z);
            }

            stats.variance_x = var_x / static_cast<double>(n);
            stats.variance_y = var_y / static_cast<double>(n);
            stats.variance_z = var_z / static_cast<double>(n);

            return stats;
        }

    } // namespace core
} // namespace rc_vision