#include "rc_vision/core/point_cloud.hpp"
#include "rc_vision/core/logger.hpp"
#include <iostream>

using namespace rc_vision::core;

int main() {
    // 初始化日志
    Logger& logger = Logger::getInstance();
    logger.setLogLevel(Logger::LogLevel::DEBUG);
    if (!logger.setLogFile("point_cloud_example.log")) {
        std::cerr << "Failed to open log file." << std::endl;
    }

    // 创建并加载第一个点云
    PointCloud cloud1;
    if (!cloud1.loadFromPLY("data/cloud1.ply")) {
        logger.log(Logger::LogLevel::ERROR, "Failed to load cloud1.ply");
        return -1;
    }

    // 创建并加载第二个点云
    PointCloud cloud2;
    if (!cloud2.loadFromPLY("data/cloud2.ply")) {
        logger.log(Logger::LogLevel::ERROR, "Failed to load cloud2.ply");
        return -1;
    }

    // 点云下采样
    logger.log(Logger::LogLevel::INFO, "Downsampling cloud1...");
    cloud1.downsample(0.05); // 体素大小为 5 cm

    logger.log(Logger::LogLevel::INFO, "Downsampling cloud2...");
    cloud2.downsample(0.05);

    // 点云去噪
    logger.log(Logger::LogLevel::INFO, "Applying Statistical Outlier Removal on cloud1...");
    cloud1.statisticalOutlierRemoval(50, 1.0);

    logger.log(Logger::LogLevel::INFO, "Applying Statistical Outlier Removal on cloud2...");
    cloud2.statisticalOutlierRemoval(50, 1.0);

    // 法向量估计
    logger.log(Logger::LogLevel::INFO, "Estimating normals for cloud1...");
    cloud1.computeNormals(10);

    logger.log(Logger::LogLevel::INFO, "Estimating normals for cloud2...");
    cloud2.computeNormals(10);

    // 点云配准（ICP）
    logger.log(Logger::LogLevel::INFO, "Performing ICP registration of cloud2 to cloud1...");
    Eigen::Matrix4d transformation = cloud2.icp(cloud1, 50, 1e-6);
    if (transformation.isIdentity()) {
        logger.log(Logger::LogLevel::ERROR, "ICP did not converge.");
        return -1;
    }

    // 合并点云
    logger.log(Logger::LogLevel::INFO, "Merging cloud2 into cloud1...");
    cloud1.mergeWith(cloud2, transformation);

    // 点云分割（欧式聚类）
    logger.log(Logger::LogLevel::INFO, "Performing Euclidean Clustering on merged cloud...");
    std::vector<PointCloud> clusters = cloud1.euclideanClustering(0.05, 100, 25000);
    logger.log(Logger::LogLevel::INFO, "Number of clusters found: " + std::to_string(clusters.size()));

    // 打印统计信息
    PointCloudStats stats = cloud1.computeStats();
    std::cout << "Merged PointCloud Stats:\n"
              << "X: min=" << stats.min_x << ", max=" << stats.max_x << ", mean=" << stats.mean_x << ", var=" << stats.variance_x << "\n"
              << "Y: min=" << stats.min_y << ", max=" << stats.max_y << ", mean=" << stats.mean_y << ", var=" << stats.variance_y << "\n"
              << "Z: min=" << stats.min_z << ", max=" << stats.max_z << ", mean=" << stats.mean_z << ", var=" << stats.variance_z << std::endl;

    // 保存处理后的点云
    if (!cloud1.saveToPLY("output/merged_cloud.ply")) {
        logger.log(Logger::LogLevel::ERROR, "Failed to save merged_cloud.ply");
        return -1;
    }

    logger.log(Logger::LogLevel::INFO, "PointCloud processing completed successfully.");
    return 0;
}