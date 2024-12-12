#include "rc_vision/core/point_cloud.hpp"
#include "rc_vision/core/logger.hpp"
#include <gtest/gtest.h>
#include <cmath>

using namespace rc_vision::core;

// 测试添加点和大小
TEST(PointCloudTest, AddPointAndSize) {
PointCloud cloud;
EXPECT_EQ(cloud.size(), 0);

cloud.addPoint(1.0, 2.0, 3.0);
EXPECT_EQ(cloud.size(), 1);

Eigen::Vector3d color(255, 0, 0);
cloud.addPoint(4.0, 5.0, 6.0, color);
EXPECT_EQ(cloud.size(), 2);
}

// 测试计算质心
TEST(PointCloudTest, ComputeCentroid) {
PointCloud cloud;
cloud.addPoint(1.0, 2.0, 3.0);
cloud.addPoint(4.0, 5.0, 6.0);
cloud.addPoint(7.0, 8.0, 9.0);

Eigen::Vector3d centroid = cloud.computeCentroid();
EXPECT_DOUBLE_EQ(centroid.x(), 4.0);
EXPECT_DOUBLE_EQ(centroid.y(), 5.0);
EXPECT_DOUBLE_EQ(centroid.z(), 6.0);
}

// 测试点云变换
TEST(PointCloudTest, TransformPointCloud) {
PointCloud cloud;
cloud.addPoint(1.0, 0.0, 0.0);
cloud.addPoint(0.0, 1.0, 0.0);
cloud.addPoint(0.0, 0.0, 1.0);

// 旋转 90 度绕 Z 轴
Eigen::Matrix4d rotation = Eigen::Matrix4d::Identity();
double angle = M_PI / 2; // 90 度
rotation(0, 0) = std::cos(angle);
rotation(0, 1) = -std::sin(angle);
rotation(1, 0) = std::sin(angle);
rotation(1, 1) = std::cos(angle);

cloud.transform(rotation);

// 检查变换后的点
const auto& points = cloud.getPoints();
EXPECT_NEAR(points[0].x, 0.0, 1e-6);
EXPECT_NEAR(points[0].y, 1.0, 1e-6);
EXPECT_NEAR(points[0].z, 0.0, 1e-6);

EXPECT_NEAR(points[1].x, -1.0, 1e-6);
EXPECT_NEAR(points[1].y, 0.0, 1e-6);
EXPECT_NEAR(points[1].z, 0.0, 1e-6);

EXPECT_NEAR(points[2].x, 0.0, 1e-6);
EXPECT_NEAR(points[2].y, 0.0, 1e-6);
EXPECT_NEAR(points[2].z, 1.0, 1e-6);
}

// 测试下采样
TEST(PointCloudTest, Downsample) {
PointCloud cloud;
// 创建一个立方体网格
for (int x = 0; x < 10; x += 2) {
for (int y = 0; y < 10; y += 2) {
for (int z = 0; z < 10; z += 2) {
cloud.addPoint(x * 0.1, y * 0.1, z * 0.1);
}
}
}
size_t original_size = cloud.size();
EXPECT_EQ(original_size, 125); // 5 * 5 * 5

cloud.downsample(0.2); // 体素大小为 0.2
size_t downsampled_size = cloud.size();
EXPECT_LT(downsampled_size, original_size);
// 具体大小取决于点的分布
}

// 测试统计滤波
TEST(PointCloudTest, StatisticalOutlierRemoval) {
PointCloud cloud;
// 添加正常点
for (int i = 0; i < 100; ++i) {
cloud.addPoint(i * 0.1, 0.0, 0.0);
}
// 添加离群点
cloud.addPoint(100.0, 100.0, 100.0);
cloud.addPoint(-100.0, -100.0, -100.0);

EXPECT_EQ(cloud.size(), 102);

cloud.statisticalOutlierRemoval(50, 1.0);
EXPECT_EQ(cloud.size(), 100);
}

// 测试法向量估计
TEST(PointCloudTest, ComputeNormals) {
PointCloud cloud;
// 创建一个平面
for (double x = -1.0; x <= 1.0; x += 0.5) {
for (double y = -1.0; y <= 1.0; y += 0.5) {
cloud.addPoint(x, y, 0.0);
}
}

cloud.computeNormals(3);

const auto& points = cloud.getPoints();
for (const auto& point : points) {
ASSERT_TRUE(point.normal.has_value());
EXPECT_NEAR(point.normal.value().x(), 0.0, 1e-6);
EXPECT_NEAR(point.normal.value().y(), 0.0, 1e-6);
EXPECT_NEAR(point.normal.value().z(), 1.0, 1e-6);
}
}

// 测试点云加载与保存
TEST(PointCloudTest, LoadAndSavePLY) {
PointCloud cloud;
cloud.addPoint(1.0, 2.0, 3.0);
Eigen::Vector3d color(255, 0, 0);
cloud.addPoint(4.0, 5.0, 6.0, color);

EXPECT_TRUE(cloud.saveToPLY("test_output.ply"));

PointCloud loaded_cloud;
EXPECT_TRUE(loaded_cloud.loadFromPLY("test_output.ply"));
EXPECT_EQ(loaded_cloud.size(), 2);

const auto& points = loaded_cloud.getPoints();
EXPECT_DOUBLE_EQ(points[0].x, 1.0);
EXPECT_DOUBLE_EQ(points[0].y, 2.0);
EXPECT_DOUBLE_EQ(points[0].z, 3.0);
EXPECT_FALSE(points[0].color.has_value());

EXPECT_DOUBLE_EQ(points[1].x, 4.0);
EXPECT_DOUBLE_EQ(points[1].y, 5.0);
EXPECT_DOUBLE_EQ(points[1].z, 6.0);
ASSERT_TRUE(points[1].color.has_value());
EXPECT_DOUBLE_EQ(points[1].color.value().x(), 255.0);
EXPECT_DOUBLE_EQ(points[1].color.value().y(), 0.0);
EXPECT_DOUBLE_EQ(points[1].color.value().z(), 0.0);
}

// 测试点云配准
TEST(PointCloudTest, ICPRegistration) {
PointCloud cloud1;
// 创建一个简单的点云
cloud1.addPoint(0.0, 0.0, 0.0);
cloud1.addPoint(1.0, 0.0, 0.0);
cloud1.addPoint(0.0, 1.0, 0.0);

PointCloud cloud2;
// 创建一个旋转90度绕Z轴的点云
cloud2.addPoint(0.0, 0.0, 0.0);
cloud2.addPoint(0.0, 1.0, 0.0);
cloud2.addPoint(-1.0, 0.0, 0.0);

Eigen::Matrix4d transformation = cloud2.icp(cloud1, 50, 1e-6);
// 理论上的变换矩阵
Eigen::Matrix4d expected = Eigen::Matrix4d::Identity();
expected(0,0) = 0.0;
expected(0,1) = -1.0;
expected(1,0) = 1.0;
expected(1,1) = 0.0;

// 由于 ICP 可能会有微小的误差，使用近似比较
for (int i = 0; i < 4; ++i) {
for (int j = 0; j < 4; ++j) {
EXPECT_NEAR(transformation(i, j), expected(i, j), 1e-3);
}
}
}

// 测试点云分割
TEST(PointCloudTest, EuclideanClustering) {
PointCloud cloud;
// 创建两个不同的簇
// 簇1
for (double x = 0.0; x < 1.0; x += 0.1) {
for (double y = 0.0; y < 1.0; y += 0.1) {
cloud.addPoint(x, y, 0.0);
}
}
// 簇2
for (double x = 5.0; x < 6.0; x += 0.1) {
for (double y = 5.0; y < 6.0; y += 0.1) {
cloud.addPoint(x, y, 0.0);
}
}

std::vector<PointCloud> clusters = cloud.euclideanClustering(0.5, 10, 1000);
EXPECT_EQ(clusters.size(), 2);

// 验证每个簇的大小
EXPECT_GE(clusters[0].size(), 10);
EXPECT_GE(clusters[1].size(), 10);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}