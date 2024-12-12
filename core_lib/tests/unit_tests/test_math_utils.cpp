#include <gtest/gtest.h>
#include "rc_vision/core/math_utils.hpp"
#include <Eigen/Dense>
#include <cmath>

using namespace rc_vision::core;

TEST(MathUtilsTest, CreateTransformationMatrix) {
Eigen::Vector3d t(1.0, 2.0, 3.0);
Eigen::Quaterniond q(Eigen::AngleAxisd(M_PI/4, Eigen::Vector3d::UnitZ()));
Eigen::Matrix4d T = MathUtils::createTransformationMatrix(t, q);

// Check translation
EXPECT_NEAR(T(0,3), 1.0, 1e-9);
EXPECT_NEAR(T(1,3), 2.0, 1e-9);
EXPECT_NEAR(T(2,3), 3.0, 1e-9);

// Check rotation
Eigen::Matrix3d R = q.toRotationMatrix();
for (int i = 0; i < 3; i++) {
for (int j = 0; j < 3; j++) {
EXPECT_NEAR(T(i,j), R(i,j), 1e-9);
}
}
}

TEST(MathUtilsTest, ComputeReprojectionError) {
Eigen::Vector2d observed(10.0, 15.0);
Eigen::Vector2d projected(10.5, 14.8);
double error = MathUtils::computeReprojectionError(observed, projected);
double expected = std::sqrt((10.0-10.5)*(10.0-10.5) + (15.0-14.8)*(15.0-14.8));
EXPECT_NEAR(error, expected, 1e-9);
}

TEST(MathUtilsTest, QuaternionRotationMatrixConversion) {
Eigen::Quaterniond q(Eigen::AngleAxisd(M_PI/4, Eigen::Vector3d::UnitZ()));
Eigen::Matrix3d R = MathUtils::quaternionToRotationMatrix(q);
Eigen::Quaterniond q_back = MathUtils::rotationMatrixToQuaternion(R);

EXPECT_NEAR(q.w(), q_back.w(), 1e-9);
EXPECT_NEAR(q.x(), q_back.x(), 1e-9);
EXPECT_NEAR(q.y(), q_back.y(), 1e-9);
EXPECT_NEAR(q.z(), q_back.z(), 1e-9);
}

TEST(MathUtilsTest, ClosestPointsBetweenSegments) {
Eigen::Vector3d p1(0,0,0), p2(1,0,0), p3(0,1,0), p4(1,1,0);
Eigen::Vector3d cp1, cp2;
double dist = MathUtils::closestPointsBetweenSegments(p1, p2, p3, p4, cp1, cp2);

// 两条线段是平行的，相对位置固定
EXPECT_NEAR(dist, 1.0, 1e-9);
// cp1应在p1-p2上，且x在[0,1]范围内
EXPECT_GE(cp1.x(), 0.0);
EXPECT_LE(cp1.x(), 1.0);
// cp2应在p3-p4上，且x在[0,1]范围内，y应为1
EXPECT_GE(cp2.x(), 0.0);
EXPECT_LE(cp2.x(), 1.0);
EXPECT_NEAR(cp2.y(), 1.0, 1e-9);
}

TEST(MathUtilsTest, PointToPlaneDistance) {
Eigen::Vector3d pt(0,0,5), plane_pt(0,0,0), plane_normal(0,0,1);
double dist = MathUtils::pointToPlaneDistance(pt, plane_pt, plane_normal);
EXPECT_NEAR(dist, 5.0, 1e-9);
}

TEST(MathUtilsTest, FitPlane) {
std::vector<Eigen::Vector3d> points = {
        Eigen::Vector3d(0,0,0),
        Eigen::Vector3d(1,0,0),
        Eigen::Vector3d(0,1,0),
        Eigen::Vector3d(1,1,0),
        Eigen::Vector3d(0.5,0.5,0)
};
Eigen::Vector3d plane_point, plane_normal;
bool success = MathUtils::fitPlane(points, plane_point, plane_normal);
EXPECT_TRUE(success);
// 平面应为z=0平面，normal接近(0,0,1)或(0,0,-1)
EXPECT_NEAR(plane_point.z(), 0.0, 1e-9);
double dot = plane_normal.dot(Eigen::Vector3d(0,0,1));
EXPECT_NEAR(std::fabs(dot), 1.0, 1e-9);
}

TEST(MathUtilsTest, EulerAnglesConversion) {
double roll = M_PI/6, pitch = M_PI/4, yaw = M_PI/3;
Eigen::Matrix3d R = MathUtils::eulerAnglesToRotationMatrix(roll, pitch, yaw);
double r2, p2, y2;
MathUtils::rotationMatrixToEulerAngles(R, r2, p2, y2);

// 由于浮点误差，允许一定范围内的偏差
EXPECT_NEAR(roll, r2, 1e-9);
EXPECT_NEAR(pitch, p2, 1e-9);
EXPECT_NEAR(yaw, y2, 1e-9);
}