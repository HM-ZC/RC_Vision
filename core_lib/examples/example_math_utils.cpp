#include <iostream>
#include <vector>
#include "rc_vision/core/math_utils.hpp"
#include "rc_vision/core/logger.hpp"
#include <Eigen/Dense>

using namespace rc_vision::core;

int main() {
    Logger::getInstance().setLogLevel(Logger::LogLevel::INFO);

    // 1. createTransformationMatrix
    Eigen::Vector3d translation(1.0, 2.0, 3.0);
    Eigen::Quaterniond rotation(Eigen::AngleAxisd(M_PI/4, Eigen::Vector3d::UnitZ()));
    Eigen::Matrix4d T = MathUtils::createTransformationMatrix(translation, rotation);
    LOG_INFO("Transformation Matrix:\n" + std::to_string(T(0,0)) + " ...");
    // 为简洁，此处仅打印部分信息，可根据需要打印整个矩阵。

    // 2. computeReprojectionError
    Eigen::Vector2d observed(10.0, 15.0);
    Eigen::Vector2d projected(10.5, 14.8);
    double error = MathUtils::computeReprojectionError(observed, projected);
    LOG_INFO("Reprojection Error: " + std::to_string(error));

    // 3. quaternionToRotationMatrix & rotationMatrixToQuaternion
    Eigen::Matrix3d R = MathUtils::quaternionToRotationMatrix(rotation);
    Eigen::Quaterniond q_back = MathUtils::rotationMatrixToQuaternion(R);
    LOG_INFO("Original Quaternion: " + std::to_string(rotation.w()) + ", "
             + std::to_string(rotation.x()) + ", "
             + std::to_string(rotation.y()) + ", "
             + std::to_string(rotation.z()));
    LOG_INFO("Recovered Quaternion: " + std::to_string(q_back.w()) + ", "
             + std::to_string(q_back.x()) + ", "
             + std::to_string(q_back.y()) + ", "
             + std::to_string(q_back.z()));

    // 4. closestPointsBetweenSegments
    Eigen::Vector3d p1(0,0,0), p2(1,0,0), p3(0,1,0), p4(1,1,0);
    Eigen::Vector3d cp1, cp2;
    double dist = MathUtils::closestPointsBetweenSegments(p1, p2, p3, p4, cp1, cp2);
    LOG_INFO("Closest distance between segments: " + std::to_string(dist));
    LOG_INFO("Closest Points: (" + std::to_string(cp1.x()) + "," + std::to_string(cp1.y()) + "," + std::to_string(cp1.z()) + ") "
                                                                                                                             "and (" + std::to_string(cp2.x()) + "," + std::to_string(cp2.y()) + "," + std::to_string(cp2.z()) + ")");

    // 5. pointToPlaneDistance
    Eigen::Vector3d plane_pt(0,0,0), plane_normal(0,0,1);
    double plane_dist = MathUtils::pointToPlaneDistance(Eigen::Vector3d(0,0,5), plane_pt, plane_normal);
    LOG_INFO("Point to plane distance: " + std::to_string(plane_dist));

    // 6. fitPlane
    std::vector<Eigen::Vector3d> points = {
            Eigen::Vector3d(0,0,0),
            Eigen::Vector3d(1,0,0),
            Eigen::Vector3d(0,1,0),
            Eigen::Vector3d(1,1,0),
            Eigen::Vector3d(0.5,0.5,0)
    };
    Eigen::Vector3d fitted_pt, fitted_normal;
    bool success = MathUtils::fitPlane(points, fitted_pt, fitted_normal);
    LOG_INFO(std::string("Plane fit success: ") + (success ? "true" : "false"));
    if(success) {
        LOG_INFO("Fitted plane point: " + std::to_string(fitted_pt.x()) + ", "
                 + std::to_string(fitted_pt.y()) + ", "
                 + std::to_string(fitted_pt.z()));
        LOG_INFO("Fitted plane normal: " + std::to_string(fitted_normal.x()) + ", "
                 + std::to_string(fitted_normal.y()) + ", "
                 + std::to_string(fitted_normal.z()));
    }

    // 7. eulerAnglesToRotationMatrix & rotationMatrixToEulerAngles
    double roll = M_PI/6, pitch = M_PI/4, yaw = M_PI/3;
    Eigen::Matrix3d R_euler = MathUtils::eulerAnglesToRotationMatrix(roll, pitch, yaw);
    double r2, p2, y2;
    MathUtils::rotationMatrixToEulerAngles(R_euler, r2, p2, y2);
    LOG_INFO("Original Euler Angles: roll=" + std::to_string(roll)
             + " pitch=" + std::to_string(pitch)
             + " yaw=" + std::to_string(yaw));
    LOG_INFO("Recovered Euler Angles: roll=" + std::to_string(r2)
             + " pitch=" + std::to_string(p2)
             + " yaw=" + std::to_string(y2));

    return 0;
}