/*
 * 提供数学相关的辅助函数和工具，如矩阵运算、几何计算等。
 */
#include "rc_vision/core/math_utils.hpp"
#include "rc_vision/core/logger.hpp"

namespace rc_vision {
    namespace core {

        Eigen::Matrix4d MathUtils::createTransformationMatrix(const Eigen::Vector3d& translation, const Eigen::Quaterniond& rotation) {
            Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity();
            transformation.block<3,3>(0,0) = rotation.toRotationMatrix();
            transformation.block<3,1>(0,3) = translation;
            return transformation;
        }

        double MathUtils::computeReprojectionError(const Eigen::Vector2d& observed, const Eigen::Vector2d& projected) {
            return (observed - projected).norm();
        }

        Eigen::Matrix3d MathUtils::quaternionToRotationMatrix(const Eigen::Quaterniond& q) {
            return q.toRotationMatrix();
        }

        Eigen::Quaterniond MathUtils::rotationMatrixToQuaternion(const Eigen::Matrix3d& R) {
            return Eigen::Quaterniond(R);
        }

        double MathUtils::closestPointsBetweenSegments(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2,
                                                       const Eigen::Vector3d& p3, const Eigen::Vector3d& p4,
                                                       Eigen::Vector3d& closestPoint1, Eigen::Vector3d& closestPoint2) {
            Eigen::Vector3d u = p2 - p1;
            Eigen::Vector3d v = p4 - p3;
            Eigen::Vector3d w = p1 - p3;

            double a = u.dot(u); // always >= 0
            double b = u.dot(v);
            double c = v.dot(v); // always >= 0
            double d = u.dot(w);
            double e = v.dot(w);
            double D = a*c - b*b; // always >= 0
            double sc, tc;

            // compute the line parameters of the two closest points
            if (D < 1e-8) { // the lines are almost parallel
                sc = 0.0;
                tc = (b > c ? d/b : e/c);
            }
            else {
                sc = (b*e - c*d) / D;
                tc = (a*e - b*d) / D;
            }

            // clamp sc and tc to [0,1] to restrict to segments
            sc = std::max(0.0, std::min(1.0, sc));
            tc = std::max(0.0, std::min(1.0, tc));

            closestPoint1 = p1 + sc * u;
            closestPoint2 = p3 + tc * v;

            return (closestPoint1 - closestPoint2).norm();
        }

        double MathUtils::pointToPlaneDistance(const Eigen::Vector3d& point,
                                               const Eigen::Vector3d& plane_point,
                                               const Eigen::Vector3d& plane_normal) {
            return (point - plane_point).dot(plane_normal);
        }

        bool MathUtils::fitPlane(const std::vector<Eigen::Vector3d>& points,
                                 Eigen::Vector3d& plane_point,
                                 Eigen::Vector3d& plane_normal) {
            if (points.size() < 3) {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "Not enough points to fit a plane.");
                return false;
            }

            // Compute the centroid
            Eigen::Vector3d centroid(0, 0, 0);
            for (const auto& p : points) {
                centroid += p;
            }
            centroid /= points.size();
            plane_point = centroid;

            // Compute the covariance matrix
            Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
            for (const auto& p : points) {
                Eigen::Vector3d diff = p - centroid;
                covariance += diff * diff.transpose();
            }

            // Perform Eigen decomposition
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
            if (solver.info() != Eigen::Success) {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "Eigen decomposition failed.");
                return false;
            }

            // The normal of the plane is the eigenvector with the smallest eigenvalue
            plane_normal = solver.eigenvectors().col(0).normalized();
            return true;
        }

        Eigen::Matrix3d MathUtils::eulerAnglesToRotationMatrix(double roll, double pitch, double yaw) {
            Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
            Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitY());
            Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());

            Eigen::Quaterniond q = yawAngle * pitchAngle * rollAngle;
            return q.toRotationMatrix();
        }

        void MathUtils::rotationMatrixToEulerAngles(const Eigen::Matrix3d& R, double& roll, double& pitch, double& yaw) {
            // Assuming the angles are in the range of -pi to pi
            if (R(2,0) < 1) {
                if (R(2,0) > -1) {
                    pitch = std::asin(R(2,0));
                    roll = std::atan2(-R(2,1), R(2,2));
                    yaw = std::atan2(-R(1,0), R(0,0));
                }
                else { // R(2,0) <= -1
                    pitch = -M_PI / 2;
                    roll = -std::atan2(R(1,2), R(1,1));
                    yaw = 0;
                }
            }
            else { // R(2,0) >= 1
                pitch = M_PI / 2;
                roll = std::atan2(R(1,2), R(1,1));
                yaw = 0;
            }
        }

    } // namespace core
} // namespace rc_vision