#include "rc_vision/core/math_utils.hpp"
#include "rc_vision/core/logger.hpp"

namespace rc_vision {
    namespace core {

/**
 * @brief 创建齐次变换矩阵
 *
 * 该函数根据给定的平移向量和旋转四元数创建一个4x4的齐次变换矩阵。
 *
 * @param translation 平移向量
 * @param rotation 四元数表示的旋转
 * @return Eigen::Matrix4d 齐次变换矩阵
 */
        Eigen::Matrix4d MathUtils::createTransformationMatrix(const Eigen::Vector3d& translation, const Eigen::Quaterniond& rotation) {
            Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity();
            transformation.block<3,3>(0,0) = rotation.toRotationMatrix();
            transformation.block<3,1>(0,3) = translation;
            return transformation;
        }

/**
 * @brief 计算重投影误差
 *
 * 该函数计算观测到的2D点与投影后的2D点之间的欧氏距离作为重投影误差。
 *
 * @param observed 观测到的2D点
 * @param projected 投影后的2D点
 * @return double 重投影误差（欧氏距离）
 */
        double MathUtils::computeReprojectionError(const Eigen::Vector2d& observed, const Eigen::Vector2d& projected) {
            return (observed - projected).norm();
        }

/**
 * @brief 将四元数转换为旋转矩阵
 *
 * 该函数将给定的四元数转换为对应的3x3旋转矩阵。
 *
 * @param q 四元数
 * @return Eigen::Matrix3d 旋转矩阵
 */
        Eigen::Matrix3d MathUtils::quaternionToRotationMatrix(const Eigen::Quaterniond& q) {
            return q.toRotationMatrix();
        }

/**
 * @brief 将旋转矩阵转换为四元数
 *
 * 该函数将给定的3x3旋转矩阵转换为对应的四元数。
 *
 * @param R 旋转矩阵
 * @return Eigen::Quaterniond 四元数
 */
        Eigen::Quaterniond MathUtils::rotationMatrixToQuaternion(const Eigen::Matrix3d& R) {
            Eigen::Quaterniond q(R);
            return q;
        }

/**
 * @brief 计算两条线段的最近点对
 *
 * 该函数计算两条3D线段之间的最近点对及其最小距离。
 *
 * @param p1 起点1
 * @param p2 终点1
 * @param p3 起点2
 * @param p4 终点2
 * @param closestPoint1 最近点对1
 * @param closestPoint2 最近点对2
 * @return double 两条线段之间的最小距离
 */
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
                tc = (b > c ? d/b : e/c); // use the largest denominator
            }
            else {
                sc = (b*e - c*d) / D;
                tc = (a*e - b*d) / D;
            }

            // clamp sc to [0,1] to stay within the segments
            sc = std::min(std::max(sc, 0.0), 1.0);
            tc = std::min(std::max(tc, 0.0), 1.0);

            closestPoint1 = p1 + sc * u;
            closestPoint2 = p3 + tc * v;
            return (closestPoint1 - closestPoint2).norm();
        }

/**
 * @brief 计算点到平面的距离
 *
 * 该函数计算一个3D点到给定平面的距离。平面的定义通过平面上的一点和法向量给出。
 *
 * @param point 点坐标
 * @param plane_point 平面上的一点
 * @param plane_normal 平面的法向量（需归一化）
 * @return double 点到平面的距离
 */
        double MathUtils::pointToPlaneDistance(const Eigen::Vector3d& point,
                                               const Eigen::Vector3d& plane_point,
                                               const Eigen::Vector3d& plane_normal) {
            return (point - plane_point).dot(plane_normal);
        }

/**
 * @brief 最小二乘法拟合平面
 *
 * 该函数使用最小二乘法拟合给定的点云，计算拟合平面上的一点和法向量。
 *
 * @param points 输入的点云
 * @param plane_point 拟合平面上的一点
 * @param plane_normal 拟合平面的法向量（归一化）
 * @return bool 拟合是否成功
 */
        bool MathUtils::fitPlane(const std::vector<Eigen::Vector3d>& points,
                                 Eigen::Vector3d& plane_point,
                                 Eigen::Vector3d& plane_normal) {
            if (points.size() < 3) {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "Not enough points to fit a plane.");
                return false;
            }

            // Compute the centroid
            Eigen::Vector3d centroid(0, 0, 0);
            for(const auto& p : points) {
                centroid += p;
            }
            centroid /= points.size();

            // Compute the covariance matrix
            Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
            for(const auto& p : points) {
                Eigen::Vector3d centered = p - centroid;
                cov += centered * centered.transpose();
            }

            // Perform Eigen decomposition
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
            if(solver.info() != Eigen::Success) {
                Logger::getInstance().log(Logger::LogLevel::ERROR, "Eigen decomposition failed.");
                return false;
            }

            // The normal of the plane is the eigenvector corresponding to the smallest eigenvalue
            plane_normal = solver.eigenvectors().col(0).normalized();
            plane_point = centroid;

            return true;
        }

/**
 * @brief 将欧拉角转换为旋转矩阵
 *
 * 该函数根据给定的滚转（roll）、俯仰（pitch）和偏航（yaw）欧拉角生成对应的3x3旋转矩阵。
 *
 * @param roll 绕X轴的旋转（弧度）
 * @param pitch 绕Y轴的旋转（弧度）
 * @param yaw 绕Z轴的旋转（弧度）
 * @return Eigen::Matrix3d 旋转矩阵
 */
        Eigen::Matrix3d MathUtils::eulerAnglesToRotationMatrix(double roll, double pitch, double yaw) {
            Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
            Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitY());
            Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());

            Eigen::Quaterniond q = yawAngle * pitchAngle * rollAngle;
            return q.toRotationMatrix();
        }

/**
 * @brief 将旋转矩阵转换为欧拉角
 *
 * 该函数将给定的3x3旋转矩阵转换为对应的滚转（roll）、俯仰（pitch）和偏航（yaw）欧拉角。
 *
 * @param R 旋转矩阵
 * @param roll 输出的绕X轴的旋转（弧度）
 * @param pitch 输出的绕Y轴的旋转（弧度）
 * @param yaw 输出的绕Z轴的旋转（弧度）
 */
        void MathUtils::rotationMatrixToEulerAngles(const Eigen::Matrix3d& R, double& roll, double& pitch, double& yaw) {
            // 使用 Eigen 提供的方法提取欧拉角（ZYX顺序）
            Eigen::Vector3d euler = R.eulerAngles(2, 1, 0); // yaw, pitch, roll
            yaw = euler[0];
            pitch = euler[1];
            roll = euler[2];
        }

    } // namespace core
} // namespace rc_vision
