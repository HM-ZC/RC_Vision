#include "rc_vision/core/filters.hpp"
#include "rc_vision/core/logger.hpp"
#include <iostream>
#include <cmath>

using namespace rc_vision::core::filters;
using namespace rc_vision::core;

int main() {
    // 初始化日志
    Logger& logger = Logger::getInstance();
    logger.setLogLevel(Logger::LogLevel::DEBUG);
    logger.setLogFile("filters_example.log");

    // --------------- 卡尔曼滤波器示例 ---------------
    logger.log(Logger::LogLevel::INFO, "Starting Kalman Filter Example");

    // 定义状态转移矩阵 A (例如，2D 运动模型: [x, y, vx, vy]^T)
    Eigen::Matrix<double, 4, 4> A;
    A << 1, 0, 1, 0,
            0, 1, 0, 1,
            0, 0, 1, 0,
            0, 0, 0, 1;

    // 过程噪声协方差 Q
    Eigen::Matrix<double, 4, 4> Q = Eigen::Matrix<double, 4, 4>::Identity() * 0.01;

    // 测量矩阵 H (假设我们只测量位置)
    Eigen::Matrix<double, 2, 4> H;
    H << 1, 0, 0, 0,
            0, 1, 0, 0;

    // 测量噪声协方差 R
    Eigen::Matrix<double, 2, 2> R = Eigen::Matrix<double, 2, 2>::Identity() * 0.1;

    // 初始状态
    Eigen::Matrix<double, 4, 1> initial_state;
    initial_state << 0, 0, 1, 1; // 初始位置(0,0), 速度(1,1)

    // 初始协方差
    Eigen::Matrix<double, 4, 4> initial_covariance = Eigen::Matrix<double, 4, 4>::Identity();

    // 创建卡尔曼滤波器实例
    KalmanFilter<double, 4, 2> kf;
    kf.init(A, Q, H, R, initial_state, initial_covariance);

    // 模拟测量数据
    std::vector<Eigen::Matrix<double, 2, 1>> measurements = {
            {1.0, 1.0},
            {2.0, 2.0},
            {3.0, 3.0},
            {4.0, 4.0},
            {5.0, 5.0}
    };

    for (size_t i = 0; i < measurements.size(); ++i) {
        kf.predict();
        logger.log(Logger::LogLevel::DEBUG, "Kalman Filter Prediction Step");

        kf.update(measurements[i]);
        logger.log(Logger::LogLevel::DEBUG, "Kalman Filter Update Step with Measurement: (" +
                                            std::to_string(measurements) + ", " +
                                            std::to_string(measurements) + ")");

        Eigen::Matrix<double, 4, 1> state = kf.getState();
        logger.log(Logger::LogLevel::INFO, "Kalman Filter Estimated State: (" +
                                           std::to_string(state(0)) + ", " +
                                           std::to_string(state(1)) + ", " +
                                           std::to_string(state(2)) + ", " +
                                           std::to_string(state(3)) + ")");
    }

    // --------------- 移动平均滤波器示例 ---------------
    logger.log(Logger::LogLevel::INFO, "Starting Moving Average Filter Example");

    MovingAverageFilter<double, 2> ma_filter(3); // 窗口大小为3

    for (size_t i = 0; i < measurements.size(); ++i) {
        ma_filter.addMeasurement(measurements[i]);
        Eigen::Matrix<double, 2, 1> ma_avg = ma_filter.getAverage();
        logger.log(Logger::LogLevel::INFO, "Moving Average Filter Average: (" +
                                           std::to_string(ma_avg(0)) + ", " +
                                           std::to_string(ma_avg(1)) + ")");
    }

    // --------------- 扩展卡尔曼滤波器 (EKF) 示例 ---------------
    logger.log(Logger::LogLevel::INFO, "Starting EKF Example");

    // 定义非线性状态转移函数 (例如，圆周运动)
    auto stateTransition = [](const Eigen::Matrix<double, 4, 1>& state) -> Eigen::Matrix<double, 4, 1> {
        double x = state(0);
        double y = state(1);
        double vx = state(2);
        double vy = state(3);

        double dt = 1.0; // 时间步长
        double radius = 10.0;
        double omega = vx / radius; // 角速度

        double new_x = x + vx * dt;
        double new_y = y + vy * dt;
        double new_vx = vx - omega * vy * dt;
        double new_vy = vy + omega * vx * dt;

        Eigen::Matrix<double, 4, 1> new_state;
        new_state << new_x, new_y, new_vx, new_vy;
        return new_state;
    };

    // 定义测量函数
    auto measurementFunction = [](const Eigen::Matrix<double, 4, 1>& state) -> Eigen::Matrix<double, 2, 1> {
        return state.head<2>();
    };

    // 定义雅可比矩阵函数
    auto F_jacobian = [](const Eigen::Matrix<double, 4, 1>& state) -> Eigen::Matrix<double, 4, 4> {
        double vx = state(2);
        double vy = state(3);
        double dt = 1.0;
        double radius = 10.0;
        double omega = vx / radius;

        Eigen::Matrix<double, 4, 4> F = Eigen::Matrix<double, 4, 4>::Identity();
        F(0, 2) = dt;
        F(1, 3) = dt;
        F(2, 1) = -omega * dt;
        F(3, 0) = omega * dt;
        return F;
    };

    auto H_jacobian = [](const Eigen::Matrix<double, 4, 1>& state) -> Eigen::Matrix<double, 2, 4> {
        Eigen::Matrix<double, 2, 4> H = Eigen::Matrix<double, 2, 4>::Zero();
        H(0, 0) = 1.0;
        H(1, 1) = 1.0;
        return H;
    };

    // 定义过程噪声协方差 Q
    Eigen::Matrix<double, 4, 4> Q_ekf = Eigen::Matrix<double, 4, 4>::Identity() * 0.01;

    // 定义测量噪声协方差 R
    Eigen::Matrix<double, 2, 2> R_ekf = Eigen::Matrix<double, 2, 2>::Identity() * 0.1;

    // 初始状态
    Eigen::Matrix<double, 4, 1> initial_state_ekf;
    initial_state_ekf << 0, 0, 1, 1;

    // 初始协方差
    Eigen::Matrix<double, 4, 4> initial_covariance_ekf = Eigen::Matrix<double, 4, 4>::Identity();

    // 创建 EKF 实例
    EKF<double, 4, 2> ekf;
    ekf.init(stateTransition, measurementFunction, F_jacobian, H_jacobian, Q_ekf, R_ekf, initial_state_ekf, initial_covariance_ekf);

    for (size_t i = 0; i < measurements.size(); ++i) {
        ekf.predict();
        logger.log(Logger::LogLevel::DEBUG, "EKF Prediction Step");

        ekf.update(measurements[i]);
        logger.log(Logger::LogLevel::DEBUG, "EKF Update Step with Measurement: (" +
                                            std::to_string(measurements) + ", " +
                                            std::to_string(measurements) + ")");

        Eigen::Matrix<double, 4, 1> ekf_state = ekf.getState();
        logger.log(Logger::LogLevel::INFO, "EKF Estimated State: (" +
                                           std::to_string(ekf_state(0)) + ", " +
                                           std::to_string(ekf_state(1)) + ", " +
                                           std::to_string(ekf_state(2)) + ", " +
                                           std::to_string(ekf_state(3)) + ")");
    }

    // --------------- 无迹卡尔曼滤波器 (UKF) 示例 ---------------
    logger.log(Logger::LogLevel::INFO, "Starting UKF Example");

    // 定义状态转移函数 (非线性)
    auto ukf_stateTransition = [](const Eigen::Matrix<double, 4, 1>& state) -> Eigen::Matrix<double, 4, 1> {
        double x = state(0);
        double y = state(1);
        double vx = state(2);
        double vy = state(3);

        double dt = 1.0; // 时间步长
        double radius = 10.0;
        double omega = vx / radius; // 角速度

        double new_x = x + vx * dt;
        double new_y = y + vy * dt;
        double new_vx = vx - omega * vy * dt;
        double new_vy = vy + omega * vx * dt;

        Eigen::Matrix<double, 4, 1> new_state;
        new_state << new_x, new_y, new_vx, new_vy;
        return new_state;
    };

    // 定义测量函数
    auto ukf_measurementFunction = [](const Eigen::Matrix<double, 4, 1>& state) -> Eigen::Matrix<double, 2, 1> {
        return state.head<2>();
    };

    // 定义雅可比矩阵函数
    auto ukf_F_jacobian = [](const Eigen::Matrix<double, 4, 1>& state) -> Eigen::Matrix<double, 4, 4> {
        double vx = state(2);
        double vy = state(3);
        double dt = 1.0;
        double radius = 10.0;
        double omega = vx / radius;

        Eigen::Matrix<double, 4, 4> F = Eigen::Matrix<double, 4, 4>::Identity();
        F(0, 2) = dt;
        F(1, 3) = dt;
        F(2, 1) = -omega * dt;
        F(3, 0) = omega * dt;
        return F;
    };

    auto ukf_H_jacobian = [](const Eigen::Matrix<double, 4, 1>& state) -> Eigen::Matrix<double, 2, 4> {
        Eigen::Matrix<double, 2, 4> H = Eigen::Matrix<double, 2, 4>::Zero();
        H(0, 0) = 1.0;
        H(1, 1) = 1.0;
        return H;
    };

    // 定义过程噪声协方差 Q
    Eigen::Matrix<double, 4, 4> Q_ukf = Eigen::Matrix<double, 4, 4>::Identity() * 0.01;

    // 定义测量噪声协方差 R
    Eigen::Matrix<double, 2, 2> R_ukf = Eigen::Matrix<double, 2, 2>::Identity() * 0.1;

    // 初始状态
    Eigen::Matrix<double, 4, 1> initial_state_ukf;
    initial_state_ukf << 0, 0, 1, 1;

    // 初始协方差
    Eigen::Matrix<double, 4, 4> initial_covariance_ukf = Eigen::Matrix<double, 4, 4>::Identity();

    // 创建 UKF 实例
    UKF<double, 4, 2> ukf;
    ukf.init(ukf_stateTransition, ukf_measurementFunction, ukf_F_jacobian, ukf_H_jacobian, Q_ukf, R_ukf, initial_state_ukf, initial_covariance_ukf);

    for (size_t i = 0; i < measurements.size(); ++i) {
        ukf.predict();
        logger.log(Logger::LogLevel::DEBUG, "UKF Prediction Step");

        ukf.update(measurements[i]);
        logger.log(Logger::LogLevel::DEBUG, "UKF Update Step with Measurement: (" +
                                            std::to_string(measurements) + ", " +
                                            std::to_string(measurements) + ")");

        Eigen::Matrix<double, 4, 1> ukf_state = ukf.getState();
        logger.log(Logger::LogLevel::INFO, "UKF Estimated State: (" +
                                           std::to_string(ukf_state(0)) + ", " +
                                           std::to_string(ukf_state(1)) + ", " +
                                           std::to_string(ukf_state(2)) + ", " +
                                           std::to_string(ukf_state(3)) + ")");
    }

    // --------------- 粒子滤波器示例 ---------------
    logger.log(Logger::LogLevel::INFO, "Starting Particle Filter Example");

    // 定义状态转移函数 (线性)
    auto pf_stateTransition = [](const Eigen::Matrix<double, 4, 1>& state) -> Eigen::Matrix<double, 4, 1> {
        // 简单的线性状态转移
        double x = state(0) + state(2);
        double y = state(1) + state(3);
        double vx = state(2);
        double vy = state(3);
        Eigen::Matrix<double, 4, 1> new_state;
        new_state << x, y, vx, vy;
        return new_state;
    };

    // 定义测量函数
    auto pf_measurementFunction = [](const Eigen::Matrix<double, 4, 1>& state) -> Eigen::Matrix<double, 2, 1> {
        return state.head<2>();
    };

    // 定义测量噪声协方差 R
    Eigen::Matrix<double, 2, 2> R_pf = Eigen::Matrix<double, 2, 2>::Identity() * 0.1;

    // 初始状态
    Eigen::Matrix<double, 4, 1> initial_state_pf;
    initial_state_pf << 0, 0, 1, 1;

    // 初始协方差
    Eigen::Matrix<double, 4, 4> initial_covariance_pf = Eigen::Matrix<double, 4, 4>::Identity();

    // 创建粒子滤波器实例
    ParticleFilter<double, 4, 2> pf(100); // 100 个粒子
    pf.init(initial_state_pf, initial_covariance_pf);

    for (size_t i = 0; i < measurements.size(); ++i) {
        pf.predict(pf_stateTransition, Eigen::Matrix<double, 4, 4>::Identity() * 0.01);
        logger.log(Logger::LogLevel::DEBUG, "Particle Filter Prediction Step");

        pf.update(measurements[i], pf_measurementFunction, R_pf);
        logger.log(Logger::LogLevel::DEBUG, "Particle Filter Update Step with Measurement: (" +
                                            std::to_string(measurements) + ", " +
                                            std::to_string(measurements) + ")");

        Eigen::Matrix<double, 4, 1> pf_state = pf.getState();
        logger.log(Logger::LogLevel::INFO, "Particle Filter Estimated State: (" +
                                           std::to_string(pf_state(0)) + ", " +
                                           std::to_string(pf_state(1)) + ", " +
                                           std::to_string(pf_state(2)) + ", " +
                                           std::to_string(pf_state(3)) + ")");
    }

    return 0;
}