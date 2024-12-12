#include "rc_vision/core/filters.hpp"
#include "rc_vision/core/logger.hpp"
#include <gtest/gtest.h>
#include <Eigen/Dense>

// 使用命名空间以简化代码
using namespace rc_vision::core::filters;
using namespace rc_vision::core;

// 测试卡尔曼滤波器的初始化和基本功能
TEST(KalmanFilterTest, Initialization) {
// 定义状态转移矩阵 A
Eigen::Matrix<double, 4, 4> A;
A << 1, 0, 1, 0,
0, 1, 0, 1,
0, 0, 1, 0,
0, 0, 0, 1;

// 过程噪声协方差 Q
Eigen::Matrix<double, 4, 4> Q = Eigen::Matrix<double, 4, 4>::Identity() * 0.01;

// 测量矩阵 H
Eigen::Matrix<double, 2, 4> H;
H << 1, 0, 0, 0,
0, 1, 0, 0;

// 测量噪声协方差 R
Eigen::Matrix<double, 2, 2> R = Eigen::Matrix<double, 2, 2>::Identity() * 0.1;

// 初始状态
Eigen::Matrix<double, 4, 1> initial_state;
initial_state << 0, 0, 1, 1;

// 初始协方差
Eigen::Matrix<double, 4, 4> initial_covariance = Eigen::Matrix<double, 4, 4>::Identity();

// 创建卡尔曼滤波器实例
KalmanFilter<double, 4, 2> kf;
kf.init(A, Q, H, R, initial_state, initial_covariance);

// 验证初始状态
Eigen::Matrix<double, 4, 1> state = kf.getState();
EXPECT_DOUBLE_EQ(state(0), 0.0);
EXPECT_DOUBLE_EQ(state(1), 0.0);
EXPECT_DOUBLE_EQ(state(2), 1.0);
EXPECT_DOUBLE_EQ(state(3), 1.0);

// 验证初始协方差
Eigen::Matrix<double, 4, 4> P = kf.getCovariance();
EXPECT_TRUE(P.isApprox(initial_covariance));
}

// 测试卡尔曼滤波器的预测和更新步骤
TEST(KalmanFilterTest, PredictUpdate) {
// 定义状态转移矩阵 A
Eigen::Matrix<double, 4, 4> A;
A << 1, 0, 1, 0,
0, 1, 0, 1,
0, 0, 1, 0,
0, 0, 0, 1;

// 过程噪声协方差 Q
Eigen::Matrix<double, 4, 4> Q = Eigen::Matrix<double, 4, 4>::Identity() * 0.01;

// 测量矩阵 H
Eigen::Matrix<double, 2, 4> H;
H << 1, 0, 0, 0,
0, 1, 0, 0;

// 测量噪声协方差 R
Eigen::Matrix<double, 2, 2> R = Eigen::Matrix<double, 2, 2>::Identity() * 0.1;

// 初始状态
Eigen::Matrix<double, 4, 1> initial_state;
initial_state << 0, 0, 1, 1;

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

// 逐步进行预测和更新，并验证状态估计
for (size_t i = 0; i < measurements.size(); ++i) {
kf.predict();
kf.update(measurements[i]);

Eigen::Matrix<double, 4, 1> state = kf.getState();

// 期望状态
Eigen::Matrix<double, 4, 1> expected_state;
expected_state << measurements, measurements, 1.0, 1.0;

// 由于存在噪声和协方差，状态估计应接近期望值
EXPECT_NEAR(state(0), expected_state(0), 0.2);
EXPECT_NEAR(state(1), expected_state(1), 0.2);
EXPECT_NEAR(state(2), expected_state(2), 0.2);
EXPECT_NEAR(state(3), expected_state(3), 0.2);
}
}

// 测试扩展卡尔曼滤波器的初始化和功能
TEST(EKFTest, Initialization) {
// 定义非线性状态转移函数
auto stateTransition = [](const Eigen::Matrix<double, 4, 1>& state) -> Eigen::Matrix<double, 4, 1> {
    double x = state(0);
    double y = state(1);
    double vx = state(2);
    double vy = state(3);

    double dt = 1.0;
    double radius = 10.0;
    double omega = vx / radius;

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

// 定义雅可比矩阵
auto stateTransitionJacobian = [](const Eigen::Matrix<double, 4, 1>& state) -> Eigen::Matrix<double, 4, 4> {
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

auto measurementJacobian = [](const Eigen::Matrix<double, 4, 1>& state) -> Eigen::Matrix<double, 2, 4> {
    Eigen::Matrix<double, 2, 4> H = Eigen::Matrix<double, 2, 4>::Zero();
    H(0, 0) = 1.0;
    H(1, 1) = 1.0;
    return H;
};

// 过程噪声协方差 Q
Eigen::Matrix<double, 4, 4> Q = Eigen::Matrix<double, 4, 4>::Identity() * 0.01;

// 测量噪声协方差 R
Eigen::Matrix<double, 2, 2> R = Eigen::Matrix<double, 2, 2>::Identity() * 0.1;

// 初始状态
Eigen::Matrix<double, 4, 1> initial_state;
initial_state << 0, 0, 1, 1;

// 初始协方差
Eigen::Matrix<double, 4, 4> initial_covariance = Eigen::Matrix<double, 4, 4>::Identity();

// 创建EKF实例
EKF<double, 4, 2> ekf;
ekf.init(stateTransition, measurementFunction, Q, R, initial_state, initial_covariance);

// 验证初始状态
Eigen::Matrix<double, 4, 1> state = ekf.getState();
EXPECT_DOUBLE_EQ(state(0), 0.0);
EXPECT_DOUBLE_EQ(state(1), 0.0);
EXPECT_DOUBLE_EQ(state(2), 1.0);
EXPECT_DOUBLE_EQ(state(3), 1.0);
}

// 测试扩展卡尔曼滤波器的预测和更新步骤
TEST(EKFTest, PredictUpdate) {
// 定义非线性状态转移函数
auto stateTransition = [](const Eigen::Matrix<double, 4, 1>& state) -> Eigen::Matrix<double, 4, 1> {
    double x = state(0);
    double y = state(1);
    double vx = state(2);
    double vy = state(3);

    double dt = 1.0;
    double radius = 10.0;
    double omega = vx / radius;

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

// 定义雅可比矩阵
auto stateTransitionJacobian = [](const Eigen::Matrix<double, 4, 1>& state) -> Eigen::Matrix<double, 4, 4> {
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

auto measurementJacobian = [](const Eigen::Matrix<double, 4, 1>& state) -> Eigen::Matrix<double, 2, 4> {
    Eigen::Matrix<double, 2, 4> H = Eigen::Matrix<double, 2, 4>::Zero();
    H(0, 0) = 1.0;
    H(1, 1) = 1.0;
    return H;
};

// 过程噪声协方差 Q
Eigen::Matrix<double, 4, 4> Q = Eigen::Matrix<double, 4, 4>::Identity() * 0.01;

// 测量噪声协方差 R
Eigen::Matrix<double, 2, 2> R = Eigen::Matrix<double, 2, 2>::Identity() * 0.1;

// 初始状态
Eigen::Matrix<double, 4, 1> initial_state;
initial_state << 0, 0, 1, 1;

// 初始协方差
Eigen::Matrix<double, 4, 4> initial_covariance = Eigen::Matrix<double, 4, 4>::Identity();

// 创建EKF实例
EKF<double, 4, 2> ekf;
ekf.init(stateTransition, measurementFunction, Q, R, initial_state, initial_covariance);

// 模拟测量数据
std::vector<Eigen::Matrix<double, 2, 1>> measurements = {
        {1.0, 1.1},
        {2.0, 2.1},
        {3.0, 3.0},
        {4.0, 4.1},
        {5.0, 5.2}
};

// 逐步进行预测和更新，并验证状态估计
for (size_t i = 0; i < measurements.size(); ++i) {
ekf.predict();
ekf.update(measurements[i]);

Eigen::Matrix<double, 4, 1> state = ekf.getState();

// 由于非线性，期望状态难以精确设定，这里只验证状态估计的合理性
// 检查位置估计是否接近测量值
EXPECT_NEAR(state(0), measurements, 0.5);
EXPECT_NEAR(state(1), measurements, 0.5);
}
}

// 测试移动平均滤波器的基本功能
TEST(MovingAverageFilterTest, BasicFunctionality) {
// 创建移动平均滤波器实例，窗口大小为3
MovingAverageFilter<double, 2> ma_filter(3);

// 定义测量数据
std::vector<Eigen::Matrix<double, 2, 1>> measurements = {
        {1.0, 1.0},
        {2.0, 2.0},
        {3.0, 3.0},
        {4.0, 4.0},
        {5.0, 5.0}
};

// 逐步添加测量并验证平均值
for (size_t i = 0; i < measurements.size(); ++i) {
ma_filter.addMeasurement(measurements[i]);
Eigen::Matrix<double, 2, 1> avg = ma_filter.getAverage();

if (i < 2) {
// 窗口未满，平均值应为当前测量值的平均
Eigen::Matrix<double, 2, 1> expected_avg = Eigen::Matrix<double, 2, 1>::Zero();
for (size_t j = 0; j <= i; ++j) {
expected_avg += measurements[j];
}
expected_avg /= static_cast<double>(i + 1);
EXPECT_TRUE(avg.isApprox(expected_avg));
} else {
// 窗口已满，平均值应为最近3个测量值的平均
Eigen::Matrix<double, 2, 1> expected_avg = (measurements[i - 2] + measurements[i - 1] + measurements[i]) / 3.0;
EXPECT_TRUE(avg.isApprox(expected_avg));
}
}
}