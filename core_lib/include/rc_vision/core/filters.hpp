#ifndef RC_VISION_CORE_FILTERS_HPP
#define RC_VISION_CORE_FILTERS_HPP

#include <Eigen/Dense>
#include <functional>
#include <deque>
#include <vector>
#include <random>

namespace rc_vision {
    namespace core {
        namespace filters {

            /**
             * @brief 提供各种滤波器的实现，包括卡尔曼滤波器、扩展卡尔曼滤波器、无迹卡尔曼滤波器、
             *        粒子滤波器和移动平均滤波器。
             */

            // --------------- 卡尔曼滤波器 ---------------
            /**
             * @brief 通用卡尔曼滤波器模板类。
             *
             * @tparam T 数据类型（通常为 float 或 double）。
             * @tparam StateDim 状态向量的维度。
             * @tparam MeasurementDim 测量向量的维度。
             */
            template <typename T, int StateDim, int MeasurementDim>
            class KalmanFilter {
            public:
                /**
                 * @brief 默认构造函数。
                 */
                KalmanFilter() {}

                /**
                 * @brief 初始化滤波器参数。
                 *
                 * @param A 状态转移矩阵。
                 * @param Q 过程噪声协方差矩阵。
                 * @param H 测量矩阵。
                 * @param R 测量噪声协方差矩阵。
                 * @param initial_state 初始状态向量。
                 * @param initial_covariance 初始状态协方差矩阵。
                 */
                void init(const Eigen::Matrix<T, StateDim, StateDim>& A,
                          const Eigen::Matrix<T, StateDim, StateDim>& Q,
                          const Eigen::Matrix<T, MeasurementDim, StateDim>& H,
                          const Eigen::Matrix<T, MeasurementDim, MeasurementDim>& R,
                          const Eigen::Matrix<T, StateDim, 1>& initial_state,
                          const Eigen::Matrix<T, StateDim, StateDim>& initial_covariance) {

                    A_ = A;
                    Q_ = Q;
                    H_ = H;
                    R_ = R;
                    x_ = initial_state;
                    P_ = initial_covariance;
                }

                /**
                 * @brief 执行预测步骤。
                 *
                 * 更新状态向量和协方差矩阵的预测值。
                 */
                void predict() {
                    x_ = A_ * x_;
                    P_ = A_ * P_ * A_.transpose() + Q_;
                }

                /**
                 * @brief 执行更新步骤。
                 *
                 * 使用测量向量更新状态向量和协方差矩阵。
                 *
                 * @param z 测量向量。
                 */
                void update(const Eigen::Matrix<T, MeasurementDim, 1>& z) {
                    Eigen::Matrix<T, MeasurementDim, 1> y = z - H_ * x_;
                    Eigen::Matrix<T, MeasurementDim, MeasurementDim> S = H_ * P_ * H_.transpose() + R_;
                    Eigen::Matrix<T, StateDim, MeasurementDim> K = P_ * H_.transpose() * S.inverse();
                    x_ = x_ + K * y;
                    Eigen::Matrix<T, StateDim, StateDim> I = Eigen::Matrix<T, StateDim, StateDim>::Identity();
                    P_ = (I - K * H_) * P_;
                }

                /**
                 * @brief 获取当前状态向量。
                 *
                 * @return Eigen::Matrix<T, StateDim, 1> 当前状态向量。
                 */
                Eigen::Matrix<T, StateDim, 1> getState() const {
                    return x_;
                }

                /**
                 * @brief 获取当前状态协方差矩阵。
                 *
                 * @return Eigen::Matrix<T, StateDim, StateDim> 当前状态协方差矩阵。
                 */
                Eigen::Matrix<T, StateDim, StateDim> getCovariance() const {
                    return P_;
                }

            protected:
                Eigen::Matrix<T, StateDim, StateDim> A_; /**< 状态转移矩阵。 */
                Eigen::Matrix<T, StateDim, StateDim> Q_; /**< 过程噪声协方差矩阵。 */
                Eigen::Matrix<T, MeasurementDim, StateDim> H_; /**< 测量矩阵。 */
                Eigen::Matrix<T, MeasurementDim, MeasurementDim> R_; /**< 测量噪声协方差矩阵。 */

                Eigen::Matrix<T, StateDim, 1> x_; /**< 状态向量。 */
                Eigen::Matrix<T, StateDim, StateDim> P_; /**< 状态协方差矩阵。 */
            };

            // --------------- 扩展卡尔曼滤波器 (EKF) ---------------
            /**
             * @brief 扩展卡尔曼滤波器模板类，适用于非线性系统。
             *
             * @tparam T 数据类型（通常为 float 或 double）。
             * @tparam StateDim 状态向量的维度。
             * @tparam MeasurementDim 测量向量的维度。
             */
            template <typename T, int StateDim, int MeasurementDim>
            class EKF : public KalmanFilter<T, StateDim, MeasurementDim> {
            public:
                /**
                 * @brief 默认构造函数。
                 */
                EKF() {}

                /**
                 * @brief 初始化滤波器参数。
                 *
                 * @param f 状态转移函数。
                 * @param h 测量函数。
                 * @param F_jacobian 状态转移函数的雅可比矩阵函数。
                 * @param H_jacobian 测量函数的雅可比矩阵函数。
                 * @param Q 过程噪声协方差矩阵。
                 * @param R 测量噪声协方差矩阵。
                 * @param initial_state 初始状态向量。
                 * @param initial_covariance 初始状态协方差矩阵。
                 */
                void init(
                        const std::function<Eigen::Matrix<T, StateDim, 1>(const Eigen::Matrix<T, StateDim, 1>&)>& f,
                        const std::function<Eigen::Matrix<T, MeasurementDim, 1>(const Eigen::Matrix<T, StateDim, 1>&)>& h,
                        const std::function<Eigen::Matrix<T, StateDim, StateDim>(const Eigen::Matrix<T, StateDim, 1>&)>& F_jacobian,
                        const std::function<Eigen::Matrix<T, MeasurementDim, StateDim>(const Eigen::Matrix<T, StateDim, 1>&)>& H_jacobian,
                        const Eigen::Matrix<T, StateDim, StateDim>& Q,
                        const Eigen::Matrix<T, MeasurementDim, MeasurementDim>& R,
                        const Eigen::Matrix<T, StateDim, 1>& initial_state,
                        const Eigen::Matrix<T, StateDim, StateDim>& initial_covariance) {

                    f_ = f;
                    h_ = h;
                    F_jacobian_ = F_jacobian;
                    H_jacobian_ = H_jacobian;

                    // 初始化父类
                    KalmanFilter<T, StateDim, MeasurementDim>::init(
                            Eigen::Matrix<T, StateDim, StateDim>::Identity(), // A_ 将在预测步骤中更新
                            Q,
                            Eigen::Matrix<T, MeasurementDim, StateDim>::Zero(), // H_ 将在更新步骤中更新
                            R,
                            initial_state,
                            initial_covariance);
                }

                /**
                 * @brief 执行预测步骤（非线性）。
                 *
                 * 使用状态转移函数和其雅可比矩阵进行预测。
                 */
                void predict() {
                    // 非线性状态转移
                    Eigen::Matrix<T, StateDim, 1> x_pred = f_(KalmanFilter<T, StateDim, MeasurementDim>::x_);

                    // 计算雅可比矩阵 F
                    Eigen::Matrix<T, StateDim, StateDim> F = F_jacobian_(KalmanFilter<T, StateDim, MeasurementDim>::x_);

                    // 更新状态转移矩阵
                    KalmanFilter<T, StateDim, MeasurementDim>::A_ = F;

                    // 调用基类的预测步骤
                    KalmanFilter<T, StateDim, MeasurementDim>::x_ = x_pred;
                    KalmanFilter<T, StateDim, MeasurementDim>::P_ = F * KalmanFilter<T, StateDim, MeasurementDim>::P_ * F.transpose() + KalmanFilter<T, StateDim, MeasurementDim>::Q_;
                }

                /**
                 * @brief 执行更新步骤（非线性）。
                 *
                 * 使用测量向量和测量函数的雅可比矩阵进行更新。
                 *
                 * @param z 测量向量。
                 */
                void update(const Eigen::Matrix<T, MeasurementDim, 1>& z) {
                    // 计算雅可比矩阵 H
                    Eigen::Matrix<T, MeasurementDim, StateDim> H = H_jacobian_(KalmanFilter<T, StateDim, MeasurementDim>::x_);

                    // 更新测量矩阵
                    KalmanFilter<T, StateDim, MeasurementDim>::H_ = H;

                    // 调用基类的更新步骤
                    KalmanFilter<T, StateDim, MeasurementDim>::update(z);
                }

            private:
                // 状态转移函数
                std::function<Eigen::Matrix<T, StateDim, 1>(const Eigen::Matrix<T, StateDim, 1>&)> f_;

                // 测量函数
                std::function<Eigen::Matrix<T, MeasurementDim, 1>(const Eigen::Matrix<T, StateDim, 1>&)> h_;

                // 状态转移函数的雅可比矩阵函数
                std::function<Eigen::Matrix<T, StateDim, StateDim>(const Eigen::Matrix<T, StateDim, 1>&)> F_jacobian_;

                // 测量函数的雅可比矩阵函数
                std::function<Eigen::Matrix<T, MeasurementDim, StateDim>(const Eigen::Matrix<T, StateDim, 1>&)> H_jacobian_;
            };

            // --------------- 无迹卡尔曼滤波器 (UKF) ---------------
            /**
             * @brief 无迹卡尔曼滤波器模板类，适用于非线性系统。
             *
             * @tparam T 数据类型（通常为 float 或 double）。
             * @tparam StateDim 状态向量的维度。
             * @tparam MeasurementDim 测量向量的维度。
             */
            template <typename T, int StateDim, int MeasurementDim>
            class UKF : public KalmanFilter<T, StateDim, MeasurementDim> {
            public:
                /**
                 * @brief 默认构造函数，初始化权重和参数。
                 */
                UKF()
                        : alpha_(1e-3), beta_(2.0), kappa_(0.0) {
                    // 计算权重
                    double lambda = std::pow(alpha_, 2) * (StateDim + kappa_) - StateDim;
                    weights_.resize(2 * StateDim + 1);
                    weights_(0) = lambda / (StateDim + lambda);
                    for(int i = 1; i < 2 * StateDim + 1; ++i) {
                        weights_(i) = 1 / (2 * (StateDim + lambda));
                    }
                }

                /**
                 * @brief 初始化滤波器参数。
                 *
                 * @param f 状态转移函数。
                 * @param h 测量函数。
                 * @param F_jacobian 状态转移函数的雅可比矩阵函数。
                 * @param H_jacobian 测量函数的雅可比矩阵函数。
                 * @param Q 过程噪声协方差矩阵。
                 * @param R 测量噪声协方差矩阵。
                 * @param initial_state 初始状态向量。
                 * @param initial_covariance 初始状态协方差矩阵。
                 */
                void init(
                        const std::function<Eigen::Matrix<T, StateDim, 1>(const Eigen::Matrix<T, StateDim, 1>&)>& f,
                        const std::function<Eigen::Matrix<T, MeasurementDim, 1>(const Eigen::Matrix<T, StateDim, 1>&)>& h,
                        const std::function<Eigen::Matrix<T, StateDim, StateDim>(const Eigen::Matrix<T, StateDim, 1>&)>& F_jacobian,
                        const std::function<Eigen::Matrix<T, MeasurementDim, StateDim>(const Eigen::Matrix<T, StateDim, 1>&)>& H_jacobian,
                        const Eigen::Matrix<T, StateDim, StateDim>& Q,
                        const Eigen::Matrix<T, MeasurementDim, MeasurementDim>& R,
                        const Eigen::Matrix<T, StateDim, 1>& initial_state,
                        const Eigen::Matrix<T, StateDim, StateDim>& initial_covariance) {

                    f_ = f;
                    h_ = h;
                    F_jacobian_ = F_jacobian;
                    H_jacobian_ = H_jacobian;

                    // 初始化父类
                    KalmanFilter<T, StateDim, MeasurementDim>::init(
                            Eigen::Matrix<T, StateDim, StateDim>::Identity(), // A_ 将在预测步骤中更新
                            Q,
                            Eigen::Matrix<T, MeasurementDim, StateDim>::Zero(), // H_ 将在更新步骤中更新
                            R,
                            initial_state,
                            initial_covariance);
                }

                /**
                 * @brief 执行预测步骤（非线性）。
                 *
                 * 使用无迹变换生成 sigma 点，通过状态转移函数预测状态，并更新状态向量和协方差矩阵。
                 */
                void predict() {
                    // 生成 sigma points
                    Eigen::Matrix<T, StateDim, 2 * StateDim + 1> sigma_points = computeSigmaPoints(KalmanFilter<T, StateDim, MeasurementDim>::x_, KalmanFilter<T, StateDim, MeasurementDim>::P_);

                    // 通过状态转移函数传递 sigma 点
                    Eigen::Matrix<T, StateDim, 2 * StateDim + 1> sigma_points_pred;
                    for(int i = 0; i < 2 * StateDim + 1; ++i) {
                        sigma_points_pred.col(i) = f_(sigma_points.col(i));
                    }

                    // 计算预测状态均值
                    Eigen::Matrix<T, StateDim, 1> x_pred = Eigen::Matrix<T, StateDim, 1>::Zero();
                    for(int i = 0; i < 2 * StateDim + 1; ++i) {
                        x_pred += weights_(i) * sigma_points_pred.col(i);
                    }

                    // 计算预测状态协方差
                    Eigen::Matrix<T, StateDim, StateDim> P_pred = Eigen::Matrix<T, StateDim, StateDim>::Zero();
                    for(int i = 0; i < 2 * StateDim + 1; ++i) {
                        Eigen::Matrix<T, StateDim, 1> diff = sigma_points_pred.col(i) - x_pred;
                        P_pred += weights_(i) * diff * diff.transpose();
                    }
                    P_pred += KalmanFilter<T, StateDim, MeasurementDim>::Q_;

                    // 更新状态和协方差
                    KalmanFilter<T, StateDim, MeasurementDim>::x_ = x_pred;
                    KalmanFilter<T, StateDim, MeasurementDim>::P_ = P_pred;
                }

                /**
                 * @brief 执行更新步骤（非线性）。
                 *
                 * 使用无迹变换生成测量 sigma 点，通过测量函数预测测量，并更新状态向量和协方差矩阵。
                 *
                 * @param z 测量向量。
                 */
                void update(const Eigen::Matrix<T, MeasurementDim, 1>& z) {
                    // 生成 sigma points from predicted state
                    Eigen::Matrix<T, StateDim, 2 * StateDim + 1> sigma_points = computeSigmaPoints(KalmanFilter<T, StateDim, MeasurementDim>::x_, KalmanFilter<T, StateDim, MeasurementDim>::P_);

                    // 通过测量函数传递 sigma 点
                    Eigen::Matrix<T, MeasurementDim, 2 * StateDim + 1> Z_sigma;
                    for(int i = 0; i < 2 * StateDim + 1; ++i) {
                        Z_sigma.col(i) = h_(sigma_points.col(i));
                    }

                    // 计算预测测量均值
                    Eigen::Matrix<T, MeasurementDim, 1> z_pred = Eigen::Matrix<T, MeasurementDim, 1>::Zero();
                    for(int i = 0; i < 2 * StateDim + 1; ++i) {
                        z_pred += weights_(i) * Z_sigma.col(i);
                    }

                    // 计算预测测量协方差
                    Eigen::Matrix<T, MeasurementDim, MeasurementDim> S = Eigen::Matrix<T, MeasurementDim, MeasurementDim>::Zero();
                    for(int i = 0; i < 2 * StateDim + 1; ++i) {
                        Eigen::Matrix<T, MeasurementDim, 1> diff = Z_sigma.col(i) - z_pred;
                        S += weights_(i) * diff * diff.transpose();
                    }
                    S += KalmanFilter<T, StateDim, MeasurementDim>::R_;

                    // 计算交叉协方差
                    Eigen::Matrix<T, StateDim, MeasurementDim> Tc = Eigen::Matrix<T, StateDim, MeasurementDim>::Zero();
                    for(int i = 0; i < 2 * StateDim + 1; ++i) {
                        Eigen::Matrix<T, StateDim, 1> x_diff = sigma_points.col(i) - KalmanFilter<T, StateDim, MeasurementDim>::x_;
                        Eigen::Matrix<T, MeasurementDim, 1> z_diff = Z_sigma.col(i) - z_pred;
                        Tc += weights_(i) * x_diff * z_diff.transpose();
                    }

                    // 计算卡尔曼增益
                    Eigen::Matrix<T, StateDim, MeasurementDim> K = Tc * S.inverse();

                    // 更新状态均值和协方差矩阵
                    KalmanFilter<T, StateDim, MeasurementDim>::x_ += K * (z - z_pred);
                    KalmanFilter<T, StateDim, MeasurementDim>::P_ -= K * S * K.transpose();
                }

            private:
                // 状态转移函数
                std::function<Eigen::Matrix<T, StateDim, 1>(const Eigen::Matrix<T, StateDim, 1>&)> f_;

                // 测量函数
                std::function<Eigen::Matrix<T, MeasurementDim, 1>(const Eigen::Matrix<T, StateDim, 1>&)> h_;

                // 状态转移函数的雅可比矩阵函数
                std::function<Eigen::Matrix<T, StateDim, StateDim>(const Eigen::Matrix<T, StateDim, 1>&)> F_jacobian_;

                // 测量函数的雅可比矩阵函数
                std::function<Eigen::Matrix<T, MeasurementDim, StateDim>(const Eigen::Matrix<T, StateDim, 1>&)> H_jacobian_;

                // 权重向量
                Eigen::Matrix<T, 2 * StateDim + 1, 1> weights_;

                // 无迹卡尔曼滤波器参数
                double alpha_; /**< 超参数 alpha，用于调整 sigma 点的分布。 */
                double beta_;  /**< 超参数 beta，用于考虑先验知识，通常设置为 2。 */
                double kappa_; /**< 超参数 kappa，用于调整 sigma 点的分布。 */

                /**
                 * @brief 计算 sigma 点。
                 *
                 * @param x 当前状态向量。
                 * @param P 当前状态协方差矩阵。
                 * @return Eigen::Matrix<T, StateDim, 2 * StateDim + 1> 生成的 sigma 点矩阵。
                 */
                Eigen::Matrix<T, StateDim, 2 * StateDim + 1> computeSigmaPoints(
                        const Eigen::Matrix<T, StateDim, 1>& x,
                        const Eigen::Matrix<T, StateDim, StateDim>& P) const {

                    Eigen::Matrix<T, StateDim, 2 * StateDim + 1> sigma_points;
                    Eigen::Matrix<T, StateDim, StateDim> sqrt_P = P.llt().matrixL();

                    sigma_points.col(0) = x;
                    double lambda = std::pow(alpha_, 2) * (StateDim + 0.0) - StateDim; // kappa_ = 0

                    for(int i = 0; i < StateDim; ++i) {
                        sigma_points.col(i + 1) = x + std::sqrt(StateDim + lambda) * sqrt_P.col(i);
                        sigma_points.col(i + 1 + StateDim) = x - std::sqrt(StateDim + lambda) * sqrt_P.col(i);
                    }

                    return sigma_points;
                }
            };

            // --------------- 粒子滤波器 (Particle Filter) ---------------
            /**
             * @brief 粒子滤波器模板类。
             *
             * @tparam T 数据类型（通常为 float 或 double）。
             * @tparam StateDim 状态向量的维度。
             * @tparam MeasurementDim 测量向量的维度。
             */
            template <typename T, int StateDim, int MeasurementDim>
            class ParticleFilter {
            public:
                /**
                 * @brief 构造函数，初始化粒子数量。
                 *
                 * @param num_particles 粒子数量。
                 */
                ParticleFilter(int num_particles)
                        : num_particles_(num_particles), gen_(rd_()), dist_(0.0, 1.0) {
                    particles_.resize(num_particles_);
                    weights_.resize(num_particles_, 1.0 / num_particles_);
                }

                /**
                 * @brief 初始化粒子滤波器。
                 *
                 * @param initial_state 初始状态向量。
                 * @param initial_covariance 初始状态协方差矩阵。
                 */
                void init(const Eigen::Matrix<T, StateDim, 1>& initial_state,
                          const Eigen::Matrix<T, StateDim, StateDim>& initial_covariance) {
                    Eigen::Matrix<T, StateDim, StateDim> cov_sqrt = initial_covariance.llt().matrixL();
                    for(int i = 0; i < num_particles_; ++i) {
                        particles_[i] = initial_state + cov_sqrt * Eigen::Matrix<T, StateDim, 1>::Random();
                    }
                }

                /**
                 * @brief 执行预测步骤。
                 *
                 * 使用状态转移函数和过程噪声协方差矩阵对粒子进行预测。
                 *
                 * @param f 状态转移函数。
                 * @param Q 过程噪声协方差矩阵。
                 */
                void predict(const std::function<Eigen::Matrix<T, StateDim, 1>(const Eigen::Matrix<T, StateDim, 1>&)>& f,
                             const Eigen::Matrix<T, StateDim, StateDim>& Q) {
                    Eigen::Matrix<T, StateDim, StateDim> cov_sqrt = Q.llt().matrixL();
                    for(int i = 0; i < num_particles_; ++i) {
                        particles_[i] = f(particles_[i]) + cov_sqrt * Eigen::Matrix<T, StateDim, 1>::Random();
                    }
                }

                /**
                 * @brief 执行更新步骤。
                 *
                 * 使用测量向量和测量函数更新粒子的权重，并进行重采样。
                 *
                 * @param z 测量向量。
                 * @param h 测量函数。
                 * @param R 测量噪声协方差矩阵。
                 */
                void update(const Eigen::Matrix<T, MeasurementDim, 1>& z,
                            const std::function<Eigen::Matrix<T, MeasurementDim, 1>(const Eigen::Matrix<T, StateDim, 1>&)>& h,
                            const Eigen::Matrix<T, MeasurementDim, MeasurementDim>& R) {
                    for(int i = 0; i < num_particles_; ++i) {
                        Eigen::Matrix<T, MeasurementDim, 1> z_pred = h(particles_[i]);
                        Eigen::Matrix<T, MeasurementDim, MeasurementDim> S = R;
                        Eigen::Matrix<T, MeasurementDim, 1> y = z - z_pred;
                        // 假设高斯噪声，计算似然度
                        T exponent = -0.5 * y.transpose() * S.inverse() * y;
                        T likelihood = std::exp(exponent);
                        weights_[i] *= likelihood;
                    }
                    normalizeWeights();
                    resample();
                }

                /**
                 * @brief 获取当前估计的状态向量。
                 *
                 * @return Eigen::Matrix<T, StateDim, 1> 估计的状态向量。
                 */
                Eigen::Matrix<T, StateDim, 1> getState() const {
                    Eigen::Matrix<T, StateDim, 1> state = Eigen::Matrix<T, StateDim, 1>::Zero();
                    for(int i = 0; i < num_particles_; ++i) {
                        state += weights_[i] * particles_[i];
                    }
                    return state;
                }

            private:
                int num_particles_; /**< 粒子数量。 */
                std::vector<Eigen::Matrix<T, StateDim, 1>> particles_; /**< 粒子集合。 */
                std::vector<T> weights_; /**< 粒子权重。 */
                std::random_device rd_; /**< 随机设备。 */
                std::mt19937 gen_; /**< 随机数生成器。 */
                std::uniform_real_distribution<> dist_; /**< 均匀分布，用于重采样。 */

                /**
                 * @brief 归一化权重。
                 *
                 * 将所有粒子的权重归一化，使其和为 1。
                 */
                void normalizeWeights() {
                    T sum = 0.0;
                    for(auto w : weights_) sum += w;
                    for(auto& w : weights_) w /= sum;
                }

                /**
                 * @brief 重采样粒子。
                 *
                 * 使用系统重采样算法，根据粒子的权重重新采样粒子集合。
                 */
                void resample() {
                    std::vector<Eigen::Matrix<T, StateDim, 1>> new_particles;
                    new_particles.reserve(num_particles_);

                    // 创建累积分布
                    std::vector<T> cumulative_weights(num_particles_, 0.0);
                    cumulative_weights[0] = weights_[0];
                    for(int i = 1; i < num_particles_; ++i) {
                        cumulative_weights[i] = cumulative_weights[i-1] + weights_[i];
                    }

                    // 重采样
                    for(int i = 0; i < num_particles_; ++i) {
                        T rand_weight = dist_(gen_);
                        auto it = std::lower_bound(cumulative_weights.begin(), cumulative_weights.end(), rand_weight);
                        int index = std::distance(cumulative_weights.begin(), it);
                        new_particles.push_back(particles_[index]);
                    }

                    particles_ = std::move(new_particles);
                    std::fill(weights_.begin(), weights_.end(), 1.0 / num_particles_);
                }
            };

            // --------------- 移动平均滤波器 (Moving Average Filter) ---------------
            /**
             * @brief 移动平均滤波器模板类。
             *
             * @tparam T 数据类型（通常为 float 或 double）。
             * @tparam Dim 向量的维度。
             */
            template <typename T, int Dim>
            class MovingAverageFilter {
            public:
                /**
                 * @brief 构造函数，初始化窗口大小。
                 *
                 * @param window_size 移动平均窗口的大小。
                 */
                MovingAverageFilter(int window_size)
                        : window_size_(window_size) {}

                /**
                 * @brief 添加一个新的观测值。
                 *
                 * 如果窗口已满，则移除最旧的观测值。
                 *
                 * @param measurement 新的观测向量。
                 */
                void addMeasurement(const Eigen::Matrix<T, Dim, 1>& measurement) {
                    if (measurements_.size() >= window_size_) {
                        measurements_.pop_front();
                    }
                    measurements_.push_back(measurement);
                }

                /**
                 * @brief 获取当前的平均值。
                 *
                 * @return Eigen::Matrix<T, Dim, 1> 当前窗口的平均向量。
                 */
                Eigen::Matrix<T, Dim, 1> getAverage() const {
                    Eigen::Matrix<T, Dim, 1> avg = Eigen::Matrix<T, Dim, 1>::Zero();
                    for(const auto& m : measurements_) {
                        avg += m;
                    }
                    if (!measurements_.empty()) {
                        avg /= static_cast<T>(measurements_.size());
                    }
                    return avg;
                }

            private:
                int window_size_; /**< 移动平均窗口的大小。 */
                std::deque<Eigen::Matrix<T, Dim, 1>> measurements_; /**< 存储观测值的双端队列。 */
            };

        } // namespace filters
    } // namespace core
} // namespace rc_vision

#endif // RC_VISION_CORE_FILTERS_HPP
