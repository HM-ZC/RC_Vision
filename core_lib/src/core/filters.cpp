#include "rc_vision/core/filters.hpp"

namespace rc_vision {
    namespace core {
        namespace filters {

// 显式模板实例化
            template class KalmanFilter<double, 4, 2>;
            template class KalmanFilter<float, 4, 2>;

            template class EKF<double, 4, 2>;
            template class EKF<float, 4, 2>;

            template class UKF<double, 4, 2>;
            template class UKF<float, 4, 2>;

            template class ParticleFilter<double, 4, 2>;
            template class ParticleFilter<float, 4, 2>;

            template class MovingAverageFilter<double, 2>;
            template class MovingAverageFilter<float, 2>;

        } // namespace filters
    } // namespace core
} // namespace rc_vision