#ifndef RC_VISION_CORE_EDGE_DETECTOR_HPP
#define RC_VISION_CORE_EDGE_DETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "rc_vision/core/logger.hpp"

namespace rc_vision {
    namespace core {

        /**
         * @enum EdgeDetectionMethod
         * @brief 枚举不同的边缘检测方法。
         */
        enum class EdgeDetectionMethod {
            CANNY,      /**< 使用 Canny 边缘检测方法。 */
            SOBEL,      /**< 使用 Sobel 边缘检测方法。 */
            LAPLACIAN,  /**< 使用 Laplacian 边缘检测方法。 */
            // 可以添加更多方法
        };

        /**
         * @enum EdgeType
         * @brief 枚举不同的边缘类型。
         */
        enum class EdgeType {
            GENERAL,    /**< 一般边缘。 */
            LINE,       /**< 直线边缘。 */
            CORNER,     /**< 角点边缘。 */
            CIRCLE,     /**< 圆形边缘。 */
            POLYGON,    /**< 多边形边缘。 */
            // 可以添加更多类型
        };

        /**
         * @struct EdgeDetectionResult
         * @brief 边缘检测结果结构体。
         */
        struct EdgeDetectionResult {
            cv::Mat edges;                         /**< 边缘图像。 */
            std::vector<cv::Point> edge_points;    /**< 边缘点集合。 */
            std::vector<EdgeType> edge_types;      /**< 每个边缘点对应的边缘类型。 */
        };

        /**
         * @class EdgeDetector
         * @brief 边缘检测类，支持多种边缘检测方法和边缘类型分类。
         *
         * EdgeDetector 提供了多种边缘检测算法，并能够根据检测到的边缘特征将其分类为不同类型的边缘。
         */
        class EdgeDetector {
        public:
            /**
             * @brief 默认构造函数，初始化边缘检测方法为 CANNY。
             */
            EdgeDetector();

            /**
             * @brief 默认析构函数。
             */
            ~EdgeDetector() = default;

            /**
             * @brief 设置边缘检测方法。
             * @param method 边缘检测方法枚举值。
             */
            void setMethod(EdgeDetectionMethod method);

            /**
             * @brief 配置算法参数。
             *
             * 根据不同的边缘检测方法，参数的含义可能不同。例如，Canny 边缘检测需要设置阈值。
             *
             * @param params 参数列表。
             */
            void setParameters(const std::vector<double>& params);

            /**
             * @brief 从配置文件加载边缘检测配置。
             * @param config_file 配置文件路径（例如 YAML 文件）。
             * @return 成功加载返回 true，失败返回 false。
             */
            bool loadConfig(const std::string& config_file);

            /**
             * @brief 执行边缘检测。
             * @param image 输入图像。
             * @return 边缘检测结果结构体。
             */
            EdgeDetectionResult detectEdges(const cv::Mat& image);

            /**
             * @brief 获取当前使用的边缘检测方法。
             * @return 当前边缘检测方法枚举值。
             */
            EdgeDetectionMethod getMethod() const;

            /**
             * @brief 设置要检测的边缘类型。
             * @param types 边缘类型枚举值的向量。
             */
            void setDetectEdgeTypes(const std::vector<EdgeType>& types);

        private:
            EdgeDetectionMethod current_method_;        /**< 当前使用的边缘检测方法。 */
            std::vector<double> parameters_;            /**< 边缘检测算法参数。 */
            std::vector<EdgeType> detect_edge_types_;   /**< 要检测的边缘类型。 */

            /**
             * @brief 执行 Canny 边缘检测。
             * @param image 输入图像。
             * @return 边缘检测结果结构体。
             */
            EdgeDetectionResult detectCanny(const cv::Mat& image);

            /**
             * @brief 执行 Sobel 边缘检测。
             * @param image 输入图像。
             * @return 边缘检测结果结构体。
             */
            EdgeDetectionResult detectSobel(const cv::Mat& image);

            /**
             * @brief 执行 Laplacian 边缘检测。
             * @param image 输入图像。
             * @return 边缘检测结果结构体。
             */
            EdgeDetectionResult detectLaplacian(const cv::Mat& image);

            /**
             * @brief 分类边缘类型。
             *
             * 根据检测到的边缘图像，对边缘点进行类型分类。
             *
             * @param edges 边缘图像。
             * @param result 边缘检测结果结构体。
             */
            void classifyEdgeTypes(const cv::Mat& edges, EdgeDetectionResult& result);

            /**
             * @brief 检测直线边缘。
             * @param edges 边缘图像。
             * @param result 边缘检测结果结构体。
             */
            void detectLines(const cv::Mat& edges, EdgeDetectionResult& result);

            /**
             * @brief 检测角点边缘。
             * @param edges 边缘图像。
             * @param result 边缘检测结果结构体。
             */
            void detectCorners(const cv::Mat& edges, EdgeDetectionResult& result);

            /**
             * @brief 检测圆形边缘。
             * @param edges 边缘图像。
             * @param result 边缘检测结果结构体。
             */
            void detectCircles(const cv::Mat& edges, EdgeDetectionResult& result);     // 新增

            /**
             * @brief 检测多边形边缘。
             * @param edges 边缘图像。
             * @param result 边缘检测结果结构体。
             */
            void detectPolygons(const cv::Mat& edges, EdgeDetectionResult& result);    // 新增
        };

    } // namespace core
} // namespace rc_vision

#endif // RC_VISION_CORE_EDGE_DETECTOR_HPP
