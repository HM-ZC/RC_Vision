#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>

#include "rc_vision/core/camera.hpp"
#include "rc_vision/core/image.hpp"
#include "rc_vision/core/logger.hpp"
#include "rc_vision/core/filters.hpp"
#include "rc_vision/core/config.hpp"
#include "rc_vision/core/edge_detector.hpp"
#include "rc_vision/core/logger.hpp"
#include "rc_vision/core/math_utils.hpp"
#include "rc_vision/core/point_cloud.hpp"

namespace py = pybind11;
using namespace rc_vision::core;

// Helper function to convert cv::Mat to numpy array
py::array_t<unsigned char> mat_to_numpy(const cv::Mat& mat) {
    // Ensure the image is in a format that can be converted
    if(mat.empty()) {
        throw std::runtime_error("Empty image cannot be converted to numpy array.");
    }

    // Determine the number of channels
    int channels = mat.channels();
    // Define the data type
    py::dtype dtype = py::dtype::of<unsigned char>();

    // Define the shape
    std::vector<ssize_t> shape;
    if (channels == 1) {
        shape = { mat.rows, mat.cols };
    } else {
        shape = { mat.rows, mat.cols, channels };
    }

    // Define the strides
    std::vector<ssize_t> strides;
    for(int i = 0; i < mat.channels(); ++i) {
        strides.push_back(mat.step[0]);
    }
    if (channels == 1) {
        strides = { mat.step[0], mat.step[1] };
    } else {
        strides = { mat.step[0], mat.step[1], 1 };
    }

    // Create the numpy array without copying the data
    return py::array(dtype, shape, strides, mat.data);
}

// Helper function to convert numpy array to cv::Mat
cv::Mat numpy_to_mat(py::array_t<unsigned char> array) {
    py::buffer_info buf = array.request();

    // Check dimensions
    if (buf.ndim < 2 || buf.ndim > 3)
        throw std::runtime_error("Number of dimensions must be 2 or 3");

    // Determine the type
    int channels = 1;
    if (buf.ndim == 3) {
        channels = buf.shape[2];
    }

    // Create cv::Mat header (no data copy)
    cv::Mat mat(buf.shape[0], buf.shape[1], (channels == 1) ? CV_8UC1 : CV_8UC3, (void*)buf.ptr);

    return mat.clone(); // Clone to ensure data ownership
}

PYBIND11_MODULE(rc_vision_py, m) {
m.doc() = "Python bindings for rc_vision_core";

// 绑定 Camera 类
py::class_<Camera>(m, "Camera")
.def(py::init<>())
.def(py::init<const Eigen::Matrix3d&, const Eigen::Matrix<double, 5, 1>&>(),
        py::arg("intrinsic"), py::arg("distortion"))
.def("getIntrinsic", &Camera::getIntrinsic)
.def("getDistortion", &Camera::getDistortion)
.def("getExtrinsic", &Camera::getExtrinsic)
.def("setIntrinsic", &Camera::setIntrinsic, py::arg("intrinsic"))
.def("setDistortion", &Camera::setDistortion, py::arg("distortion"))
.def("setExtrinsic", &Camera::setExtrinsic, py::arg("extrinsic"))
.def("project", (Eigen::Vector2d (Camera::*)(const Eigen::Vector3d&) const) &Camera::project, py::arg("point_3d"))
.def("project", (std::vector<Eigen::Vector2d> (Camera::*)(const std::vector<Eigen::Vector3d>&) const) &Camera::project, py::arg("points_3d"))
.def("undistort", (Eigen::Vector2d (Camera::*)(const Eigen::Vector2d&) const) &Camera::undistort, py::arg("distorted_point"))
.def("undistort", (std::vector<Eigen::Vector2d> (Camera::*)(const std::vector<Eigen::Vector2d>&) const) &Camera::undistort, py::arg("distorted_points"))
.def("undistortImage", &Camera::undistortImage, py::arg("distorted_image"))
.def("correctDistortion", &Camera::correctDistortion, py::arg("distorted_image"))
.def("saveParameters", &Camera::saveParameters, py::arg("file_path"))
.def("loadParameters", &Camera::loadParameters, py::arg("file_path"))
.def("worldToCamera", &Camera::worldToCamera, py::arg("point_world"))
.def("cameraToWorld", &Camera::cameraToWorld, py::arg("point_camera"))
.def_static("calibrateCamera", &Camera::calibrateCamera,
py::arg("object_points"),
py::arg("image_points"),
py::arg("image_size"),
py::arg("out_intrinsic"),
py::arg("out_distortion"));

// 绑定 StereoCamera 类
py::class_<StereoCamera>(m, "StereoCamera")
.def(py::init<>())
.def(py::init<const Camera&, const Camera&, const Eigen::Matrix4d&>(),
        py::arg("left_camera"), py::arg("right_camera"), py::arg("extrinsic"))
.def("getLeftCamera", &StereoCamera::getLeftCamera, py::return_value_policy::reference_internal)
.def("getRightCamera", &StereoCamera::getRightCamera, py::return_value_policy::reference_internal)
.def("getExtrinsic", &StereoCamera::getExtrinsic)
.def("setLeftCamera", &StereoCamera::setLeftCamera, py::arg("left_camera"))
.def("setRightCamera", &StereoCamera::setRightCamera, py::arg("right_camera"))
.def("setExtrinsic", &StereoCamera::setExtrinsic, py::arg("extrinsic"))
.def("rectifyImages", &StereoCamera::rectifyImages,
py::arg("left_image"), py::arg("right_image"),
py::arg("rectified_left"), py::arg("rectified_right"))
.def("computeDisparity", &StereoCamera::computeDisparity,
py::arg("rectified_left"), py::arg("rectified_right"),
py::arg("disparity_map"))
.def_static("calibrateStereoCamera", &StereoCamera::calibrateStereoCamera,
py::arg("object_points"),
py::arg("image_points_left"),
py::arg("image_points_right"),
py::arg("image_size"),
py::arg("left_camera"),
py::arg("right_camera"),
py::arg("out_extrinsic"));

// 绑定 ConfigFormat 枚举
py::enum_<ConfigFormat>(m, "ConfigFormat")
.value("YAML", ConfigFormat::YAML)
.value("JSON", ConfigFormat::JSON)
.export_values();

// 绑定 Config 类
py::class_<Config>(m, "Config")
.def(py::init<>())
.def("set_format", &Config::set_format, py::arg("fmt"),
"设置配置文件的格式。支持 'YAML' 和 'JSON'。")
.def("load_from_file", &Config::loadFromFile, py::arg("file_path"),
"从指定文件加载配置。")
// 为模板方法 get<T> 提供具体类型的方法
.def("get_int", &Config::get<int>, py::arg("key"),
"根据键获取整数类型的配置值。")
.def("get_double", &Config::get<double>, py::arg("key"),
"根据键获取双精度浮点数类型的配置值。")
.def("get_string", &Config::get<std::string>, py::arg("key"),
"根据键获取字符串类型的配置值。")
.def("get_bool", &Config::get<bool>, py::arg("key"),
"根据键获取布尔类型的配置值。");

// Bind EdgeDetectionMethod enum
py::enum_<EdgeDetectionMethod>(m, "EdgeDetectionMethod")
.value("CANNY", EdgeDetectionMethod::CANNY)
.value("SOBEL", EdgeDetectionMethod::SOBEL)
.value("LAPLACIAN", EdgeDetectionMethod::LAPLACIAN)
// .value("OTHER_METHOD", EdgeDetectionMethod::OTHER_METHOD) // 添加更多方法
.export_values();

// Bind EdgeType enum
py::enum_<EdgeType>(m, "EdgeType")
.value("GENERAL", EdgeType::GENERAL)
.value("LINE", EdgeType::LINE)
.value("CORNER", EdgeType::CORNER)
.value("CIRCLE", EdgeType::CIRCLE)
.value("POLYGON", EdgeType::POLYGON)
// .value("OTHER_TYPE", EdgeType::OTHER_TYPE) // 添加更多类型
.export_values();

// Bind EdgeDetectionResult struct
py::class_<EdgeDetectionResult>(m, "EdgeDetectionResult")
.def(py::init<>())
.def_readonly("edges", &EdgeDetectionResult::edges)
.def_readonly("edge_points", &EdgeDetectionResult::edge_points)
.def_readonly("edge_types", &EdgeDetectionResult::edge_types);

// Bind EdgeDetector class
py::class_<EdgeDetector>(m, "EdgeDetector")
.def(py::init<>())
.def("set_method", &EdgeDetector::setMethod, py::arg("method"),
"Set the edge detection method.")
.def("set_parameters", &EdgeDetector::setParameters, py::arg("params"),
"Set the parameters for the edge detection algorithm.")
.def("load_config", &EdgeDetector::loadConfig, py::arg("config_file"),
"Load edge detection configuration from a file.")
.def("detect_edges", [](EdgeDetector &self, py::array_t<unsigned char> image) -> EdgeDetectionResult {
cv::Mat mat_image = numpy_to_mat(image);
return self.detectEdges(mat_image);
}, py::arg("image"),
"Perform edge detection on the given image.")
.def("get_method", &EdgeDetector::getMethod,
"Get the current edge detection method.")
.def("set_detect_edge_types", &EdgeDetector::setDetectEdgeTypes, py::arg("types"),
"Set the types of edges to detect.");

// Bind filters namespace
py::module m_filters = m.def_submodule("filters", "Filters module");

// 绑定 KalmanFilter<double, 4, 2>
using KalmanFilterDouble4_2 = rc_vision::core::filters::KalmanFilter<double, 4, 2>;
py::class_<KalmanFilterDouble4_2>(m_filters, "KalmanFilter4_2")
.def(py::init<>())
.def("init", &KalmanFilterDouble4_2::init,
py::arg("A"), py::arg("Q"), py::arg("H"), py::arg("R"),
py::arg("initial_state"), py::arg("initial_covariance"),
"Initialize the Kalman Filter with matrices A, Q, H, R, initial state, and initial covariance.")
.def("predict", &KalmanFilterDouble4_2::predict,
"Perform the prediction step.")
.def("update", &KalmanFilterDouble4_2::update,
py::arg("z"),
"Perform the update step with measurement z.")
.def("getState", &KalmanFilterDouble4_2::getState,
"Get the current state vector.")
.def("getCovariance", &KalmanFilterDouble4_2::getCovariance,
"Get the current state covariance matrix.");

// 绑定 EKF<double, 4, 2>
using EKFDouble4_2 = rc_vision::core::filters::EKF<double, 4, 2>;
py::class_<EKFDouble4_2>(m_filters, "EKF4_2")
.def(py::init<>())
.def("init", &EKFDouble4_2::init,
py::arg("f"), py::arg("h"),
py::arg("F_jacobian"), py::arg("H_jacobian"),
py::arg("Q"), py::arg("R"),
py::arg("initial_state"), py::arg("initial_covariance"),
"Initialize the EKF with functions f, h, F_jacobian, H_jacobian, matrices Q, R, initial state, and initial covariance.")
.def("predict", &EKFDouble4_2::predict,
"Perform the prediction step.")
.def("update", &EKFDouble4_2::update,
py::arg("z"),
"Perform the update step with measurement z.")
.def("getState", &EKFDouble4_2::getState,
"Get the current state vector.")
.def("getCovariance", &EKFDouble4_2::getCovariance,
"Get the current state covariance matrix.");

// 绑定 UKF<double, 4, 2>
using UKFDouble4_2 = rc_vision::core::filters::UKF<double, 4, 2>;
py::class_<UKFDouble4_2>(m_filters, "UKF4_2")
.def(py::init<>())
.def("init", &UKFDouble4_2::init,
py::arg("f"), py::arg("h"),
py::arg("F_jacobian"), py::arg("H_jacobian"),
py::arg("Q"), py::arg("R"),
py::arg("initial_state"), py::arg("initial_covariance"),
"Initialize the UKF with functions f, h, F_jacobian, H_jacobian, matrices Q, R, initial state, and initial covariance.")
.def("predict", &UKFDouble4_2::predict,
"Perform the prediction step.")
.def("update", &UKFDouble4_2::update,
py::arg("z"),
"Perform the update step with measurement z.")
.def("getState", &UKFDouble4_2::getState,
"Get the current state vector.")
.def("getCovariance", &UKFDouble4_2::getCovariance,
"Get the current state covariance matrix.");

// 绑定 ParticleFilter<double, 4, 2>
using ParticleFilterDouble4_2 = rc_vision::core::filters::ParticleFilter<double, 4, 2>;
py::class_<ParticleFilterDouble4_2>(m_filters, "ParticleFilter4_2")
.def(py::init<int>(), py::arg("num_particles"),
"Initialize the Particle Filter with the number of particles.")
.def("init", &ParticleFilterDouble4_2::init,
py::arg("initial_state"), py::arg("initial_covariance"),
"Initialize the Particle Filter with initial state and covariance.")
.def("predict", &ParticleFilterDouble4_2::predict,
py::arg("f"), py::arg("Q"),
"Perform the prediction step with state transition function f and process noise Q.")
.def("update", &ParticleFilterDouble4_2::update,
py::arg("z"), py::arg("h"), py::arg("R"),
"Perform the update step with measurement z, measurement function h, and measurement noise R.")
.def("getState", &ParticleFilterDouble4_2::getState,
"Get the current estimated state vector.");

// 绑定 MovingAverageFilter<double, 2>
using MovingAverageFilterDouble2 = rc_vision::core::filters::MovingAverageFilter<double, 2>;
py::class_<MovingAverageFilterDouble2>(m_filters, "MovingAverageFilter2")
.def(py::init<int>(), py::arg("window_size"),
"Initialize the Moving Average Filter with a specified window size.")
.def("addMeasurement", &MovingAverageFilterDouble2::addMeasurement,
py::arg("measurement"),
"Add a new measurement to the moving average filter.")
.def("getAverage", &MovingAverageFilterDouble2::getAverage,
"Get the current average value.");

// 绑定 Image 类
py::class_<Image>(m, "Image")
.def(py::init<>())
.def(py::init<const std::string&>(), py::arg("file_path"),
"Construct an Image by loading from a file path.")
.def(py::init<const cv::Mat&>(), py::arg("mat"),
"Construct an Image from an existing cv::Mat object.")
.def("load", &Image::load, py::arg("file_path"),
"Load an image from the specified file path.")
.def("save", &Image::save, py::arg("file_path"),
"Save the image to the specified file path.")
.def("getMat", &Image::getMat, py::return_value_policy::reference_internal,
"Get the underlying cv::Mat object.")
.def("setMat", &Image::setMat, py::arg("mat"),
"Set the underlying cv::Mat object.")
.def("convertColor", &Image::convertColor, py::arg("code"),
"Convert the image to a different color space using the specified code.")
.def("resize", &Image::resize, py::arg("width"), py::arg("height"), py::arg("interpolation") = cv::INTER_LINEAR,
"Resize the image to the specified width and height with optional interpolation.")
.def("crop", &Image::crop, py::arg("x"), py::arg("y"), py::arg("width"), py::arg("height"),
"Crop a region from the image and return a new Image object.")
.def("rotate", &Image::rotate, py::arg("angle"), py::arg("center") = cv::Point2f(-1, -1), py::arg("scale") = 1.0,
"Rotate the image by the specified angle around the given center with optional scaling.")
.def("flip", &Image::flip, py::arg("flip_code"),
"Flip the image according to the specified flip code (0: vertical, 1: horizontal, -1: both).")
.def("gaussianBlur", &Image::gaussianBlur, py::arg("ksize") = 5, py::arg("sigmaX") = 1.0,
"Apply Gaussian blur to the image with specified kernel size and sigmaX.")
.def("medianBlur", &Image::medianBlur, py::arg("ksize") = 5,
"Apply median blur to the image with specified kernel size.")
.def("edgeDetection", &Image::edgeDetection, py::arg("threshold1") = 100, py::arg("threshold2") = 200,
"Apply Canny edge detection to the image with specified thresholds.")
.def("equalizeHist", &Image::equalizeHist,
"Apply histogram equalization to the image (only for single-channel images).")
.def("adjustContrast", &Image::adjustContrast, py::arg("alpha"), py::arg("beta"),
"Adjust the image contrast and brightness: new_image = alpha * image + beta.")
.def("detectKeypoints", &Image::detectKeypoints, py::arg("detector_type") = "ORB",
"Detect keypoints in the image using the specified detector type (e.g., ORB, SIFT, SURF).")
.def("computeDescriptors", &Image::computeDescriptors, py::arg("descriptor_type") = "ORB",
"Compute descriptors for the detected keypoints using the specified descriptor type (e.g., ORB, SIFT, SURF).")
.def("mean", &Image::mean,
"Compute the mean value of the image pixels.")
.def("stddev", &Image::stddev,
"Compute the standard deviation of the image pixels.")
.def("histogram", &Image::histogram, py::arg("histSize") = 256, py::arg("range") = std::vector<float>{0, 256},
"Compute the histogram of the image with specified histogram size and range.")
.def("__add__", &Image::operator+, py::arg("other"),
"Add two images.")
.def("__sub__", &Image::operator-, py::arg("other"),
"Subtract two images.")
.def("__mul__", &Image::operator*, py::arg("other"),
"Multiply two images.")
.def("__truediv__", &Image::operator/, py::arg("other"),
"Divide two images.")
.def("setROI", &Image::setROI, py::arg("roi"),
"Set the region of interest (ROI) for the image.")
.def("getROI", &Image::getROI,
"Get the current region of interest (ROI) of the image.")
.def("getROIImage", &Image::getROIImage,
"Get a new Image object representing the ROI area.")
.def("matchTemplate", &Image::matchTemplate, py::arg("template_img"), py::arg("method") = cv::TM_CCOEFF,
"Perform template matching using the specified method.")
.def("findBestMatch", &Image::findBestMatch, py::arg("template_img"), py::arg("method") = cv::TM_CCOEFF, py::arg("min_max_vals") = nullptr,
"Find the best match location of the template in the image.")
.def("drawMatch", &Image::drawMatch, py::arg("template_img"), py::arg("top_left"), py::arg("color") = cv::Scalar(0, 0, 255), py::arg("thickness") = 2,
"Draw a rectangle around the matched template area.");

m.def("log_debug", [](const std::string& message) {
// 获取调用者信息（文件名、行号、函数名）
// 使用 Python 的 inspect 模块可以在 Python 端获取
// 这里仅记录消息，文件、行、函数可以留空或通过其他方式传递
Logger::getInstance().log(Logger::LogLevel::DEBUG, message);
}, py::arg("message"),
"Log a DEBUG level message.");

m.def("log_info", [](const std::string& message) {
Logger::getInstance().log(Logger::LogLevel::INFO, message);
}, py::arg("message"),
"Log an INFO level message.");

m.def("log_warn", [](const std::string& message) {
Logger::getInstance().log(Logger::LogLevel::WARN, message);
}, py::arg("message"),
"Log a WARN level message.");

m.def("log_error", [](const std::string& message) {
Logger::getInstance().log(Logger::LogLevel::ERROR, message);
}, py::arg("message"),
"Log an ERROR level message.");

// 绑定 MathUtils 类
py::class_<MathUtils>(m, "MathUtils")
// 创建齐次变换矩阵
.def_static("create_transformation_matrix", &MathUtils::createTransformationMatrix,
py::arg("translation"), py::arg("rotation"),
"Create a 4x4 homogeneous transformation matrix from translation and rotation.")

// 计算重投影误差
.def_static("compute_reprojection_error", &MathUtils::computeReprojectionError,
py::arg("observed"), py::arg("projected"),
"Compute the Euclidean distance between observed and projected 2D points as reprojection error.")

// 四元数与旋转矩阵转换
.def_static("quaternion_to_rotation_matrix", &MathUtils::quaternionToRotationMatrix,
py::arg("quaternion"),
"Convert a quaternion to a 3x3 rotation matrix.")
.def_static("rotation_matrix_to_quaternion", &MathUtils::rotationMatrixToQuaternion,
py::arg("rotation_matrix"),
"Convert a 3x3 rotation matrix to a quaternion.")

// 计算两条线段的最近点对
.def_static("closest_points_between_segments",
[](const Eigen::Vector3d& p1, const Eigen::Vector3d& p2,
const Eigen::Vector3d& p3, const Eigen::Vector3d& p4) -> std::tuple<Eigen::Vector3d, Eigen::Vector3d, double> {
Eigen::Vector3d cp1, cp2;
double distance = MathUtils::closestPointsBetweenSegments(p1, p2, p3, p4, cp1, cp2);
return std::make_tuple(cp1, cp2, distance);
},
py::arg("p1"), py::arg("p2"), py::arg("p3"), py::arg("p4"),
"Compute the closest points between two 3D segments and their minimum distance.")

// 点到平面的距离
.def_static("point_to_plane_distance", &MathUtils::pointToPlaneDistance,
py::arg("point"), py::arg("plane_point"), py::arg("plane_normal"),
"Compute the distance from a 3D point to a plane defined by a point and a normal vector.")

// 最小二乘法拟合平面
.def_static("fit_plane",
[](const std::vector<Eigen::Vector3d>& points) -> std::tuple<bool, Eigen::Vector3d, Eigen::Vector3d> {
Eigen::Vector3d plane_point, plane_normal;
bool success = MathUtils::fitPlane(points, plane_point, plane_normal);
return std::make_tuple(success, plane_point, plane_normal);
},
py::arg("points"),
"Fit a plane to a point cloud using least squares. Returns (success, plane_point, plane_normal).")

// 欧拉角与旋转矩阵转换
.def_static("euler_angles_to_rotation_matrix", &MathUtils::eulerAnglesToRotationMatrix,
py::arg("roll"), py::arg("pitch"), py::arg("yaw"),
"Convert Euler angles (roll, pitch, yaw) to a 3x3 rotation matrix.")
.def_static("rotation_matrix_to_euler_angles",
[](const Eigen::Matrix3d& R) -> std::tuple<double, double, double> {
double roll, pitch, yaw;
MathUtils::rotationMatrixToEulerAngles(R, roll, pitch, yaw);
return std::make_tuple(roll, pitch, yaw);
},
py::arg("rotation_matrix"),
"Convert a 3x3 rotation matrix to Euler angles (roll, pitch, yaw) in radians.");

// 绑定 Point 结构体
py::class_<Point>(m, "Point")
.def(py::init<double, double, double>(), py::arg("x"), py::arg("y"), py::arg("z"),
"Construct a Point with coordinates (x, y, z).")
.def(py::init<double, double, double, const Eigen::Vector3d&>(), py::arg("x"), py::arg("y"), py::arg("z"), py::arg("color"),
"Construct a Point with coordinates (x, y, z) and color.")
.def_readwrite("x", &Point::x, "X coordinate of the point.")
.def_readwrite("y", &Point::y, "Y coordinate of the point.")
.def_readwrite("z", &Point::z, "Z coordinate of the point.")
.def_property("color",
[](const Point& p) -> py::object {
if (p.color.has_value()) {
return py::cast(p.color.value());
} else {
return py::none();
}
},
[](Point& p, const py::object& obj) {
if (!obj.is_none()) {
p.color = obj.cast<Eigen::Vector3d>();
} else {
p.color.reset();
}
},
"Optional RGB color of the point.")
.def_property("normal",
[](const Point& p) -> py::object {
if (p.normal.has_value()) {
return py::cast(p.normal.value());
} else {
return py::none();
}
},
[](Point& p, const py::object& obj) {
if (!obj.is_none()) {
p.normal = obj.cast<Eigen::Vector3d>();
} else {
p.normal.reset();
}
},
"Optional normal vector of the point.")
.def_property("intensity",
[](const Point& p) -> py::object {
if (p.intensity.has_value()) {
return py::cast(p.intensity.value());
} else {
return py::none();
}
},
[](Point& p, const py::object& obj) {
if (!obj.is_none()) {
p.intensity = obj.cast<double>();
} else {
p.intensity.reset();
}
},
"Optional intensity of the point.");

// 绑定 PointCloudStats 结构体
py::class_<PointCloudStats>(m, "PointCloudStats")
.def(py::init<>())
.def_readwrite("min_x", &PointCloudStats::min_x, "Minimum X value.")
.def_readwrite("max_x", &PointCloudStats::max_x, "Maximum X value.")
.def_readwrite("min_y", &PointCloudStats::min_y, "Minimum Y value.")
.def_readwrite("max_y", &PointCloudStats::max_y, "Maximum Y value.")
.def_readwrite("min_z", &PointCloudStats::min_z, "Minimum Z value.")
.def_readwrite("max_z", &PointCloudStats::max_z, "Maximum Z value.")
.def_readwrite("mean_x", &PointCloudStats::mean_x, "Mean X value.")
.def_readwrite("mean_y", &PointCloudStats::mean_y, "Mean Y value.")
.def_readwrite("mean_z", &PointCloudStats::mean_z, "Mean Z value.")
.def_readwrite("variance_x", &PointCloudStats::variance_x, "Variance in X.")
.def_readwrite("variance_y", &PointCloudStats::variance_y, "Variance in Y.")
.def_readwrite("variance_z", &PointCloudStats::variance_z, "Variance in Z.");

// 绑定 PointCloud 类
py::class_<PointCloud>(m, "PointCloud")
.def(py::init<>(),
"Construct an empty PointCloud.")
.def("addPoint", (void (PointCloud::*)(const Point&)) &PointCloud::addPoint, py::arg("point"),
"Add a Point to the PointCloud.")
.def("addPoint", (void (PointCloud::*)(double, double, double)) &PointCloud::addPoint, py::arg("x"), py::arg("y"), py::arg("z"),
"Add a point with coordinates (x, y, z) to the PointCloud.")
.def("addPoint", (void (PointCloud::*)(double, double, double, const Eigen::Vector3d&)) &PointCloud::addPoint,
py::arg("x"), py::arg("y"), py::arg("z"), py::arg("color"),
"Add a point with coordinates (x, y, z) and color to the PointCloud.")
.def("size", &PointCloud::size,
"Get the number of points in the PointCloud.")
.def("getPoints", &PointCloud::getPoints, py::return_value_policy::reference_internal,
"Get the list of points in the PointCloud.")
.def("computeCentroid", &PointCloud::computeCentroid,
"Compute the centroid of the PointCloud.")
.def("clear", &PointCloud::clear,
"Clear all points from the PointCloud.")
.def("transform", &PointCloud::transform, py::arg("transformation"),
"Apply a 4x4 transformation matrix to the PointCloud.")
.def("rotate", &PointCloud::rotate, py::arg("rotation_matrix"),
"Rotate the PointCloud using a 3x3 rotation matrix.")
.def("translate", &PointCloud::translate, py::arg("translation_vector"),
"Translate the PointCloud using a 3D translation vector.")
.def("downsample", &PointCloud::downsample, py::arg("voxel_size"),
"Downsample the PointCloud using a voxel grid filter with the specified voxel size.")
.def("statisticalOutlierRemoval", &PointCloud::statisticalOutlierRemoval, py::arg("mean_k") = 50, py::arg("std_dev_mul_thresh") = 1.0,
"Remove statistical outliers from the PointCloud.")
.def("radiusOutlierRemoval", &PointCloud::radiusOutlierRemoval, py::arg("radius"), py::arg("min_neighbors"),
"Remove points with fewer than min_neighbors within the specified radius.")
.def("computeNormals", &PointCloud::computeNormals, py::arg("k") = 10,
"Compute normals for each point in the PointCloud using k nearest neighbors.")
.def("icp", &PointCloud::icp, py::arg("target"), py::arg("max_iterations") = 50, py::arg("tolerance") = 1e-6,
"Register the PointCloud to a target PointCloud using the ICP algorithm. Returns the transformation matrix.")
.def("euclideanClustering", &PointCloud::euclideanClustering, py::arg("cluster_tolerance"), py::arg("min_size"), py::arg("max_size"),
"Perform Euclidean clustering on the PointCloud. Returns a list of clustered PointClouds.")
.def("merge", &PointCloud::merge, py::arg("other"),
"Merge another PointCloud into this PointCloud.")
.def("mergeWith", &PointCloud::mergeWith, py::arg("other"), py::arg("transformation"),
"Merge another PointCloud into this PointCloud with a specified transformation.")
.def("computeStats", &PointCloud::computeStats,
"Compute statistical information of the PointCloud.")
.def("loadFromPLY", &PointCloud::loadFromPLY, py::arg("file_path"),
"Load PointCloud data from a PLY file.")
.def("saveToPLY", &PointCloud::saveToPLY, py::arg("file_path"),
"Save PointCloud data to a PLY file.")
.def("loadFromPCD", &PointCloud::loadFromPCD, py::arg("file_path"),
"Load PointCloud data from a PCD file.")
.def("saveToPCD", &PointCloud::saveToPCD, py::arg("file_path"),
"Save PointCloud data to a PCD file.");

// Optionally, add helper functions to convert between cv::Mat and numpy arrays
m.def("mat_to_numpy", &mat_to_numpy, py::arg("mat"),
"Convert a cv::Mat to a numpy array.");
m.def("numpy_to_mat", &numpy_to_mat, py::arg("array"),
"Convert a numpy array to a cv::Mat.");

}