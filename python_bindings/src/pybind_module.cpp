#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "rc_vision/core/camera.hpp"
#include "rc_vision/core/image.hpp"
#include "rc_vision/core/logger.hpp"

namespace py = pybind11;
using namespace rc_vision::core;

PYBIND11_MODULE(rc_vision_py, m) {
m.doc() = "Python bindings for rc_vision_core";

// 绑定 Logger 单例
py::class_<Logger>(m, "Logger")
.def_static("get_instance", &Logger::getInstance)
.def("set_log_level", &Logger::setLogLevel)
.def("log", &Logger::log);

// 绑定 Camera 类
py::class_<Camera>(m, "Camera")
.def(py::init<>())
.def(py::init<const Eigen::Matrix3d&, const Eigen::Matrix<double, 1, 5>&>())
.def("get_intrinsic", &Camera::getIntrinsic)
.def("get_distortion", &Camera::getDistortion)
.def("set_intrinsic", &Camera::setIntrinsic)
.def("set_distortion", &Camera::setDistortion)
.def("project", &Camera::project)
.def("undistort", &Camera::undistort);

// 绑定 Image 类
py::class_<Image>(m, "Image")
.def(py::init<>())
.def(py::init<const std::string&>())
.def("load", &Image::load)
.def("save", &Image::save)
.def("get_mat", &Image::getMat, py::return_value_policy::reference_internal)
.def("set_mat", &Image::setMat);
}
