cmake_minimum_required(VERSION 3.14)
project(rc_vision_python_bindings)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 寻找 pybind11
find_package(pybind11 REQUIRED)

# 寻找 OpenCV
find_package(OpenCV REQUIRED)

# 寻找 Eigen3
find_package(Eigen3 REQUIRED)

# 寻找 YAML-CPP
find_package(yaml-cpp REQUIRED)

# 寻找 rc_vision_core
find_package(rc_vision_core REQUIRED)

# 包含头文件
include_directories(
        ${pybind11_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIRS}
        ${YAML_CPP_INCLUDE_DIRS}
        ${rc_vision_core_INCLUDE_DIRS}
)

# 源文件
set(SOURCES
        src/pybind_module.cpp
)

# 创建 Python 模块
pybind11_add_module(rc_vision_py ${SOURCES})

# 链接库
target_link_libraries(rc_vision_py PRIVATE
        rc_vision_core
        ${OpenCV_LIBS}
        Eigen3::Eigen
        yaml-cpp
)

# 设置模块属性（可选）
set_target_properties(rc_vision_py PROPERTIES
        PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${PYTHON_MODULE_EXTENSION}"
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

# 安装规则
install(TARGETS rc_vision_py
        LIBRARY DESTINATION lib/python_bindings
        ARCHIVE DESTINATION lib/python_bindings
        RUNTIME DESTINATION bin/python_bindings
)