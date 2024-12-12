cmake_minimum_required(VERSION 3.10)
project(rc_vision_python_bindings)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 pybind11
find_package(pybind11 REQUIRED)
find_package(rc_vision_core REQUIRED)

# 包含头文件
include_directories(
        ${rc_vision_core_INCLUDE_DIRS}
)

# 创建 Python 模块
pybind11_add_module(rc_vision_py src/pybind_module.cpp)

# 链接核心库
target_link_libraries(rc_vision_py PRIVATE rc_vision_core)

# 安装规则
install(TARGETS rc_vision_py
        LIBRARY DESTINATION lib/python3/dist-packages
)