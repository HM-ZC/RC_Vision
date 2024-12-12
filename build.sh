#!/bin/bash

# build_all.sh - 自动化构建 RC Vision 项目

# 退出脚本时发生错误
set -e

# 函数：打印消息
echo_info() {
    echo -e "\033[1;34m[INFO]\033[0m $1"
}

# 函数：打印错误并退出
echo_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
    exit 1
}

# 项目根目录
PROJECT_ROOT=$(pwd)

# 安装目录
INSTALL_CORE_DIR="${PROJECT_ROOT}/install_core"
INSTALL_ROS2_DIR="${PROJECT_ROOT}/install_ros2"

# 构建并安装 core_lib
build_core_lib() {
    echo_info "开始构建并安装 core_lib..."
    mkdir -p "${PROJECT_ROOT}/core_lib/build"
    cd "${PROJECT_ROOT}/core_lib/build"

    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${INSTALL_CORE_DIR}"
    make -j$(nproc)
    make install

    echo_info "core_lib 构建和安装完成。"
    cd "${PROJECT_ROOT}"
}

# 构建 ROS1 工作空间
build_ros1_workspace() {
    echo_info "开始构建 ROS1 工作空间..."

    # 确保 ROS1 环境已被 source
    if [ -z "$ROS_DISTRO" ]; then
        echo_error "ROS1 环境未被 source。请先 source ROS1 的 setup.bash，例如：source /opt/ros/noetic/setup.bash"
    fi

    # 设置 CMAKE_PREFIX_PATH 以包含 core_lib 的安装路径
    export CMAKE_PREFIX_PATH="${INSTALL_CORE_DIR}:${CMAKE_PREFIX_PATH}"

    # 进入 ROS1 工作空间
    cd "${PROJECT_ROOT}/ros1"

    # 初始化 Catkin 工作空间（如果尚未初始化）
    if [ ! -f "build/CMakeCache.txt" ]; then
        echo_info "初始化 Catkin 工作空间..."
        catkin_make
    fi

    # 构建 ROS1 包
    echo_info "构建 ROS1 包..."
    catkin_make -DCMAKE_PREFIX_PATH="${INSTALL_CORE_DIR}:${CMAKE_PREFIX_PATH}"
    catkin_make install

    echo_info "ROS1 工作空间构建完成。"
    cd "${PROJECT_ROOT}"
}

# 构建 ROS2 工作空间
build_ros2_workspace() {
    echo_info "开始构建 ROS2 工作空间..."

    # 确保 ROS2 环境已被 source
    if [ -z "$AMENT_PREFIX_PATH" ]; then
        echo_error "ROS2 环境未被 source。请先 source ROS2 的 setup.bash，例如：source /opt/ros/foxy/setup.bash"
    fi

    # 设置 CMAKE_PREFIX_PATH 以包含 core_lib 的安装路径
    export CMAKE_PREFIX_PATH="${INSTALL_CORE_DIR}:${CMAKE_PREFIX_PATH}"

    # 进入 ROS2 工作空间
    cd "${PROJECT_ROOT}/ros2"

    # 构建 ROS2 包
    echo_info "构建 ROS2 包..."
    colcon build --install-base "${INSTALL_ROS2_DIR}" --cmake-args -DCMAKE_PREFIX_PATH="${INSTALL_CORE_DIR}:${CMAKE_PREFIX_PATH}"

    echo_info "ROS2 工作空间构建完成。"
    cd "${PROJECT_ROOT}"
}

# 构建 示例程序
build_examples() {
    echo_info "开始构建示例程序..."
    mkdir -p "${PROJECT_ROOT}/examples/build"
    cd "${PROJECT_ROOT}/examples/build"

    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="${INSTALL_CORE_DIR}:${CMAKE_PREFIX_PATH}"
    make -j$(nproc)
    make install

    echo_info "示例程序构建完成。可执行文件已安装到 bin/examples 目录。"
    cd "${PROJECT_ROOT}"
}

# 构建 Python 绑定
build_python_bindings() {
    echo_info "开始构建 Python 绑定..."
    mkdir -p "${PROJECT_ROOT}/python_bindings/build"
    cd "${PROJECT_ROOT}/python_bindings/build"

    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="${INSTALL_CORE_DIR}:${CMAKE_PREFIX_PATH}"
    make -j$(nproc)
    make install

    echo_info "Python 绑定构建完成。"
    cd "${PROJECT_ROOT}"
}

# 运行单元测试
run_tests() {
    echo_info "开始运行单元测试..."
    cd "${PROJECT_ROOT}/core_lib/tests"

    mkdir -p build
    cd build
    cmake ..
    make -j$(nproc)
    ctest --output-on-failure

    echo_info "所有单元测试通过。"
    cd "${PROJECT_ROOT}"
}

# 主流程
main() {
    build_core_lib
    build_ros1_workspace
    build_ros2_workspace
    build_examples
    build_python_bindings
    run_tests
    echo_info "所有构建步骤已完成。"
}

# 执行主流程
main