cmake_minimum_required(VERSION 3.14)
project(rc_vision_core LANGUAGES CXX)

# 包含头文件目录
include_directories(
        include
        ${OpenCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIRS}
        ${YAML_CPP_INCLUDE_DIRS}
        ${nlohmann_json_INCLUDE_DIRS}
)

# 定义核心库的头文件和源文件
set(CORE_HEADERS
        include/rc_vision/core/camera.hpp
        include/rc_vision/core/config.hpp
        include/rc_vision/core/edge_detector.hpp
        include/rc_vision/core/filters.hpp
        include/rc_vision/core/image.hpp
        include/rc_vision/core/logger.hpp
        include/rc_vision/core/math_utils.hpp
        include/rc_vision/core/point_cloud.hpp
)

set(CORE_SOURCES
        src/camera.cpp
        src/config.cpp
        src/edge_detector.cpp
        src/filters.cpp
        src/image.cpp
        src/logger.cpp
        src/math_utils.cpp
        src/point_cloud.cpp
)

# 创建核心库
add_library(rc_vision_core ${CORE_SOURCES})

# 设置公共头文件目录
target_include_directories(rc_vision_core PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# 链接依赖库
target_link_libraries(rc_vision_core PUBLIC
        OpenCV::OpenCV
        Eigen3::Eigen
        yaml-cpp
        nlohmann_json::nlohmann_json
)

# 设置 C++ 标准
set_target_properties(rc_vision_core PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
)

# 安装核心库
install(TARGETS rc_vision_core
        EXPORT rc_vision_coreTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
)

# 安装头文件
install(DIRECTORY include/rc_vision/core/
        DESTINATION include/rc_vision/core
)

# 导出核心库目标，以便其他项目使用
install(EXPORT rc_vision_coreTargets
        FILE rc_vision_coreTargets.cmake
        NAMESPACE rc_vision::core::
        DESTINATION lib/cmake/rc_vision_core
)

# 生成核心库的配置文件
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/rc_vision_coreConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
        "${CMAKE_CURRENT_BINARY_DIR}/rc_vision_coreConfig.cmake"
        INSTALL_DESTINATION lib/cmake/rc_vision_core
)

install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/rc_vision_coreConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/rc_vision_coreConfigVersion.cmake"
        DESTINATION lib/cmake/rc_vision_core
)