cmake_minimum_required(VERSION 3.14)
project(rc_vision_utils LANGUAGES CXX)

# 包含头文件目录
include_directories(
        ${CMAKE_SOURCE_DIR}/core_lib/include
)

# 定义工具程序
add_executable(camera_calibration camera_calibration.cpp)
add_executable(generate_camera_params generate_camera_params.cpp)
add_executable(stereo_camera_calibration stereo_camera_calibration.cpp)

# 定义一个列表包含所有工具程序的名称
set(UTILS_EXECUTABLES
        camera_calibration
        generate_camera_params
        stereo_camera_calibration
)

# 为每个工具程序设置链接库和 C++ 标准
foreach(EXE ${UTILS_EXECUTABLES})
    target_link_libraries(${EXE} PRIVATE rc_vision_core OpenCV::OpenCV Eigen3::Eigen)

    set_target_properties(${EXE} PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
    )

    # 安装每个工具程序
    install(TARGETS ${EXE}
            RUNTIME DESTINATION bin
    )
endforeach()