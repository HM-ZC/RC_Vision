cmake_minimum_required(VERSION 3.14)
project(rc_vision_examples LANGUAGES CXX)

# 定义一个列表，包含所有示例程序的源文件
set(EXAMPLE_SOURCES
        example_filters.cpp
        example_camera.cpp
        example_edge_detector.cpp
        example_image.cpp
        example_logger.cpp
        example_math_utils.cpp
        example_point_cloud.cpp
)

# 遍历每个示例程序，定义可执行文件、链接库、设置属性并安装
foreach(EXAMPLE_SRC ${EXAMPLE_SOURCES})
    # 获取不带扩展名的可执行文件名
    get_filename_component(EXAMPLE_NAME ${EXAMPLE_SRC} NAME_WE)

    # 定义可执行文件
    add_executable(${EXAMPLE_NAME} ${EXAMPLE_SRC})

    # 链接核心库和其他依赖库
    target_link_libraries(${EXAMPLE_NAME} PRIVATE rc_vision_core OpenCV::OpenCV Eigen3::Eigen)

    # 设置 C++ 标准
    set_target_properties(${EXAMPLE_NAME} PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
    )

    # 安装可执行文件到 bin 目录
    install(TARGETS ${EXAMPLE_NAME}
            RUNTIME DESTINATION bin
    )
endforeach()