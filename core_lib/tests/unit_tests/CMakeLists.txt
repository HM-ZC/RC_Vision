cmake_minimum_required(VERSION 3.14)
project(rc_vision_tests LANGUAGES CXX)

# 查找 Google Test 库
if(NOT TARGET GTest::gtest_main)
    find_package(GTest REQUIRED)
endif()

# 查找核心库
find_package(rc_vision_core REQUIRED)

# 定义所有测试源文件
set(TEST_SOURCES
        test_filters.cpp
        test_camera.cpp
        test_edge_detector.cpp
        test_image.cpp
        test_logger.cpp
        test_math_utils.cpp
        test_point_cloud.cpp
        test_stereo_camera.cpp
)

# 为每个测试源文件创建一个可执行文件并设置相关属性
foreach(test_src ${TEST_SOURCES})
    # 提取测试名称（去除文件扩展名）
    get_filename_component(test_name ${test_src} NAME_WE)

    # 添加可执行文件
    add_executable(${test_name} ${test_src})

    # 链接核心库和 Google Test
    target_link_libraries(${test_name} PRIVATE rc_vision_core GTest::gtest_main)

    # 设置 C++ 标准
    set_target_properties(${test_name} PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
    )

    # 包含核心库的头文件目录
    target_include_directories(${test_name} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../../core_lib/include
    )

    # 发现并注册测试
    gtest_discover_tests(${test_name})
endforeach()