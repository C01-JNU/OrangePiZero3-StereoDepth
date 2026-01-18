cmake_minimum_required(VERSION 3.16)
project(OrangePiZero3-StereoDepth VERSION 1.0.0)

# 设置项目信息
set(PROJECT_DESCRIPTION "Real-time Stereo Depth Calculation on OrangePi Zero3")
set(PROJECT_AUTHOR "C01-JNU")
set(PROJECT_DATE "2026-01-18")

# C++标准设置
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# 编译器设置
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
        -Werror
        -O3
        -march=native
        -mtune=native
        -ffast-math
        -fno-exceptions
        -fno-rtti
    )
    
    # ARM平台特定优化
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64")
        add_compile_options(
            -mfpu=neon
            -mfloat-abi=hard
            -ftree-vectorize
        )
    endif()
endif()

# 包含自定义CMake模块
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# 查找依赖
find_package(OpenCV 4.5 REQUIRED COMPONENTS core imgproc calib3d)
find_package(Vulkan REQUIRED)
find_package(yaml-cpp 0.7 REQUIRED)

# 可选依赖
option(WITH_ROS2 "Enable ROS2 support" OFF)
if(WITH_ROS2)
    find_package(ament_cmake REQUIRED)
    find_package(rclcpp REQUIRED)
    find_package(sensor_msgs REQUIRED)
    find_package(cv_bridge REQUIRED)
endif()

# 包含目录
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/vulkan
    ${CMAKE_SOURCE_DIR}/utils
    ${OpenCV_INCLUDE_DIRS}
    ${Vulkan_INCLUDE_DIRS}
    ${yaml-cpp_INCLUDE_DIRS}
)

# 添加可执行文件
add_executable(stereo_depth_main
    src/main.cpp
    src/stereo_pipeline.cpp
    src/camera_interface.cpp
    src/calibration_loader.cpp
    src/disparity_processor.cpp
    src/depth_calculator.cpp
)

# 添加工具库
add_library(stereo_utils STATIC
    utils/config_manager.cpp
    utils/logger.cpp
    utils/image_utils.cpp
    utils/file_utils.cpp
    utils/timer.cpp
    utils/profiler.cpp
)

# 添加Vulkan核心库
add_library(vulkan_core STATIC
    vulkan/vulkan_context.cpp
    vulkan/vulkan_device.cpp
    vulkan/vulkan_pipeline.cpp
    vulkan/vulkan_shader.cpp
    vulkan/vulkan_buffer.cpp
    vulkan/vulkan_image.cpp
    vulkan/stereo_compute_pipeline.cpp
    vulkan/census_compute.cpp
    vulkan/cost_compute.cpp
    vulkan/wta_compute.cpp
    vulkan/postprocess_compute.cpp
)

# 链接库
target_link_libraries(stereo_depth_main
    stereo_utils
    vulkan_core
    ${OpenCV_LIBS}
    ${Vulkan_LIBRARIES}
    ${yaml-cpp_LIBRARIES}
    pthread
    dl
)

# 如果启用ROS2，添加相关链接
if(WITH_ROS2)
    target_link_libraries(stereo_depth_main
        ${rclcpp_LIBRARIES}
        ${sensor_msgs_LIBRARIES}
        ${cv_bridge_LIBRARIES}
    )
endif()

# 着色器编译步骤
add_custom_command(
    OUTPUT ${CMAKE_SOURCE_DIR}/shaders/compiled/census_transform.spv
    COMMAND glslangValidator -V ${CMAKE_SOURCE_DIR}/shaders/source/census_transform.comp -o ${CMAKE_SOURCE_DIR}/shaders/compiled/census_transform.spv
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/source/census_transform.comp
    COMMENT "Compiling census transform shader"
)

add_custom_command(
    OUTPUT ${CMAKE_SOURCE_DIR}/shaders/compiled/cost_computation.spv
    COMMAND glslangValidator -V ${CMAKE_SOURCE_DIR}/shaders/source/cost_computation.comp -o ${CMAKE_SOURCE_DIR}/shaders/compiled/cost_computation.spv
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/source/cost_computation.comp
    COMMENT "Compiling cost computation shader"
)

add_custom_command(
    OUTPUT ${CMAKE_SOURCE_DIR}/shaders/compiled/wta_optimization.spv
    COMMAND glslangValidator -V ${CMAKE_SOURCE_DIR}/shaders/source/wta_optimization.comp -o ${CMAKE_SOURCE_DIR}/shaders/compiled/wta_optimization.spv
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/source/wta_optimization.comp
    COMMENT "Compiling WTA optimization shader"
)

add_custom_command(
    OUTPUT ${CMAKE_SOURCE_DIR}/shaders/compiled/post_processing.spv
    COMMAND glslangValidator -V ${CMAKE_SOURCE_DIR}/shaders/source/post_processing.comp -o ${CMAKE_SOURCE_DIR}/shaders/compiled/post_processing.spv
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/source/post_processing.comp
    COMMENT "Compiling post processing shader"
)

# 添加着色器编译目标
add_custom_target(compile_shaders ALL
    DEPENDS 
        ${CMAKE_SOURCE_DIR}/shaders/compiled/census_transform.spv
        ${CMAKE_SOURCE_DIR}/shaders/compiled/cost_computation.spv
        ${CMAKE_SOURCE_DIR}/shaders/compiled/wta_optimization.spv
        ${CMAKE_SOURCE_DIR}/shaders/compiled/post_processing.spv
)

# 安装规则
install(TARGETS stereo_depth_main DESTINATION bin)
install(DIRECTORY config/ DESTINATION ${CMAKE_INSTALL_PREFIX}/config)
install(DIRECTORY shaders/compiled/ DESTINATION ${CMAKE_INSTALL_PREFIX}/shaders/compiled)
install(FILES calibration_results/stereo_calibration.yml DESTINATION ${CMAKE_INSTALL_PREFIX}/calibration_results)

# 测试配置
option(BUILD_TESTS "Build unit tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ARM平台特定设置
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64")
    message(STATUS "Building for ARM platform")
    add_definitions(-DTARGET_ARM=1)
else()
    add_definitions(-DTARGET_ARM=0)
endif()
