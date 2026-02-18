cmake_minimum_required(VERSION 3.16)
project(orangepizero3_stereodepth VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ============================================================================
# 读取配置文件中的 ros2.enabled
# ============================================================================
find_program(PYTHON3 python3 REQUIRED)

execute_process(
    COMMAND ${PYTHON3} -c "
import yaml, sys
try:
    with open('${CMAKE_CURRENT_SOURCE_DIR}/config/global_config.yaml') as f:
        cfg = yaml.safe_load(f)
    ros2_enabled = cfg.get('ros2', {}).get('enabled', False)
    print('1' if ros2_enabled else '0')
except Exception as e:
    print('0')
"
    OUTPUT_VARIABLE ROS2_ENABLED_NUM
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(ROS2_ENABLED_NUM EQUAL 1)
    set(BUILD_ROS2_NODE_DEFAULT ON)
else()
    set(BUILD_ROS2_NODE_DEFAULT OFF)
endif()

option(BUILD_ROS2_NODE "Build ROS2 node" ${BUILD_ROS2_NODE_DEFAULT})

message(STATUS "BUILD_ROS2_NODE: ${BUILD_ROS2_NODE}")

# ============================================================================
# 查找依赖
# ============================================================================
find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)   # 保留 Eigen

if(BUILD_ROS2_NODE)
    find_package(rclcpp REQUIRED)
    find_package(sensor_msgs REQUIRED)
    find_package(cv_bridge REQUIRED)
    find_package(image_transport REQUIRED)
    find_package(std_msgs REQUIRED)
    find_package(ament_cmake REQUIRED)
endif()

# ============================================================================
# 项目包含目录
# ============================================================================
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${EIGEN3_INCLUDE_DIRS})  # Eigen 头文件路径

# ============================================================================
# 通用工具模块
# ============================================================================
set(UTILS_SOURCES
    src/utils/config.cpp
    src/utils/logger.cpp
)

set(MAIN_SOURCES src/main.cpp)
set(TEST_SOURCES src/test.cpp)

# ============================================================================
# 添加子模块
# ============================================================================
add_subdirectory(src/cpu_stereo)
add_subdirectory(src/calibration)
add_subdirectory(src/camera)

# ============================================================================
# 主可执行文件（独立程序）
# ============================================================================
add_executable(stereo_depth ${MAIN_SOURCES} ${UTILS_SOURCES})
target_compile_definitions(stereo_depth PRIVATE
    ENABLE_GPU=OFF
    ENABLE_CPU=ON
)
target_link_libraries(stereo_depth PRIVATE
    ${OpenCV_LIBS}
    spdlog::spdlog_header_only
    yaml-cpp
    Threads::Threads
    camera
    cpu_stereo
    ${EIGEN3_LIBRARIES}
)
if(TARGET calibration_utils)
    target_link_libraries(stereo_depth PRIVATE calibration_utils)
endif()
set_target_properties(stereo_depth PROPERTIES
    OUTPUT_NAME "stereo_depth"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# ============================================================================
# 测试可执行文件
# ============================================================================
add_executable(stereo_depth_test ${TEST_SOURCES} ${UTILS_SOURCES})
target_compile_definitions(stereo_depth_test PRIVATE
    ENABLE_GPU=OFF
    ENABLE_CPU=ON
)
target_link_libraries(stereo_depth_test PRIVATE
    ${OpenCV_LIBS}
    spdlog::spdlog_header_only
    yaml-cpp
    Threads::Threads
    camera
    cpu_stereo
    ${EIGEN3_LIBRARIES}
)
if(TARGET calibration_utils)
    target_link_libraries(stereo_depth_test PRIVATE calibration_utils)
endif()
set_target_properties(stereo_depth_test PROPERTIES
    OUTPUT_NAME "test_stereo_depth"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# ============================================================================
# ROS2 节点（根据 BUILD_ROS2_NODE 编译）
# ============================================================================
if(BUILD_ROS2_NODE)
    add_executable(stereo_depth_node
        src/ros2_node/stereo_depth_node.cpp
        ${UTILS_SOURCES}
    )
    target_compile_definitions(stereo_depth_node PRIVATE
        ENABLE_GPU=OFF
        ENABLE_CPU=ON
    )

    ament_target_dependencies(stereo_depth_node
        rclcpp
        sensor_msgs
        cv_bridge
        image_transport
        std_msgs
    )

    target_link_libraries(stereo_depth_node
        ${OpenCV_LIBS}
        spdlog::spdlog_header_only
        yaml-cpp
        Threads::Threads
        camera
        cpu_stereo
        ${EIGEN3_LIBRARIES}
    )
    if(TARGET calibration_utils)
        target_link_libraries(stereo_depth_node calibration_utils)
    endif()

    install(TARGETS stereo_depth_node
        DESTINATION lib/${PROJECT_NAME}
    )
    install(DIRECTORY src/ros2_node/launch/
        DESTINATION share/${PROJECT_NAME}/launch
    )
    install(DIRECTORY src/ros2_node/config/
        DESTINATION share/${PROJECT_NAME}/config
    )
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/config/global_config.yaml
        DESTINATION share/${PROJECT_NAME}/config
    )
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/calibration_results/
        DESTINATION share/${PROJECT_NAME}/calibration_results
    )
    install(FILES package.xml
        DESTINATION share/${PROJECT_NAME}
    )
endif()

# ============================================================================
# 复制配置文件和数据到可执行文件输出目录
# ============================================================================
add_custom_command(TARGET stereo_depth POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:stereo_depth>/config
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/config/global_config.yaml $<TARGET_FILE_DIR:stereo_depth>/config/
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:stereo_depth>/calibration_results
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/calibration_results/stereo_calibration.yml $<TARGET_FILE_DIR:stereo_depth>/calibration_results/
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:stereo_depth>/images/test
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/images/test $<TARGET_FILE_DIR:stereo_depth>/images/test
    COMMENT "复制数据到 stereo_depth 目录"
)

add_custom_command(TARGET stereo_depth_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:stereo_depth_test>/config
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/config/global_config.yaml $<TARGET_FILE_DIR:stereo_depth_test>/config/
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:stereo_depth_test>/calibration_results
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/calibration_results/stereo_calibration.yml $<TARGET_FILE_DIR:stereo_depth_test>/calibration_results/
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:stereo_depth_test>/images/test
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/images/test $<TARGET_FILE_DIR:stereo_depth_test>/images/test
    COMMENT "复制数据到 stereo_depth_test 目录"
)

install(TARGETS stereo_depth DESTINATION bin)

ament_package()

message(STATUS "=========================================")
message(STATUS "orangepizero3_stereodepth Configuration")
message(STATUS "=========================================")
message(STATUS "Project version: ${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Output binaries: ${CMAKE_BINARY_DIR}/bin")
message(STATUS "BUILD_ROS2_NODE: ${BUILD_ROS2_NODE}")
message(STATUS "=========================================")
