cmake_minimum_required(VERSION 3.16)
project(orangepizero3_stereodepth VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ============================================================================
# 读取配置文件中的 system.mode 和 ros2.enabled
# ============================================================================
find_program(PYTHON3 python3 REQUIRED)

execute_process(
    COMMAND ${PYTHON3} -c "
import yaml, sys
try:
    with open('${CMAKE_CURRENT_SOURCE_DIR}/config/global_config.yaml') as f:
        cfg = yaml.safe_load(f)
    mode = cfg.get('system', {}).get('mode', 'cpu')
    ros2_enabled = cfg.get('ros2', {}).get('enabled', False)
    print(f'{mode} {1 if ros2_enabled else 0}')
except Exception as e:
    print('cpu 0')
"
    OUTPUT_VARIABLE CONFIG_VALUES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

string(REPLACE " " ";" CONFIG_LIST ${CONFIG_VALUES})
list(GET CONFIG_LIST 0 SYSTEM_MODE)
list(GET CONFIG_LIST 1 ROS2_ENABLED_NUM)

message(STATUS "Detected system.mode from config: ${SYSTEM_MODE}")
message(STATUS "Detected ros2.enabled from config: ${ROS2_ENABLED_NUM}")

if(SYSTEM_MODE STREQUAL "gpu")
    set(ENABLE_GPU_DEFAULT ON)
    set(ENABLE_CPU_DEFAULT OFF)
else()
    set(ENABLE_GPU_DEFAULT OFF)
    set(ENABLE_CPU_DEFAULT ON)
endif()

option(ENABLE_GPU "Build Vulkan GPU stereo module" ${ENABLE_GPU_DEFAULT})
option(ENABLE_CPU "Build CPU stereo module" ${ENABLE_CPU_DEFAULT})

if(ROS2_ENABLED_NUM EQUAL 1)
    set(BUILD_ROS2_NODE_DEFAULT ON)
else()
    set(BUILD_ROS2_NODE_DEFAULT OFF)
endif()

option(BUILD_ROS2_NODE "Build ROS2 node" ${BUILD_ROS2_NODE_DEFAULT})

message(STATUS "ENABLE_GPU: ${ENABLE_GPU}")
message(STATUS "ENABLE_CPU: ${ENABLE_CPU}")
message(STATUS "BUILD_ROS2_NODE: ${BUILD_ROS2_NODE}")

# ============================================================================
# 查找依赖
# ============================================================================
find_package(Vulkan REQUIRED)
find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Threads REQUIRED)

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
# 添加子模块（根据选项）
# ============================================================================
if(ENABLE_GPU)
    add_subdirectory(src/vulkan)
endif()

if(ENABLE_CPU)
    add_subdirectory(src/cpu_stereo)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/calibration/CMakeLists.txt)
    add_subdirectory(src/calibration)
endif()

add_subdirectory(src/camera)

# ============================================================================
# 主可执行文件（独立程序）
# ============================================================================
add_executable(stereo_depth ${MAIN_SOURCES} ${UTILS_SOURCES})
target_compile_definitions(stereo_depth PRIVATE
    ENABLE_GPU=$<BOOL:${ENABLE_GPU}>
    ENABLE_CPU=$<BOOL:${ENABLE_CPU}>
)
target_link_libraries(stereo_depth PRIVATE
    ${OpenCV_LIBS}
    spdlog::spdlog_header_only
    yaml-cpp
    Threads::Threads
    camera
)
if(ENABLE_GPU AND TARGET vulkan_stereo)
    target_link_libraries(stereo_depth PRIVATE vulkan_stereo)
endif()
if(ENABLE_CPU AND TARGET cpu_stereo)
    target_link_libraries(stereo_depth PRIVATE cpu_stereo)
endif()
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
    ENABLE_GPU=$<BOOL:${ENABLE_GPU}>
    ENABLE_CPU=$<BOOL:${ENABLE_CPU}>
)
target_link_libraries(stereo_depth_test PRIVATE
    ${OpenCV_LIBS}
    spdlog::spdlog_header_only
    yaml-cpp
    Threads::Threads
    camera
)
if(ENABLE_GPU AND TARGET vulkan_stereo)
    target_link_libraries(stereo_depth_test PRIVATE vulkan_stereo)
endif()
if(ENABLE_CPU AND TARGET cpu_stereo)
    target_link_libraries(stereo_depth_test PRIVATE cpu_stereo)
endif()
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
        ENABLE_GPU=$<BOOL:${ENABLE_GPU}>
        ENABLE_CPU=$<BOOL:${ENABLE_CPU}>
    )

    # 首先调用 ament_target_dependencies（内部使用 plain 签名）
    ament_target_dependencies(stereo_depth_node
        rclcpp
        sensor_msgs
        cv_bridge
        image_transport
        std_msgs
    )

    # 然后链接非 ROS 依赖，必须使用 plain 签名（无关键字）
    target_link_libraries(stereo_depth_node
        ${OpenCV_LIBS}
        spdlog::spdlog_header_only
        yaml-cpp
        Threads::Threads
        camera
    )
    if(ENABLE_GPU AND TARGET vulkan_stereo)
        target_link_libraries(stereo_depth_node vulkan_stereo)
    endif()
    if(ENABLE_CPU AND TARGET cpu_stereo)
        target_link_libraries(stereo_depth_node cpu_stereo)
    endif()
    if(TARGET calibration_utils)
        target_link_libraries(stereo_depth_node calibration_utils)
    endif()

    # 安装节点和相关文件
    install(TARGETS stereo_depth_node
        DESTINATION lib/${PROJECT_NAME}
    )
    install(DIRECTORY src/ros2_node/launch/
        DESTINATION share/${PROJECT_NAME}/launch
    )
    install(DIRECTORY src/ros2_node/config/
        DESTINATION share/${PROJECT_NAME}/config
    )
    # 安装全局配置文件和标定结果到 share 目录
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
# GPU 着色器生成与编译（仅在 GPU 启用时执行）
# ============================================================================
if(ENABLE_GPU)
    find_program(PYTHON3 python3)
    find_program(GLSLANG_VALIDATOR glslangValidator)

    if(NOT PYTHON3)
        message(FATAL_ERROR "Python3 not found, required for shader generation")
    endif()
    if(NOT GLSLANG_VALIDATOR)
        message(FATAL_ERROR "glslangValidator not found, required for shader compilation")
    endif()

    set(CONFIG_FILE ${PROJECT_SOURCE_DIR}/config/global_config.yaml)
    set(GENERATED_DIR ${PROJECT_SOURCE_DIR}/src/vulkan/generated)
    set(SPV_DIR ${PROJECT_SOURCE_DIR}/src/vulkan/spv)
    set(BUILD_SHADER_DIR ${CMAKE_BINARY_DIR}/shaders)

    # 获取金字塔层数
    execute_process(
        COMMAND ${PYTHON3} -c "
import yaml
with open('${CONFIG_FILE}') as f: cfg = yaml.safe_load(f)
print(cfg.get('pyramid', {}).get('levels', 1))"
        OUTPUT_VARIABLE PYRAMID_LEVELS
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(STATUS "Pyramid levels: ${PYRAMID_LEVELS}")

    # 为每层生成参数文件
    foreach(LEVEL RANGE 0 ${PYRAMID_LEVELS})
        set(LAYER_GEN_DIR ${GENERATED_DIR}/layer${LEVEL})
        add_custom_command(
            OUTPUT ${LAYER_GEN_DIR}/shader_params.hpp ${LAYER_GEN_DIR}/shader_params.glsl
            COMMAND ${CMAKE_COMMAND} -E make_directory ${LAYER_GEN_DIR}
            COMMAND ${PYTHON3} ${PROJECT_SOURCE_DIR}/tools/generate_shader_config.py
                    ${CONFIG_FILE} --layer ${LEVEL} --outdir ${LAYER_GEN_DIR}
            DEPENDS ${CONFIG_FILE} ${PROJECT_SOURCE_DIR}/tools/generate_shader_config.py
            COMMENT "生成第 ${LEVEL} 层着色器参数"
        )
        list(APPEND ALL_LAYER_PARAMS ${LAYER_GEN_DIR}/shader_params.hpp)
    endforeach()

    # 生成金字塔配置头文件
    set(PYRAMID_CONFIG_FILE ${GENERATED_DIR}/pyramid_config.hpp)
    add_custom_command(
        OUTPUT ${PYRAMID_CONFIG_FILE}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${GENERATED_DIR}
        COMMAND ${PYTHON3} ${PROJECT_SOURCE_DIR}/tools/generate_pyramid_config.py
                ${CONFIG_FILE} --outfile ${PYRAMID_CONFIG_FILE}
        DEPENDS ${CONFIG_FILE} ${PROJECT_SOURCE_DIR}/tools/generate_pyramid_config.py
        COMMENT "生成金字塔配置类"
    )
    list(APPEND ALL_GENERATED_HEADERS ${PYRAMID_CONFIG_FILE})

    add_custom_target(generate_shader_params ALL DEPENDS ${ALL_LAYER_PARAMS} ${PYRAMID_CONFIG_FILE})

    # 编译每层着色器
    file(GLOB SHADER_COMP_FILES ${PROJECT_SOURCE_DIR}/src/vulkan/shaders/*.comp)
    foreach(SHADER_FILE ${SHADER_COMP_FILES})
        get_filename_component(SHADER_NAME ${SHADER_FILE} NAME_WE)
        foreach(LEVEL RANGE 0 ${PYRAMID_LEVELS})
            set(LAYER_GEN_DIR ${GENERATED_DIR}/layer${LEVEL})
            set(MERGED_SHADER ${BUILD_SHADER_DIR}/layer${LEVEL}/${SHADER_NAME}_merged.comp)
            set(SPV_FILE ${SPV_DIR}/${SHADER_NAME}_layer${LEVEL}.comp.spv)

            add_custom_command(
                OUTPUT ${MERGED_SHADER}
                COMMAND ${CMAKE_COMMAND} -E make_directory ${BUILD_SHADER_DIR}/layer${LEVEL}
                COMMAND ${PYTHON3} ${PROJECT_SOURCE_DIR}/tools/merge_shader.py
                        ${SHADER_FILE}
                        ${LAYER_GEN_DIR}/shader_params.glsl
                        ${MERGED_SHADER}
                DEPENDS ${SHADER_FILE} ${LAYER_GEN_DIR}/shader_params.glsl
                        ${PROJECT_SOURCE_DIR}/tools/merge_shader.py
                COMMENT "合并着色器: ${SHADER_NAME} (层 ${LEVEL})"
            )

            add_custom_command(
                OUTPUT ${SPV_FILE}
                COMMAND ${CMAKE_COMMAND} -E make_directory ${SPV_DIR}
                COMMAND ${GLSLANG_VALIDATOR} -V ${MERGED_SHADER} -o ${SPV_FILE}
                DEPENDS ${MERGED_SHADER}
                COMMENT "编译着色器: ${SHADER_NAME}_layer${LEVEL}.comp.spv"
            )
            list(APPEND ALL_SPV_FILES ${SPV_FILE})
        endforeach()
    endforeach()

    add_custom_target(compile_shaders ALL DEPENDS ${ALL_SPV_FILES})
    add_dependencies(compile_shaders generate_shader_params)

    if(TARGET stereo_depth)
        add_dependencies(stereo_depth compile_shaders)
    endif()
    if(TARGET stereo_depth_test)
        add_dependencies(stereo_depth_test compile_shaders)
    endif()
    if(BUILD_ROS2_NODE AND TARGET stereo_depth_node)
        add_dependencies(stereo_depth_node compile_shaders)
    endif()
endif()

# ============================================================================
# 复制配置文件和数据到可执行文件输出目录（仅用于独立程序，不影响 install）
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

if(ENABLE_GPU)
    add_custom_command(TARGET stereo_depth POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:stereo_depth>/spv
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/spv $<TARGET_FILE_DIR:stereo_depth>/spv
        COMMENT "复制 SPIR-V 着色器到 stereo_depth 目录"
    )
    add_custom_command(TARGET stereo_depth_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:stereo_depth_test>/spv
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/spv $<TARGET_FILE_DIR:stereo_depth_test>/spv
        COMMENT "复制 SPIR-V 着色器到 stereo_depth_test 目录"
    )
endif()

# ============================================================================
# 安装独立程序（可选）
# ============================================================================
install(TARGETS stereo_depth DESTINATION bin)

# ============================================================================
# 必须调用 ament_package() 以生成 ROS2 包所需的文件
# ============================================================================
ament_package()

message(STATUS "=========================================")
message(STATUS "orangepizero3_stereodepth Configuration")
message(STATUS "=========================================")
message(STATUS "Project version: ${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Output binaries: ${CMAKE_BINARY_DIR}/bin")
message(STATUS "ENABLE_GPU: ${ENABLE_GPU}")
message(STATUS "ENABLE_CPU: ${ENABLE_CPU}")
message(STATUS "BUILD_ROS2_NODE: ${BUILD_ROS2_NODE}")
message(STATUS "=========================================")
