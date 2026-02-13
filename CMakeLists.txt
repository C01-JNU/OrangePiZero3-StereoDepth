cmake_minimum_required(VERSION 3.16)
project(OrangePiZero3-StereoDepth VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

# 查找依赖包
find_package(Vulkan REQUIRED)
find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Threads REQUIRED)

# ============================================
# 添加子模块
# ============================================
# CPU 立体匹配模块
add_subdirectory(src/cpu_stereo)

# 标定模块（如果存在）
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/calibration/CMakeLists.txt)
    add_subdirectory(src/calibration)
endif()

# 项目包含目录
include_directories(${PROJECT_SOURCE_DIR}/src)

# 源文件分组
set(VULKAN_SOURCES
    src/vulkan/context.cpp
    src/vulkan/buffer_manager.cpp
    src/vulkan/compute_pipeline.cpp
    src/vulkan/stereo_pipeline.cpp
)

set(UTILS_SOURCES
    src/utils/config.cpp
    src/utils/logger.cpp
)

set(MAIN_SOURCES src/main.cpp)
set(TEST_SOURCES src/test.cpp)

# 主可执行文件
add_executable(stereo_depth ${MAIN_SOURCES} ${VULKAN_SOURCES} ${UTILS_SOURCES})
target_link_libraries(stereo_depth
    Vulkan::Vulkan
    ${OpenCV_LIBS}
    spdlog::spdlog_header_only
    yaml-cpp
    Threads::Threads
    cpu_stereo          # CPU 立体匹配库
)
set_target_properties(stereo_depth PROPERTIES
    OUTPUT_NAME "stereo_depth"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# 测试可执行文件
add_executable(stereo_depth_test ${TEST_SOURCES} ${VULKAN_SOURCES} ${UTILS_SOURCES})
target_link_libraries(stereo_depth_test
    Vulkan::Vulkan
    ${OpenCV_LIBS}
    spdlog::spdlog_header_only
    yaml-cpp
    Threads::Threads
    cpu_stereo          # CPU 立体匹配库
)
set_target_properties(stereo_depth_test PROPERTIES
    OUTPUT_NAME "test_stereo_depth"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# 安装规则（可选）
install(TARGETS stereo_depth DESTINATION bin)

# ------------------------------------------------------------
# 着色器生成与编译（GPU 专用，保持不变）
# ------------------------------------------------------------
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
add_dependencies(stereo_depth compile_shaders)
add_dependencies(stereo_depth_test compile_shaders)
if(TARGET stereo_calibrator)
    add_dependencies(stereo_calibrator compile_shaders)
endif()

# ------------------------------------------------------------
# 配置摘要
# ------------------------------------------------------------
message(STATUS "=========================================")
message(STATUS "OrangePiZero3-StereoDepth Configuration")
message(STATUS "=========================================")
message(STATUS "Project version: ${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Output binaries: ${CMAKE_BINARY_DIR}/bin")
message(STATUS "Vulkan: YES")
message(STATUS "OpenCV: YES")
message(STATUS "spdlog: YES")
message(STATUS "yaml-cpp: YES")
message(STATUS "Pyramid levels: ${PYRAMID_LEVELS}")
message(STATUS "Shader SPIR-V dir: ${SPV_DIR}")
message(STATUS "CPU stereo module: YES (src/cpu_stereo)")
message(STATUS "=========================================")
