cmake_minimum_required(VERSION 3.16)
project(OrangePiZero3-StereoDepth VERSION 1.0.0 LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 编译选项
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# 项目配置
set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

# 查找依赖包
find_package(Vulkan REQUIRED)
find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)
find_package(yaml-cpp REQUIRED)

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

set(MAIN_SOURCES
    src/main.cpp
)

set(TEST_SOURCES
    src/test.cpp
)

# 主可执行文件
add_executable(stereo_depth ${MAIN_SOURCES} ${VULKAN_SOURCES} ${UTILS_SOURCES})

# 链接库
target_link_libraries(stereo_depth
    Vulkan::Vulkan
    ${OpenCV_LIBS}
    spdlog::spdlog_header_only
    yaml-cpp
    Threads::Threads
)

# 添加pthread库，因为Vulkan可能需要
find_package(Threads REQUIRED)

# 设置输出目录
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
)

set_target_properties(stereo_depth_test PROPERTIES
    OUTPUT_NAME "test_stereo_depth"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# 安装规则（可选）
install(TARGETS stereo_depth DESTINATION bin)

# 配置生成
find_program(PYTHON3 python3)
if(PYTHON3)
    # 配置生成目标
    set(GENERATED_DIR ${PROJECT_SOURCE_DIR}/src/vulkan/generated)
    
    add_custom_command(
        OUTPUT ${GENERATED_DIR}/shader_params.hpp
               ${GENERATED_DIR}/shader_params.glsl
        COMMAND ${PYTHON3} ${PROJECT_SOURCE_DIR}/tools/generate_shader_config.py
                ${PROJECT_SOURCE_DIR}/config/global_config.yaml
        DEPENDS ${PROJECT_SOURCE_DIR}/config/global_config.yaml
                ${PROJECT_SOURCE_DIR}/tools/generate_shader_config.py
        COMMENT "生成着色器参数配置"
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    )
    
    add_custom_target(generate_shader_params ALL
        DEPENDS ${GENERATED_DIR}/shader_params.hpp
                ${GENERATED_DIR}/shader_params.glsl
    )
    
    # 添加到可执行文件依赖
    add_dependencies(stereo_depth generate_shader_params)
    add_dependencies(stereo_depth_test generate_shader_params)
    
    message(STATUS "Python3 found: 着色器参数生成启用")
else()
    message(WARNING "Python3 not found: 着色器参数生成禁用")
endif()

# 着色器编译
find_program(GLSLANG_VALIDATOR glslangValidator)
if(GLSLANG_VALIDATOR)
    # 创建SPV输出目录
    set(SPV_DIR ${PROJECT_SOURCE_DIR}/src/vulkan/spv)
    set(SHADER_DIR ${PROJECT_SOURCE_DIR}/src/vulkan/shaders)
    set(BUILD_SHADER_DIR ${CMAKE_BINARY_DIR}/shaders)
    
    # 查找着色器文件
    file(GLOB SHADER_FILES ${SHADER_DIR}/*.comp)
    
    # 为每个着色器创建编译规则
    foreach(SHADER_FILE ${SHADER_FILES})
        get_filename_component(SHADER_NAME ${SHADER_FILE} NAME_WE)
        
        # 合并后的着色器文件（在构建目录中）
        set(MERGED_SHADER ${BUILD_SHADER_DIR}/${SHADER_NAME}_merged.comp)
        set(SPV_FILE ${SPV_DIR}/${SHADER_NAME}.comp.spv)
        
        # 合并着色器和参数文件
        add_custom_command(
            OUTPUT ${MERGED_SHADER}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${BUILD_SHADER_DIR}
            COMMAND ${PYTHON3} ${PROJECT_SOURCE_DIR}/tools/merge_shader.py
                    ${SHADER_FILE}
                    ${GENERATED_DIR}/shader_params.glsl
                    ${MERGED_SHADER}
            DEPENDS ${SHADER_FILE} ${GENERATED_DIR}/shader_params.glsl
                    ${PROJECT_SOURCE_DIR}/tools/merge_shader.py
            COMMENT "合并着色器: ${SHADER_NAME}.comp"
        )
        
        # 编译合并后的着色器
        add_custom_command(
            OUTPUT ${SPV_FILE}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${SPV_DIR}
            COMMAND ${GLSLANG_VALIDATOR} -V ${MERGED_SHADER} -o ${SPV_FILE}
            DEPENDS ${MERGED_SHADER}
            COMMENT "编译着色器: ${SHADER_NAME}.comp -> ${SHADER_NAME}.comp.spv"
        )
        
        list(APPEND ALL_SPV_FILES ${SPV_FILE})
    endforeach()
    
    # 创建着色器编译目标
    if(ALL_SPV_FILES)
        add_custom_target(compile_shaders ALL DEPENDS ${ALL_SPV_FILES})
        
        # 添加依赖关系
        add_dependencies(compile_shaders generate_shader_params)
        add_dependencies(stereo_depth compile_shaders)
        add_dependencies(stereo_depth_test compile_shaders)
        
        message(STATUS "Found glslangValidator: 着色器编译启用")
        message(STATUS "SPV输出目录: ${SPV_DIR}")
    endif()
else()
    message(WARNING "glslangValidator not found: 着色器编译禁用")
endif()

# 显示配置信息
message(STATUS "=========================================")
message(STATUS "OrangePiZero3-StereoDepth Configuration")
message(STATUS "=========================================")
message(STATUS "Project version: ${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Output binaries: ${CMAKE_BINARY_DIR}/bin")
message(STATUS "Vulkan found: YES")
message(STATUS "OpenCV found: YES")
message(STATUS "spdlog found: YES")
message(STATUS "yaml-cpp found: YES")
if(PYTHON3)
    message(STATUS "着色器参数生成: 启用")
else()
    message(STATUS "着色器参数生成: 禁用")
endif()
if(GLSLANG_VALIDATOR)
    message(STATUS "着色器编译: 启用")
else()
    message(STATUS "着色器编译: 禁用")
endif()
message(STATUS "=========================================")
