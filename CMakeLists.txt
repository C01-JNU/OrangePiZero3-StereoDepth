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
find_package(yaml-cpp REQUIRED)  # 添加yaml-cpp

# 项目包含目录
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)

# 源文件
set(SOURCES
    src/main.cpp
    src/vulkan/context.cpp
    src/utils/config.cpp
    src/utils/logger.cpp
)

# 可执行文件
add_executable(stereo_depth ${SOURCES})

# 链接库
target_link_libraries(stereo_depth
    Vulkan::Vulkan
    ${OpenCV_LIBS}
    spdlog::spdlog_header_only
    yaml-cpp  # 链接yaml-cpp
)

# 添加pthread库，因为Vulkan可能需要
find_package(Threads REQUIRED)
target_link_libraries(stereo_depth Threads::Threads)

# 安装规则（可选）
install(TARGETS stereo_depth DESTINATION bin)
