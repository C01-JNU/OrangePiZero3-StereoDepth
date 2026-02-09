# 标定模块CMake配置文件
cmake_minimum_required(VERSION 3.16)

# 查找依赖
find_package(OpenCV REQUIRED)

# 标定工具可执行文件 - 直接编译所有源文件
add_executable(stereo_calibrator
    stereo_calibrator_main.cpp
    stereo_calibrator.cpp
    ../utils/config.cpp
    ../utils/logger.cpp
)

# 链接依赖 - 与根目录主程序保持一致
target_link_libraries(stereo_calibrator
    ${OpenCV_LIBS}
    spdlog::spdlog_header_only
    yaml-cpp
)

# 设置包含目录
target_include_directories(stereo_calibrator PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${OpenCV_INCLUDE_DIRS}
)

# 设置输出
set_target_properties(stereo_calibrator PROPERTIES
    OUTPUT_NAME "stereo_calibrator"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

message(STATUS "标定工具配置完成")
