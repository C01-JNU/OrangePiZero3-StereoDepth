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

# 标定测试程序 - 测试校正功能
add_executable(test_calibration
    test_calibration.cpp
    calibration_loader.cpp
    stereo_rectifier.cpp
    ../utils/config.cpp
    ../utils/logger.cpp
)

# 标定库（如果其他模块需要）
add_library(calibration_utils STATIC
    calibration_loader.cpp
    stereo_rectifier.cpp
)

# 链接依赖 - 与根目录主程序保持一致
target_link_libraries(stereo_calibrator
    ${OpenCV_LIBS}
    spdlog::spdlog_header_only
    yaml-cpp
)

target_link_libraries(test_calibration
    ${OpenCV_LIBS}
    spdlog::spdlog_header_only
    yaml-cpp
)

target_link_libraries(calibration_utils
    ${OpenCV_LIBS}
    spdlog::spdlog_header_only
    yaml-cpp
)

# 设置包含目录 - 修复头文件搜索路径
# 可执行文件：需要当前目录和上级目录（src）
target_include_directories(stereo_calibrator PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${OpenCV_INCLUDE_DIRS}
)

target_include_directories(test_calibration PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${OpenCV_INCLUDE_DIRS}
)

# 库：需要当前目录和上级目录（src），设为 PUBLIC 以便链接者继承
target_include_directories(calibration_utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${OpenCV_INCLUDE_DIRS}
)

# 设置输出
set_target_properties(stereo_calibrator PROPERTIES
    OUTPUT_NAME "stereo_calibrator"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

set_target_properties(test_calibration PROPERTIES
    OUTPUT_NAME "test_calibration"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# 设置编译属性（与主项目一致）
set_target_properties(stereo_calibrator test_calibration calibration_utils
    PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

message(STATUS "标定模块配置完成（修复包含路径）")
