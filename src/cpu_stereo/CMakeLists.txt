# CPU立体匹配模块
# 使用者需在根 CMakeLists.txt 中添加: add_subdirectory(src/cpu_stereo)

add_library(cpu_stereo
    cpu_stereo_matcher.cpp
)

target_include_directories(cpu_stereo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(cpu_stereo PUBLIC
    ${OpenCV_LIBS}
    yaml-cpp
    spdlog::spdlog_header_only
)

# 暴露给上层可执行文件
target_include_directories(cpu_stereo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
