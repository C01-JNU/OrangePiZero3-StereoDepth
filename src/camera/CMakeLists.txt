add_library(camera
    mock_camera.cpp
    camera_factory.cpp
)

target_include_directories(camera PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(camera
    ${OpenCV_LIBS}
    spdlog::spdlog_header_only
)

set_target_properties(camera PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

message(STATUS "摄像头模块配置完成")
