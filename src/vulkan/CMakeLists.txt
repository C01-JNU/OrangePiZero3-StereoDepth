# Vulkan GPU立体匹配模块
# 仅在 ENABLE_GPU=ON 时编译

if(ENABLE_GPU)
    message(STATUS "启用 Vulkan GPU 立体匹配模块")

    set(VULKAN_SOURCES
        context.cpp
        buffer_manager.cpp
        compute_pipeline.cpp
        stereo_pipeline.cpp
    )

    add_library(vulkan_stereo STATIC ${VULKAN_SOURCES})
    target_include_directories(vulkan_stereo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
    target_link_libraries(vulkan_stereo PUBLIC Vulkan::Vulkan)

    set(VULKAN_STEREO_LIB vulkan_stereo CACHE INTERNAL "Vulkan stereo library target")
else()
    message(STATUS "禁用 Vulkan GPU 立体匹配模块")
    set(VULKAN_STEREO_LIB "" CACHE INTERNAL "Vulkan stereo library target (disabled)")
endif()
