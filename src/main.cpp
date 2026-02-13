#include <iostream>
#include "utils/logger.hpp"
#include "utils/config.hpp"

#if ENABLE_GPU
#include "vulkan/context.hpp"
using namespace stereo_depth::vulkan;
#endif

int main() {
    stereo_depth::utils::Logger::initialize("stereo_depth", spdlog::level::info);
    LOG_INFO("OrangePiZero3-StereoDepth 启动");

    auto& cfg_mgr = stereo_depth::utils::ConfigManager::getInstance();
    if (!cfg_mgr.loadGlobalConfig("config/global_config.yaml")) {
        LOG_WARN("加载配置文件失败，使用默认参数");
    }

#if ENABLE_GPU
    LOG_INFO("GPU模块已启用，初始化Vulkan上下文...");
    VulkanContext ctx;
    if (ctx.initialize()) {
        LOG_INFO("Vulkan设备: {}", ctx.getDeviceName());
        LOG_INFO("Vulkan版本: {}", ctx.getVulkanVersion());
    } else {
        LOG_WARN("Vulkan初始化失败，请检查驱动和权限");
    }
#else
    LOG_INFO("GPU模块未编译，仅支持CPU模式");
    LOG_INFO("请通过 test_stereo_depth 程序运行立体匹配");
#endif

    LOG_INFO("程序结束");
    return 0;
}
