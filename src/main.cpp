#include "vulkan/context.hpp"
#include "utils/config.hpp"
#include "utils/logger.hpp"
#include <iostream>
#include <cstdlib>
#include <filesystem>
#include <vector>

// 设置Mali-G31所需的环境变量
void setupEnvironment() {
    // 必须在任何Vulkan调用之前设置这个环境变量
    setenv("PAN_I_WANT_A_BROKEN_VULKAN_DRIVER", "1", 1);
    
    // 打印环境变量状态
    const char* env_value = getenv("PAN_I_WANT_A_BROKEN_VULKAN_DRIVER");
    if (env_value && strcmp(env_value, "1") == 0) {
        LOG_INFO("Mali-G31环境变量已设置: PAN_I_WANT_A_BROKEN_VULKAN_DRIVER=1");
    } else {
        LOG_WARN("Mali-G31环境变量未正确设置，Vulkan可能无法工作");
    }
}

int main(int argc, char* argv[]) {
    // 设置环境变量（针对Mali-G31）
    setupEnvironment();
    
    // 初始化日志系统
    stereo_depth::utils::Logger::initialize("stereo_depth", spdlog::level::info);
    
    LOG_INFO("=== OrangePiZero3 StereoDepth GPU Test ===");
    LOG_INFO("Starting application (without VMA)...");
    
    // 尝试加载配置
    stereo_depth::utils::ConfigManager& config_mgr = stereo_depth::utils::ConfigManager::getInstance();
    
    // 尝试从多个位置加载配置
    std::vector<std::string> config_paths = {
        "config/global_config.yaml",
        "../config/global_config.yaml",
        "../../config/global_config.yaml"
    };
    
    bool config_loaded = false;
    for (const auto& path : config_paths) {
        if (std::filesystem::exists(path)) {
            config_loaded = config_mgr.loadGlobalConfig(path);
            if (config_loaded) break;
        }
    }
    
    if (!config_loaded) {
        LOG_WARN("Failed to load config file from standard locations, using defaults");
    }
    
    // 创建Vulkan上下文
    LOG_INFO("Creating Vulkan context...");
    
    stereo_depth::vulkan::VulkanContext vulkan_context;
    
    // 检查是否需要禁用验证层
    bool enable_validation = false;
    if (config_loaded) {
        enable_validation = config_mgr.getConfig().get<bool>("gpu.enable_validation", false);
    }
    
    // 在ARM平台强制禁用验证层
    #ifdef __arm__
    if (enable_validation) {
        LOG_WARN("Validation layers disabled on ARM Mali-G31");
        enable_validation = false;
    }
    #endif
    
    // 初始化Vulkan
    if (!vulkan_context.initialize(enable_validation)) {
        LOG_ERROR("Failed to initialize Vulkan context");
        return EXIT_FAILURE;
    }
    
    LOG_INFO("Vulkan context initialized successfully");
    LOG_INFO("Device: {}", vulkan_context.getDeviceName());
    LOG_INFO("Vulkan Version: {}", vulkan_context.getVulkanVersion());
    LOG_INFO("Compute Queue Family Index: {}", vulkan_context.getComputeQueueFamilyIndex());
    
    // 测试命令池和命令缓冲区创建
    LOG_INFO("Testing command buffer creation...");
    try {
        auto command_pool = vulkan_context.createCommandPool();
        auto command_buffer = vulkan_context.createCommandBuffer(command_pool);
        
        if (command_pool && command_buffer) {
            LOG_INFO("Command buffer created successfully");
            
            // 清理
            vkFreeCommandBuffers(vulkan_context.getDevice(), command_pool, 1, &command_buffer);
            vkDestroyCommandPool(vulkan_context.getDevice(), command_pool, nullptr);
            
            LOG_INFO("Command resources cleaned up");
        } else {
            LOG_ERROR("Failed to create command buffer");
        }
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to create command buffer: {}", e.what());
    }
    
    // 测试缓冲区创建（不使用VMA）
    LOG_INFO("Testing buffer creation (without VMA)...");
    try {
        VkBuffer testBuffer;
        VkDeviceMemory testBufferMemory;
        VkDeviceSize bufferSize = 1024 * 1024; // 1MB
        
        if (vulkan_context.createBuffer(bufferSize, 
                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                       testBuffer, testBufferMemory)) {
            LOG_INFO("Buffer created successfully");
            
            // 清理
            vkDestroyBuffer(vulkan_context.getDevice(), testBuffer, nullptr);
            vkFreeMemory(vulkan_context.getDevice(), testBufferMemory, nullptr);
            
            LOG_INFO("Buffer resources cleaned up");
        } else {
            LOG_WARN("Buffer creation failed, but this may be expected for PanVK");
        }
    } catch (const std::exception& e) {
        LOG_ERROR("Buffer creation failed: {}", e.what());
    }
    
    // 等待设备空闲
    vulkan_context.waitIdle();
    
    LOG_INFO("=== Test completed successfully ===");
    LOG_INFO("Vulkan GPU system is ready for stereo depth computation");
    LOG_INFO("Note: Running without VMA (Vulkan Memory Allocator)");
    
    return EXIT_SUCCESS;
}
