// minimal_test_simple/simple_test.cpp
#include <iostream>
#include <vector>
#include <fstream>
#include <cstring>
#include <cstdlib>

// 直接包含Vulkan头文件
#define VULKAN_HPP_NO_EXCEPTIONS
#include <vulkan/vulkan.h>

// 简单的日志宏
#define LOG_INFO(msg) std::cout << "[INFO] " << msg << std::endl
#define LOG_ERROR(msg) std::cerr << "[ERROR] " << msg << std::endl
#define LOG_SUCCESS(msg) std::cout << "[SUCCESS] " << msg << std::endl

// 加载着色器文件
std::vector<uint32_t> loadShader(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        LOG_ERROR("无法打开文件: " << filename);
        return {};
    }
    
    size_t fileSize = file.tellg();
    file.seekg(0);
    
    std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
    
    LOG_INFO("加载着色器: " << filename << " (" << fileSize << " bytes)");
    return buffer;
}

int main() {
    LOG_INFO("=== 最简单的Vulkan测试 ===");
    
    // 设置环境变量
    setenv("PAN_I_WANT_A_BROKEN_VULKAN_DRIVER", "1", 1);
    LOG_INFO("设置环境变量: PAN_I_WANT_A_BROKEN_VULKAN_DRIVER=1");
    
    VkResult result;
    
    // 1. 创建Vulkan实例
    LOG_INFO("1. 创建Vulkan实例...");
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "MinimalTest";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "NoEngine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;
    
    VkInstanceCreateInfo instanceInfo = {};
    instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceInfo.pApplicationInfo = &appInfo;
    instanceInfo.enabledExtensionCount = 0;
    instanceInfo.ppEnabledExtensionNames = nullptr;
    instanceInfo.enabledLayerCount = 0;
    instanceInfo.ppEnabledLayerNames = nullptr;
    
    VkInstance instance = VK_NULL_HANDLE;
    result = vkCreateInstance(&instanceInfo, nullptr, &instance);
    if (result != VK_SUCCESS) {
        LOG_ERROR("创建实例失败: " << result);
        return 1;
    }
    LOG_SUCCESS("Vulkan实例创建成功");
    
    // 2. 选择物理设备
    LOG_INFO("2. 选择物理设备...");
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    
    if (deviceCount == 0) {
        LOG_ERROR("没有找到Vulkan设备");
        vkDestroyInstance(instance, nullptr);
        return 1;
    }
    
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
    
    VkPhysicalDevice physicalDevice = devices[0];
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physicalDevice, &props);
    LOG_INFO("选择设备: " << props.deviceName);
    
    // 3. 查找计算队列
    LOG_INFO("3. 查找计算队列...");
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
    
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());
    
    uint32_t computeQueueFamilyIndex = UINT32_MAX;
    for (uint32_t i = 0; i < queueFamilyCount; ++i) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            computeQueueFamilyIndex = i;
            LOG_INFO("找到计算队列族: " << i);
            break;
        }
    }
    
    if (computeQueueFamilyIndex == UINT32_MAX) {
        LOG_ERROR("没有找到计算队列族");
        vkDestroyInstance(instance, nullptr);
        return 1;
    }
    
    // 4. 创建设备
    LOG_INFO("4. 创建设备...");
    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo = {};
    queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfo.queueFamilyIndex = computeQueueFamilyIndex;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &queuePriority;
    
    // 关键：使用全零特性结构体
    VkPhysicalDeviceFeatures deviceFeatures = {};
    
    VkDeviceCreateInfo deviceInfo = {};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    deviceInfo.pEnabledFeatures = &deviceFeatures;  // 全零特性
    deviceInfo.enabledExtensionCount = 0;
    deviceInfo.ppEnabledExtensionNames = nullptr;
    deviceInfo.enabledLayerCount = 0;
    deviceInfo.ppEnabledLayerNames = nullptr;
    
    VkDevice device = VK_NULL_HANDLE;
    result = vkCreateDevice(physicalDevice, &deviceInfo, nullptr, &device);
    if (result != VK_SUCCESS) {
        LOG_ERROR("创建设备失败: " << result);
        vkDestroyInstance(instance, nullptr);
        return 1;
    }
    LOG_SUCCESS("设备创建成功");
    
    // 5. 获取队列
    VkQueue computeQueue;
    vkGetDeviceQueue(device, computeQueueFamilyIndex, 0, &computeQueue);
    
    // 6. 加载着色器
    LOG_INFO("5. 加载着色器...");
    auto shaderCode = loadShader("minimal.comp.spv");
    if (shaderCode.empty()) {
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
        return 1;
    }
    
    // 7. 创建着色器模块
    LOG_INFO("6. 创建着色器模块...");
    VkShaderModule shaderModule;
    VkShaderModuleCreateInfo shaderInfo = {};
    shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderInfo.codeSize = shaderCode.size() * sizeof(uint32_t);
    shaderInfo.pCode = shaderCode.data();
    
    result = vkCreateShaderModule(device, &shaderInfo, nullptr, &shaderModule);
    if (result != VK_SUCCESS) {
        LOG_ERROR("创建着色器模块失败: " << result);
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
        return 1;
    }
    LOG_SUCCESS("着色器模块创建成功");
    
    // 8. 创建管线布局（空布局）
    LOG_INFO("7. 创建管线布局...");
    VkPipelineLayout pipelineLayout;
    VkPipelineLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 0;
    layoutInfo.pSetLayouts = nullptr;
    layoutInfo.pushConstantRangeCount = 0;
    layoutInfo.pPushConstantRanges = nullptr;
    
    result = vkCreatePipelineLayout(device, &layoutInfo, nullptr, &pipelineLayout);
    if (result != VK_SUCCESS) {
        LOG_ERROR("创建管线布局失败: " << result);
        vkDestroyShaderModule(device, shaderModule, nullptr);
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
        return 1;
    }
    LOG_SUCCESS("管线布局创建成功");
    
    // 9. 创建计算管线（关键测试！）
    LOG_INFO("8. 创建计算管线...");
    VkPipeline computePipeline;
    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    
    VkPipelineShaderStageCreateInfo stageInfo = {};
    stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = shaderModule;
    stageInfo.pName = "main";
    
    pipelineInfo.stage = stageInfo;
    pipelineInfo.layout = pipelineLayout;
    
    result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &computePipeline);
    
    if (result == VK_SUCCESS) {
        LOG_SUCCESS("🎉🎉🎉 计算管线创建成功！");
        LOG_SUCCESS("这意味着PanVK驱动可以工作！");
        
        // 清理资源
        vkDestroyPipeline(device, computePipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyShaderModule(device, shaderModule, nullptr);
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
        
        std::cout << "\n========================================\n";
        std::cout << "✅ 测试完全成功！\n";
        std::cout << "说明：\n";
        std::cout << "1. PanVK驱动可以创建计算管线\n";
        std::cout << "2. 问题出在项目环境的配置上\n";
        std::cout << "3. 需要比较项目与最小测试的差异\n";
        std::cout << "========================================\n";
        return 0;
    } else {
        LOG_ERROR("计算管线创建失败: " << result);
        
        // 输出错误详情
        if (result == -13) {
            LOG_ERROR("错误-13: VK_ERROR_INCOMPATIBLE_DRIVER");
            LOG_ERROR("PanVK驱动与应用程序不兼容");
        }
        
        // 清理资源
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyShaderModule(device, shaderModule, nullptr);
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
        
        std::cout << "\n========================================\n";
        std::cout << "❌ 测试失败！\n";
        std::cout << "说明：\n";
        std::cout << "1. PanVK驱动无法创建计算管线\n";
        std::cout << "2. 可能是驱动bug或硬件限制\n";
        std::cout << "========================================\n";
        return 1;
    }
}
