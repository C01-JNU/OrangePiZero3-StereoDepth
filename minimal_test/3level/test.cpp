// minimal_test/3level/test.cpp
#include <iostream>
#include <vector>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <cassert>

#define VULKAN_HPP_NO_EXCEPTIONS
#include <vulkan/vulkan.h>

#define LOG_INFO(msg) std::cout << "[INFO] " << msg << std::endl
#define LOG_ERROR(msg) std::cerr << "[ERROR] " << msg << std::endl
#define LOG_SUCCESS(msg) std::cout << "[SUCCESS] " << msg << std::endl
#define LOG_WARN(msg) std::cout << "[WARN] " << msg << std::endl

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
    LOG_INFO("=== 3level测试：增加推送常量 ===");
    LOG_INFO("测试目标：推送常量（push constants）");
    
    // 设置环境变量
    setenv("PAN_I_WANT_A_BROKEN_VULKAN_DRIVER", "1", 1);
    
    VkResult result;
    
    // 1. 创建实例
    LOG_INFO("1. 创建Vulkan实例...");
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Level3Test";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;
    
    VkInstanceCreateInfo instanceInfo = {};
    instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceInfo.pApplicationInfo = &appInfo;
    
    VkInstance instance;
    result = vkCreateInstance(&instanceInfo, nullptr, &instance);
    if (result != VK_SUCCESS) {
        LOG_ERROR("创建实例失败: " << result);
        return 1;
    }
    LOG_SUCCESS("实例创建成功");
    
    // 2. 选择物理设备
    LOG_INFO("2. 选择物理设备...");
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    
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
    
    uint32_t computeQueueFamilyIndex = 0;
    for (uint32_t i = 0; i < queueFamilyCount; ++i) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            computeQueueFamilyIndex = i;
            break;
        }
    }
    
    // 4. 创建设备
    LOG_INFO("4. 创建设备...");
    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo = {};
    queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfo.queueFamilyIndex = computeQueueFamilyIndex;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &queuePriority;
    
    VkPhysicalDeviceFeatures deviceFeatures = {};
    
    VkDeviceCreateInfo deviceInfo = {};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    deviceInfo.pEnabledFeatures = &deviceFeatures;
    
    VkDevice device;
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
    auto shaderCode = loadShader("compute.comp.spv");
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
    
    // 8. 创建描述符集布局
    LOG_INFO("7. 创建描述符集布局...");
    VkDescriptorSetLayout descriptorSetLayout;
    
    std::vector<VkDescriptorSetLayoutBinding> bindings(2);
    
    // 绑定0：存储缓冲区
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[0].pImmutableSamplers = nullptr;
    
    // 绑定1：uniform缓冲区
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].pImmutableSamplers = nullptr;
    
    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    
    result = vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout);
    if (result != VK_SUCCESS) {
        LOG_ERROR("创建描述符集布局失败: " << result);
        vkDestroyShaderModule(device, shaderModule, nullptr);
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
        return 1;
    }
    LOG_SUCCESS("描述符集布局创建成功");
    
    // 9. 创建推送常量范围
    LOG_INFO("8. 创建推送常量范围...");
    VkPushConstantRange pushConstantRange = {};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = 8; // sizeof(float) + sizeof(uint) = 4 + 4 = 8 字节
    
    // 10. 创建管线布局（包含推送常量）
    LOG_INFO("9. 创建管线布局（包含推送常量）...");
    VkPipelineLayout pipelineLayout;
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
    
    result = vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout);
    if (result != VK_SUCCESS) {
        LOG_ERROR("创建管线布局失败: " << result);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        vkDestroyShaderModule(device, shaderModule, nullptr);
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
        return 1;
    }
    LOG_SUCCESS("管线布局创建成功（包含推送常量）");
    
    // 11. 创建计算管线
    LOG_INFO("10. 创建计算管线...");
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
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    pipelineInfo.basePipelineIndex = -1;
    
    result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &computePipeline);
    
    if (result == VK_SUCCESS) {
        LOG_SUCCESS("🎉 计算管线创建成功！");
        LOG_INFO("  推送常量测试通过");
        
        // 清理资源
        vkDestroyPipeline(device, computePipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        vkDestroyShaderModule(device, shaderModule, nullptr);
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
        
        std::cout << "\n========================================\n";
        std::cout << "✅ 3level测试成功！\n";
        std::cout << "推送常量创建通过\n";
        std::cout << "========================================\n";
        return 0;
    } else {
        LOG_ERROR("计算管线创建失败: " << result);
        
        if (result == -13) {
            LOG_ERROR("❌ 错误-13: VK_ERROR_INCOMPATIBLE_DRIVER");
            LOG_ERROR("问题出现在：推送常量创建阶段");
        }
        
        // 清理资源
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        vkDestroyShaderModule(device, shaderModule, nullptr);
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
        
        std::cout << "\n========================================\n";
        std::cout << "❌ 3level测试失败！\n";
        std::cout << "问题出现在：推送常量\n";
        std::cout << "========================================\n";
        return 1;
    }
}
