// minimal_test/4level/test.cpp
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
#define LOG_DEBUG(msg) std::cout << "[DEBUG] " << msg << std::endl

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
    LOG_INFO("=== 4level测试：存储缓冲区与描述符池 ===");
    LOG_INFO("测试目标：创建存储缓冲区、描述符池、描述符集");
    
    // 设置环境变量
    setenv("PAN_I_WANT_A_BROKEN_VULKAN_DRIVER", "1", 1);
    
    VkResult result;
    
    // 1. 创建实例
    LOG_INFO("1. 创建Vulkan实例...");
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Level4Test";
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
    
    // 3. 获取内存属性
    VkPhysicalDeviceMemoryProperties memoryProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);
    
    // 4. 查找计算队列
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
    
    // 5. 创建设备
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
    
    // 6. 获取队列
    VkQueue computeQueue;
    vkGetDeviceQueue(device, computeQueueFamilyIndex, 0, &computeQueue);
    
    // 7. 加载着色器
    LOG_INFO("5. 加载着色器...");
    auto shaderCode = loadShader("compute.comp.spv");
    if (shaderCode.empty()) {
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
        return 1;
    }
    
    // 8. 创建着色器模块
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
    
    // 9. 创建描述符集布局
    LOG_INFO("7. 创建描述符集布局...");
    VkDescriptorSetLayout descriptorSetLayout;
    
    VkDescriptorSetLayoutBinding layoutBinding = {};
    layoutBinding.binding = 0;
    layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    layoutBinding.descriptorCount = 1;
    layoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    layoutBinding.pImmutableSamplers = nullptr;
    
    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &layoutBinding;
    
    result = vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout);
    if (result != VK_SUCCESS) {
        LOG_ERROR("创建描述符集布局失败: " << result);
        vkDestroyShaderModule(device, shaderModule, nullptr);
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
        return 1;
    }
    LOG_SUCCESS("描述符集布局创建成功");
    
    // 10. 创建推送常量范围
    LOG_INFO("8. 创建推送常量范围...");
    VkPushConstantRange pushConstantRange = {};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = 8; // 2个uint32_t = 8字节
    
    // 11. 创建管线布局
    LOG_INFO("9. 创建管线布局...");
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
    LOG_SUCCESS("管线布局创建成功");
    
    // 12. ⭐️ 创建存储缓冲区 - 4level新特性！
    LOG_INFO("10. 创建存储缓冲区...");
    const VkDeviceSize bufferSize = 1024 * sizeof(uint32_t); // 4KB
    
    VkBuffer storageBuffer;
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = bufferSize;
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    result = vkCreateBuffer(device, &bufferInfo, nullptr, &storageBuffer);
    if (result != VK_SUCCESS) {
        LOG_ERROR("创建缓冲区失败: " << result);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        vkDestroyShaderModule(device, shaderModule, nullptr);
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
        return 1;
    }
    LOG_SUCCESS("缓冲区创建成功");
    
    // 13. 为缓冲区分配内存
    LOG_INFO("11. 分配缓冲区内存...");
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, storageBuffer, &memRequirements);
    
    // 查找合适的内存类型
    uint32_t memoryTypeIndex = UINT32_MAX;
    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
        if ((memRequirements.memoryTypeBits & (1 << i)) &&
            (memoryProperties.memoryTypes[i].propertyFlags & 
             (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
            (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            memoryTypeIndex = i;
            break;
        }
    }
    
    if (memoryTypeIndex == UINT32_MAX) {
        LOG_ERROR("没有找到合适的内存类型");
        vkDestroyBuffer(device, storageBuffer, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        vkDestroyShaderModule(device, shaderModule, nullptr);
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
        return 1;
    }
    
    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = memoryTypeIndex;
    
    VkDeviceMemory bufferMemory;
    result = vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory);
    if (result != VK_SUCCESS) {
        LOG_ERROR("分配内存失败: " << result);
        vkDestroyBuffer(device, storageBuffer, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        vkDestroyShaderModule(device, shaderModule, nullptr);
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
        return 1;
    }
    
    result = vkBindBufferMemory(device, storageBuffer, bufferMemory, 0);
    if (result != VK_SUCCESS) {
        LOG_ERROR("绑定内存失败: " << result);
        vkFreeMemory(device, bufferMemory, nullptr);
        vkDestroyBuffer(device, storageBuffer, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        vkDestroyShaderModule(device, shaderModule, nullptr);
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
        return 1;
    }
    LOG_SUCCESS("内存分配并绑定成功");
    
    // 14. ⭐️ 创建描述符池 - 4level新特性！
    LOG_INFO("12. 创建描述符池...");
    VkDescriptorPoolSize poolSize = {};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 1;
    
    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = 1;
    
    VkDescriptorPool descriptorPool;
    result = vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool);
    if (result != VK_SUCCESS) {
        LOG_ERROR("创建描述符池失败: " << result);
        vkFreeMemory(device, bufferMemory, nullptr);
        vkDestroyBuffer(device, storageBuffer, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        vkDestroyShaderModule(device, shaderModule, nullptr);
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
        return 1;
    }
    LOG_SUCCESS("描述符池创建成功");
    
    // 15. ⭐️ 分配描述符集 - 4level新特性！
    LOG_INFO("13. 分配描述符集...");
    VkDescriptorSetAllocateInfo allocSetInfo = {};
    allocSetInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocSetInfo.descriptorPool = descriptorPool;
    allocSetInfo.descriptorSetCount = 1;
    allocSetInfo.pSetLayouts = &descriptorSetLayout;
    
    VkDescriptorSet descriptorSet;
    result = vkAllocateDescriptorSets(device, &allocSetInfo, &descriptorSet);
    if (result != VK_SUCCESS) {
        LOG_ERROR("分配描述符集失败: " << result);
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        vkFreeMemory(device, bufferMemory, nullptr);
        vkDestroyBuffer(device, storageBuffer, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        vkDestroyShaderModule(device, shaderModule, nullptr);
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
        return 1;
    }
    LOG_SUCCESS("描述符集分配成功");
    
    // 16. ⭐️ 更新描述符集 - 4level新特性！
    LOG_INFO("14. 更新描述符集...");
    VkDescriptorBufferInfo bufferDescriptorInfo = {};
    bufferDescriptorInfo.buffer = storageBuffer;
    bufferDescriptorInfo.offset = 0;
    bufferDescriptorInfo.range = VK_WHOLE_SIZE;
    
    VkWriteDescriptorSet descriptorWrite = {};
    descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrite.dstSet = descriptorSet;
    descriptorWrite.dstBinding = 0;
    descriptorWrite.dstArrayElement = 0;
    descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrite.descriptorCount = 1;
    descriptorWrite.pBufferInfo = &bufferDescriptorInfo;
    
    vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);
    LOG_SUCCESS("描述符集更新成功");
    
    // 17. 创建计算管线
    LOG_INFO("15. 创建计算管线...");
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
        LOG_INFO("   存储缓冲区、描述符池、描述符集测试通过");
        
        // 18. 测试实际执行（可选）
        LOG_INFO("16. 测试命令缓冲区执行...");
        
        // 创建命令池
        VkCommandPoolCreateInfo cmdPoolInfo = {};
        cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cmdPoolInfo.queueFamilyIndex = computeQueueFamilyIndex;
        cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        
        VkCommandPool commandPool;
        result = vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &commandPool);
        if (result != VK_SUCCESS) {
            LOG_WARN("创建命令池失败，跳过执行测试");
        } else {
            LOG_SUCCESS("命令池创建成功");
            
            // 创建命令缓冲区
            VkCommandBufferAllocateInfo cmdAllocInfo = {};
            cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            cmdAllocInfo.commandPool = commandPool;
            cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cmdAllocInfo.commandBufferCount = 1;
            
            VkCommandBuffer commandBuffer;
            result = vkAllocateCommandBuffers(device, &cmdAllocInfo, &commandBuffer);
            if (result != VK_SUCCESS) {
                LOG_WARN("分配命令缓冲区失败，跳过执行测试");
            } else {
                LOG_SUCCESS("命令缓冲区分配成功");
                
                // 开始记录命令
                VkCommandBufferBeginInfo beginInfo = {};
                beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                
                result = vkBeginCommandBuffer(commandBuffer, &beginInfo);
                if (result == VK_SUCCESS) {
                    LOG_SUCCESS("命令缓冲区开始记录成功");
                    
                    // 绑定管线
                    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
                    
                    // 绑定描述符集
                    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                           pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
                    
                    // 推送常量
                    uint32_t pushConstants[2] = {1024, 2};
                    vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                                      0, sizeof(pushConstants), pushConstants);
                    
                    // 分派计算
                    vkCmdDispatch(commandBuffer, 4, 1, 1); // 1024/256 = 4工作组
                    
                    result = vkEndCommandBuffer(commandBuffer);
                    if (result == VK_SUCCESS) {
                        LOG_SUCCESS("命令缓冲区记录完成");
                        
                        // 提交命令
                        VkSubmitInfo submitInfo = {};
                        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                        submitInfo.commandBufferCount = 1;
                        submitInfo.pCommandBuffers = &commandBuffer;
                        
                        result = vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE);
                        if (result == VK_SUCCESS) {
                            LOG_SUCCESS("命令提交成功");
                            
                            // 等待完成
                            result = vkQueueWaitIdle(computeQueue);
                            if (result == VK_SUCCESS) {
                                LOG_SUCCESS("🎉 GPU计算执行完成！");
                            } else {
                                LOG_WARN("等待队列空闲失败");
                            }
                        } else {
                            LOG_WARN("命令提交失败");
                        }
                    } else {
                        LOG_WARN("结束命令缓冲区失败");
                    }
                } else {
                    LOG_WARN("开始命令缓冲区失败");
                }
                
                // 销毁命令池
                vkDestroyCommandPool(device, commandPool, nullptr);
                LOG_SUCCESS("命令池已销毁");
            }
        }
        
        // 清理资源
        vkDestroyPipeline(device, computePipeline, nullptr);
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        vkFreeMemory(device, bufferMemory, nullptr);
        vkDestroyBuffer(device, storageBuffer, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        vkDestroyShaderModule(device, shaderModule, nullptr);
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
        
        std::cout << "\n========================================\n";
        std::cout << "✅ 4level测试成功！\n";
        std::cout << "存储缓冲区、描述符池、描述符集创建通过\n";
        std::cout << "实际GPU执行可能已成功\n";
        std::cout << "========================================\n";
        return 0;
    } else {
        LOG_ERROR("计算管线创建失败: " << result);
        
        if (result == -13) {
            LOG_ERROR("❌ 错误-13: VK_ERROR_INCOMPATIBLE_DRIVER");
            LOG_ERROR("问题出现在：描述符池/描述符集/缓冲区创建阶段");
        }
        
        // 清理资源
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        vkFreeMemory(device, bufferMemory, nullptr);
        vkDestroyBuffer(device, storageBuffer, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        vkDestroyShaderModule(device, shaderModule, nullptr);
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
        
        std::cout << "\n========================================\n";
        std::cout << "❌ 4level测试失败！\n";
        std::cout << "问题出现在：存储缓冲区/描述符池/描述符集\n";
        std::cout << "========================================\n";
        return 1;
    }
}
