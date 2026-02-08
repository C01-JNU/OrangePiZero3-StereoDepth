#include <iostream>
#include <vector>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <memory>

#define VULKAN_HPP_NO_EXCEPTIONS
#include <vulkan/vulkan.h>

#define LOG_INFO(msg) std::cout << "[INFO] " << msg << std::endl
#define LOG_ERROR(msg) std::cerr << "[ERROR] " << msg << std::endl
#define LOG_SUCCESS(msg) std::cout << "[SUCCESS] " << msg << std::endl
#define LOG_WARN(msg) std::cout << "[WARN] " << msg << std::endl

// 简单的RAII包装器
struct BufferResource {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    size_t size = 0;
    void* mapped = nullptr;
    
    ~BufferResource() {
        if (mapped) {
            LOG_WARN("缓冲区在映射状态下被销毁，可能有问题");
        }
    }
};

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

VkShaderModule createShaderModule(VkDevice device, const std::vector<uint32_t>& code) {
    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size() * sizeof(uint32_t);
    createInfo.pCode = code.data();
    
    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    return shaderModule;
}

bool createBuffer(VkDevice device, VkPhysicalDevice physicalDevice, 
                  VkDeviceSize size, VkBufferUsageFlags usage, 
                  VkMemoryPropertyFlags properties, 
                  BufferResource& buffer) {
    
    // 创建缓冲区
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer.buffer) != VK_SUCCESS) {
        LOG_ERROR("创建缓冲区失败");
        return false;
    }
    
    // 获取内存要求
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer.buffer, &memRequirements);
    
    // 查找内存类型
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
    
    uint32_t memoryTypeIndex = UINT32_MAX;
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((memRequirements.memoryTypeBits & (1 << i)) && 
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            memoryTypeIndex = i;
            break;
        }
    }
    
    if (memoryTypeIndex == UINT32_MAX) {
        LOG_ERROR("找不到合适的内存类型");
        vkDestroyBuffer(device, buffer.buffer, nullptr);
        buffer.buffer = VK_NULL_HANDLE;
        return false;
    }
    
    // 分配内存
    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = memoryTypeIndex;
    
    if (vkAllocateMemory(device, &allocInfo, nullptr, &buffer.memory) != VK_SUCCESS) {
        LOG_ERROR("分配内存失败");
        vkDestroyBuffer(device, buffer.buffer, nullptr);
        buffer.buffer = VK_NULL_HANDLE;
        return false;
    }
    
    buffer.size = size;
    
    // 绑定内存
    if (vkBindBufferMemory(device, buffer.buffer, buffer.memory, 0) != VK_SUCCESS) {
        LOG_ERROR("绑定内存失败");
        vkDestroyBuffer(device, buffer.buffer, nullptr);
        vkFreeMemory(device, buffer.memory, nullptr);
        buffer.buffer = VK_NULL_HANDLE;
        buffer.memory = VK_NULL_HANDLE;
        return false;
    }
    
    return true;
}

void cleanupBuffer(VkDevice device, BufferResource& buffer) {
    if (buffer.mapped) {
        vkUnmapMemory(device, buffer.memory);
        buffer.mapped = nullptr;
    }
    
    if (buffer.buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, buffer.buffer, nullptr);
        buffer.buffer = VK_NULL_HANDLE;
    }
    
    if (buffer.memory != VK_NULL_HANDLE) {
        vkFreeMemory(device, buffer.memory, nullptr);
        buffer.memory = VK_NULL_HANDLE;
    }
    
    buffer.size = 0;
}

bool copyToBuffer(VkDevice device, BufferResource& buffer, const void* data, size_t size) {
    if (size > buffer.size) {
        LOG_ERROR("复制数据大小超过缓冲区大小");
        return false;
    }
    
    // 映射内存
    void* mapped;
    if (vkMapMemory(device, buffer.memory, 0, buffer.size, 0, &mapped) != VK_SUCCESS) {
        LOG_ERROR("映射内存失败");
        return false;
    }
    
    // 复制数据
    memcpy(mapped, data, size);
    
    // 确保数据写入（对于非一致内存可能需要刷新）
    VkMappedMemoryRange memoryRange = {};
    memoryRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    memoryRange.memory = buffer.memory;
    memoryRange.offset = 0;
    memoryRange.size = buffer.size;
    vkFlushMappedMemoryRanges(device, 1, &memoryRange);
    
    // 解除映射
    vkUnmapMemory(device, buffer.memory);
    
    return true;
}

bool copyFromBuffer(VkDevice device, BufferResource& buffer, void* data, size_t size) {
    if (size > buffer.size) {
        LOG_ERROR("读取数据大小超过缓冲区大小");
        return false;
    }
    
    // 映射内存
    void* mapped;
    if (vkMapMemory(device, buffer.memory, 0, buffer.size, 0, &mapped) != VK_SUCCESS) {
        LOG_ERROR("映射内存失败");
        return false;
    }
    
    // 复制数据
    memcpy(data, mapped, size);
    
    // 解除映射
    vkUnmapMemory(device, buffer.memory);
    
    return true;
}

int main() {
    LOG_INFO("=== 5level测试：多阶段计算管线与多个缓冲区 ===");
    LOG_INFO("模拟项目中的立体匹配流水线：多个阶段、多个缓冲区、资源管理");
    
    // 设置环境变量
    setenv("PAN_I_WANT_A_BROKEN_VULKAN_DRIVER", "1", 1);
    
    VkResult result;
    bool success = true;
    
    // Vulkan句柄
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue computeQueue = VK_NULL_HANDLE;
    uint32_t computeQueueFamilyIndex = 0;
    
    // 着色器模块
    VkShaderModule shaderModule1 = VK_NULL_HANDLE;
    VkShaderModule shaderModule2 = VK_NULL_HANDLE;
    
    // 描述符集布局
    VkDescriptorSetLayout descriptorSetLayout1 = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout2 = VK_NULL_HANDLE;
    
    // 管线布局
    VkPipelineLayout pipelineLayout1 = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout2 = VK_NULL_HANDLE;
    
    // 计算管线
    VkPipeline computePipeline1 = VK_NULL_HANDLE;
    VkPipeline computePipeline2 = VK_NULL_HANDLE;
    
    // 描述符池和描述符集
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet1 = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet2 = VK_NULL_HANDLE;
    
    // 缓冲区资源
    std::vector<BufferResource> buffers;
    const size_t bufferCount = 3;  // 输入、中间结果、最终结果
    const size_t dataSize = 1024 * sizeof(uint32_t);
    
    // 命令池和命令缓冲区
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    
    try {
        // 1. 创建实例
        LOG_INFO("1. 创建Vulkan实例...");
        VkApplicationInfo appInfo = {};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "MultiStageTest";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;
        
        VkInstanceCreateInfo instanceInfo = {};
        instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instanceInfo.pApplicationInfo = &appInfo;
        
        result = vkCreateInstance(&instanceInfo, nullptr, &instance);
        if (result != VK_SUCCESS) throw std::runtime_error("创建实例失败: " + std::to_string(result));
        LOG_SUCCESS("实例创建成功");
        
        // 2. 选择物理设备
        LOG_INFO("2. 选择物理设备...");
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
        
        physicalDevice = devices[0];
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(physicalDevice, &props);
        LOG_INFO("选择设备: " + std::string(props.deviceName));
        
        // 3. 查找计算队列
        LOG_INFO("3. 查找计算队列...");
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
        
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());
        
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
        
        result = vkCreateDevice(physicalDevice, &deviceInfo, nullptr, &device);
        if (result != VK_SUCCESS) throw std::runtime_error("创建设备失败: " + std::to_string(result));
        LOG_SUCCESS("设备创建成功");
        
        // 5. 获取队列
        vkGetDeviceQueue(device, computeQueueFamilyIndex, 0, &computeQueue);
        
        // 6. 加载着色器
        LOG_INFO("5. 加载着色器...");
        auto shaderCode1 = loadShader("compute_stage1.comp.spv");
        auto shaderCode2 = loadShader("compute_stage2.comp.spv");
        
        if (shaderCode1.empty() || shaderCode2.empty()) {
            throw std::runtime_error("加载着色器失败");
        }
        
        // 7. 创建着色器模块
        LOG_INFO("6. 创建着色器模块...");
        shaderModule1 = createShaderModule(device, shaderCode1);
        shaderModule2 = createShaderModule(device, shaderCode2);
        
        if (!shaderModule1 || !shaderModule2) {
            throw std::runtime_error("创建着色器模块失败");
        }
        LOG_SUCCESS("两个着色器模块创建成功");
        
        // 8. 创建多个描述符集布局
        LOG_INFO("7. 创建描述符集布局...");
        
        // 第一个描述符集布局（阶段1）：输入缓冲区 + 阶段1输出
        VkDescriptorSetLayoutBinding bindings1[2] = {};
        bindings1[0].binding = 0;
        bindings1[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings1[0].descriptorCount = 1;
        bindings1[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        
        bindings1[1].binding = 1;
        bindings1[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings1[1].descriptorCount = 1;
        bindings1[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        
        VkDescriptorSetLayoutCreateInfo layoutInfo1 = {};
        layoutInfo1.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo1.bindingCount = 2;
        layoutInfo1.pBindings = bindings1;
        
        result = vkCreateDescriptorSetLayout(device, &layoutInfo1, nullptr, &descriptorSetLayout1);
        if (result != VK_SUCCESS) throw std::runtime_error("创建描述符集布局1失败");
        
        // 第二个描述符集布局（阶段2）：阶段1输出 + 最终输出
        VkDescriptorSetLayoutBinding bindings2[2] = {};
        bindings2[0].binding = 0;
        bindings2[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings2[0].descriptorCount = 1;
        bindings2[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        
        bindings2[1].binding = 1;
        bindings2[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings2[1].descriptorCount = 1;
        bindings2[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        
        VkDescriptorSetLayoutCreateInfo layoutInfo2 = {};
        layoutInfo2.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo2.bindingCount = 2;
        layoutInfo2.pBindings = bindings2;
        
        result = vkCreateDescriptorSetLayout(device, &layoutInfo2, nullptr, &descriptorSetLayout2);
        if (result != VK_SUCCESS) throw std::runtime_error("创建描述符集布局2失败");
        
        LOG_SUCCESS("两个描述符集布局创建成功");
        
        // 9. 创建多个管线布局
        LOG_INFO("8. 创建管线布局...");
        
        VkPipelineLayoutCreateInfo pipelineLayoutInfo1 = {};
        pipelineLayoutInfo1.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo1.setLayoutCount = 1;
        pipelineLayoutInfo1.pSetLayouts = &descriptorSetLayout1;
        
        result = vkCreatePipelineLayout(device, &pipelineLayoutInfo1, nullptr, &pipelineLayout1);
        if (result != VK_SUCCESS) throw std::runtime_error("创建管线布局1失败");
        
        VkPipelineLayoutCreateInfo pipelineLayoutInfo2 = {};
        pipelineLayoutInfo2.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo2.setLayoutCount = 1;
        pipelineLayoutInfo2.pSetLayouts = &descriptorSetLayout2;
        
        result = vkCreatePipelineLayout(device, &pipelineLayoutInfo2, nullptr, &pipelineLayout2);
        if (result != VK_SUCCESS) throw std::runtime_error("创建管线布局2失败");
        
        LOG_SUCCESS("两个管线布局创建成功");
        
        // 10. 创建多个计算管线
        LOG_INFO("9. 创建计算管线...");
        
        // 第一阶段管线
        VkComputePipelineCreateInfo pipelineInfo1 = {};
        pipelineInfo1.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        
        VkPipelineShaderStageCreateInfo stageInfo1 = {};
        stageInfo1.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stageInfo1.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stageInfo1.module = shaderModule1;
        stageInfo1.pName = "main";
        
        pipelineInfo1.stage = stageInfo1;
        pipelineInfo1.layout = pipelineLayout1;
        
        result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo1, nullptr, &computePipeline1);
        if (result != VK_SUCCESS) throw std::runtime_error("创建计算管线1失败");
        
        // 第二阶段管线
        VkComputePipelineCreateInfo pipelineInfo2 = {};
        pipelineInfo2.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        
        VkPipelineShaderStageCreateInfo stageInfo2 = {};
        stageInfo2.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stageInfo2.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stageInfo2.module = shaderModule2;
        stageInfo2.pName = "main";
        
        pipelineInfo2.stage = stageInfo2;
        pipelineInfo2.layout = pipelineLayout2;
        
        result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo2, nullptr, &computePipeline2);
        if (result != VK_SUCCESS) throw std::runtime_error("创建计算管线2失败");
        
        LOG_SUCCESS("两个计算管线创建成功");
        
        // 11. 创建多个缓冲区（模拟立体匹配流水线）
        LOG_INFO("10. 创建多个缓冲区...");
        buffers.resize(bufferCount);
        
        for (size_t i = 0; i < bufferCount; ++i) {
            LOG_INFO("  创建缓冲区 " + std::to_string(i) + "...");
            if (!createBuffer(device, physicalDevice, dataSize, 
                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | 
                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                             buffers[i])) {
                throw std::runtime_error("创建缓冲区 " + std::to_string(i) + " 失败");
            }
            LOG_SUCCESS("  缓冲区 " + std::to_string(i) + " 创建成功");
        }
        LOG_SUCCESS("所有缓冲区创建成功");
        
        // 12. 创建描述符池
        LOG_INFO("11. 创建描述符池...");
        
        VkDescriptorPoolSize poolSize = {};
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize.descriptorCount = 4;  // 两个描述符集 × 每个2个绑定
        
        VkDescriptorPoolCreateInfo poolInfo = {};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
        poolInfo.maxSets = 2;  // 两个描述符集
        
        result = vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool);
        if (result != VK_SUCCESS) throw std::runtime_error("创建描述符池失败");
        LOG_SUCCESS("描述符池创建成功");
        
        // 13. 分配描述符集
        LOG_INFO("12. 分配描述符集...");
        
        VkDescriptorSetLayout layouts[2] = {descriptorSetLayout1, descriptorSetLayout2};
        VkDescriptorSetAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = 2;
        allocInfo.pSetLayouts = layouts;
        
        VkDescriptorSet descriptorSets[2];
        result = vkAllocateDescriptorSets(device, &allocInfo, descriptorSets);
        if (result != VK_SUCCESS) throw std::runtime_error("分配描述符集失败");
        
        descriptorSet1 = descriptorSets[0];
        descriptorSet2 = descriptorSets[1];
        LOG_SUCCESS("两个描述符集分配成功");
        
        // 14. 更新描述符集
        LOG_INFO("13. 更新描述符集...");
        
        // 第一阶段描述符集：绑定缓冲区0和1
        VkDescriptorBufferInfo bufferInfo1[2] = {};
        bufferInfo1[0].buffer = buffers[0].buffer;  // 输入
        bufferInfo1[0].offset = 0;
        bufferInfo1[0].range = dataSize;
        
        bufferInfo1[1].buffer = buffers[1].buffer;  // 中间结果
        bufferInfo1[1].offset = 0;
        bufferInfo1[1].range = dataSize;
        
        VkWriteDescriptorSet descriptorWrites1[2] = {};
        descriptorWrites1[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites1[0].dstSet = descriptorSet1;
        descriptorWrites1[0].dstBinding = 0;
        descriptorWrites1[0].dstArrayElement = 0;
        descriptorWrites1[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites1[0].descriptorCount = 1;
        descriptorWrites1[0].pBufferInfo = &bufferInfo1[0];
        
        descriptorWrites1[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites1[1].dstSet = descriptorSet1;
        descriptorWrites1[1].dstBinding = 1;
        descriptorWrites1[1].dstArrayElement = 0;
        descriptorWrites1[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites1[1].descriptorCount = 1;
        descriptorWrites1[1].pBufferInfo = &bufferInfo1[1];
        
        // 第二阶段描述符集：绑定缓冲区1和2
        VkDescriptorBufferInfo bufferInfo2[2] = {};
        bufferInfo2[0].buffer = buffers[1].buffer;  // 中间结果（来自阶段1）
        bufferInfo2[0].offset = 0;
        bufferInfo2[0].range = dataSize;
        
        bufferInfo2[1].buffer = buffers[2].buffer;  // 最终结果
        bufferInfo2[1].offset = 0;
        bufferInfo2[1].range = dataSize;
        
        VkWriteDescriptorSet descriptorWrites2[2] = {};
        descriptorWrites2[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites2[0].dstSet = descriptorSet2;
        descriptorWrites2[0].dstBinding = 0;
        descriptorWrites2[0].dstArrayElement = 0;
        descriptorWrites2[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites2[0].descriptorCount = 1;
        descriptorWrites2[0].pBufferInfo = &bufferInfo2[0];
        
        descriptorWrites2[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites2[1].dstSet = descriptorSet2;
        descriptorWrites2[1].dstBinding = 1;
        descriptorWrites2[1].dstArrayElement = 0;
        descriptorWrites2[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites2[1].descriptorCount = 1;
        descriptorWrites2[1].pBufferInfo = &bufferInfo2[1];
        
        // 更新描述符集
        vkUpdateDescriptorSets(device, 2, descriptorWrites1, 0, nullptr);
        vkUpdateDescriptorSets(device, 2, descriptorWrites2, 0, nullptr);
        
        LOG_SUCCESS("描述符集更新成功");
        
        // 15. 初始化输入数据
        LOG_INFO("14. 初始化输入数据...");
        std::vector<uint32_t> inputData(1024);
        for (size_t i = 0; i < 1024; ++i) {
            inputData[i] = static_cast<uint32_t>(i + 1);
        }
        
        if (!copyToBuffer(device, buffers[0], inputData.data(), dataSize)) {
            throw std::runtime_error("复制输入数据失败");
        }
        LOG_SUCCESS("输入数据初始化成功");
        
        // 16. 创建命令池和命令缓冲区
        LOG_INFO("15. 创建命令池和命令缓冲区...");
        
        VkCommandPoolCreateInfo poolCreateInfo = {};
        poolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolCreateInfo.queueFamilyIndex = computeQueueFamilyIndex;
        poolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        
        result = vkCreateCommandPool(device, &poolCreateInfo, nullptr, &commandPool);
        if (result != VK_SUCCESS) throw std::runtime_error("创建命令池失败");
        LOG_SUCCESS("命令池创建成功");
        
        VkCommandBufferAllocateInfo allocInfoCB = {};
        allocInfoCB.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfoCB.commandPool = commandPool;
        allocInfoCB.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfoCB.commandBufferCount = 1;
        
        result = vkAllocateCommandBuffers(device, &allocInfoCB, &commandBuffer);
        if (result != VK_SUCCESS) throw std::runtime_error("分配命令缓冲区失败");
        LOG_SUCCESS("命令缓冲区分配成功");
        
        // 17. 记录命令缓冲区
        LOG_INFO("16. 记录命令缓冲区（模拟多阶段流水线）...");
        
        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        
        result = vkBeginCommandBuffer(commandBuffer, &beginInfo);
        if (result != VK_SUCCESS) throw std::runtime_error("开始命令缓冲区失败");
        
        // 第一阶段计算
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline1);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, 
                               pipelineLayout1, 0, 1, &descriptorSet1, 0, nullptr);
        vkCmdDispatch(commandBuffer, 1024, 1, 1);  // 1024个工作组
        
        // 内存屏障确保第一阶段完成
        VkMemoryBarrier memoryBarrier = {};
        memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(commandBuffer,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);
        
        // 第二阶段计算
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline2);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipelineLayout2, 0, 1, &descriptorSet2, 0, nullptr);
        vkCmdDispatch(commandBuffer, 1024, 1, 1);  // 1024个工作组
        
        result = vkEndCommandBuffer(commandBuffer);
        if (result != VK_SUCCESS) throw std::runtime_error("结束命令缓冲区失败");
        LOG_SUCCESS("命令缓冲区记录完成");
        
        // 18. 提交命令
        LOG_INFO("17. 提交命令到队列...");
        
        VkSubmitInfo submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;
        
        result = vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE);
        if (result != VK_SUCCESS) throw std::runtime_error("提交命令失败");
        
        // 19. 等待GPU完成
        LOG_INFO("18. 等待GPU完成计算...");
        result = vkQueueWaitIdle(computeQueue);
        if (result != VK_SUCCESS) throw std::runtime_error("等待队列空闲失败");
        LOG_SUCCESS("GPU计算完成");
        
        // 20. 验证结果
        LOG_INFO("19. 验证计算结果...");
        
        // 读取最终结果
        std::vector<uint32_t> outputData(1024);
        if (!copyFromBuffer(device, buffers[2], outputData.data(), dataSize)) {
            throw std::runtime_error("读取输出数据失败");
        }
        
        // 验证：每个元素应该是 (i+1) * 2 * 3 = (i+1) * 6
        bool allCorrect = true;
        for (size_t i = 0; i < 1024; ++i) {
            uint32_t expected = static_cast<uint32_t>((i + 1) * 6);
            if (outputData[i] != expected) {
                LOG_ERROR("计算结果错误: 位置 " + std::to_string(i) + 
                         ", 期望 " + std::to_string(expected) + 
                         ", 实际 " + std::to_string(outputData[i]));
                allCorrect = false;
                break;
            }
        }
        
        if (allCorrect) {
            LOG_SUCCESS("🎉 所有计算结果正确！");
            LOG_SUCCESS("多阶段计算管线测试完全通过");
        } else {
            LOG_WARN("⚠️ 计算结果有误");
        }
        
        // 21. 资源清理
        LOG_INFO("20. 清理资源...");
        
        if (commandPool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device, commandPool, nullptr);
            commandPool = VK_NULL_HANDLE;
            LOG_SUCCESS("命令池已销毁");
        }
        
        if (descriptorPool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(device, descriptorPool, nullptr);
            descriptorPool = VK_NULL_HANDLE;
            LOG_SUCCESS("描述符池已销毁");
        }
        
        for (auto& buffer : buffers) {
            cleanupBuffer(device, buffer);
        }
        LOG_SUCCESS("所有缓冲区已清理");
        
        if (computePipeline1 != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, computePipeline1, nullptr);
            computePipeline1 = VK_NULL_HANDLE;
        }
        
        if (computePipeline2 != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, computePipeline2, nullptr);
            computePipeline2 = VK_NULL_HANDLE;
        }
        LOG_SUCCESS("计算管线已销毁");
        
        if (pipelineLayout1 != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device, pipelineLayout1, nullptr);
            pipelineLayout1 = VK_NULL_HANDLE;
        }
        
        if (pipelineLayout2 != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device, pipelineLayout2, nullptr);
            pipelineLayout2 = VK_NULL_HANDLE;
        }
        LOG_SUCCESS("管线布局已销毁");
        
        if (descriptorSetLayout1 != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(device, descriptorSetLayout1, nullptr);
            descriptorSetLayout1 = VK_NULL_HANDLE;
        }
        
        if (descriptorSetLayout2 != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(device, descriptorSetLayout2, nullptr);
            descriptorSetLayout2 = VK_NULL_HANDLE;
        }
        LOG_SUCCESS("描述符集布局已销毁");
        
        if (shaderModule1 != VK_NULL_HANDLE) {
            vkDestroyShaderModule(device, shaderModule1, nullptr);
            shaderModule1 = VK_NULL_HANDLE;
        }
        
        if (shaderModule2 != VK_NULL_HANDLE) {
            vkDestroyShaderModule(device, shaderModule2, nullptr);
            shaderModule2 = VK_NULL_HANDLE;
        }
        LOG_SUCCESS("着色器模块已销毁");
        
        if (device != VK_NULL_HANDLE) {
            vkDestroyDevice(device, nullptr);
            device = VK_NULL_HANDLE;
            LOG_SUCCESS("设备已销毁");
        }
        
        if (instance != VK_NULL_HANDLE) {
            vkDestroyInstance(instance, nullptr);
            instance = VK_NULL_HANDLE;
            LOG_SUCCESS("实例已销毁");
        }
        
        LOG_SUCCESS("所有资源清理完成");
        
        std::cout << "\n========================================\n";
        std::cout << "✅ 5level测试完全成功！\n";
        std::cout << "多阶段计算管线、多个缓冲区、资源管理全部通过\n";
        std::cout << "模拟项目中的立体匹配流水线成功\n";
        std::cout << "========================================\n";
        
        return 0;
        
    } catch (const std::exception& e) {
        LOG_ERROR("测试失败: " + std::string(e.what()));
        
        // 紧急清理资源
        LOG_WARN("执行紧急资源清理...");
        
        for (auto& buffer : buffers) {
            cleanupBuffer(device, buffer);
        }
        
        if (commandPool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device, commandPool, nullptr);
        }
        
        if (descriptorPool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        }
        
        if (computePipeline1 != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, computePipeline1, nullptr);
        }
        
        if (computePipeline2 != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, computePipeline2, nullptr);
        }
        
        if (pipelineLayout1 != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device, pipelineLayout1, nullptr);
        }
        
        if (pipelineLayout2 != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device, pipelineLayout2, nullptr);
        }
        
        if (descriptorSetLayout1 != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(device, descriptorSetLayout1, nullptr);
        }
        
        if (descriptorSetLayout2 != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(device, descriptorSetLayout2, nullptr);
        }
        
        if (shaderModule1 != VK_NULL_HANDLE) {
            vkDestroyShaderModule(device, shaderModule1, nullptr);
        }
        
        if (shaderModule2 != VK_NULL_HANDLE) {
            vkDestroyShaderModule(device, shaderModule2, nullptr);
        }
        
        if (device != VK_NULL_HANDLE) {
            vkDestroyDevice(device, nullptr);
        }
        
        if (instance != VK_NULL_HANDLE) {
            vkDestroyInstance(instance, nullptr);
        }
        
        std::cout << "\n========================================\n";
        std::cout << "❌ 5level测试失败！\n";
        std::cout << "错误出现在: " << e.what() << "\n";
        std::cout << "========================================\n";
        
        return 1;
    }
}
