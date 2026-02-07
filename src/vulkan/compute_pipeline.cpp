#include "vulkan/compute_pipeline.hpp"
#include "utils/logger.hpp"
#include <fstream>
#include <stdexcept>
#include <cstring>

namespace stereo_depth {
namespace vulkan {

ComputePipeline::ComputePipeline(const VulkanContext& context)
    : m_context(context)
    , m_shaderModule(VK_NULL_HANDLE)
    , m_descriptorSetLayout(VK_NULL_HANDLE)
    , m_pipelineLayout(VK_NULL_HANDLE)
    , m_pipeline(VK_NULL_HANDLE)
    , m_descriptorPool(VK_NULL_HANDLE)
    , m_descriptorSet(VK_NULL_HANDLE) {
    LOG_DEBUG("创建ComputePipeline对象");
}

ComputePipeline::~ComputePipeline() {
    cleanup();
}

ComputePipeline::ComputePipeline(ComputePipeline&& other) noexcept
    : m_context(other.m_context)
    , m_shaderModule(other.m_shaderModule)
    , m_descriptorSetLayout(other.m_descriptorSetLayout)
    , m_pipelineLayout(other.m_pipelineLayout)
    , m_pipeline(other.m_pipeline)
    , m_descriptorPool(other.m_descriptorPool)
    , m_descriptorSet(other.m_descriptorSet) {
    
    // 将原对象的句柄置为空，防止双重释放
    other.m_shaderModule = VK_NULL_HANDLE;
    other.m_descriptorSetLayout = VK_NULL_HANDLE;
    other.m_pipelineLayout = VK_NULL_HANDLE;
    other.m_pipeline = VK_NULL_HANDLE;
    other.m_descriptorPool = VK_NULL_HANDLE;
    other.m_descriptorSet = VK_NULL_HANDLE;
    
    LOG_DEBUG("移动构造ComputePipeline");
}

ComputePipeline& ComputePipeline::operator=(ComputePipeline&& other) noexcept {
    if (this != &other) {
        // 清理当前对象的资源
        cleanup();
        
        // 移动资源
        m_shaderModule = other.m_shaderModule;
        m_descriptorSetLayout = other.m_descriptorSetLayout;
        m_pipelineLayout = other.m_pipelineLayout;
        m_pipeline = other.m_pipeline;
        m_descriptorPool = other.m_descriptorPool;
        m_descriptorSet = other.m_descriptorSet;
        
        // 将原对象的句柄置为空
        other.m_shaderModule = VK_NULL_HANDLE;
        other.m_descriptorSetLayout = VK_NULL_HANDLE;
        other.m_pipelineLayout = VK_NULL_HANDLE;
        other.m_pipeline = VK_NULL_HANDLE;
        other.m_descriptorPool = VK_NULL_HANDLE;
        other.m_descriptorSet = VK_NULL_HANDLE;
        
        LOG_DEBUG("移动赋值ComputePipeline");
    }
    return *this;
}

bool ComputePipeline::loadShaderFromFile(const std::string& shaderPath) {
    LOG_DEBUG("从文件加载着色器: {}", shaderPath);
    
    std::ifstream file(shaderPath, std::ios::ate | std::ios::binary);
    
    if (!file.is_open()) {
        LOG_ERROR("无法打开着色器文件: {}", shaderPath);
        return false;
    }
    
    size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));
    
    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
    file.close();
    
    return loadShaderFromMemory(buffer.data(), fileSize);
}

bool ComputePipeline::loadShaderFromMemory(const uint32_t* shaderCode, size_t codeSize) {
    LOG_DEBUG("从内存加载着色器，大小: {} 字节", codeSize);
    
    if (codeSize == 0 || codeSize % 4 != 0) {
        LOG_ERROR("无效的着色器大小: {} 字节", codeSize);
        return false;
    }
    
    if (m_shaderModule != VK_NULL_HANDLE) {
        LOG_WARN("着色器模块已存在，先清理");
        vkDestroyShaderModule(m_context.getDevice(), m_shaderModule, nullptr);
        m_shaderModule = VK_NULL_HANDLE;
    }
    
    m_shaderModule = createShaderModule(shaderCode, codeSize);
    if (m_shaderModule == VK_NULL_HANDLE) {
        LOG_ERROR("创建着色器模块失败");
        return false;
    }
    
    LOG_INFO("✅ 着色器模块创建成功");
    return true;
}

void ComputePipeline::setDescriptorSetLayout(VkDescriptorSetLayout layout) {
    LOG_DEBUG("设置描述符集布局: {}", reinterpret_cast<void*>(layout));
    m_descriptorSetLayout = layout;
}

bool ComputePipeline::createPipeline(size_t pushConstantSize) {
    LOG_DEBUG("开始创建计算管线");
    
    if (m_shaderModule == VK_NULL_HANDLE) {
        LOG_ERROR("着色器模块未加载");
        return false;
    }
    
    if (m_pipelineLayout != VK_NULL_HANDLE || m_pipeline != VK_NULL_HANDLE) {
        LOG_WARN("管线已存在，先清理");
        cleanup();
    }
    
    VkDevice device = m_context.getDevice();
    
    // 调试信息：只在debug级别大于2时输出
    if (spdlog::get_level() <= spdlog::level::debug) {
        LOG_DEBUG("创建计算管线参数:");
        LOG_DEBUG("  设备: {}", reinterpret_cast<void*>(device));
        LOG_DEBUG("  着色器模块: {}", reinterpret_cast<void*>(m_shaderModule));
        LOG_DEBUG("  描述符集布局: {}", reinterpret_cast<void*>(m_descriptorSetLayout));
        LOG_DEBUG("  推送常量大小: {} 字节", pushConstantSize);
    }
    
    // 创建管线布局
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.pNext = nullptr;
    pipelineLayoutInfo.flags = 0;
    
    // 设置描述符集布局
    if (m_descriptorSetLayout != VK_NULL_HANDLE) {
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &m_descriptorSetLayout;
        LOG_DEBUG("使用描述符集布局创建管线布局");
    } else {
        pipelineLayoutInfo.setLayoutCount = 0;
        pipelineLayoutInfo.pSetLayouts = nullptr;
        LOG_DEBUG("创建无描述符集的管线布局");
    }
    
    // 设置推送常量范围
    VkPushConstantRange pushConstantRange = {};
    if (pushConstantSize > 0) {
        pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = static_cast<uint32_t>(pushConstantSize);
        
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
        LOG_DEBUG("设置推送常量: {} 字节", pushConstantSize);
    } else {
        pipelineLayoutInfo.pushConstantRangeCount = 0;
        pipelineLayoutInfo.pPushConstantRanges = nullptr;
    }
    
    VkResult result = vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &m_pipelineLayout);
    if (result != VK_SUCCESS) {
        LOG_ERROR("创建管线布局失败: {}", result);
        return false;
    }
    
    LOG_INFO("✅ 管线布局创建成功");
    
    if (spdlog::get_level() <= spdlog::level::debug) {
        LOG_DEBUG("管线布局创建成功: {}", reinterpret_cast<void*>(m_pipelineLayout));
    }
    
    // 创建计算管线 - 关键修复：完整初始化所有字段
    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.pNext = nullptr;  // 重要！
    pipelineInfo.flags = 0;        // 重要！
    pipelineInfo.layout = m_pipelineLayout;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;  // 重要！
    pipelineInfo.basePipelineIndex = -1;               // 重要！
    
    // 着色器阶段信息 - 完整初始化
    VkPipelineShaderStageCreateInfo shaderStageInfo = {};
    shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageInfo.pNext = nullptr;
    shaderStageInfo.flags = 0;
    shaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageInfo.module = m_shaderModule;
    shaderStageInfo.pName = "main";
    shaderStageInfo.pSpecializationInfo = nullptr;  // 重要！
    
    pipelineInfo.stage = shaderStageInfo;
    
    // 调试信息：输出完整的管线创建信息
    if (spdlog::get_level() <= spdlog::level::debug) {
        LOG_DEBUG("计算管线创建信息:");
        LOG_DEBUG("  sType: {}", pipelineInfo.sType);
        LOG_DEBUG("  pNext: {}", reinterpret_cast<void*>(pipelineInfo.pNext));
        LOG_DEBUG("  flags: {}", pipelineInfo.flags);
        LOG_DEBUG("  阶段: compute");
        LOG_DEBUG("  布局: {}", reinterpret_cast<void*>(pipelineInfo.layout));
        LOG_DEBUG("  basePipelineHandle: {}", reinterpret_cast<void*>(pipelineInfo.basePipelineHandle));
        LOG_DEBUG("  basePipelineIndex: {}", pipelineInfo.basePipelineIndex);
        
        LOG_DEBUG("着色器阶段信息:");
        LOG_DEBUG("  着色器模块: {}", reinterpret_cast<void*>(shaderStageInfo.module));
        LOG_DEBUG("  入口点: {}", shaderStageInfo.pName);
        LOG_DEBUG("  特化信息: {}", reinterpret_cast<void*>(shaderStageInfo.pSpecializationInfo));
    }
    
    LOG_DEBUG("正在创建计算管线...");
    
    result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_pipeline);
    if (result != VK_SUCCESS) {
        LOG_ERROR("创建计算管线失败: {}", result);
        
        // 更详细的错误信息
        if (result == VK_ERROR_INCOMPATIBLE_DRIVER) {
            LOG_ERROR("错误 -13: VK_ERROR_INCOMPATIBLE_DRIVER");
            LOG_ERROR("可能原因:");
            LOG_ERROR("  1. 驱动程序版本太旧");
            LOG_ERROR("  2. 驱动程序不支持某些Vulkan特性");
            LOG_ERROR("  3. 创建管线时的参数不完整");
            LOG_ERROR("  4. 着色器使用了驱动程序不支持的SPIR-V特性");
        } else {
            LOG_ERROR("错误代码: {}", result);
            
            // 常见错误代码解释
            switch (result) {
                case VK_ERROR_OUT_OF_HOST_MEMORY:
                    LOG_ERROR("主机内存不足");
                    break;
                case VK_ERROR_OUT_OF_DEVICE_MEMORY:
                    LOG_ERROR("设备内存不足");
                    break;
                case VK_ERROR_INITIALIZATION_FAILED:
                    LOG_ERROR("初始化失败");
                    break;
                case VK_ERROR_DEVICE_LOST:
                    LOG_ERROR("设备丢失");
                    break;
                case VK_ERROR_FEATURE_NOT_PRESENT:
                    LOG_ERROR("不支持的特性");
                    break;
                case VK_ERROR_EXTENSION_NOT_PRESENT:
                    LOG_ERROR("扩展不存在");
                    break;
                default:
                    LOG_ERROR("未知错误代码");
                    break;
            }
        }
        
        vkDestroyPipelineLayout(device, m_pipelineLayout, nullptr);
        m_pipelineLayout = VK_NULL_HANDLE;
        return false;
    }
    
    LOG_INFO("✅ 计算管线创建成功");
    
    if (spdlog::get_level() <= spdlog::level::debug) {
        LOG_DEBUG("计算管线创建成功: {}", reinterpret_cast<void*>(m_pipeline));
    }
    
    return true;
}

bool ComputePipeline::createDescriptorSet(const std::vector<VkBuffer>& buffers,
                                        const std::vector<VkDescriptorType>& bufferTypes) {
    LOG_DEBUG("创建描述符集，缓冲区数量: {}", buffers.size());
    
    if (buffers.size() != bufferTypes.size()) {
        LOG_ERROR("缓冲区数量 ({}) 与缓冲区类型数量 ({}) 不匹配", 
                  buffers.size(), bufferTypes.size());
        return false;
    }
    
    if (!createDescriptorPool(bufferTypes)) {
        return false;
    }
    
    VkDevice device = m_context.getDevice();
    
    // 分配描述符集
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = m_descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &m_descriptorSetLayout;
    
    VkResult result = vkAllocateDescriptorSets(device, &allocInfo, &m_descriptorSet);
    if (result != VK_SUCCESS) {
        LOG_ERROR("分配描述符集失败: {}", result);
        return false;
    }
    
    // 更新描述符集
    std::vector<VkDescriptorBufferInfo> bufferInfos(buffers.size());
    std::vector<VkWriteDescriptorSet> descriptorWrites(buffers.size());
    
    for (size_t i = 0; i < buffers.size(); ++i) {
        bufferInfos[i].buffer = buffers[i];
        bufferInfos[i].offset = 0;
        bufferInfos[i].range = VK_WHOLE_SIZE;
        
        descriptorWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[i].dstSet = m_descriptorSet;
        descriptorWrites[i].dstBinding = static_cast<uint32_t>(i);
        descriptorWrites[i].dstArrayElement = 0;
        descriptorWrites[i].descriptorType = bufferTypes[i];
        descriptorWrites[i].descriptorCount = 1;
        descriptorWrites[i].pBufferInfo = &bufferInfos[i];
        descriptorWrites[i].pImageInfo = nullptr;
        descriptorWrites[i].pTexelBufferView = nullptr;
    }
    
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), 
                          descriptorWrites.data(), 0, nullptr);
    
    LOG_INFO("✅ 描述符集创建成功，包含 {} 个缓冲区", buffers.size());
    return true;
}

void ComputePipeline::recordCommands(VkCommandBuffer commandBuffer,
                                    uint32_t groupCountX,
                                    uint32_t groupCountY,
                                    uint32_t groupCountZ,
                                    const void* pushConstants,
                                    size_t pushConstantSize) {
    if (m_pipeline == VK_NULL_HANDLE) {
        LOG_ERROR("无法记录命令: 管线未创建");
        return;
    }
    
    // 绑定管线
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
    
    // 绑定描述符集
    if (m_descriptorSet != VK_NULL_HANDLE) {
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               m_pipelineLayout, 0, 1, &m_descriptorSet, 0, nullptr);
    }
    
    // 设置推送常量
    if (pushConstants != nullptr && pushConstantSize > 0) {
        vkCmdPushConstants(commandBuffer, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                          0, static_cast<uint32_t>(pushConstantSize), pushConstants);
    }
    
    // 分派计算
    vkCmdDispatch(commandBuffer, groupCountX, groupCountY, groupCountZ);
    
    LOG_DEBUG("记录计算命令: 工作组=({}, {}, {})", 
              groupCountX, groupCountY, groupCountZ);
}

VkShaderModule ComputePipeline::createShaderModule(const uint32_t* code, size_t codeSize) {
    LOG_DEBUG("创建着色器模块，大小: {} 字节", codeSize);
    
    VkDevice device = m_context.getDevice();
    
    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = codeSize;
    createInfo.pCode = code;
    
    VkShaderModule shaderModule;
    VkResult result = vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule);
    
    if (result != VK_SUCCESS) {
        LOG_ERROR("创建着色器模块失败: {}", result);
        return VK_NULL_HANDLE;
    }
    
    LOG_DEBUG("着色器模块创建成功");
    return shaderModule;
}

bool ComputePipeline::createDescriptorPool(const std::vector<VkDescriptorType>& bufferTypes) {
    LOG_DEBUG("创建描述符池，类型数量: {}", bufferTypes.size());
    
    if (m_descriptorPool != VK_NULL_HANDLE) {
        LOG_WARN("描述符池已存在");
        return true;
    }
    
    VkDevice device = m_context.getDevice();
    
    // 统计每种类型的描述符数量
    std::vector<VkDescriptorPoolSize> poolSizes;
    for (const auto& type : bufferTypes) {
        bool found = false;
        for (auto& poolSize : poolSizes) {
            if (poolSize.type == type) {
                poolSize.descriptorCount++;
                found = true;
                break;
            }
        }
        
        if (!found) {
            VkDescriptorPoolSize poolSize = {};
            poolSize.type = type;
            poolSize.descriptorCount = 1;
            poolSizes.push_back(poolSize);
        }
    }
    
    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = 1; // 我们只需要一个描述符集
    
    VkResult result = vkCreateDescriptorPool(device, &poolInfo, nullptr, &m_descriptorPool);
    if (result != VK_SUCCESS) {
        LOG_ERROR("创建描述符池失败: {}", result);
        return false;
    }
    
    LOG_DEBUG("描述符池创建成功，包含 {} 种类型", poolSizes.size());
    return true;
}

void ComputePipeline::cleanup() {
    LOG_DEBUG("清理ComputePipeline资源");
    
    VkDevice device = m_context.getDevice();
    
    if (m_pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, m_pipeline, nullptr);
        m_pipeline = VK_NULL_HANDLE;
        LOG_DEBUG("销毁管线");
    }
    
    if (m_pipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device, m_pipelineLayout, nullptr);
        m_pipelineLayout = VK_NULL_HANDLE;
        LOG_DEBUG("销毁管线布局");
    }
    
    if (m_shaderModule != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device, m_shaderModule, nullptr);
        m_shaderModule = VK_NULL_HANDLE;
        LOG_DEBUG("销毁着色器模块");
    }
    
    if (m_descriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, m_descriptorPool, nullptr);
        m_descriptorPool = VK_NULL_HANDLE;
        LOG_DEBUG("销毁描述符池");
    }
    
    // 注意：m_descriptorSetLayout 由调用者管理，这里不销毁
    m_descriptorSet = VK_NULL_HANDLE;
    
    LOG_DEBUG("ComputePipeline资源清理完成");
}

} // namespace vulkan
} // namespace stereo_depth
