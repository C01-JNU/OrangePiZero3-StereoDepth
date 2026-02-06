#include "vulkan/compute_pipeline.hpp"
#include "utils/logger.hpp"
#include <fstream>
#include <stdexcept>
#include <cstring>

namespace stereo_depth {
namespace vulkan {

ComputePipeline::ComputePipeline(const VulkanContext& context)
    : m_context(context) {
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
    
    other.m_shaderModule = VK_NULL_HANDLE;
    other.m_descriptorSetLayout = VK_NULL_HANDLE;
    other.m_pipelineLayout = VK_NULL_HANDLE;
    other.m_pipeline = VK_NULL_HANDLE;
    other.m_descriptorPool = VK_NULL_HANDLE;
    other.m_descriptorSet = VK_NULL_HANDLE;
}

ComputePipeline& ComputePipeline::operator=(ComputePipeline&& other) noexcept {
    if (this != &other) {
        cleanup();
        
        m_shaderModule = other.m_shaderModule;
        m_descriptorSetLayout = other.m_descriptorSetLayout;
        m_pipelineLayout = other.m_pipelineLayout;
        m_pipeline = other.m_pipeline;
        m_descriptorPool = other.m_descriptorPool;
        m_descriptorSet = other.m_descriptorSet;
        
        other.m_shaderModule = VK_NULL_HANDLE;
        other.m_descriptorSetLayout = VK_NULL_HANDLE;
        other.m_pipelineLayout = VK_NULL_HANDLE;
        other.m_pipeline = VK_NULL_HANDLE;
        other.m_descriptorPool = VK_NULL_HANDLE;
        other.m_descriptorSet = VK_NULL_HANDLE;
    }
    return *this;
}

bool ComputePipeline::loadShaderFromFile(const std::string& shaderPath) {
    std::ifstream file(shaderPath, std::ios::ate | std::ios::binary);
    
    if (!file.is_open()) {
        LOG_ERROR("Failed to open shader file: {}", shaderPath);
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
    if (m_shaderModule != VK_NULL_HANDLE) {
        LOG_WARN("Shader module already exists, cleaning up first");
        vkDestroyShaderModule(m_context.getDevice(), m_shaderModule, nullptr);
        m_shaderModule = VK_NULL_HANDLE;
    }
    
    m_shaderModule = createShaderModule(shaderCode, codeSize);
    return m_shaderModule != VK_NULL_HANDLE;
}

void ComputePipeline::setDescriptorSetLayout(VkDescriptorSetLayout layout) {
    m_descriptorSetLayout = layout;
}

bool ComputePipeline::createPipeline(size_t pushConstantSize) {
    if (m_shaderModule == VK_NULL_HANDLE) {
        LOG_ERROR("Shader module not loaded");
        return false;
    }
    
    if (m_pipelineLayout != VK_NULL_HANDLE || m_pipeline != VK_NULL_HANDLE) {
        LOG_WARN("Pipeline already exists, cleaning up first");
        cleanup();
    }
    
    VkDevice device = m_context.getDevice();
    
    // 创建管线布局
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    
    // 设置描述符集布局
    VkDescriptorSetLayout layouts[] = { m_descriptorSetLayout };
    if (m_descriptorSetLayout != VK_NULL_HANDLE) {
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = layouts;
    } else {
        pipelineLayoutInfo.setLayoutCount = 0;
        pipelineLayoutInfo.pSetLayouts = nullptr;
    }
    
    // 设置推送常量范围
    VkPushConstantRange pushConstantRange = {};
    if (pushConstantSize > 0) {
        pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = static_cast<uint32_t>(pushConstantSize);
        
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
    } else {
        pipelineLayoutInfo.pushConstantRangeCount = 0;
        pipelineLayoutInfo.pPushConstantRanges = nullptr;
    }
    
    VkResult result = vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &m_pipelineLayout);
    if (result != VK_SUCCESS) {
        LOG_ERROR("Failed to create pipeline layout: {}", result);
        return false;
    }
    
    // 创建计算管线
    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.layout = m_pipelineLayout;
    
    // 着色器阶段信息
    VkPipelineShaderStageCreateInfo shaderStageInfo = {};
    shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageInfo.module = m_shaderModule;
    shaderStageInfo.pName = "main"; // 着色器入口函数名
    
    pipelineInfo.stage = shaderStageInfo;
    
    result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_pipeline);
    if (result != VK_SUCCESS) {
        LOG_ERROR("Failed to create compute pipeline: {}", result);
        vkDestroyPipelineLayout(device, m_pipelineLayout, nullptr);
        m_pipelineLayout = VK_NULL_HANDLE;
        return false;
    }
    
    LOG_DEBUG("Compute pipeline created successfully");
    if (pushConstantSize > 0) {
        LOG_DEBUG("  Push constants: {} bytes", pushConstantSize);
    }
    
    return true;
}

bool ComputePipeline::createDescriptorSet(const std::vector<VkBuffer>& buffers,
                                        const std::vector<VkDescriptorType>& bufferTypes) {
    if (buffers.size() != bufferTypes.size()) {
        LOG_ERROR("Buffer count ({}) does not match buffer type count ({})", 
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
        LOG_ERROR("Failed to allocate descriptor set: {}", result);
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
    
    LOG_DEBUG("Descriptor set created with {} buffers", buffers.size());
    return true;
}

void ComputePipeline::recordCommands(VkCommandBuffer commandBuffer,
                                    uint32_t groupCountX,
                                    uint32_t groupCountY,
                                    uint32_t groupCountZ,
                                    const void* pushConstants,
                                    size_t pushConstantSize) {
    if (m_pipeline == VK_NULL_HANDLE) {
        LOG_ERROR("Cannot record commands: pipeline not created");
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
    
    LOG_DEBUG("Recorded compute command: groups=({}, {}, {})", 
              groupCountX, groupCountY, groupCountZ);
}

VkShaderModule ComputePipeline::createShaderModule(const uint32_t* code, size_t codeSize) {
    VkDevice device = m_context.getDevice();
    
    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = codeSize;
    createInfo.pCode = code;
    
    VkShaderModule shaderModule;
    VkResult result = vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule);
    
    if (result != VK_SUCCESS) {
        LOG_ERROR("Failed to create shader module: {}", result);
        return VK_NULL_HANDLE;
    }
    
    LOG_DEBUG("Shader module created: size={} bytes", codeSize);
    return shaderModule;
}

bool ComputePipeline::createDescriptorPool(const std::vector<VkDescriptorType>& bufferTypes) {
    if (m_descriptorPool != VK_NULL_HANDLE) {
        LOG_WARN("Descriptor pool already exists");
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
        LOG_ERROR("Failed to create descriptor pool: {}", result);
        return false;
    }
    
    LOG_DEBUG("Descriptor pool created with {} pool sizes", poolSizes.size());
    return true;
}

void ComputePipeline::cleanup() {
    VkDevice device = m_context.getDevice();
    
    if (m_pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, m_pipeline, nullptr);
        m_pipeline = VK_NULL_HANDLE;
    }
    
    if (m_pipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device, m_pipelineLayout, nullptr);
        m_pipelineLayout = VK_NULL_HANDLE;
    }
    
    if (m_shaderModule != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device, m_shaderModule, nullptr);
        m_shaderModule = VK_NULL_HANDLE;
    }
    
    if (m_descriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, m_descriptorPool, nullptr);
        m_descriptorPool = VK_NULL_HANDLE;
    }
    
    // 注意：m_descriptorSetLayout 由调用者管理，这里不销毁
    m_descriptorSet = VK_NULL_HANDLE;
}

} // namespace vulkan
} // namespace stereo_depth
