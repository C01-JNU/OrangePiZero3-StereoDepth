#include "vulkan/compute_pipeline.hpp"
#include "utils/logger.hpp"
#include <fstream>
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
        
        LOG_DEBUG("移动赋值ComputePipeline");
    }
    return *this;
}

bool ComputePipeline::loadShaderFromFile(const std::string& shaderPath) {
    LOG_DEBUG("从文件加载着色器: {}", shaderPath);
    
    std::ifstream file(shaderPath, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        LOG_ERROR("❌ 无法打开着色器文件: {}", shaderPath);
        return false;   // 关键修复：必须返回 false！
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
    m_descriptorSetLayout = layout;
}

bool ComputePipeline::createPipeline(size_t pushConstantSize) {
    if (m_shaderModule == VK_NULL_HANDLE) {
        LOG_ERROR("着色器模块未加载");
        return false;
    }
    
    VkDevice device = m_context.getDevice();
    
    // 创建管线布局
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = (m_descriptorSetLayout != VK_NULL_HANDLE) ? 1 : 0;
    pipelineLayoutInfo.pSetLayouts = &m_descriptorSetLayout;
    
    VkPushConstantRange pushConstantRange = {};
    if (pushConstantSize > 0) {
        pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = static_cast<uint32_t>(pushConstantSize);
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
    }
    
    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &m_pipelineLayout) != VK_SUCCESS) {
        LOG_ERROR("创建管线布局失败");
        return false;
    }
    
    // 创建计算管线
    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.layout = m_pipelineLayout;
    
    VkPipelineShaderStageCreateInfo stageInfo = {};
    stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = m_shaderModule;
    stageInfo.pName = "main";
    
    pipelineInfo.stage = stageInfo;
    
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_pipeline) != VK_SUCCESS) {
        LOG_ERROR("创建计算管线失败");
        vkDestroyPipelineLayout(device, m_pipelineLayout, nullptr);
        m_pipelineLayout = VK_NULL_HANDLE;
        return false;
    }
    
    LOG_INFO("✅ 计算管线创建成功");
    return true;
}

bool ComputePipeline::createDescriptorSet(const std::vector<VkBuffer>& buffers,
                                          const std::vector<VkDescriptorType>& bufferTypes) {
    if (buffers.size() != bufferTypes.size()) {
        LOG_ERROR("缓冲区数量与类型数量不匹配");
        return false;
    }
    
    VkDevice device = m_context.getDevice();
    
    // 创建描述符池
    std::vector<VkDescriptorPoolSize> poolSizes;
    for (const auto& type : bufferTypes) {
        bool found = false;
        for (auto& ps : poolSizes) {
            if (ps.type == type) {
                ps.descriptorCount++;
                found = true;
                break;
            }
        }
        if (!found) {
            poolSizes.push_back({type, 1});
        }
    }
    
    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = 1;
    
    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &m_descriptorPool) != VK_SUCCESS) {
        LOG_ERROR("创建描述符池失败");
        return false;
    }
    
    // 分配描述符集
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = m_descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &m_descriptorSetLayout;
    
    if (vkAllocateDescriptorSets(device, &allocInfo, &m_descriptorSet) != VK_SUCCESS) {
        LOG_ERROR("分配描述符集失败");
        vkDestroyDescriptorPool(device, m_descriptorPool, nullptr);
        m_descriptorPool = VK_NULL_HANDLE;
        return false;
    }
    
    // 更新描述符集
    std::vector<VkDescriptorBufferInfo> bufferInfos(buffers.size());
    std::vector<VkWriteDescriptorSet> writes(buffers.size());
    
    for (size_t i = 0; i < buffers.size(); ++i) {
        bufferInfos[i].buffer = buffers[i];
        bufferInfos[i].offset = 0;
        bufferInfos[i].range = VK_WHOLE_SIZE;
        
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = m_descriptorSet;
        writes[i].dstBinding = static_cast<uint32_t>(i);
        writes[i].dstArrayElement = 0;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = bufferTypes[i];
        writes[i].pBufferInfo = &bufferInfos[i];
    }
    
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    
    LOG_INFO("✅ 描述符集创建成功，包含 {} 个缓冲区", buffers.size());
    return true;
}

void ComputePipeline::recordCommands(VkCommandBuffer commandBuffer,
                                     uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ,
                                     const void* pushConstants, size_t pushConstantSize) {
    if (m_pipeline == VK_NULL_HANDLE) {
        LOG_ERROR("管线未创建，无法记录命令");
        return;
    }
    
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
    
    if (m_descriptorSet != VK_NULL_HANDLE) {
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                m_pipelineLayout, 0, 1, &m_descriptorSet, 0, nullptr);
    }
    
    if (pushConstants != nullptr && pushConstantSize > 0) {
        vkCmdPushConstants(commandBuffer, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, static_cast<uint32_t>(pushConstantSize), pushConstants);
    }
    
    vkCmdDispatch(commandBuffer, groupCountX, groupCountY, groupCountZ);
}

VkShaderModule ComputePipeline::createShaderModule(const uint32_t* code, size_t codeSize) {
    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = codeSize;
    createInfo.pCode = code;
    
    VkShaderModule shaderModule;
    if (vkCreateShaderModule(m_context.getDevice(), &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    return shaderModule;
}

void ComputePipeline::cleanup() {
    VkDevice device = m_context.getDevice();
    if (device == VK_NULL_HANDLE) return;
    
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
    m_descriptorSet = VK_NULL_HANDLE;
}

} // namespace vulkan
} // namespace stereo_depth
