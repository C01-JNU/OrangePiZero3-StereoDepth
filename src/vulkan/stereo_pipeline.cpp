#include "vulkan/stereo_pipeline.hpp"
#include "utils/logger.hpp"
#include "utils/config.hpp"
#include <filesystem>

namespace stereo_depth {
namespace vulkan {

StereoPipeline::StereoPipeline(const VulkanContext& context)
    : m_context(context) {
}

StereoPipeline::~StereoPipeline() {
    cleanup();
}

StereoPipeline::StereoPipeline(StereoPipeline&& other) noexcept
    : m_context(other.m_context)
    , m_imageWidth(other.m_imageWidth)
    , m_imageHeight(other.m_imageHeight)
    , m_maxDisparity(other.m_maxDisparity)
    , m_initialized(other.m_initialized)
    , m_leftImageBuffer(std::move(other.m_leftImageBuffer))
    , m_rightImageBuffer(std::move(other.m_rightImageBuffer))
    , m_costVolumeBuffer(std::move(other.m_costVolumeBuffer))
    , m_disparityBuffer(std::move(other.m_disparityBuffer))
    , m_tempBuffer1(std::move(other.m_tempBuffer1))
    , m_tempBuffer2(std::move(other.m_tempBuffer2))
    , m_paramsBuffer(std::move(other.m_paramsBuffer))
    , m_censusPipeline(std::move(other.m_censusPipeline))
    , m_costPipeline(std::move(other.m_costPipeline))
    , m_aggregationPipeline(std::move(other.m_aggregationPipeline))
    , m_wtaPipeline(std::move(other.m_wtaPipeline))
    , m_postprocessPipeline(std::move(other.m_postprocessPipeline))
    , m_censusLayout(other.m_censusLayout)
    , m_costLayout(other.m_costLayout)
    , m_aggregationLayout(other.m_aggregationLayout)
    , m_wtaLayout(other.m_wtaLayout)
    , m_postprocessLayout(other.m_postprocessLayout) {
    
    other.m_initialized = false;
    other.m_censusLayout = VK_NULL_HANDLE;
    other.m_costLayout = VK_NULL_HANDLE;
    other.m_aggregationLayout = VK_NULL_HANDLE;
    other.m_wtaLayout = VK_NULL_HANDLE;
    other.m_postprocessLayout = VK_NULL_HANDLE;
}

StereoPipeline& StereoPipeline::operator=(StereoPipeline&& other) noexcept {
    if (this != &other) {
        cleanup();
        
        m_imageWidth = other.m_imageWidth;
        m_imageHeight = other.m_imageHeight;
        m_maxDisparity = other.m_maxDisparity;
        m_initialized = other.m_initialized;
        
        m_leftImageBuffer = std::move(other.m_leftImageBuffer);
        m_rightImageBuffer = std::move(other.m_rightImageBuffer);
        m_costVolumeBuffer = std::move(other.m_costVolumeBuffer);
        m_disparityBuffer = std::move(other.m_disparityBuffer);
        m_tempBuffer1 = std::move(other.m_tempBuffer1);
        m_tempBuffer2 = std::move(other.m_tempBuffer2);
        m_paramsBuffer = std::move(other.m_paramsBuffer);
        
        m_censusPipeline = std::move(other.m_censusPipeline);
        m_costPipeline = std::move(other.m_costPipeline);
        m_aggregationPipeline = std::move(other.m_aggregationPipeline);
        m_wtaPipeline = std::move(other.m_wtaPipeline);
        m_postprocessPipeline = std::move(other.m_postprocessPipeline);
        
        m_censusLayout = other.m_censusLayout;
        m_costLayout = other.m_costLayout;
        m_aggregationLayout = other.m_aggregationLayout;
        m_wtaLayout = other.m_wtaLayout;
        m_postprocessLayout = other.m_postprocessLayout;
        
        other.m_initialized = false;
        other.m_censusLayout = VK_NULL_HANDLE;
        other.m_costLayout = VK_NULL_HANDLE;
        other.m_aggregationLayout = VK_NULL_HANDLE;
        other.m_wtaLayout = VK_NULL_HANDLE;
        other.m_postprocessLayout = VK_NULL_HANDLE;
    }
    return *this;
}

bool StereoPipeline::initialize(uint32_t imageWidth, uint32_t imageHeight, uint32_t maxDisparity) {
    if (m_initialized) {
        LOG_WARN("Pipeline already initialized");
        return true;
    }
    
    m_imageWidth = imageWidth;
    m_imageHeight = imageHeight;
    m_maxDisparity = maxDisparity;
    
    LOG_INFO("Initializing stereo pipeline");
    LOG_INFO("  Image size: {} x {}", m_imageWidth, m_imageHeight);
    LOG_INFO("  Max disparity: {}", m_maxDisparity);
    
    try {
        if (!createBuffers()) {
            LOG_ERROR("Failed to create buffers");
            return false;
        }
        
        if (!createDescriptorSetLayouts()) {
            LOG_ERROR("Failed to create descriptor set layouts");
            return false;
        }
        
        if (!createPipelines()) {
            LOG_ERROR("Failed to create compute pipelines");
            return false;
        }
        
        // 设置计算参数
        PipelineParams params = {};
        params.imageWidth = m_imageWidth;
        params.imageHeight = m_imageHeight;
        params.maxDisparity = m_maxDisparity;
        params.windowSize = 9; // 默认窗口大小
        params.uniquenessRatio = 0.6f; // 默认唯一性比率
        
        if (!m_paramsBuffer->copyToBuffer(&params, sizeof(params))) {
            LOG_ERROR("Failed to copy pipeline parameters");
            return false;
        }
        
        m_initialized = true;
        LOG_INFO("Stereo pipeline initialized successfully");
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("Exception during pipeline initialization: {}", e.what());
        cleanup();
        return false;
    }
}

bool StereoPipeline::setLeftImage(const uint8_t* data) {
    if (!m_initialized) {
        LOG_ERROR("Pipeline not initialized");
        return false;
    }
    
    size_t imageSize = m_imageWidth * m_imageHeight;
    return m_leftImageBuffer->copyToBuffer(data, imageSize);
}

bool StereoPipeline::setRightImage(const uint8_t* data) {
    if (!m_initialized) {
        LOG_ERROR("Pipeline not initialized");
        return false;
    }
    
    size_t imageSize = m_imageWidth * m_imageHeight;
    return m_rightImageBuffer->copyToBuffer(data, imageSize);
}

bool StereoPipeline::compute() {
    if (!m_initialized) {
        LOG_ERROR("Pipeline not initialized");
        return false;
    }
    
    LOG_DEBUG("Starting stereo matching computation");
    
    try {
        // 步骤1: Census变换
        LOG_DEBUG("Step 1: Census transform");
        if (!executePipelineStep(*m_censusPipeline, 
                                (m_imageWidth + 15) / 16, 
                                (m_imageHeight + 15) / 16)) {
            LOG_ERROR("Failed to execute census pipeline");
            return false;
        }
        
        // 步骤2: 代价计算
        LOG_DEBUG("Step 2: Cost computation");
        if (!executePipelineStep(*m_costPipeline,
                                (m_imageWidth + 7) / 8,
                                (m_imageHeight + 7) / 8)) {
            LOG_ERROR("Failed to execute cost pipeline");
            return false;
        }
        
        // 步骤3: 代价聚合
        LOG_DEBUG("Step 3: Cost aggregation");
        if (!executePipelineStep(*m_aggregationPipeline,
                                (m_imageWidth + 7) / 8,
                                (m_imageHeight + 7) / 8)) {
            LOG_ERROR("Failed to execute aggregation pipeline");
            return false;
        }
        
        // 步骤4: WTA（赢家通吃）优化
        LOG_DEBUG("Step 4: WTA optimization");
        if (!executePipelineStep(*m_wtaPipeline,
                                (m_imageWidth + 15) / 16,
                                (m_imageHeight + 15) / 16)) {
            LOG_ERROR("Failed to execute WTA pipeline");
            return false;
        }
        
        // 步骤5: 后处理
        LOG_DEBUG("Step 5: Post-processing");
        if (!executePipelineStep(*m_postprocessPipeline,
                                (m_imageWidth + 15) / 16,
                                (m_imageHeight + 15) / 16)) {
            LOG_ERROR("Failed to execute post-processing pipeline");
            return false;
        }
        
        LOG_INFO("Stereo matching completed successfully");
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("Exception during computation: {}", e.what());
        return false;
    }
}

bool StereoPipeline::getDisparityMap(uint16_t* output) {
    if (!m_initialized || !m_disparityBuffer) {
        LOG_ERROR("Pipeline not initialized or disparity buffer not available");
        return false;
    }
    
    size_t disparitySize = m_imageWidth * m_imageHeight * sizeof(uint16_t);
    return m_disparityBuffer->copyFromBuffer(output, disparitySize);
}

bool StereoPipeline::getIntermediateResult(uint32_t bufferIndex, void* output, size_t size) {
    if (!m_initialized) {
        LOG_ERROR("Pipeline not initialized");
        return false;
    }
    
    BufferManager* buffer = nullptr;
    
    switch (bufferIndex) {
        case 0: buffer = m_leftImageBuffer.get(); break;
        case 1: buffer = m_rightImageBuffer.get(); break;
        case 2: buffer = m_costVolumeBuffer.get(); break;
        case 3: buffer = m_disparityBuffer.get(); break;
        case 4: buffer = m_tempBuffer1.get(); break;
        case 5: buffer = m_tempBuffer2.get(); break;
        default:
            LOG_ERROR("Invalid buffer index: {}", bufferIndex);
            return false;
    }
    
    if (!buffer || !buffer->isValid()) {
        LOG_ERROR("Buffer {} not available", bufferIndex);
        return false;
    }
    
    return buffer->copyFromBuffer(output, size);
}

bool StereoPipeline::createBuffers() {
    size_t imageSize = m_imageWidth * m_imageHeight;
    size_t costVolumeSize = m_imageWidth * m_imageHeight * m_maxDisparity * sizeof(uint16_t);
    size_t disparitySize = m_imageWidth * m_imageHeight * sizeof(uint16_t);
    
    try {
        // 左图像缓冲区
        m_leftImageBuffer = std::make_unique<BufferManager>(m_context);
        if (!m_leftImageBuffer->createStorageBuffer(imageSize)) {
            LOG_ERROR("Failed to create left image buffer");
            return false;
        }
        
        // 右图像缓冲区
        m_rightImageBuffer = std::make_unique<BufferManager>(m_context);
        if (!m_rightImageBuffer->createStorageBuffer(imageSize)) {
            LOG_ERROR("Failed to create right image buffer");
            return false;
        }
        
        // 代价体缓冲区
        m_costVolumeBuffer = std::make_unique<BufferManager>(m_context);
        if (!m_costVolumeBuffer->createStorageBuffer(costVolumeSize)) {
            LOG_ERROR("Failed to create cost volume buffer");
            return false;
        }
        
        // 视差图缓冲区
        m_disparityBuffer = std::make_unique<BufferManager>(m_context);
        if (!m_disparityBuffer->createStorageBuffer(disparitySize)) {
            LOG_ERROR("Failed to create disparity buffer");
            return false;
        }
        
        // 临时缓冲区1（用于中间计算）
        m_tempBuffer1 = std::make_unique<BufferManager>(m_context);
        if (!m_tempBuffer1->createStorageBuffer(imageSize * sizeof(uint32_t))) {
            LOG_ERROR("Failed to create temp buffer 1");
            return false;
        }
        
        // 临时缓冲区2（用于中间计算）
        m_tempBuffer2 = std::make_unique<BufferManager>(m_context);
        if (!m_tempBuffer2->createStorageBuffer(imageSize * sizeof(uint32_t))) {
            LOG_ERROR("Failed to create temp buffer 2");
            return false;
        }
        
        // 参数缓冲区
        m_paramsBuffer = std::make_unique<BufferManager>(m_context);
        if (!m_paramsBuffer->createUniformBuffer(sizeof(PipelineParams))) {
            LOG_ERROR("Failed to create params buffer");
            return false;
        }
        
        LOG_DEBUG("Created {} buffers for stereo pipeline", 7);
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("Exception creating buffers: {}", e.what());
        return false;
    }
}

bool StereoPipeline::createDescriptorSetLayouts() {
    DescriptorSetLayoutBuilder builder(m_context);
    
    // Census变换布局：左图、右图、左Census、右Census
    builder.addStorageBuffer(0)  // 左图像
           .addStorageBuffer(1)  // 右图像
           .addStorageBuffer(2)  // 左Census特征
           .addStorageBuffer(3)  // 右Census特征
           .addUniformBuffer(4); // 参数
    
    m_censusLayout = builder.build();
    if (m_censusLayout == VK_NULL_HANDLE) return false;
    
    // 代价计算布局：左Census、右Census、代价体
    DescriptorSetLayoutBuilder builder2(m_context);
    builder2.addStorageBuffer(0)  // 左Census特征
            .addStorageBuffer(1)  // 右Census特征
            .addStorageBuffer(2)  // 代价体
            .addUniformBuffer(3); // 参数
    
    m_costLayout = builder2.build();
    if (m_costLayout == VK_NULL_HANDLE) return false;
    
    // 代价聚合布局：代价体、聚合后的代价、临时缓冲区
    DescriptorSetLayoutBuilder builder3(m_context);
    builder3.addStorageBuffer(0)  // 输入代价体
            .addStorageBuffer(1)  // 输出聚合代价
            .addStorageBuffer(2)  // 临时缓冲区1
            .addUniformBuffer(3); // 参数
    
    m_aggregationLayout = builder3.build();
    if (m_aggregationLayout == VK_NULL_HANDLE) return false;
    
    // WTA布局：聚合代价、视差图
    DescriptorSetLayoutBuilder builder4(m_context);
    builder4.addStorageBuffer(0)  // 聚合代价
            .addStorageBuffer(1)  // 视差图
            .addUniformBuffer(2); // 参数
    
    m_wtaLayout = builder4.build();
    if (m_wtaLayout == VK_NULL_HANDLE) return false;
    
    // 后处理布局：视差图、后处理后的视差、临时缓冲区
    DescriptorSetLayoutBuilder builder5(m_context);
    builder5.addStorageBuffer(0)  // 输入视差图
            .addStorageBuffer(1)  // 输出视差图
            .addStorageBuffer(2)  // 临时缓冲区2
            .addUniformBuffer(3); // 参数
    
    m_postprocessLayout = builder5.build();
    if (m_postprocessLayout == VK_NULL_HANDLE) return false;
    
    LOG_DEBUG("Created {} descriptor set layouts", 5);
    return true;
}

bool StereoPipeline::createPipelines() {
    // 创建Census变换管线
    m_censusPipeline = std::make_unique<ComputePipeline>(m_context);
    m_censusPipeline->setDescriptorSetLayout(m_censusLayout);
    
    if (!loadShader(*m_censusPipeline, "census.comp.spv") ||
        !m_censusPipeline->createPipeline()) {
        LOG_ERROR("Failed to create census pipeline");
        return false;
    }
    
    // 创建代价计算管线
    m_costPipeline = std::make_unique<ComputePipeline>(m_context);
    m_costPipeline->setDescriptorSetLayout(m_costLayout);
    
    if (!loadShader(*m_costPipeline, "cost.comp.spv") ||
        !m_costPipeline->createPipeline()) {
        LOG_ERROR("Failed to create cost pipeline");
        return false;
    }
    
    // 创建代价聚合管线
    m_aggregationPipeline = std::make_unique<ComputePipeline>(m_context);
    m_aggregationPipeline->setDescriptorSetLayout(m_aggregationLayout);
    
    if (!loadShader(*m_aggregationPipeline, "aggregation.comp.spv") ||
        !m_aggregationPipeline->createPipeline()) {
        LOG_ERROR("Failed to create aggregation pipeline");
        return false;
    }
    
    // 创建WTA管线
    m_wtaPipeline = std::make_unique<ComputePipeline>(m_context);
    m_wtaPipeline->setDescriptorSetLayout(m_wtaLayout);
    
    if (!loadShader(*m_wtaPipeline, "wta.comp.spv") ||
        !m_wtaPipeline->createPipeline()) {
        LOG_ERROR("Failed to create WTA pipeline");
        return false;
    }
    
    // 创建后处理管线
    m_postprocessPipeline = std::make_unique<ComputePipeline>(m_context);
    m_postprocessPipeline->setDescriptorSetLayout(m_postprocessLayout);
    
    if (!loadShader(*m_postprocessPipeline, "postprocess.comp.spv") ||
        !m_postprocessPipeline->createPipeline()) {
        LOG_ERROR("Failed to create postprocess pipeline");
        return false;
    }
    
    LOG_DEBUG("Created {} compute pipelines", 5);
    return true;
}

bool StereoPipeline::loadShader(ComputePipeline& pipeline, const std::string& shaderName) {
    // 尝试从多个位置查找着色器文件
    std::vector<std::string> searchPaths = {
        "shaders/" + shaderName,
        "../shaders/" + shaderName,
        "../../shaders/" + shaderName,
        "third_party/shaders/" + shaderName
    };
    
    for (const auto& path : searchPaths) {
        if (std::filesystem::exists(path)) {
            if (pipeline.loadShaderFromFile(path)) {
                LOG_DEBUG("Loaded shader from: {}", path);
                return true;
            }
        }
    }
    
    LOG_ERROR("Shader not found: {}", shaderName);
    return false;
}

bool StereoPipeline::executePipelineStep(ComputePipeline& pipeline, 
                                        uint32_t groupCountX, 
                                        uint32_t groupCountY) {
    // 为这个管线步骤创建命令缓冲区
    VkCommandPool commandPool = m_context.createCommandPool();
    if (!commandPool) {
        LOG_ERROR("Failed to create command pool");
        return false;
    }
    
    VkCommandBuffer commandBuffer = m_context.createCommandBuffer(commandPool);
    if (!commandBuffer) {
        LOG_ERROR("Failed to create command buffer");
        vkDestroyCommandPool(m_context.getDevice(), commandPool, nullptr);
        return false;
    }
    
    // 开始记录命令
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    
    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        LOG_ERROR("Failed to begin command buffer");
        vkDestroyCommandPool(m_context.getDevice(), commandPool, nullptr);
        return false;
    }
    
    // 记录计算命令
    pipeline.recordCommands(commandBuffer, groupCountX, groupCountY, 1);
    
    // 结束记录命令
    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        LOG_ERROR("Failed to end command buffer");
        vkDestroyCommandPool(m_context.getDevice(), commandPool, nullptr);
        return false;
    }
    
    // 提交命令
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    
    if (vkQueueSubmit(m_context.getComputeQueue(), 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
        LOG_ERROR("Failed to submit command buffer");
        vkDestroyCommandPool(m_context.getDevice(), commandPool, nullptr);
        return false;
    }
    
    // 等待计算完成
    vkQueueWaitIdle(m_context.getComputeQueue());
    
    // 清理
    vkFreeCommandBuffers(m_context.getDevice(), commandPool, 1, &commandBuffer);
    vkDestroyCommandPool(m_context.getDevice(), commandPool, nullptr);
    
    return true;
}

void StereoPipeline::cleanup() {
    VkDevice device = m_context.getDevice();
    
    // 销毁描述符集布局
    if (m_censusLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, m_censusLayout, nullptr);
        m_censusLayout = VK_NULL_HANDLE;
    }
    
    if (m_costLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, m_costLayout, nullptr);
        m_costLayout = VK_NULL_HANDLE;
    }
    
    if (m_aggregationLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, m_aggregationLayout, nullptr);
        m_aggregationLayout = VK_NULL_HANDLE;
    }
    
    if (m_wtaLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, m_wtaLayout, nullptr);
        m_wtaLayout = VK_NULL_HANDLE;
    }
    
    if (m_postprocessLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, m_postprocessLayout, nullptr);
        m_postprocessLayout = VK_NULL_HANDLE;
    }
    
    // 缓冲区和管理器会被智能指针自动清理
    
    m_initialized = false;
    LOG_DEBUG("Stereo pipeline cleaned up");
}

} // namespace vulkan
} // namespace stereo_depth
