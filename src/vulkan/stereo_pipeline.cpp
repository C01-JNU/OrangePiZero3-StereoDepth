#include "vulkan/stereo_pipeline.hpp"
#include "utils/logger.hpp"
#include "utils/config.hpp"
#include <sys/stat.h>

namespace stereo_depth {
namespace vulkan {

// 辅助函数：检查文件是否存在
bool fileExists(const std::string& filename) {
    struct stat buffer;
    return (stat(filename.c_str(), &buffer) == 0);
}

StereoPipeline::StereoPipeline(const VulkanContext& context)
    : m_context(context)
    , m_imageWidth(0)
    , m_imageHeight(0)
    , m_maxDisparity(0)
    , m_initialized(false)
    , m_censusLayout(VK_NULL_HANDLE)
    , m_costLayout(VK_NULL_HANDLE)
    , m_aggregationLayout(VK_NULL_HANDLE)
    , m_wtaLayout(VK_NULL_HANDLE)
    , m_postprocessLayout(VK_NULL_HANDLE) {
}

StereoPipeline::~StereoPipeline() {
    if (m_initialized) {
        cleanup();
    }
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
        LOG_WARN("流水线已经初始化");
        return true;
    }
    
    m_imageWidth = imageWidth;
    m_imageHeight = imageHeight;
    m_maxDisparity = maxDisparity;
    
    LOG_INFO("正在初始化立体匹配流水线");
    LOG_INFO("  图像尺寸: {} x {}", m_imageWidth, m_imageHeight);
    LOG_INFO("  最大视差: {}", m_maxDisparity);
    
    try {
        if (!createBuffers()) {
            LOG_ERROR("创建缓冲区失败");
            return false;
        }
        
        if (!createDescriptorSetLayouts()) {
            LOG_ERROR("创建描述符集布局失败");
            return false;
        }
        
        if (!createPipelines()) {
            LOG_ERROR("创建计算管线失败");
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
            LOG_ERROR("复制流水线参数失败");
            return false;
        }
        
        m_initialized = true;
        LOG_INFO("立体匹配流水线初始化成功");
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("流水线初始化异常: {}", e.what());
        cleanup();
        return false;
    }
}

bool StereoPipeline::setLeftImage(const uint8_t* data) {
    if (!m_initialized) {
        LOG_ERROR("流水线未初始化");
        return false;
    }
    
    size_t imageSize = m_imageWidth * m_imageHeight;
    return m_leftImageBuffer->copyToBuffer(data, imageSize);
}

bool StereoPipeline::setRightImage(const uint8_t* data) {
    if (!m_initialized) {
        LOG_ERROR("流水线未初始化");
        return false;
    }
    
    size_t imageSize = m_imageWidth * m_imageHeight;
    return m_rightImageBuffer->copyToBuffer(data, imageSize);
}

bool StereoPipeline::compute() {
    if (!m_initialized) {
        LOG_ERROR("流水线未初始化");
        return false;
    }
    
    LOG_DEBUG("开始立体匹配计算");
    
    try {
        // 步骤1: Census变换
        LOG_DEBUG("步骤1: Census变换");
        if (!executePipelineStep(*m_censusPipeline, 
                                (m_imageWidth + 15) / 16, 
                                (m_imageHeight + 15) / 16)) {
            LOG_ERROR("执行Census变换管线失败");
            return false;
        }
        
        // 步骤2: 代价计算
        LOG_DEBUG("步骤2: 代价计算");
        if (!executePipelineStep(*m_costPipeline,
                                (m_imageWidth + 7) / 8,
                                (m_imageHeight + 7) / 8)) {
            LOG_ERROR("执行代价计算管线失败");
            return false;
        }
        
        // 步骤3: 代价聚合
        LOG_DEBUG("步骤3: 代价聚合");
        if (!executePipelineStep(*m_aggregationPipeline,
                                (m_imageWidth + 7) / 8,
                                (m_imageHeight + 7) / 8)) {
            LOG_ERROR("执行代价聚合管线失败");
            return false;
        }
        
        // 步骤4: WTA（赢家通吃）优化
        LOG_DEBUG("步骤4: WTA优化");
        if (!executePipelineStep(*m_wtaPipeline,
                                (m_imageWidth + 15) / 16,
                                (m_imageHeight + 15) / 16)) {
            LOG_ERROR("执行WTA优化管线失败");
            return false;
        }
        
        // 步骤5: 后处理
        LOG_DEBUG("步骤5: 后处理");
        if (!executePipelineStep(*m_postprocessPipeline,
                                (m_imageWidth + 15) / 16,
                                (m_imageHeight + 15) / 16)) {
            LOG_ERROR("执行后处理管线失败");
            return false;
        }
        
        LOG_INFO("立体匹配计算完成");
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("计算过程中发生异常: {}", e.what());
        return false;
    }
}

bool StereoPipeline::getDisparityMap(uint16_t* output) {
    if (!m_initialized || !m_disparityBuffer) {
        LOG_ERROR("流水线未初始化或视差图缓冲区不可用");
        return false;
    }
    
    size_t disparitySize = m_imageWidth * m_imageHeight * sizeof(uint16_t);
    return m_disparityBuffer->copyFromBuffer(output, disparitySize);
}

bool StereoPipeline::getIntermediateResult(uint32_t bufferIndex, void* output, size_t size) {
    if (!m_initialized) {
        LOG_ERROR("流水线未初始化");
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
            LOG_ERROR("无效的缓冲区索引: {}", bufferIndex);
            return false;
    }
    
    if (!buffer || !buffer->isValid()) {
        LOG_ERROR("缓冲区 {} 不可用", bufferIndex);
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
            LOG_ERROR("创建左图像缓冲区失败");
            return false;
        }
        
        // 右图像缓冲区
        m_rightImageBuffer = std::make_unique<BufferManager>(m_context);
        if (!m_rightImageBuffer->createStorageBuffer(imageSize)) {
            LOG_ERROR("创建右图像缓冲区失败");
            return false;
        }
        
        // 代价体缓冲区
        m_costVolumeBuffer = std::make_unique<BufferManager>(m_context);
        if (!m_costVolumeBuffer->createStorageBuffer(costVolumeSize)) {
            LOG_ERROR("创建代价体缓冲区失败");
            return false;
        }
        
        // 视差图缓冲区
        m_disparityBuffer = std::make_unique<BufferManager>(m_context);
        if (!m_disparityBuffer->createStorageBuffer(disparitySize)) {
            LOG_ERROR("创建视差图缓冲区失败");
            return false;
        }
        
        // 临时缓冲区1（用于中间计算）
        m_tempBuffer1 = std::make_unique<BufferManager>(m_context);
        if (!m_tempBuffer1->createStorageBuffer(imageSize * sizeof(uint32_t))) {
            LOG_ERROR("创建临时缓冲区1失败");
            return false;
        }
        
        // 临时缓冲区2（用于中间计算）
        m_tempBuffer2 = std::make_unique<BufferManager>(m_context);
        if (!m_tempBuffer2->createStorageBuffer(imageSize * sizeof(uint32_t))) {
            LOG_ERROR("创建临时缓冲区2失败");
            return false;
        }
        
        // 参数缓冲区
        m_paramsBuffer = std::make_unique<BufferManager>(m_context);
        if (!m_paramsBuffer->createUniformBuffer(sizeof(PipelineParams))) {
            LOG_ERROR("创建参数缓冲区失败");
            return false;
        }
        
        LOG_DEBUG("为立体匹配流水线创建了 {} 个缓冲区", 7);
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("创建缓冲区时发生异常: {}", e.what());
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
    
    LOG_DEBUG("创建了 {} 个描述符集布局", 5);
    return true;
}

bool StereoPipeline::createPipelines() {
    // 首先检查所有必需的着色器文件
    std::vector<std::pair<std::string, std::string>> requiredShaders = {
        {"Census变换", "census.comp.spv"},
        {"代价计算", "cost.comp.spv"},
        {"代价聚合", "aggregation.comp.spv"},
        {"WTA优化", "wta.comp.spv"},
        {"后处理", "postprocess.comp.spv"}
    };
    
    bool allShadersExist = true;
    
    for (const auto& [shaderName, fileName] : requiredShaders) {
        if (!checkShaderExists(fileName)) {
            LOG_ERROR("❌ 缺少着色器: {} ({})", shaderName, fileName);
            LOG_ERROR("   请先编译着色器: {}", fileName);
            allShadersExist = false;
        } else {
            LOG_INFO("✅ 找到着色器: {} ({})", shaderName, fileName);
        }
    }
    
    if (!allShadersExist) {
        LOG_ERROR("无法创建立体匹配流水线 - 缺少必需的着色器文件");
        LOG_ERROR("请先运行以下命令编译着色器:");
        LOG_ERROR("   1. 在项目根目录执行: cmake --build build --target shaders");
        LOG_ERROR("   2. 或者手动编译着色器到: src/vulkan/spv/ 目录");
        return false;
    }
    
    LOG_INFO("开始创建计算管线...");
    
    // 创建Census变换管线
    m_censusPipeline = std::make_unique<ComputePipeline>(m_context);
    m_censusPipeline->setDescriptorSetLayout(m_censusLayout);
    
    if (!loadShader(*m_censusPipeline, "census.comp.spv")) {
        LOG_ERROR("❌ 无法加载Census变换着色器");
        return false;
    }
    
    if (!m_censusPipeline->createPipeline()) {
        LOG_ERROR("❌ 无法创建Census变换管线");
        return false;
    }
    
    LOG_INFO("✅ Census变换管线创建成功");
    
    // 创建代价计算管线
    m_costPipeline = std::make_unique<ComputePipeline>(m_context);
    m_costPipeline->setDescriptorSetLayout(m_costLayout);
    
    if (!loadShader(*m_costPipeline, "cost.comp.spv")) {
        LOG_ERROR("❌ 无法加载代价计算着色器");
        return false;
    }
    
    if (!m_costPipeline->createPipeline()) {
        LOG_ERROR("❌ 无法创建代价计算管线");
        return false;
    }
    
    LOG_INFO("✅ 代价计算管线创建成功");
    
    // 创建代价聚合管线
    m_aggregationPipeline = std::make_unique<ComputePipeline>(m_context);
    m_aggregationPipeline->setDescriptorSetLayout(m_aggregationLayout);
    
    if (!loadShader(*m_aggregationPipeline, "aggregation.comp.spv")) {
        LOG_ERROR("❌ 无法加载代价聚合着色器");
        return false;
    }
    
    if (!m_aggregationPipeline->createPipeline()) {
        LOG_ERROR("❌ 无法创建代价聚合管线");
        return false;
    }
    
    LOG_INFO("✅ 代价聚合管线创建成功");
    
    // 创建WTA管线
    m_wtaPipeline = std::make_unique<ComputePipeline>(m_context);
    m_wtaPipeline->setDescriptorSetLayout(m_wtaLayout);
    
    if (!loadShader(*m_wtaPipeline, "wta.comp.spv")) {
        LOG_ERROR("❌ 无法加载WTA优化着色器");
        return false;
    }
    
    if (!m_wtaPipeline->createPipeline()) {
        LOG_ERROR("❌ 无法创建WTA优化管线");
        return false;
    }
    
    LOG_INFO("✅ WTA优化管线创建成功");
    
    // 创建后处理管线
    m_postprocessPipeline = std::make_unique<ComputePipeline>(m_context);
    m_postprocessPipeline->setDescriptorSetLayout(m_postprocessLayout);
    
    if (!loadShader(*m_postprocessPipeline, "postprocess.comp.spv")) {
        LOG_ERROR("❌ 无法加载后处理着色器");
        return false;
    }
    
    if (!m_postprocessPipeline->createPipeline()) {
        LOG_ERROR("❌ 无法创建后处理管线");
        return false;
    }
    
    LOG_INFO("✅ 后处理管线创建成功");
    
    LOG_INFO("所有计算管线创建完成 (共5个)");
    return true;
}

bool StereoPipeline::checkShaderExists(const std::string& shaderName) {
    // 尝试从多个位置查找着色器文件
    std::vector<std::string> searchPaths = {
        // 从程序所在目录开始搜索
        "shaders/" + shaderName,                     // build/shaders/
        "../shaders/" + shaderName,                  // project/shaders/
        "../../shaders/" + shaderName,               // project/../shaders/
        
        // SPIR-V编译输出目录（关键！）
        "src/vulkan/spv/" + shaderName,              // src/vulkan/spv/
        "../src/vulkan/spv/" + shaderName,           // build/src/vulkan/spv/
        "../../src/vulkan/spv/" + shaderName,        // project/src/vulkan/spv/
        
        // 构建系统可能放置的位置
        "build/shaders/" + shaderName,               // build/shaders/
        "../build/shaders/" + shaderName,            // project/build/shaders/
        "../../build/shaders/" + shaderName,         // project/../build/shaders/
        
        // 可能的其他位置
        "third_party/shaders/" + shaderName,         // third_party/shaders/
        "../third_party/shaders/" + shaderName,      // project/third_party/shaders/
        
        // 直接在当前目录
        shaderName
    };
    
    LOG_DEBUG("正在查找着色器: {}", shaderName);
    
    for (const auto& path : searchPaths) {
        if (fileExists(path)) {
            LOG_DEBUG("找到着色器: {} -> {}", shaderName, path);
            return true;
        }
    }
    
    return false;
}

bool StereoPipeline::loadShader(ComputePipeline& pipeline, const std::string& shaderName) {
    // 尝试从多个位置查找着色器文件
    std::vector<std::string> searchPaths = {
        // 从程序所在目录开始搜索
        "shaders/" + shaderName,                     // build/shaders/
        "../shaders/" + shaderName,                  // project/shaders/
        "../../shaders/" + shaderName,               // project/../shaders/
        
        // SPIR-V编译输出目录（关键！）
        "src/vulkan/spv/" + shaderName,              // src/vulkan/spv/
        "../src/vulkan/spv/" + shaderName,           // build/src/vulkan/spv/
        "../../src/vulkan/spv/" + shaderName,        // project/src/vulkan/spv/
        
        // 构建系统可能放置的位置
        "build/shaders/" + shaderName,               // build/shaders/
        "../build/shaders/" + shaderName,            // project/build/shaders/
        "../../build/shaders/" + shaderName,         // project/../build/shaders/
        
        // 可能的其他位置
        "third_party/shaders/" + shaderName,         // third_party/shaders/
        "../third_party/shaders/" + shaderName,      // project/third_party/shaders/
        
        // 直接在当前目录
        shaderName
    };
    
    LOG_DEBUG("正在加载着色器: {}", shaderName);
    
    for (const auto& path : searchPaths) {
        if (fileExists(path)) {
            LOG_DEBUG("尝试加载着色器: {}", path);
            if (pipeline.loadShaderFromFile(path)) {
                LOG_INFO("成功加载着色器: {} (来自: {})", shaderName, path);
                return true;
            }
        }
    }
    
    LOG_ERROR("无法加载着色器: {}", shaderName);
    LOG_ERROR("已搜索以下路径:");
    for (const auto& path : searchPaths) {
        LOG_ERROR("  • {}", path);
    }
    
    return false;
}

bool StereoPipeline::executePipelineStep(ComputePipeline& pipeline, 
                                        uint32_t groupCountX, 
                                        uint32_t groupCountY) {
    // 为这个管线步骤创建命令缓冲区
    VkCommandPool commandPool = m_context.createCommandPool();
    if (!commandPool) {
        LOG_ERROR("创建命令池失败");
        return false;
    }
    
    VkCommandBuffer commandBuffer = m_context.createCommandBuffer(commandPool);
    if (!commandBuffer) {
        LOG_ERROR("创建命令缓冲区失败");
        vkDestroyCommandPool(m_context.getDevice(), commandPool, nullptr);
        return false;
    }
    
    // 开始记录命令
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    
    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        LOG_ERROR("开始命令缓冲区失败");
        vkDestroyCommandPool(m_context.getDevice(), commandPool, nullptr);
        return false;
    }
    
    // 记录计算命令
    pipeline.recordCommands(commandBuffer, groupCountX, groupCountY, 1);
    
    // 结束记录命令
    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        LOG_ERROR("结束命令缓冲区失败");
        vkDestroyCommandPool(m_context.getDevice(), commandPool, nullptr);
        return false;
    }
    
    // 提交命令
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    
    if (vkQueueSubmit(m_context.getComputeQueue(), 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
        LOG_ERROR("提交命令缓冲区失败");
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
    // 只有在设备可用时才进行清理
    if (!m_context.getDevice()) {
        LOG_DEBUG("Vulkan设备不可用，跳过清理");
        return;
    }
    
    VkDevice device = m_context.getDevice();
    
    // 安全地销毁描述符集布局
    auto safeDestroyLayout = [device](VkDescriptorSetLayout& layout) {
        if (layout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(device, layout, nullptr);
            layout = VK_NULL_HANDLE;
        }
    };
    
    safeDestroyLayout(m_censusLayout);
    safeDestroyLayout(m_costLayout);
    safeDestroyLayout(m_aggregationLayout);
    safeDestroyLayout(m_wtaLayout);
    safeDestroyLayout(m_postprocessLayout);
    
    // 清除所有缓冲区（智能指针会自动清理）
    m_leftImageBuffer.reset();
    m_rightImageBuffer.reset();
    m_costVolumeBuffer.reset();
    m_disparityBuffer.reset();
    m_tempBuffer1.reset();
    m_tempBuffer2.reset();
    m_paramsBuffer.reset();
    
    // 清除所有管线（智能指针会自动清理）
    m_censusPipeline.reset();
    m_costPipeline.reset();
    m_aggregationPipeline.reset();
    m_wtaPipeline.reset();
    m_postprocessPipeline.reset();
    
    m_initialized = false;
    LOG_DEBUG("立体匹配流水线清理完成");
}

} // namespace vulkan
} // namespace stereo_depth
