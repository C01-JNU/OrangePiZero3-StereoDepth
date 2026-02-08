#include "vulkan/stereo_pipeline.hpp"
#include "utils/logger.hpp"
#include "utils/config.hpp"
#include <sys/stat.h>
#include <algorithm>
#include <cassert>

namespace stereo_depth {
namespace vulkan {

// 辅助函数：检查文件是否存在
bool fileExists(const std::string& filename) {
    struct stat buffer;
    return (stat(filename.c_str(), &buffer) == 0);
}

StereoPipeline::StereoPipeline(const VulkanContext& context)
    : m_context(context)
    , m_originalImageWidth(0)
    , m_originalImageHeight(0)
    , m_compressedImageWidth(0)
    , m_compressedImageHeight(0)
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
    , m_originalImageWidth(other.m_originalImageWidth)
    , m_originalImageHeight(other.m_originalImageHeight)
    , m_compressedImageWidth(other.m_compressedImageWidth)
    , m_compressedImageHeight(other.m_compressedImageHeight)
    , m_maxDisparity(other.m_maxDisparity)
    , m_initialized(other.m_initialized)
    , m_leftImageBuffer(std::move(other.m_leftImageBuffer))
    , m_rightImageBuffer(std::move(other.m_rightImageBuffer))
    , m_stitchedImageBuffer(std::move(other.m_stitchedImageBuffer))
    , m_leftCensusBuffer(std::move(other.m_leftCensusBuffer))
    , m_rightCensusBuffer(std::move(other.m_rightCensusBuffer))
    , m_censusDebugBuffer(std::move(other.m_censusDebugBuffer))
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
        
        m_originalImageWidth = other.m_originalImageWidth;
        m_originalImageHeight = other.m_originalImageHeight;
        m_compressedImageWidth = other.m_compressedImageWidth;
        m_compressedImageHeight = other.m_compressedImageHeight;
        m_maxDisparity = other.m_maxDisparity;
        m_initialized = other.m_initialized;
        
        m_leftImageBuffer = std::move(other.m_leftImageBuffer);
        m_rightImageBuffer = std::move(other.m_rightImageBuffer);
        m_stitchedImageBuffer = std::move(other.m_stitchedImageBuffer);
        m_leftCensusBuffer = std::move(other.m_leftCensusBuffer);
        m_rightCensusBuffer = std::move(other.m_rightCensusBuffer);
        m_censusDebugBuffer = std::move(other.m_censusDebugBuffer);
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
    
    // 存储原始图像尺寸
    m_originalImageWidth = imageWidth;
    m_originalImageHeight = imageHeight;
    
    // 计算压缩后尺寸（宽度减半，高度不变）
    m_compressedImageWidth = imageWidth / 2;
    m_compressedImageHeight = imageHeight;
    m_maxDisparity = maxDisparity;
    
    LOG_INFO("正在初始化立体匹配流水线");
    LOG_INFO("  原始图像尺寸: {} x {}", m_originalImageWidth, m_originalImageHeight);
    LOG_INFO("  压缩后尺寸: {} x {}", m_compressedImageWidth, m_compressedImageHeight);
    LOG_INFO("  最大视差: {}", m_maxDisparity);
    
    // 验证压缩比例
    if (m_originalImageWidth % 2 != 0) {
        LOG_ERROR("原始图像宽度必须是偶数，以便压缩为一半");
        return false;
    }
    
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
        
        // 设置计算参数（使用压缩后尺寸）
        PipelineParams params = {};
        params.compressedImageWidth = m_compressedImageWidth;   // 对应着色器imageWidth
        params.compressedImageHeight = m_compressedImageHeight; // 对应着色器imageHeight
        params.maxDisparity = m_maxDisparity;
        params.windowSize = 9; // 默认窗口大小
        params.uniquenessRatio = 0.15f; // 从配置文件读取
        params.penaltyP1 = 8.0f;
        params.penaltyP2 = 32.0f;
        params.flags = 0;
        params.speckleWindow = 100;
        params.speckleRange = 32;
        params.medianSize = 3;
        params.padding[0] = 0; // 着色器使用的padding
        params.padding[1] = 0;
        params.padding[2] = 0;
        params.reserved1 = 0;  // 额外字段
        params.reserved2 = 0;
        
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
    
    size_t imageBytes = static_cast<size_t>(m_originalImageWidth) * 
                       static_cast<size_t>(m_originalImageHeight) * sizeof(uint8_t);
    LOG_DEBUG("设置左图像数据: {} 字节", imageBytes);
    return m_leftImageBuffer->copyToBuffer(data, imageBytes);
}

bool StereoPipeline::setRightImage(const uint8_t* data) {
    if (!m_initialized) {
        LOG_ERROR("流水线未初始化");
        return false;
    }
    
    size_t imageBytes = static_cast<size_t>(m_originalImageWidth) * 
                       static_cast<size_t>(m_originalImageHeight) * sizeof(uint8_t);
    LOG_DEBUG("设置右图像数据: {} 字节", imageBytes);
    return m_rightImageBuffer->copyToBuffer(data, imageBytes);
}

bool StereoPipeline::compute() {
    if (!m_initialized) {
        LOG_ERROR("流水线未初始化");
        return false;
    }
    
    LOG_INFO("开始立体匹配计算");
    
    try {
        // 步骤0: 压缩并拼接左右图像
        LOG_INFO("步骤0: 压缩并拼接左右图像");
        if (!compressAndStitchImages()) {
            LOG_ERROR("图像压缩拼接失败");
            return false;
        }
        LOG_INFO("✅ 图像压缩拼接完成");
        
        // 步骤1: Census变换（双输出版本）
        LOG_INFO("步骤1: Census变换（双输出）");
        if (!executePipelineStep(*m_censusPipeline, 
                                (m_compressedImageWidth + 15) / 16, 
                                (m_compressedImageHeight + 15) / 16)) {
            LOG_ERROR("执行Census变换管线失败");
            return false;
        }
        LOG_INFO("✅ Census变换完成");
        
        // 步骤2: 代价计算
        LOG_INFO("步骤2: 代价计算");
        if (!executePipelineStep(*m_costPipeline,
                                (m_compressedImageWidth + 7) / 8,
                                (m_compressedImageHeight + 7) / 8)) {
            LOG_ERROR("执行代价计算管线失败");
            return false;
        }
        LOG_INFO("✅ 代价计算完成");
        
        // 步骤3: 代价聚合
        LOG_INFO("步骤3: 代价聚合");
        if (!executePipelineStep(*m_aggregationPipeline,
                                (m_compressedImageWidth + 7) / 8,
                                (m_compressedImageHeight + 7) / 8)) {
            LOG_ERROR("执行代价聚合管线失败");
            return false;
        }
        LOG_INFO("✅ 代价聚合完成");
        
        // 步骤4: WTA（赢家通吃）优化
        LOG_INFO("步骤4: WTA优化");
        if (!executePipelineStep(*m_wtaPipeline,
                                (m_compressedImageWidth + 15) / 16,
                                (m_compressedImageHeight + 15) / 16)) {
            LOG_ERROR("执行WTA优化管线失败");
            return false;
        }
        LOG_INFO("✅ WTA优化完成");
        
        // 步骤5: 后处理
        LOG_INFO("步骤5: 后处理");
        if (!executePipelineStep(*m_postprocessPipeline,
                                (m_compressedImageWidth + 15) / 16,
                                (m_compressedImageHeight + 15) / 16)) {
            LOG_ERROR("执行后处理管线失败");
            return false;
        }
        LOG_INFO("✅ 后处理完成");
        
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
    
    // 视差图是压缩后尺寸
    size_t disparitySize = static_cast<size_t>(m_compressedImageWidth) * 
                          static_cast<size_t>(m_compressedImageHeight) * sizeof(uint16_t);
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
        case 2: buffer = m_stitchedImageBuffer.get(); break;
        case 3: buffer = m_leftCensusBuffer.get(); break;
        case 4: buffer = m_rightCensusBuffer.get(); break;
        case 5: buffer = m_censusDebugBuffer.get(); break;
        case 6: buffer = m_costVolumeBuffer.get(); break;
        case 7: buffer = m_disparityBuffer.get(); break;
        case 8: buffer = m_tempBuffer1.get(); break;
        case 9: buffer = m_tempBuffer2.get(); break;
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
    // 计算原始图像的像素数和字节数
    size_t originalPixelCount = static_cast<size_t>(m_originalImageWidth) * 
                               static_cast<size_t>(m_originalImageHeight);
    size_t originalImageBytes = originalPixelCount * sizeof(uint8_t);
    
    // 计算压缩后图像的像素数和字节数
    size_t compressedPixelCount = static_cast<size_t>(m_compressedImageWidth) * 
                                 static_cast<size_t>(m_compressedImageHeight);
    
    size_t stitchedImageBytes = compressedPixelCount * 2 * sizeof(uint8_t); // 拼接图像：2×压缩图像
    size_t censusBufferBytes = compressedPixelCount * 2 * sizeof(uint32_t); // Census描述符：每个像素2个uint32_t
    size_t debugBufferBytes = 8 * sizeof(uint32_t);        // 调试缓冲区：8个uint32_t
    size_t costVolumeBytes = compressedPixelCount * static_cast<size_t>(m_maxDisparity) * sizeof(uint16_t);
    size_t disparityBytes = compressedPixelCount * sizeof(uint16_t); // 视差图：16位/像素
    size_t tempBufferBytes = compressedPixelCount * sizeof(uint32_t); // 临时缓冲区：32位/像素
    
    LOG_INFO("缓冲区大小计算:");
    LOG_INFO("  原始图像: {}x{} = {} 像素", m_originalImageWidth, m_originalImageHeight, originalPixelCount);
    LOG_INFO("  压缩后图像: {}x{} = {} 像素", m_compressedImageWidth, m_compressedImageHeight, compressedPixelCount);
    LOG_INFO("  原始图像缓冲区: {} 字节 ({} KB)", originalImageBytes, originalImageBytes/1024);
    LOG_INFO("  拼接图像缓冲区: {} 字节 ({} KB)", stitchedImageBytes, stitchedImageBytes/1024);
    LOG_INFO("  Census缓冲区: {} 字节 ({} KB)", censusBufferBytes, censusBufferBytes/1024);
    LOG_INFO("  代价体缓冲区: {} 字节 ({} MB)", costVolumeBytes, costVolumeBytes/(1024*1024));
    LOG_INFO("  视差图缓冲区: {} 字节 ({} KB)", disparityBytes, disparityBytes/1024);
    LOG_INFO("  临时缓冲区: {} 字节 ({} KB)", tempBufferBytes, tempBufferBytes/1024);
    
    try {
        // 左图像缓冲区（原始尺寸）
        m_leftImageBuffer = std::make_unique<BufferManager>(m_context);
        if (!m_leftImageBuffer->createStorageBuffer(originalImageBytes)) {
            LOG_ERROR("创建左图像缓冲区失败 ({} 字节)", originalImageBytes);
            return false;
        }
        LOG_DEBUG("左图像缓冲区创建成功: {} 字节", m_leftImageBuffer->getSize());
        
        // 右图像缓冲区（原始尺寸）
        m_rightImageBuffer = std::make_unique<BufferManager>(m_context);
        if (!m_rightImageBuffer->createStorageBuffer(originalImageBytes)) {
            LOG_ERROR("创建右图像缓冲区失败 ({} 字节)", originalImageBytes);
            return false;
        }
        LOG_DEBUG("右图像缓冲区创建成功: {} 字节", m_rightImageBuffer->getSize());
        
        // 拼接图像缓冲区（压缩后左右拼接）
        m_stitchedImageBuffer = std::make_unique<BufferManager>(m_context);
        if (!m_stitchedImageBuffer->createStorageBuffer(stitchedImageBytes)) {
            LOG_ERROR("创建拼接图像缓冲区失败 ({} 字节)", stitchedImageBytes);
            return false;
        }
        LOG_DEBUG("拼接图像缓冲区创建成功: {} 字节", m_stitchedImageBuffer->getSize());
        
        // 左眼Census缓冲区（压缩后尺寸）
        m_leftCensusBuffer = std::make_unique<BufferManager>(m_context);
        if (!m_leftCensusBuffer->createStorageBuffer(censusBufferBytes)) {
            LOG_ERROR("创建左眼Census缓冲区失败 ({} 字节)", censusBufferBytes);
            return false;
        }
        LOG_DEBUG("左眼Census缓冲区创建成功: {} 字节", m_leftCensusBuffer->getSize());
        
        // 右眼Census缓冲区（压缩后尺寸）
        m_rightCensusBuffer = std::make_unique<BufferManager>(m_context);
        if (!m_rightCensusBuffer->createStorageBuffer(censusBufferBytes)) {
            LOG_ERROR("创建右眼Census缓冲区失败 ({} 字节)", censusBufferBytes);
            return false;
        }
        LOG_DEBUG("右眼Census缓冲区创建成功: {} 字节", m_rightCensusBuffer->getSize());
        
        // Census调试缓冲区
        m_censusDebugBuffer = std::make_unique<BufferManager>(m_context);
        if (!m_censusDebugBuffer->createStorageBuffer(debugBufferBytes)) {
            LOG_ERROR("创建Census调试缓冲区失败 ({} 字节)", debugBufferBytes);
            return false;
        }
        LOG_DEBUG("Census调试缓冲区创建成功: {} 字节", m_censusDebugBuffer->getSize());
        
        // 代价体缓冲区（压缩后尺寸）
        m_costVolumeBuffer = std::make_unique<BufferManager>(m_context);
        if (!m_costVolumeBuffer->createStorageBuffer(costVolumeBytes)) {
            LOG_ERROR("创建代价体缓冲区失败 ({} 字节)", costVolumeBytes);
            return false;
        }
        LOG_DEBUG("代价体缓冲区创建成功: {} 字节", m_costVolumeBuffer->getSize());
        
        // 视差图缓冲区（压缩后尺寸）
        m_disparityBuffer = std::make_unique<BufferManager>(m_context);
        if (!m_disparityBuffer->createStorageBuffer(disparityBytes)) {
            LOG_ERROR("创建视差图缓冲区失败 ({} 字节)", disparityBytes);
            return false;
        }
        LOG_DEBUG("视差图缓冲区创建成功: {} 字节", m_disparityBuffer->getSize());
        
        // 临时缓冲区1（压缩后尺寸）
        m_tempBuffer1 = std::make_unique<BufferManager>(m_context);
        if (!m_tempBuffer1->createStorageBuffer(tempBufferBytes)) {
            LOG_ERROR("创建临时缓冲区1失败 ({} 字节)", tempBufferBytes);
            return false;
        }
        LOG_DEBUG("临时缓冲区1创建成功: {} 字节", m_tempBuffer1->getSize());
        
        // 临时缓冲区2（压缩后尺寸）
        m_tempBuffer2 = std::make_unique<BufferManager>(m_context);
        if (!m_tempBuffer2->createStorageBuffer(tempBufferBytes)) {
            LOG_ERROR("创建临时缓冲区2失败 ({} 字节)", tempBufferBytes);
            return false;
        }
        LOG_DEBUG("临时缓冲区2创建成功: {} 字节", m_tempBuffer2->getSize());
        
        // 参数缓冲区
        m_paramsBuffer = std::make_unique<BufferManager>(m_context);
        size_t paramsSize = sizeof(PipelineParams);
        if (!m_paramsBuffer->createUniformBuffer(paramsSize)) {
            LOG_ERROR("创建参数缓冲区失败 ({} 字节)", paramsSize);
            return false;
        }
        LOG_DEBUG("参数缓冲区创建成功: {} 字节", m_paramsBuffer->getSize());
        
        LOG_INFO("✅ 立体匹配流水线缓冲区创建完成 (共11个缓冲区)");
        LOG_INFO("  总内存使用: {:.2f} MB", 
                 (originalImageBytes*2 + stitchedImageBytes + censusBufferBytes*2 + debugBufferBytes + 
                  costVolumeBytes + disparityBytes + tempBufferBytes*2 + 
                  paramsSize) / (1024.0 * 1024.0));
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("创建缓冲区时发生异常: {}", e.what());
        return false;
    }
}

bool StereoPipeline::createDescriptorSetLayouts() {
    // Census变换布局（双输出版本）
    // binding 0: Uniform参数
    // binding 1: 拼接图像（压缩后左右拼接）
    // binding 2: 左眼Census输出（压缩后尺寸）
    // binding 3: 右眼Census输出（压缩后尺寸）
    // binding 4: 调试缓冲区
    DescriptorSetLayoutBuilder censusBuilder(m_context);
    censusBuilder.addUniformBuffer(0, 1, VK_SHADER_STAGE_COMPUTE_BIT)
                .addStorageBuffer(1, 1, VK_SHADER_STAGE_COMPUTE_BIT)
                .addStorageBuffer(2, 1, VK_SHADER_STAGE_COMPUTE_BIT)
                .addStorageBuffer(3, 1, VK_SHADER_STAGE_COMPUTE_BIT)
                .addStorageBuffer(4, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    
    m_censusLayout = censusBuilder.build();
    if (m_censusLayout == VK_NULL_HANDLE) {
        LOG_ERROR("创建Census变换描述符集布局失败");
        return false;
    }
    
    LOG_DEBUG("Census变换描述符集布局创建成功 (5个绑定)");
    
    // 代价计算布局：左Census、右Census、代价体、参数
    DescriptorSetLayoutBuilder costBuilder(m_context);
    costBuilder.addStorageBuffer(0, 1, VK_SHADER_STAGE_COMPUTE_BIT)  // 左Census特征
               .addStorageBuffer(1, 1, VK_SHADER_STAGE_COMPUTE_BIT)  // 右Census特征
               .addStorageBuffer(2, 1, VK_SHADER_STAGE_COMPUTE_BIT)  // 代价体
               .addUniformBuffer(3, 1, VK_SHADER_STAGE_COMPUTE_BIT); // 参数
    
    m_costLayout = costBuilder.build();
    if (m_costLayout == VK_NULL_HANDLE) {
        LOG_ERROR("创建代价计算描述符集布局失败");
        return false;
    }
    
    LOG_DEBUG("代价计算描述符集布局创建成功 (4个绑定)");
    
    // 代价聚合布局：输入代价体、输出聚合代价、临时缓冲区、参数
    DescriptorSetLayoutBuilder aggregationBuilder(m_context);
    aggregationBuilder.addStorageBuffer(0, 1, VK_SHADER_STAGE_COMPUTE_BIT)  // 输入代价体
                      .addStorageBuffer(1, 1, VK_SHADER_STAGE_COMPUTE_BIT)  // 输出聚合代价
                      .addStorageBuffer(2, 1, VK_SHADER_STAGE_COMPUTE_BIT)  // 临时缓冲区1
                      .addUniformBuffer(3, 1, VK_SHADER_STAGE_COMPUTE_BIT); // 参数
    
    m_aggregationLayout = aggregationBuilder.build();
    if (m_aggregationLayout == VK_NULL_HANDLE) {
        LOG_ERROR("创建代价聚合描述符集布局失败");
        return false;
    }
    
    LOG_DEBUG("代价聚合描述符集布局创建成功 (4个绑定)");
    
    // WTA布局：聚合代价、视差图、参数
    DescriptorSetLayoutBuilder wtaBuilder(m_context);
    wtaBuilder.addStorageBuffer(0, 1, VK_SHADER_STAGE_COMPUTE_BIT)  // 聚合代价
               .addStorageBuffer(1, 1, VK_SHADER_STAGE_COMPUTE_BIT)  // 视差图
               .addUniformBuffer(2, 1, VK_SHADER_STAGE_COMPUTE_BIT); // 参数
    
    m_wtaLayout = wtaBuilder.build();
    if (m_wtaLayout == VK_NULL_HANDLE) {
        LOG_ERROR("创建WTA优化描述符集布局失败");
        return false;
    }
    
    LOG_DEBUG("WTA优化描述符集布局创建成功 (3个绑定)");
    
    // 后处理布局：输入视差图、输出视差图、临时缓冲区、参数
    DescriptorSetLayoutBuilder postprocessBuilder(m_context);
    postprocessBuilder.addStorageBuffer(0, 1, VK_SHADER_STAGE_COMPUTE_BIT)  // 输入视差图
                       .addStorageBuffer(1, 1, VK_SHADER_STAGE_COMPUTE_BIT)  // 输出视差图
                       .addStorageBuffer(2, 1, VK_SHADER_STAGE_COMPUTE_BIT)  // 临时缓冲区2
                       .addUniformBuffer(3, 1, VK_SHADER_STAGE_COMPUTE_BIT); // 参数
    
    m_postprocessLayout = postprocessBuilder.build();
    if (m_postprocessLayout == VK_NULL_HANDLE) {
        LOG_ERROR("创建后处理描述符集布局失败");
        return false;
    }
    
    LOG_DEBUG("后处理描述符集布局创建成功 (4个绑定)");
    LOG_INFO("✅ 所有描述符集布局创建完成 (共5个布局)");
    return true;
}

bool StereoPipeline::createPipelines() {
    // 检查所有必需的着色器文件
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
    
    // 创建Census变换管线（双输出版本）
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
    
    // 为Census管线创建描述符集
    std::vector<VkBuffer> censusBuffers = {
        m_paramsBuffer->getBuffer(),        // binding 0: 参数
        m_stitchedImageBuffer->getBuffer(), // binding 1: 拼接图像
        m_leftCensusBuffer->getBuffer(),    // binding 2: 左眼Census输出
        m_rightCensusBuffer->getBuffer(),   // binding 3: 右眼Census输出
        m_censusDebugBuffer->getBuffer()    // binding 4: 调试缓冲区
    };
    
    std::vector<VkDescriptorType> censusBufferTypes = {
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,  // binding 0
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,  // binding 1
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,  // binding 2
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,  // binding 3
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER   // binding 4
    };
    
    LOG_INFO("为Census管线创建描述符集...");
    if (!m_censusPipeline->createDescriptorSet(censusBuffers, censusBufferTypes)) {
        LOG_ERROR("❌ 无法为Census变换创建描述符集");
        return false;
    }
    
    LOG_INFO("✅ Census变换管线创建成功 (双输出版本)");
    
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
    
    // 为代价计算管线创建描述符集
    std::vector<VkBuffer> costBuffers = {
        m_leftCensusBuffer->getBuffer(),    // binding 0: 左Census特征
        m_rightCensusBuffer->getBuffer(),   // binding 1: 右Census特征
        m_costVolumeBuffer->getBuffer(),    // binding 2: 代价体
        m_paramsBuffer->getBuffer()         // binding 3: 参数
    };
    
    std::vector<VkDescriptorType> costBufferTypes = {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,  // binding 0
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,  // binding 1
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,  // binding 2
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER   // binding 3
    };
    
    LOG_INFO("为代价计算管线创建描述符集...");
    if (!m_costPipeline->createDescriptorSet(costBuffers, costBufferTypes)) {
        LOG_ERROR("❌ 无法为代价计算创建描述符集");
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
    
    // 为代价聚合管线创建描述符集
    std::vector<VkBuffer> aggregationBuffers = {
        m_costVolumeBuffer->getBuffer(),    // binding 0: 输入代价体
        m_tempBuffer1->getBuffer(),         // binding 1: 输出聚合代价
        m_tempBuffer2->getBuffer(),         // binding 2: 临时缓冲区
        m_paramsBuffer->getBuffer()         // binding 3: 参数
    };
    
    std::vector<VkDescriptorType> aggregationBufferTypes = {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,  // binding 0
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,  // binding 1
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,  // binding 2
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER   // binding 3
    };
    
    LOG_INFO("为代价聚合管线创建描述符集...");
    if (!m_aggregationPipeline->createDescriptorSet(aggregationBuffers, aggregationBufferTypes)) {
        LOG_ERROR("❌ 无法为代价聚合创建描述符集");
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
    
    // 为WTA管线创建描述符集
    std::vector<VkBuffer> wtaBuffers = {
        m_tempBuffer1->getBuffer(),         // binding 0: 聚合代价
        m_disparityBuffer->getBuffer(),     // binding 1: 视差图
        m_paramsBuffer->getBuffer()         // binding 2: 参数
    };
    
    std::vector<VkDescriptorType> wtaBufferTypes = {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,  // binding 0
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,  // binding 1
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER   // binding 2
    };
    
    LOG_INFO("为WTA管线创建描述符集...");
    if (!m_wtaPipeline->createDescriptorSet(wtaBuffers, wtaBufferTypes)) {
        LOG_ERROR("❌ 无法为WTA优化创建描述符集");
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
    
    // 为后处理管线创建描述符集
    std::vector<VkBuffer> postprocessBuffers = {
        m_disparityBuffer->getBuffer(),     // binding 0: 输入视差图
        m_disparityBuffer->getBuffer(),     // binding 1: 输出视差图（原地处理）
        m_tempBuffer2->getBuffer(),         // binding 2: 临时缓冲区2
        m_paramsBuffer->getBuffer()         // binding 3: 参数
    };
    
    std::vector<VkDescriptorType> postprocessBufferTypes = {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,  // binding 0
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,  // binding 1
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,  // binding 2
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER   // binding 3
    };
    
    LOG_INFO("为后处理管线创建描述符集...");
    if (!m_postprocessPipeline->createDescriptorSet(postprocessBuffers, postprocessBufferTypes)) {
        LOG_ERROR("❌ 无法为后处理创建描述符集");
        return false;
    }
    
    LOG_INFO("✅ 后处理管线创建成功");
    
    LOG_INFO("所有计算管线创建完成 (共5个管线，均已创建描述符集)");
    return true;
}

bool StereoPipeline::checkShaderExists(const std::string& shaderName) {
    std::vector<std::string> searchPaths = {
        "shaders/" + shaderName,
        "src/vulkan/spv/" + shaderName,
        "../src/vulkan/spv/" + shaderName,
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
    std::vector<std::string> searchPaths = {
        "shaders/" + shaderName,
        "src/vulkan/spv/" + shaderName,
        "../src/vulkan/spv/" + shaderName,
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
    // 检查管线是否有效
    if (!pipeline.isValid()) {
        LOG_ERROR("管线无效，无法执行");
        return false;
    }
    
    LOG_DEBUG("执行管线步骤: 工作组=({}, {}, 1)", groupCountX, groupCountY);
    
    // 为这个管线步骤创建命令缓冲区
    VkCommandPool commandPool = m_context.createCommandPool();
    if (commandPool == VK_NULL_HANDLE) {
        LOG_ERROR("创建命令池失败");
        return false;
    }
    
    VkCommandBuffer commandBuffer = m_context.createCommandBuffer(commandPool);
    if (commandBuffer == VK_NULL_HANDLE) {
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

std::vector<uint8_t> StereoPipeline::compressImage(const uint8_t* src, 
                                                  uint32_t srcWidth, uint32_t srcHeight,
                                                  uint32_t dstWidth, uint32_t dstHeight) {
    // 简单的平均压缩算法（宽度减半）
    // 每个目标像素 = (src[x*2] + src[x*2+1]) / 2
    
    std::vector<uint8_t> dst(dstWidth * dstHeight);
    
    for (uint32_t y = 0; y < dstHeight; ++y) {
        for (uint32_t x = 0; x < dstWidth; ++x) {
            // 原始图像中的两个像素位置
            uint32_t srcX1 = x * 2;
            uint32_t srcX2 = x * 2 + 1;
            
            // 计算平均值
            uint32_t sum = src[y * srcWidth + srcX1] + src[y * srcWidth + srcX2];
            dst[y * dstWidth + x] = static_cast<uint8_t>(sum / 2);
        }
    }
    
    return dst;
}

bool StereoPipeline::compressAndStitchImages() {
    // 从左右图像缓冲区获取原始数据
    size_t originalImageSize = static_cast<size_t>(m_originalImageWidth) * 
                              static_cast<size_t>(m_originalImageHeight);
    
    LOG_DEBUG("压缩拼接参数:");
    LOG_DEBUG("  原始尺寸: {}x{} = {} 像素", 
              m_originalImageWidth, m_originalImageHeight, originalImageSize);
    LOG_DEBUG("  压缩后尺寸: {}x{} = {} 像素",
              m_compressedImageWidth, m_compressedImageHeight,
              m_compressedImageWidth * m_compressedImageHeight);
    
    std::vector<uint8_t> leftImage(originalImageSize);
    std::vector<uint8_t> rightImage(originalImageSize);
    
    LOG_DEBUG("从缓冲区读取左图像数据...");
    if (!m_leftImageBuffer->copyFromBuffer(leftImage.data(), originalImageSize)) {
        LOG_ERROR("获取左图像数据失败");
        return false;
    }
    
    LOG_DEBUG("从缓冲区读取右图像数据...");
    if (!m_rightImageBuffer->copyFromBuffer(rightImage.data(), originalImageSize)) {
        LOG_ERROR("获取右图像数据失败");
        return false;
    }
    
    // 压缩左右图像
    LOG_DEBUG("压缩左图像...");
    auto compressedLeft = compressImage(leftImage.data(), 
                                       m_originalImageWidth, m_originalImageHeight,
                                       m_compressedImageWidth, m_compressedImageHeight);
    
    LOG_DEBUG("压缩右图像...");
    auto compressedRight = compressImage(rightImage.data(), 
                                        m_originalImageWidth, m_originalImageHeight,
                                        m_compressedImageWidth, m_compressedImageHeight);
    
    // 拼接图像：左半部分 = 压缩后的左眼，右半部分 = 压缩后的右眼
    size_t compressedSize = compressedLeft.size(); // compressedWidth * compressedHeight
    std::vector<uint8_t> stitchedImage(compressedSize * 2);
    
    LOG_DEBUG("拼接图像，压缩大小: {} 字节", compressedSize);
    
    // 复制左眼数据到拼接图像左半部分
    std::copy(compressedLeft.begin(), compressedLeft.end(), stitchedImage.begin());
    
    // 复制右眼数据到拼接图像右半部分
    std::copy(compressedRight.begin(), compressedRight.end(), 
              stitchedImage.begin() + compressedSize);
    
    LOG_DEBUG("拼接图像总大小: {} 字节", stitchedImage.size());
    
    // 将拼接图像复制到缓冲区
    if (!m_stitchedImageBuffer->copyToBuffer(stitchedImage.data(), stitchedImage.size())) {
        LOG_ERROR("复制拼接图像到缓冲区失败");
        return false;
    }
    
    LOG_INFO("✅ 图像压缩拼接完成: {}x{} -> {}x{} (拼接后: {}x{})", 
              m_originalImageWidth, m_originalImageHeight,
              m_compressedImageWidth, m_compressedImageHeight,
              m_compressedImageWidth * 2, m_compressedImageHeight);
    
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
    m_stitchedImageBuffer.reset();
    m_leftCensusBuffer.reset();
    m_rightCensusBuffer.reset();
    m_censusDebugBuffer.reset();
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
