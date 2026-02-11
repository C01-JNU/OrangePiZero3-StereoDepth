#include "vulkan/stereo_pipeline.hpp"
#include "utils/logger.hpp"
#include "utils/config.hpp"
#include <cstring>
#include <algorithm>

namespace stereo_depth {
namespace vulkan {

// -----------------------------------------------------------------
// 辅助宏：错误日志 + 返回 false
// -----------------------------------------------------------------
#define CHECK(expr, msg) \
    do { \
        if (!(expr)) { \
            LOG_ERROR(msg); \
            return false; \
        } \
    } while(0)

// -----------------------------------------------------------------
// 构造函数 / 析构函数
// -----------------------------------------------------------------
StereoPipeline::StereoPipeline(const VulkanContext& context) : m_ctx(context) {}

StereoPipeline::~StereoPipeline() {
    VkDevice dev = m_ctx.getDevice();
    auto destroyLayout = [dev](VkDescriptorSetLayout& l) {
        if (l != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(dev, l, nullptr);
            l = VK_NULL_HANDLE;
        }
    };
    destroyLayout(m_censusLayout);
    destroyLayout(m_costLayout);
    destroyLayout(m_aggregationLayout);
    destroyLayout(m_wtaLayout);
    destroyLayout(m_postLayout);
}

// -----------------------------------------------------------------
// 初始化
// -----------------------------------------------------------------
bool StereoPipeline::initialize(uint32_t camWidth, uint32_t camHeight, uint32_t maxDisparity) {
    if (m_initialized) return true;
    CHECK(camWidth % 2 == 0, "原始图像宽度必须是偶数");

    m_origW = camWidth;
    m_origH = camHeight;
    m_compW = camWidth / 2;
    m_compH = camHeight;
    m_maxDisp = maxDisparity;

    // 清空CPU缓存
    m_leftCpu.clear();
    m_rightCpu.clear();

    LOG_INFO("初始化立体匹配流水线: 压缩尺寸 {}x{}, 最大视差 {}", m_compW, m_compH, m_maxDisp);

    CHECK(createAllBuffers(),           "创建缓冲区失败");
    CHECK(createAllDescriptorSetLayouts(), "创建描述符集布局失败");
    CHECK(createAllPipelines(),         "创建计算管线失败");

    // 从配置读取参数
    PipelineParams params = {};
    params.width        = m_compW;
    params.height       = m_compH;
    params.maxDisparity = m_maxDisp;

    auto& cfg = utils::ConfigManager::getInstance().getConfig();
    params.windowSize = cfg.get<uint32_t>("stereo.window_size", 9);
    float uniquenessPercent = cfg.get<float>("stereo.uniqueness_ratio", 15.0f);
    params.uniquenessRatio = uniquenessPercent / 100.0f;
    params.penaltyP1 = cfg.get<float>("stereo.penalty_p1", 8.0f);
    params.penaltyP2 = cfg.get<float>("stereo.penalty_p2", 32.0f);
    params.flags = 0;
    if (cfg.get<bool>("stereo.use_census", true))          params.flags |= 0x01;
    if (cfg.get<bool>("stereo.use_median_filter", true))   params.flags |= 0x02;
    if (cfg.get<bool>("stereo.enable_postprocessing", true)) params.flags |= 0x04;
    params.speckleWindow = cfg.get<uint32_t>("stereo.speckle_window_size", 100);
    params.speckleRange  = cfg.get<uint32_t>("stereo.speckle_range", 32);
    params.medianSize    = cfg.get<uint32_t>("stereo.median_filter_size", 3);
    memset(params.padding, 0, sizeof(params.padding));
    params.reserved1 = params.reserved2 = 0;

    CHECK(m_paramsBuf->copyToBuffer(&params, sizeof(params)), "上传参数到GPU失败");

    m_initialized = true;
    LOG_INFO("立体匹配流水线初始化完成");
    return true;
}

// -----------------------------------------------------------------
// 创建所有缓冲区（统一 uint32_t）
// -----------------------------------------------------------------
bool StereoPipeline::createAllBuffers() {
    size_t pixels = static_cast<size_t>(m_compW) * m_compH;
    size_t imgBytes    = pixels * sizeof(uint32_t);
    size_t stitchBytes = pixels * 2 * sizeof(uint32_t);
    size_t censusBytes = pixels * 2 * sizeof(uint32_t);
    size_t debugBytes  = 8 * sizeof(uint32_t);
    size_t costVolBytes = pixels * m_maxDisp * sizeof(uint32_t);
    size_t tempBytes    = pixels * sizeof(uint32_t);
    size_t paramsBytes  = sizeof(PipelineParams);

    auto createStorage = [this](VkDeviceSize size, const char* name) {
        auto buf = std::make_unique<BufferManager>(m_ctx);
        if (!buf->createStorageBuffer(size)) {
            LOG_ERROR("❌ 创建存储缓冲区失败: {} ({} 字节)", name, size);
            return std::unique_ptr<BufferManager>(nullptr);
        }
        LOG_DEBUG("✅ 创建存储缓冲区: {} ({} 字节)", name, size);
        return buf;
    };

    auto createUniform = [this](VkDeviceSize size, const char* name) {
        auto buf = std::make_unique<BufferManager>(m_ctx);
        if (!buf->createUniformBuffer(size)) {
            LOG_ERROR("❌ 创建 Uniform 缓冲区失败: {} ({} 字节)", name, size);
            return std::unique_ptr<BufferManager>(nullptr);
        }
        LOG_DEBUG("✅ 创建 Uniform 缓冲区: {} ({} 字节)", name, size);
        return buf;
    };

    m_leftImgBuf   = createStorage(imgBytes,    "LeftImage");
    if (!m_leftImgBuf)   return false;
    m_rightImgBuf  = createStorage(imgBytes,    "RightImage");
    if (!m_rightImgBuf)  return false;
    m_stitchedBuf  = createStorage(stitchBytes, "Stitched");
    if (!m_stitchedBuf)  return false;

    m_leftCensusBuf = createStorage(censusBytes, "LeftCensus");
    if (!m_leftCensusBuf) return false;
    m_rightCensusBuf = createStorage(censusBytes, "RightCensus");
    if (!m_rightCensusBuf) return false;

    m_censusDbgBuf = createStorage(debugBytes, "CensusDebug");
    if (!m_censusDbgBuf) return false;
    m_costDbgBuf   = createStorage(debugBytes, "CostDebug");
    if (!m_costDbgBuf) return false;
    m_wtaDbgBuf    = createStorage(debugBytes, "WTADebug");
    if (!m_wtaDbgBuf) return false;
    m_postDbgBuf   = createStorage(debugBytes, "PostDebug");
    if (!m_postDbgBuf) return false;

    m_costVolBuf   = createStorage(costVolBytes, "CostVolume");
    if (!m_costVolBuf) return false;
    m_disparityBuf = createStorage(tempBytes,    "Disparity");
    if (!m_disparityBuf) return false;
    m_tempBuf1     = createStorage(tempBytes,    "Temp1");
    if (!m_tempBuf1) return false;
    m_tempBuf2     = createStorage(tempBytes,    "Temp2");
    if (!m_tempBuf2) return false;

    m_paramsBuf    = createUniform(paramsBytes, "Params");
    if (!m_paramsBuf) return false;

    LOG_INFO("✅ 所有缓冲区创建成功，总内存约 {:.2f} MB",
             (imgBytes*2 + stitchBytes + censusBytes*2 + debugBytes*4 + costVolBytes + tempBytes*2 + paramsBytes) / (1024.0*1024.0));
    return true;
}

// -----------------------------------------------------------------
// 手动创建描述符集布局
// -----------------------------------------------------------------
bool StereoPipeline::createAllDescriptorSetLayouts() {
    VkDevice dev = m_ctx.getDevice();

    // Census 布局：5个 binding
    VkDescriptorSetLayoutBinding censusBindings[5] = {};
    for (uint32_t i = 0; i < 5; i++) {
        censusBindings[i].binding = i;
        censusBindings[i].descriptorType = (i == 0) ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        censusBindings[i].descriptorCount = 1;
        censusBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo censusInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    censusInfo.bindingCount = 5;
    censusInfo.pBindings = censusBindings;
    CHECK(vkCreateDescriptorSetLayout(dev, &censusInfo, nullptr, &m_censusLayout) == VK_SUCCESS,
          "创建 Census 布局失败");

    // Cost 布局：5个 binding
    VkDescriptorSetLayoutBinding costBindings[5] = {};
    for (uint32_t i = 0; i < 5; i++) {
        costBindings[i].binding = i;
        costBindings[i].descriptorType = (i == 3) ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        costBindings[i].descriptorCount = 1;
        costBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo costInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    costInfo.bindingCount = 5;
    costInfo.pBindings = costBindings;
    CHECK(vkCreateDescriptorSetLayout(dev, &costInfo, nullptr, &m_costLayout) == VK_SUCCESS,
          "创建 Cost 布局失败");

    // 聚合布局：4个 binding
    VkDescriptorSetLayoutBinding aggBindings[4] = {};
    for (uint32_t i = 0; i < 4; i++) {
        aggBindings[i].binding = i;
        aggBindings[i].descriptorType = (i == 3) ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        aggBindings[i].descriptorCount = 1;
        aggBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo aggInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    aggInfo.bindingCount = 4;
    aggInfo.pBindings = aggBindings;
    CHECK(vkCreateDescriptorSetLayout(dev, &aggInfo, nullptr, &m_aggregationLayout) == VK_SUCCESS,
          "创建聚合布局失败");

    // WTA 布局：4个 binding
    VkDescriptorSetLayoutBinding wtaBindings[4] = {};
    for (uint32_t i = 0; i < 4; i++) {
        wtaBindings[i].binding = i;
        wtaBindings[i].descriptorType = (i == 2) ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wtaBindings[i].descriptorCount = 1;
        wtaBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo wtaInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    wtaInfo.bindingCount = 4;
    wtaInfo.pBindings = wtaBindings;
    CHECK(vkCreateDescriptorSetLayout(dev, &wtaInfo, nullptr, &m_wtaLayout) == VK_SUCCESS,
          "创建 WTA 布局失败");

    // 后处理布局：5个 binding
    VkDescriptorSetLayoutBinding postBindings[5] = {};
    for (uint32_t i = 0; i < 5; i++) {
        postBindings[i].binding = i;
        postBindings[i].descriptorType = (i == 3) ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        postBindings[i].descriptorCount = 1;
        postBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo postInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    postInfo.bindingCount = 5;
    postInfo.pBindings = postBindings;
    CHECK(vkCreateDescriptorSetLayout(dev, &postInfo, nullptr, &m_postLayout) == VK_SUCCESS,
          "创建后处理布局失败");

    LOG_INFO("✅ 所有描述符集布局创建成功");
    return true;
}

// -----------------------------------------------------------------
// 创建管线并绑定描述符集
// -----------------------------------------------------------------
bool StereoPipeline::createAllPipelines() {
    // 1. Census 管线
    m_censusPipe = std::make_unique<ComputePipeline>(m_ctx);
    m_censusPipe->setDescriptorSetLayout(m_censusLayout);
    CHECK(loadShader(*m_censusPipe, "census.comp.spv"), "加载 Census 着色器失败");
    CHECK(m_censusPipe->createPipeline(), "创建 Census 管线失败");

    std::vector<VkBuffer> censusBufs = {
        m_paramsBuf->getBuffer(),
        m_stitchedBuf->getBuffer(),
        m_leftCensusBuf->getBuffer(),
        m_rightCensusBuf->getBuffer(),
        m_censusDbgBuf->getBuffer()
    };
    std::vector<VkDescriptorType> censusTypes = {
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
    };
    CHECK(m_censusPipe->createDescriptorSet(censusBufs, censusTypes), "Census 描述符集创建失败");

    // 2. Cost 管线
    m_costPipe = std::make_unique<ComputePipeline>(m_ctx);
    m_costPipe->setDescriptorSetLayout(m_costLayout);
    CHECK(loadShader(*m_costPipe, "cost.comp.spv"), "加载 Cost 着色器失败");
    CHECK(m_costPipe->createPipeline(), "创建 Cost 管线失败");

    std::vector<VkBuffer> costBufs = {
        m_leftCensusBuf->getBuffer(),
        m_rightCensusBuf->getBuffer(),
        m_costVolBuf->getBuffer(),
        m_paramsBuf->getBuffer(),
        m_costDbgBuf->getBuffer()
    };
    std::vector<VkDescriptorType> costTypes = {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
    };
    CHECK(m_costPipe->createDescriptorSet(costBufs, costTypes), "Cost 描述符集创建失败");

    // 3. 聚合管线
    m_aggregationPipe = std::make_unique<ComputePipeline>(m_ctx);
    m_aggregationPipe->setDescriptorSetLayout(m_aggregationLayout);
    CHECK(loadShader(*m_aggregationPipe, "aggregation.comp.spv"), "加载聚合着色器失败");
    CHECK(m_aggregationPipe->createPipeline(), "创建聚合管线失败");

    std::vector<VkBuffer> aggBufs = {
        m_costVolBuf->getBuffer(),
        m_tempBuf1->getBuffer(),
        m_tempBuf2->getBuffer(),
        m_paramsBuf->getBuffer()
    };
    std::vector<VkDescriptorType> aggTypes = {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
    };
    CHECK(m_aggregationPipe->createDescriptorSet(aggBufs, aggTypes), "聚合描述符集创建失败");

    // 4. WTA 管线
    m_wtaPipe = std::make_unique<ComputePipeline>(m_ctx);
    m_wtaPipe->setDescriptorSetLayout(m_wtaLayout);
    CHECK(loadShader(*m_wtaPipe, "wta.comp.spv"), "加载 WTA 着色器失败");
    CHECK(m_wtaPipe->createPipeline(), "创建 WTA 管线失败");

    std::vector<VkBuffer> wtaBufs = {
        m_tempBuf1->getBuffer(),
        m_disparityBuf->getBuffer(),
        m_paramsBuf->getBuffer(),
        m_wtaDbgBuf->getBuffer()
    };
    std::vector<VkDescriptorType> wtaTypes = {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
    };
    CHECK(m_wtaPipe->createDescriptorSet(wtaBufs, wtaTypes), "WTA 描述符集创建失败");

    // 5. 后处理管线
    m_postPipe = std::make_unique<ComputePipeline>(m_ctx);
    m_postPipe->setDescriptorSetLayout(m_postLayout);
    CHECK(loadShader(*m_postPipe, "postprocess.comp.spv"), "加载后处理着色器失败");
    CHECK(m_postPipe->createPipeline(), "创建后处理管线失败");

    std::vector<VkBuffer> postBufs = {
        m_disparityBuf->getBuffer(),
        m_disparityBuf->getBuffer(),
        m_tempBuf2->getBuffer(),
        m_paramsBuf->getBuffer(),
        m_postDbgBuf->getBuffer()
    };
    std::vector<VkDescriptorType> postTypes = {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
    };
    CHECK(m_postPipe->createDescriptorSet(postBufs, postTypes), "后处理描述符集创建失败");

    LOG_INFO("✅ 所有计算管线创建成功");
    return true;
}

// -----------------------------------------------------------------
// 加载着色器（优先搜索 src/vulkan/spv/）
// -----------------------------------------------------------------
bool StereoPipeline::loadShader(ComputePipeline& pipe, const std::string& name) {
    std::vector<std::string> paths = {
        "src/vulkan/spv/" + name,           // ✅ 源目录下的SPIR-V
        "../src/vulkan/spv/" + name,        // 从 build/bin 回退
        "shaders/" + name,                 // 原始着色器（非SPIR-V）
        name
    };
    for (const auto& p : paths) {
        if (pipe.loadShaderFromFile(p)) {
            LOG_DEBUG("✅ 加载着色器: {}", p);
            return true;
        }
    }
    LOG_ERROR("❌ 无法加载着色器: {}", name);
    return false;
}

// -----------------------------------------------------------------
// 8位扩展为32位
// -----------------------------------------------------------------
void StereoPipeline::expand8To32(const uint8_t* src, uint32_t* dst, size_t count) {
    for (size_t i = 0; i < count; ++i) dst[i] = static_cast<uint32_t>(src[i]);
}

// -----------------------------------------------------------------
// 设置左/右图像（同时存入CPU缓存）
// -----------------------------------------------------------------
bool StereoPipeline::setLeftImage(const uint8_t* data) {
    CHECK(m_initialized, "流水线未初始化");
    size_t count = static_cast<size_t>(m_compW) * m_compH;
    m_leftCpu.resize(count);
    expand8To32(data, m_leftCpu.data(), count);
    return m_leftImgBuf->copyToBuffer(m_leftCpu.data(), count * sizeof(uint32_t));
}

bool StereoPipeline::setRightImage(const uint8_t* data) {
    CHECK(m_initialized, "流水线未初始化");
    size_t count = static_cast<size_t>(m_compW) * m_compH;
    m_rightCpu.resize(count);
    expand8To32(data, m_rightCpu.data(), count);
    return m_rightImgBuf->copyToBuffer(m_rightCpu.data(), count * sizeof(uint32_t));
}

// -----------------------------------------------------------------
// 上传并拼接（直接使用CPU缓存，零回读）
// -----------------------------------------------------------------
bool StereoPipeline::uploadAndStitch() {
    CHECK(!m_leftCpu.empty() && !m_rightCpu.empty(), "左右图像CPU缓存为空");
    size_t pixels = static_cast<size_t>(m_compW) * m_compH;
    
    std::vector<uint32_t> stitched(pixels * 2);
    std::copy(m_leftCpu.begin(), m_leftCpu.end(), stitched.begin());
    std::copy(m_rightCpu.begin(), m_rightCpu.end(), stitched.begin() + pixels);
    
    return m_stitchedBuf->copyToBuffer(stitched.data(), stitched.size() * sizeof(uint32_t));
}

// -----------------------------------------------------------------
// 执行单个管线
// -----------------------------------------------------------------
bool StereoPipeline::executePipeline(ComputePipeline& pipe, uint32_t gx, uint32_t gy) {
    VkCommandPool pool = m_ctx.createCommandPool();
    CHECK(pool != VK_NULL_HANDLE, "创建命令池失败");

    VkCommandBuffer cmd = m_ctx.createCommandBuffer(pool);
    CHECK(cmd != VK_NULL_HANDLE, "创建命令缓冲区失败");

    VkCommandBufferBeginInfo beginInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CHECK(vkBeginCommandBuffer(cmd, &beginInfo) == VK_SUCCESS, "开始命令缓冲区失败");

    pipe.recordCommands(cmd, gx, gy, 1);
    CHECK(vkEndCommandBuffer(cmd) == VK_SUCCESS, "结束命令缓冲区失败");

    VkSubmitInfo submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    CHECK(vkQueueSubmit(m_ctx.getComputeQueue(), 1, &submitInfo, VK_NULL_HANDLE) == VK_SUCCESS,
          "提交命令失败");

    vkQueueWaitIdle(m_ctx.getComputeQueue());

    vkFreeCommandBuffers(m_ctx.getDevice(), pool, 1, &cmd);
    vkDestroyCommandPool(m_ctx.getDevice(), pool, nullptr);
    return true;
}

// -----------------------------------------------------------------
// 主计算流程
// -----------------------------------------------------------------
bool StereoPipeline::compute() {
    CHECK(m_initialized, "流水线未初始化");

    // 步骤0：拼接图像
    CHECK(uploadAndStitch(), "拼接图像失败");

    uint32_t gx = (m_compW + 15) / 16;
    uint32_t gy = (m_compH + 15) / 16;

    // 步骤1：Census
    CHECK(executePipeline(*m_censusPipe, gx, gy), "Census 失败");

    // 步骤2：代价计算
    CHECK(executePipeline(*m_costPipe, gx, gy), "代价计算失败");

    // 步骤3：代价聚合
    CHECK(executePipeline(*m_aggregationPipe, gx, gy), "代价聚合失败");

    // 步骤4：WTA
    CHECK(executePipeline(*m_wtaPipe, gx, gy), "WTA 失败");

    // 步骤5：后处理
    CHECK(executePipeline(*m_postPipe, gx, gy), "后处理失败");

    LOG_INFO("✅ 立体匹配计算完成");
    return true;
}

// -----------------------------------------------------------------
// 获取视差图（32位转16位）
// -----------------------------------------------------------------
bool StereoPipeline::getDisparityMap(uint16_t* output) {
    CHECK(m_initialized && m_disparityBuf, "流水线未初始化");
    size_t pixels = static_cast<size_t>(m_compW) * m_compH;
    std::vector<uint32_t> gpuDisp(pixels);
    CHECK(m_disparityBuf->copyFromBuffer(gpuDisp.data(), pixels * sizeof(uint32_t)),
          "从GPU读取视差图失败");

    for (size_t i = 0; i < pixels; ++i) {
        uint32_t val = gpuDisp[i];
        output[i] = static_cast<uint16_t>(val > m_maxDisp ? m_maxDisp : val);
    }
    return true;
}

// -----------------------------------------------------------------
// 获取中间结果（调试）
// -----------------------------------------------------------------
bool StereoPipeline::getIntermediateResult(uint32_t index, void* output, size_t size) {
    CHECK(m_initialized, "流水线未初始化");
    BufferManager* buf = nullptr;
    switch (index) {
        case 0:  buf = m_leftImgBuf.get();   break;
        case 1:  buf = m_rightImgBuf.get();  break;
        case 2:  buf = m_stitchedBuf.get();  break;
        case 3:  buf = m_leftCensusBuf.get(); break;
        case 4:  buf = m_rightCensusBuf.get(); break;
        case 5:  buf = m_censusDbgBuf.get(); break;
        case 6:  buf = m_costDbgBuf.get();   break;
        case 7:  buf = m_wtaDbgBuf.get();    break;
        case 8:  buf = m_postDbgBuf.get();   break;
        case 9:  buf = m_costVolBuf.get();   break;
        case 10: buf = m_disparityBuf.get(); break;
        case 11: buf = m_tempBuf1.get();     break;
        case 12: buf = m_tempBuf2.get();     break;
        default: LOG_ERROR("无效缓冲区索引: {}", index); return false;
    }
    CHECK(buf && buf->isValid(), "缓冲区无效");
    return buf->copyFromBuffer(output, std::min(size, buf->getSize()));
}

} // namespace vulkan
} // namespace stereo_depth
