#include "vulkan/stereo_pipeline.hpp"
#include "utils/logger.hpp"
#include "vulkan/generated/pyramid_config.hpp"
#include <cstring>
#include <algorithm>
namespace stereo_depth::vulkan {
#define CHECK(expr, msg) do { if (!(expr)) { LOG_ERROR(msg); return false; } } while(0)

StereoPipeline::StereoPipeline(const VulkanContext& context) : m_ctx(context) {}

StereoPipeline::~StereoPipeline() {
    VkDevice dev = m_ctx.getDevice();
    for (auto& lvl : m_levels) {
        if (lvl.layout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(dev, lvl.layout, nullptr);
            lvl.layout = VK_NULL_HANDLE;
        }
    }
}

bool StereoPipeline::initialize() {
    if (m_initialized) return true;
    m_levels.resize(PYRAMID_LEVEL_COUNT);
    m_baseWidth  = PYRAMID_LEVELS[0].width;
    m_baseHeight = PYRAMID_LEVELS[0].height;
    m_leftCpu.clear(); m_rightCpu.clear();
    LOG_INFO("初始化立体匹配流水线，金字塔层数: {}", PYRAMID_LEVEL_COUNT);
    for (uint32_t i = 0; i < PYRAMID_LEVEL_COUNT; ++i) {
        LOG_DEBUG("  层{}: {}x{} 视差={}", i, PYRAMID_LEVELS[i].width,
                  PYRAMID_LEVELS[i].height, PYRAMID_LEVELS[i].max_disparity);
        CHECK(createLevelResources(i, PYRAMID_LEVELS[i]), "创建层资源失败");
    }
    m_initialized = true;
    LOG_INFO("立体匹配流水线初始化完成");
    return true;
}

bool StereoPipeline::createLevelResources(uint32_t level, const PyramidLevelParams& params) {
    LevelResources& res = m_levels[level];
    res.width = params.width; res.height = params.height; res.maxDisparity = params.max_disparity;
    size_t pixels = static_cast<size_t>(res.width) * res.height;
    size_t imgBytes = pixels * sizeof(uint32_t);
    size_t censusBytes = pixels * 2 * sizeof(uint32_t);
    size_t costVolBytes = pixels * res.maxDisparity * sizeof(uint32_t);
    size_t tempBytes = pixels * sizeof(uint32_t);
    size_t debugBytes = 256 * sizeof(uint32_t);
    size_t paramsBytes = sizeof(PipelineParams);
    auto createStorage = [this](VkDeviceSize size, const char* name) {
        auto buf = std::make_unique<BufferManager>(m_ctx);
        if (!buf->createStorageBuffer(size)) {
            LOG_ERROR("创建存储缓冲区失败: {}", name);
            return std::unique_ptr<BufferManager>(nullptr);
        }
        return buf;
    };
    auto createUniform = [this](VkDeviceSize size, const char* name) {
        auto buf = std::make_unique<BufferManager>(m_ctx);
        if (!buf->createUniformBuffer(size)) {
            LOG_ERROR("创建Uniform缓冲区失败: {}", name);
            return std::unique_ptr<BufferManager>(nullptr);
        }
        return buf;
    };
    res.leftImg    = createStorage(imgBytes,     "LeftImage");
    res.rightImg   = createStorage(imgBytes,     "RightImage");
    res.leftCensus = createStorage(censusBytes,  "LeftCensus");
    res.rightCensus= createStorage(censusBytes,  "RightCensus");
    res.costVolume = createStorage(costVolBytes, "CostVolume");
    res.disparity  = createStorage(tempBytes,    "Disparity");
    res.temp       = createStorage(tempBytes,    "Temp");
    res.params     = createUniform(paramsBytes,  "Params");
    res.debug      = createStorage(debugBytes,   "Debug");
    if (!res.leftImg || !res.rightImg || !res.leftCensus || !res.rightCensus ||
        !res.costVolume || !res.disparity || !res.temp || !res.params || !res.debug)
        return false;

    PipelineParams uniformParams = {};
    uniformParams.imageWidth     = res.width;
    uniformParams.imageHeight    = res.height;
    uniformParams.maxDisparity   = res.maxDisparity;
    uniformParams.windowSize     = params.window_size;
    uniformParams.uniquenessRatio = params.uniqueness_ratio;
    uniformParams.penaltyP1      = params.penalty_p1;
    uniformParams.penaltyP2      = params.penalty_p2;
    uniformParams.flags          = params.flags;
    uniformParams.speckleWindow  = params.speckle_window;
    uniformParams.speckleRange   = params.speckle_range;
    uniformParams.medianSize     = params.median_size;
    uniformParams.padding[0] = uniformParams.padding[1] = uniformParams.padding[2] = 0;

    CHECK(res.params->copyToBuffer(&uniformParams, sizeof(uniformParams)), "上传Uniform参数失败");
    CHECK(createDescriptorSetLayout(res), "创建描述符集布局失败");
    CHECK(createPipelines(res, level), "创建管线失败");
    LOG_DEBUG("层{} 资源创建成功: {}x{}", level, res.width, res.height);
    return true;
}

bool StereoPipeline::createDescriptorSetLayout(LevelResources& res) {
    VkDescriptorSetLayoutBinding bindings[6] = {};
    for (uint32_t i = 0; i < 6; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = (i == 0) ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                              : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    info.bindingCount = 6; info.pBindings = bindings;
    VkDevice dev = m_ctx.getDevice();
    CHECK(vkCreateDescriptorSetLayout(dev, &info, nullptr, &res.layout) == VK_SUCCESS,
          "创建描述符集布局失败");
    return true;
}

bool StereoPipeline::createPipelines(LevelResources& res, uint32_t level) {
    res.censusPipe    = std::make_unique<ComputePipeline>(m_ctx);
    res.costPipe      = std::make_unique<ComputePipeline>(m_ctx);
    res.aggregatePipe = std::make_unique<ComputePipeline>(m_ctx);
    res.wtaPipe       = std::make_unique<ComputePipeline>(m_ctx);
    res.postPipe      = std::make_unique<ComputePipeline>(m_ctx);

    res.censusPipe->setDescriptorSetLayout(res.layout);
    res.costPipe->setDescriptorSetLayout(res.layout);
    res.aggregatePipe->setDescriptorSetLayout(res.layout);
    res.wtaPipe->setDescriptorSetLayout(res.layout);
    res.postPipe->setDescriptorSetLayout(res.layout);

    // 修正：传入不含 .comp 后缀的基本名
    CHECK(loadShader(*res.censusPipe,    "census",      level), "加载 Census 着色器失败");
    CHECK(loadShader(*res.costPipe,      "cost",        level), "加载 Cost 着色器失败");
    CHECK(loadShader(*res.aggregatePipe, "aggregation", level), "加载 Aggregate 着色器失败");
    CHECK(loadShader(*res.wtaPipe,       "wta",         level), "加载 WTA 着色器失败");
    CHECK(loadShader(*res.postPipe,      "postprocess", level), "加载 Postprocess 着色器失败");

    CHECK(res.censusPipe->createPipeline(),    "创建 Census 管线失败");
    CHECK(res.costPipe->createPipeline(),      "创建 Cost 管线失败");
    CHECK(res.aggregatePipe->createPipeline(), "创建 Aggregate 管线失败");
    CHECK(res.wtaPipe->createPipeline(),       "创建 WTA 管线失败");
    CHECK(res.postPipe->createPipeline(),      "创建 Postprocess 管线失败");

    auto bindBuffers = [&](ComputePipeline& pipe,
                           VkBuffer b1, VkBuffer b2, VkBuffer b3, VkBuffer b4) -> bool {
        std::vector<VkBuffer> bufs = {
            res.params->getBuffer(), b1, b2, b3, b4, res.debug->getBuffer()
        };
        std::vector<VkDescriptorType> types(6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        types[0] = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        return pipe.createDescriptorSet(bufs, types);
    };

    CHECK(bindBuffers(*res.censusPipe,
        res.leftImg->getBuffer(), res.rightImg->getBuffer(),
        res.leftCensus->getBuffer(), res.rightCensus->getBuffer()), "Census 描述符集失败");
    CHECK(bindBuffers(*res.costPipe,
        res.leftCensus->getBuffer(), res.rightCensus->getBuffer(),
        res.costVolume->getBuffer(), res.temp->getBuffer()), "Cost 描述符集失败");
    CHECK(bindBuffers(*res.aggregatePipe,
        res.costVolume->getBuffer(), res.temp->getBuffer(),
        res.costVolume->getBuffer(), res.temp->getBuffer()), "Aggregate 描述符集失败");
    CHECK(bindBuffers(*res.wtaPipe,
        res.costVolume->getBuffer(), res.temp->getBuffer(),
        res.disparity->getBuffer(), res.temp->getBuffer()), "WTA 描述符集失败");
    CHECK(bindBuffers(*res.postPipe,
        res.disparity->getBuffer(), res.temp->getBuffer(),
        res.disparity->getBuffer(), res.temp->getBuffer()), "Postprocess 描述符集失败");

    return true;
}

bool StereoPipeline::loadShader(ComputePipeline& pipe, const std::string& name, uint32_t level) {
    // 构建文件名：name_layer{level}.comp.spv
    std::string spvFile = "src/vulkan/spv/" + name + "_layer" + std::to_string(level) + ".comp.spv";
    if (pipe.loadShaderFromFile(spvFile)) {
        LOG_DEBUG("加载着色器: {}", spvFile);
        return true;
    }
    LOG_ERROR("无法加载着色器: {}", spvFile);
    return false;
}

bool StereoPipeline::executePipeline(ComputePipeline& pipe, uint32_t level,
                                     uint32_t gx, uint32_t gy) {
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
    submitInfo.commandBufferCount = 1; submitInfo.pCommandBuffers = &cmd;
    CHECK(vkQueueSubmit(m_ctx.getComputeQueue(), 1, &submitInfo, VK_NULL_HANDLE) == VK_SUCCESS,
          "提交命令失败");
    vkQueueWaitIdle(m_ctx.getComputeQueue());
    vkFreeCommandBuffers(m_ctx.getDevice(), pool, 1, &cmd);
    vkDestroyCommandPool(m_ctx.getDevice(), pool, nullptr);
    return true;
}

void StereoPipeline::expand8To32(const uint8_t* src, uint32_t* dst, size_t count) {
    for (size_t i = 0; i < count; ++i) dst[i] = static_cast<uint32_t>(src[i]);
}

bool StereoPipeline::setLeftImage(const uint8_t* data) {
    CHECK(m_initialized, "流水线未初始化");
    size_t count = static_cast<size_t>(m_baseWidth) * m_baseHeight;
    m_leftCpu.resize(count);
    expand8To32(data, m_leftCpu.data(), count);
    return m_levels[0].leftImg->copyToBuffer(m_leftCpu.data(), count * sizeof(uint32_t));
}

bool StereoPipeline::setRightImage(const uint8_t* data) {
    CHECK(m_initialized, "流水线未初始化");
    size_t count = static_cast<size_t>(m_baseWidth) * m_baseHeight;
    m_rightCpu.resize(count);
    expand8To32(data, m_rightCpu.data(), count);
    return m_levels[0].rightImg->copyToBuffer(m_rightCpu.data(), count * sizeof(uint32_t));
}

bool StereoPipeline::uploadAndStitch() { return true; }

bool StereoPipeline::downsampleDisparity(uint32_t fromLevel, uint32_t toLevel) {
    LOG_WARN("视差下采样使用最近邻复制");
    auto& from = m_levels[fromLevel];
    auto& to   = m_levels[toLevel];
    size_t fromPixels = from.width * from.height;
    size_t toPixels   = to.width * to.height;
    std::vector<uint32_t> fromData(fromPixels);
    CHECK(from.disparity->copyFromBuffer(fromData.data(), fromPixels * sizeof(uint32_t)),
          "读取上层视差失败");
    std::vector<uint32_t> toData(toPixels);
    float scale_x = static_cast<float>(from.width) / to.width;
    float scale_y = static_cast<float>(from.height) / to.height;
    for (uint32_t ty = 0; ty < to.height; ++ty) {
        uint32_t sy = static_cast<uint32_t>(ty * scale_y);
        for (uint32_t tx = 0; tx < to.width; ++tx) {
            uint32_t sx = static_cast<uint32_t>(tx * scale_x);
            toData[ty * to.width + tx] = fromData[sy * from.width + sx];
        }
    }
    return to.disparity->copyToBuffer(toData.data(), toPixels * sizeof(uint32_t));
}

bool StereoPipeline::compute() {
    CHECK(m_initialized, "流水线未初始化");
    CHECK(!m_leftCpu.empty() && !m_rightCpu.empty(), "左右图像未设置");
    for (int32_t level = static_cast<int32_t>(m_levels.size()) - 1; level >= 0; --level) {
        auto& res = m_levels[level];
        if (level != static_cast<int32_t>(m_levels.size()) - 1) {
            CHECK(downsampleDisparity(level + 1, level), "视差下采样失败");
        }
        uint32_t gx = (res.width  + WORKGROUP_X - 1) / WORKGROUP_X;
        uint32_t gy = (res.height + WORKGROUP_Y - 1) / WORKGROUP_Y;
        LOG_DEBUG("执行层{}: {}x{}, 工作组 {}x{}", level, res.width, res.height, gx, gy);
        CHECK(executePipeline(*res.censusPipe,    level, gx, gy), "Census 失败");
        CHECK(executePipeline(*res.costPipe,      level, gx, gy), "Cost 失败");
        CHECK(executePipeline(*res.aggregatePipe, level, gx, gy), "Aggregate 失败");
        CHECK(executePipeline(*res.wtaPipe,       level, gx, gy), "WTA 失败");
        CHECK(executePipeline(*res.postPipe,      level, gx, gy), "Postprocess 失败");
    }
    LOG_INFO("立体匹配计算完成");
    return true;
}

bool StereoPipeline::getDisparityMap(uint16_t* output) {
    CHECK(m_initialized && !m_levels.empty(), "流水线未初始化");
    auto& res = m_levels[0];
    size_t pixels = static_cast<size_t>(res.width) * res.height;
    std::vector<uint32_t> gpuDisp(pixels);
    CHECK(res.disparity->copyFromBuffer(gpuDisp.data(), pixels * sizeof(uint32_t)),
          "从GPU读取视差图失败");
    for (size_t i = 0; i < pixels; ++i) {
        uint32_t val = gpuDisp[i];
        output[i] = static_cast<uint16_t>(val > res.maxDisparity ? res.maxDisparity : val);
    }
    return true;
}

bool StereoPipeline::getDebugBuffer(uint32_t level, uint32_t stage, void* output, size_t size) {
    CHECK(level < m_levels.size(), "无效层索引");
    auto& res = m_levels[level];
    BufferManager* dbg = res.debug.get();
    CHECK(dbg && dbg->isValid(), "调试缓冲区无效");
    return dbg->copyFromBuffer(output, std::min(size, dbg->getSize()));
}

} // namespace stereo_depth::vulkan
