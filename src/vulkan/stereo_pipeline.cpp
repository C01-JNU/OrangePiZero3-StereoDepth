#include "vulkan/stereo_pipeline.hpp"
#include "utils/logger.hpp"
#include "vulkan/generated/pyramid_config.hpp"
#include <cstring>
#include <algorithm>
#include <unistd.h>
#include <limits.h>
#include <filesystem>

namespace stereo_depth::vulkan {

#define CHECK(expr, msg) do { if (!(expr)) { LOG_ERROR(msg); return false; } } while(0)

// 获取可执行文件所在目录（绝对路径）
static std::string getExeDir() {
    char result[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
    if (count != -1) {
        result[count] = '\0';
        std::string exePath(result);
        size_t pos = exePath.find_last_of('/');
        if (pos != std::string::npos) {
            return exePath.substr(0, pos);
        }
    }
    return ".";
}

// -----------------------------------------------------------------
// 构造函数 / 析构函数
// -----------------------------------------------------------------
StereoPipeline::StereoPipeline(const VulkanContext& context)
    : m_ctx(context), m_device(context.getDevice()), m_queue(context.getComputeQueue()) {}

StereoPipeline::~StereoPipeline() {
    VkDevice dev = m_device;
    for (auto& lvl : m_levels) {
        if (lvl.layout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(dev, lvl.layout, nullptr);
            lvl.layout = VK_NULL_HANDLE;
        }
        auto destroyCmd = [dev](LevelResources::CmdResources& c) {
            if (c.fence)  vkDestroyFence(dev, c.fence, nullptr);
            if (c.buffer) vkFreeCommandBuffers(dev, c.pool, 1, &c.buffer);
            if (c.pool)   vkDestroyCommandPool(dev, c.pool, nullptr);
            c.pool = VK_NULL_HANDLE; c.buffer = VK_NULL_HANDLE; c.fence = VK_NULL_HANDLE;
        };
        destroyCmd(lvl.censusCmd);
        destroyCmd(lvl.costWtaCmd);
        destroyCmd(lvl.postCmd);
        destroyCmd(lvl.downsampleCmd);
    }
}

// -----------------------------------------------------------------
// 初始化
// -----------------------------------------------------------------
bool StereoPipeline::initialize() {
    if (m_initialized) return true;
    m_levels.resize(PYRAMID_LEVEL_COUNT);
    m_baseWidth  = PYRAMID_LEVELS[0].width;
    m_baseHeight = PYRAMID_LEVELS[0].height;
    m_leftCpu.clear(); m_rightCpu.clear();

    LOG_INFO("初始化立体匹配流水线（设备本地内存优化版），金字塔层数: {}", PYRAMID_LEVEL_COUNT);
    for (uint32_t i = 0; i < PYRAMID_LEVEL_COUNT; ++i) {
        LOG_DEBUG("  层{}: {}x{} 视差={} 搜索半径={}", i,
                  PYRAMID_LEVELS[i].width, PYRAMID_LEVELS[i].height,
                  PYRAMID_LEVELS[i].max_disparity, PYRAMID_LEVELS[i].search_radius);
        CHECK(createLevelResources(i, PYRAMID_LEVELS[i]), "创建层资源失败");
    }
    m_initialized = true;
    LOG_INFO("立体匹配流水线初始化完成");
    return true;
}

// -----------------------------------------------------------------
// 创建单层资源（全部使用设备本地缓冲区）
// -----------------------------------------------------------------
bool StereoPipeline::createLevelResources(uint32_t level, const PyramidLevelParams& params) {
    LevelResources& res = m_levels[level];
    res.width = params.width;
    res.height = params.height;
    res.maxDisparity = params.max_disparity;

    size_t pixels = static_cast<size_t>(res.width) * res.height;
    size_t imgBytes    = pixels * sizeof(uint32_t);
    size_t censusBytes = pixels * 2 * sizeof(uint32_t);
    size_t tempBytes   = pixels * sizeof(uint32_t);
    size_t debugBytes  = 256 * sizeof(uint32_t);
    size_t paramsBytes = sizeof(PipelineParams);

    // 设备本地缓冲区创建辅助函数
    auto createDeviceLocal = [this](VkDeviceSize size, VkBufferUsageFlags usage, const char* name) {
        auto buf = std::make_unique<BufferManager>(m_ctx);
        if (!buf->createDeviceLocalBuffer(size, usage)) {
            LOG_ERROR("创建设备本地缓冲区失败: {}", name);
            return std::unique_ptr<BufferManager>(nullptr);
        }
        return buf;
    };

    // Uniform 缓冲区（主机可见，小数据量）
    auto createUniform = [this](VkDeviceSize size, const char* name) {
        auto buf = std::make_unique<BufferManager>(m_ctx);
        if (!buf->createUniformBuffer(size)) {
            LOG_ERROR("创建Uniform缓冲区失败: {}", name);
            return std::unique_ptr<BufferManager>(nullptr);
        }
        return buf;
    };

    // 所有存储缓冲区均使用 DEVICE_LOCAL，并添加 TRANSFER_DST 以便上传数据
    res.leftImg    = createDeviceLocal(imgBytes,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      "LeftImage");
    res.rightImg   = createDeviceLocal(imgBytes,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      "RightImage");
    res.leftCensus = createDeviceLocal(censusBytes,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      "LeftCensus");
    res.rightCensus= createDeviceLocal(censusBytes,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      "RightCensus");
    res.disparity  = createDeviceLocal(tempBytes,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                      "Disparity");
    res.priorDisparity = createDeviceLocal(tempBytes,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      "PriorDisparity");
    res.temp       = createDeviceLocal(tempBytes,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      "Temp");
    res.params     = createUniform(paramsBytes, "Params");
    res.debug      = createDeviceLocal(debugBytes,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      "Debug");

    if (!res.leftImg || !res.rightImg || !res.leftCensus || !res.rightCensus ||
        !res.disparity || !res.priorDisparity || !res.temp || !res.params || !res.debug)
        return false;

    // 填充 Uniform 参数
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
    uniformParams.searchRadius   = params.search_radius;
    uniformParams.padding[0] = uniformParams.padding[1] = 0;

    CHECK(res.params->copyToBuffer(&uniformParams, sizeof(uniformParams)), "上传Uniform参数失败");

    CHECK(createDescriptorSetLayout(res), "创建描述符集布局失败");
    CHECK(createPipelines(res, level), "创建管线失败");
    CHECK(createCmdResources(res.censusCmd) && createCmdResources(res.costWtaCmd) &&
          createCmdResources(res.postCmd) && createCmdResources(res.downsampleCmd),
          "创建命令资源失败");

    LOG_DEBUG("层{} 资源创建成功: {}x{}", level, res.width, res.height);
    return true;
}

// -----------------------------------------------------------------
// 创建描述符集布局（统一6个binding）
// -----------------------------------------------------------------
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
    info.bindingCount = 6;
    info.pBindings = bindings;
    CHECK(vkCreateDescriptorSetLayout(m_device, &info, nullptr, &res.layout) == VK_SUCCESS,
          "创建描述符集布局失败");
    return true;
}

// -----------------------------------------------------------------
// 创建管线及描述符集（绑定设备本地缓冲区）
// -----------------------------------------------------------------
bool StereoPipeline::createPipelines(LevelResources& res, uint32_t level) {
    res.censusPipe    = std::make_unique<ComputePipeline>(m_ctx);
    res.costWtaPipe   = std::make_unique<ComputePipeline>(m_ctx);
    res.postPipe      = std::make_unique<ComputePipeline>(m_ctx);
    res.downsamplePipe = std::make_unique<ComputePipeline>(m_ctx);

    res.censusPipe->setDescriptorSetLayout(res.layout);
    res.costWtaPipe->setDescriptorSetLayout(res.layout);
    res.postPipe->setDescriptorSetLayout(res.layout);
    res.downsamplePipe->setDescriptorSetLayout(res.layout);

    CHECK(loadShader(*res.censusPipe,    "census",      level), "加载 Census 着色器失败");
    CHECK(loadShader(*res.costWtaPipe,   "cost_wta",    level), "加载 CostWTA 着色器失败");
    CHECK(loadShader(*res.postPipe,      "postprocess", level), "加载 Postprocess 着色器失败");
    CHECK(loadShader(*res.downsamplePipe, "downsample", level), "加载 Downsample 着色器失败");

    CHECK(res.censusPipe->createPipeline(0),    "创建 Census 管线失败");
    CHECK(res.costWtaPipe->createPipeline(0),   "创建 CostWTA 管线失败");
    CHECK(res.postPipe->createPipeline(0),      "创建 Postprocess 管线失败");
    CHECK(res.downsamplePipe->createPipeline(16), "创建 Downsample 管线失败");

    // 绑定描述符集辅助函数
    auto bindBuffers = [&](ComputePipeline& pipe,
                           VkBuffer b1, VkBuffer b2, VkBuffer b3, VkBuffer b4) -> bool {
        std::vector<VkBuffer> bufs = {
            res.params->getBuffer(), b1, b2, b3, b4, res.debug->getBuffer()
        };
        std::vector<VkDescriptorType> types(6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        types[0] = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        return pipe.createDescriptorSet(bufs, types);
    };

    // Census: binding1=左图, binding2=右图, binding3=左Census, binding4=右Census
    CHECK(bindBuffers(*res.censusPipe,
        res.leftImg->getBuffer(), res.rightImg->getBuffer(),
        res.leftCensus->getBuffer(), res.rightCensus->getBuffer()), "Census 描述符集失败");

    // CostWTA: binding1=左Census, binding2=右Census, binding3=视差图, binding4=先验视差图
    CHECK(bindBuffers(*res.costWtaPipe,
        res.leftCensus->getBuffer(), res.rightCensus->getBuffer(),
        res.disparity->getBuffer(), res.priorDisparity->getBuffer()), "CostWTA 描述符集失败");

    // Postprocess: 输入(binding1) = 视差图, 输出(binding3) = 视差图（原位更新）
    CHECK(bindBuffers(*res.postPipe,
        res.disparity->getBuffer(), res.temp->getBuffer(),
        res.disparity->getBuffer(), res.temp->getBuffer()), "Postprocess 描述符集失败");

    // Downsample: 绑定占位，实际执行时重新绑定（因为输入来自上层）
    CHECK(bindBuffers(*res.downsamplePipe,
        res.disparity->getBuffer(), res.temp->getBuffer(),
        res.disparity->getBuffer(), res.priorDisparity->getBuffer()), "Downsample 描述符集失败");

    return true;
}

// -----------------------------------------------------------------
// 创建命令池/缓冲/Fence（复用）
// -----------------------------------------------------------------
bool StereoPipeline::createCmdResources(LevelResources::CmdResources& cmd) {
    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = m_ctx.getComputeQueueFamilyIndex();
    CHECK(vkCreateCommandPool(m_device, &poolInfo, nullptr, &cmd.pool) == VK_SUCCESS,
          "创建命令池失败");

    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = cmd.pool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    CHECK(vkAllocateCommandBuffers(m_device, &allocInfo, &cmd.buffer) == VK_SUCCESS,
          "分配命令缓冲失败");

    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    CHECK(vkCreateFence(m_device, &fenceInfo, nullptr, &cmd.fence) == VK_SUCCESS,
          "创建Fence失败");
    return true;
}

// -----------------------------------------------------------------
// 加载着色器（使用绝对路径，基于可执行文件目录下的 spv 文件夹）
// -----------------------------------------------------------------
bool StereoPipeline::loadShader(ComputePipeline& pipe, const std::string& name, uint32_t level) {
    std::string exeDir = getExeDir();                     // build/bin
    std::string spvFile = exeDir + "/spv/" + name + "_layer" + std::to_string(level) + ".comp.spv";
    if (pipe.loadShaderFromFile(spvFile)) {
        LOG_DEBUG("加载着色器: {}", spvFile);
        return true;
    }
    LOG_ERROR("无法加载着色器: {}", spvFile);
    return false;
}

// -----------------------------------------------------------------
// 执行单个管线（复用命令缓冲）
// -----------------------------------------------------------------
bool StereoPipeline::executePipeline(ComputePipeline& pipe, LevelResources::CmdResources& cmd,
                                     uint32_t level, uint32_t gx, uint32_t gy) {
    vkWaitForFences(m_device, 1, &cmd.fence, VK_TRUE, UINT64_MAX);
    vkResetFences(m_device, 1, &cmd.fence);
    vkResetCommandBuffer(cmd.buffer, 0);

    VkCommandBufferBeginInfo beginInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CHECK(vkBeginCommandBuffer(cmd.buffer, &beginInfo) == VK_SUCCESS, "开始命令缓冲区失败");
    pipe.recordCommands(cmd.buffer, gx, gy, 1);
    CHECK(vkEndCommandBuffer(cmd.buffer) == VK_SUCCESS, "结束命令缓冲区失败");

    VkSubmitInfo submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd.buffer;
    CHECK(vkQueueSubmit(m_queue, 1, &submitInfo, cmd.fence) == VK_SUCCESS, "提交命令失败");
    return true;
}

// -----------------------------------------------------------------
// 8位扩展32位（CPU端）
// -----------------------------------------------------------------
void StereoPipeline::expand8To32(const uint8_t* src, uint32_t* dst, size_t count) {
    for (size_t i = 0; i < count; ++i) dst[i] = static_cast<uint32_t>(src[i]);
}

// -----------------------------------------------------------------
// 设置左图像（使用暂存缓冲上传到设备本地）
// -----------------------------------------------------------------
bool StereoPipeline::setLeftImage(const uint8_t* data) {
    CHECK(m_initialized, "流水线未初始化");
    size_t count = static_cast<size_t>(m_baseWidth) * m_baseHeight;
    m_leftCpu.resize(count);
    expand8To32(data, m_leftCpu.data(), count);
    return m_levels[0].leftImg->copyToDevice(m_leftCpu.data(), count * sizeof(uint32_t));
}

bool StereoPipeline::setRightImage(const uint8_t* data) {
    CHECK(m_initialized, "流水线未初始化");
    size_t count = static_cast<size_t>(m_baseWidth) * m_baseHeight;
    m_rightCpu.resize(count);
    expand8To32(data, m_rightCpu.data(), count);
    return m_levels[0].rightImg->copyToDevice(m_rightCpu.data(), count * sizeof(uint32_t));
}

// -----------------------------------------------------------------
// 下采样：将上层视差图作为先验，输出到当前层的 priorDisparity
// -----------------------------------------------------------------
bool StereoPipeline::downsamplePriorGPU(uint32_t fromLevel, uint32_t toLevel) {
    auto& from = m_levels[fromLevel];
    auto& to   = m_levels[toLevel];

    struct PushParams {
        uint32_t fromWidth, fromHeight, toWidth, toHeight;
    } pushParams = { from.width, from.height, to.width, to.height };

    uint32_t gx = (to.width  + WORKGROUP_X - 1) / WORKGROUP_X;
    uint32_t gy = (to.height + WORKGROUP_Y - 1) / WORKGROUP_Y;

    auto& cmd = to.downsampleCmd;
    vkWaitForFences(m_device, 1, &cmd.fence, VK_TRUE, UINT64_MAX);
    vkResetFences(m_device, 1, &cmd.fence);
    vkResetCommandBuffer(cmd.buffer, 0);

    VkCommandBufferBeginInfo beginInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CHECK(vkBeginCommandBuffer(cmd.buffer, &beginInfo) == VK_SUCCESS, "开始命令缓冲区失败");

    // 动态绑定：输入 = from.disparity，输出 = to.priorDisparity
    std::vector<VkBuffer> bufs = {
        to.params->getBuffer(),          // binding 0
        from.disparity->getBuffer(),     // binding 1 (输入)
        to.temp->getBuffer(),            // binding 2 (未使用)
        to.disparity->getBuffer(),       // binding 3 (未使用)
        to.priorDisparity->getBuffer(),  // binding 4 (输出)
        to.debug->getBuffer()            // binding 5
    };
    std::vector<VkDescriptorType> types(6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    types[0] = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    if (!to.downsamplePipe->createDescriptorSet(bufs, types)) {
        LOG_ERROR("创建下采样描述符集失败");
        return false;
    }

    to.downsamplePipe->recordCommands(cmd.buffer, gx, gy, 1, &pushParams, sizeof(pushParams));

    CHECK(vkEndCommandBuffer(cmd.buffer) == VK_SUCCESS, "结束命令缓冲区失败");

    VkSubmitInfo submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd.buffer;
    CHECK(vkQueueSubmit(m_queue, 1, &submitInfo, cmd.fence) == VK_SUCCESS, "提交下采样命令失败");
    return true;
}

// -----------------------------------------------------------------
// 主计算流程（从粗到细）
// -----------------------------------------------------------------
bool StereoPipeline::compute() {
    CHECK(m_initialized, "流水线未初始化");
    CHECK(!m_leftCpu.empty() && !m_rightCpu.empty(), "左右图像未设置");

    for (int32_t level = static_cast<int32_t>(m_levels.size()) - 1; level >= 0; --level) {
        auto& res = m_levels[level];

        if (level != static_cast<int32_t>(m_levels.size()) - 1) {
            CHECK(downsamplePriorGPU(level + 1, level), "视差下采样失败");
            vkWaitForFences(m_device, 1, &res.downsampleCmd.fence, VK_TRUE, UINT64_MAX);
        }

        uint32_t gx = (res.width  + WORKGROUP_X - 1) / WORKGROUP_X;
        uint32_t gy = (res.height + WORKGROUP_Y - 1) / WORKGROUP_Y;

        CHECK(executePipeline(*res.censusPipe,    res.censusCmd,    level, gx, gy), "Census 失败");
        CHECK(executePipeline(*res.costWtaPipe,   res.costWtaCmd,   level, gx, gy), "CostWTA 失败");
        CHECK(executePipeline(*res.postPipe,      res.postCmd,      level, gx, gy), "Postprocess 失败");

        VkFence fences[] = {res.censusCmd.fence, res.costWtaCmd.fence, res.postCmd.fence};
        vkWaitForFences(m_device, 3, fences, VK_TRUE, UINT64_MAX);
    }

    LOG_INFO("立体匹配计算完成");
    return true;
}

// -----------------------------------------------------------------
// 获取视差图（从设备本地缓冲区下载）
// -----------------------------------------------------------------
bool StereoPipeline::getDisparityMap(uint16_t* output) {
    CHECK(m_initialized && !m_levels.empty(), "流水线未初始化");
    auto& res = m_levels[0];
    size_t pixels = static_cast<size_t>(res.width) * res.height;
    std::vector<uint32_t> gpuDisp(pixels);
    CHECK(res.disparity->copyFromDevice(gpuDisp.data(), pixels * sizeof(uint32_t)),
          "从GPU读取视差图失败");
    for (size_t i = 0; i < pixels; ++i) {
        uint32_t val = gpuDisp[i];
        output[i] = static_cast<uint16_t>(val > res.maxDisparity ? res.maxDisparity : val);
    }
    return true;
}

// -----------------------------------------------------------------
// 获取调试缓冲区（直接从设备本地读回）
// -----------------------------------------------------------------
bool StereoPipeline::getDebugBuffer(uint32_t level, uint32_t stage, void* output, size_t size) {
    CHECK(level < m_levels.size(), "无效层索引");
    auto& res = m_levels[level];
    BufferManager* dbg = res.debug.get();
    CHECK(dbg && dbg->isValid(), "调试缓冲区无效");
    return dbg->copyFromDevice(output, std::min(size, dbg->getSize()));
}

} // namespace stereo_depth::vulkan
