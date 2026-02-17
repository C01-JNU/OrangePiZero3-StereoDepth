#pragma once
#include "vulkan/context.hpp"
#include "vulkan/buffer_manager.hpp"
#include "vulkan/compute_pipeline.hpp"
#include "vulkan/generated/pyramid_config.hpp"
#include <memory>
#include <vector>

namespace stereo_depth::vulkan {

class StereoPipeline {
public:
    explicit StereoPipeline(const VulkanContext& context);
    ~StereoPipeline();

    StereoPipeline(const StereoPipeline&) = delete;
    StereoPipeline& operator=(const StereoPipeline&) = delete;
    StereoPipeline(StereoPipeline&&) = default;
    StereoPipeline& operator=(StereoPipeline&&) = default;

    bool initialize();
    bool setLeftImage(const uint8_t* data);
    bool setRightImage(const uint8_t* data);
    bool compute();
    bool getDisparityMap(uint16_t* output);
    bool getDebugBuffer(uint32_t level, uint32_t stage, void* output, size_t size);

    uint32_t getLevelCount() const { return PYRAMID_LEVEL_COUNT; }
    uint32_t getBaseWidth()  const { return m_baseWidth; }
    uint32_t getBaseHeight() const { return m_baseHeight; }
    bool isInitialized()     const { return m_initialized; }

private:
    const VulkanContext& m_ctx;
    VkDevice m_device = VK_NULL_HANDLE;
    VkQueue  m_queue  = VK_NULL_HANDLE;

    std::vector<uint32_t> m_leftCpu;
    std::vector<uint32_t> m_rightCpu;
    uint32_t m_baseWidth  = 0;
    uint32_t m_baseHeight = 0;
    bool m_initialized = false;

    struct LevelResources {
        uint32_t width, height, maxDisparity;

        // 缓冲区（全部使用设备本地内存，Uniform除外）
        std::unique_ptr<BufferManager> leftImg, rightImg;
        std::unique_ptr<BufferManager> leftCensus, rightCensus;
        std::unique_ptr<BufferManager> disparity;
        std::unique_ptr<BufferManager> priorDisparity;
        std::unique_ptr<BufferManager> temp;
        std::unique_ptr<BufferManager> params;   // Uniform，主机可见
        std::unique_ptr<BufferManager> debug;

        VkDescriptorSetLayout layout = VK_NULL_HANDLE;

        std::unique_ptr<ComputePipeline> censusPipe;
        std::unique_ptr<ComputePipeline> costWtaPipe;
        std::unique_ptr<ComputePipeline> postPipe;
        std::unique_ptr<ComputePipeline> downsamplePipe;

        struct CmdResources {
            VkCommandPool   pool   = VK_NULL_HANDLE;
            VkCommandBuffer buffer = VK_NULL_HANDLE;
            VkFence         fence  = VK_NULL_HANDLE;
        };
        CmdResources censusCmd, costWtaCmd, postCmd, downsampleCmd;
    };
    std::vector<LevelResources> m_levels;

    // Uniform结构体（56字节）
    struct PipelineParams {
        uint32_t imageWidth, imageHeight, maxDisparity, windowSize;
        float uniquenessRatio, penaltyP1, penaltyP2;
        uint32_t flags, speckleWindow, speckleRange, medianSize, searchRadius;
        uint32_t padding[2];
    };
    static_assert(sizeof(PipelineParams) == 56, "PipelineParams must be 56 bytes");

    bool createLevelResources(uint32_t level, const PyramidLevelParams& params);
    bool createDescriptorSetLayout(LevelResources& res);
    bool createPipelines(LevelResources& res, uint32_t level);
    bool createCmdResources(LevelResources::CmdResources& cmd);
    bool loadShader(ComputePipeline& pipe, const std::string& name, uint32_t level);
    bool executePipeline(ComputePipeline& pipe, LevelResources::CmdResources& cmd,
                         uint32_t level, uint32_t gx, uint32_t gy);
    void expand8To32(const uint8_t* src, uint32_t* dst, size_t count);
    bool downsamplePriorGPU(uint32_t fromLevel, uint32_t toLevel);
};

} // namespace stereo_depth::vulkan
