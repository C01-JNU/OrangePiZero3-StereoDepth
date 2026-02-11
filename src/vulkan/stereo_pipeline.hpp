#pragma once

#include "vulkan/context.hpp"
#include "vulkan/buffer_manager.hpp"
#include "vulkan/compute_pipeline.hpp"
#include <memory>
#include <vector>

namespace stereo_depth {
namespace vulkan {

/**
 * @brief 立体匹配流水线（精简可靠版）
 *
 * 核心原则：
 * - 每个着色器独占一个描述符集，布局严格匹配着色器
 * - 使用手动创建描述符集布局，避免 Builder 潜在问题
 * - 所有缓冲区均为 uint32_t 元素
 * - CPU 缓存扩展图像数据，避免 GPU 回读
 */
class StereoPipeline {
public:
    explicit StereoPipeline(const VulkanContext& context);
    ~StereoPipeline();

    // 禁止拷贝，允许移动
    StereoPipeline(const StereoPipeline&) = delete;
    StereoPipeline& operator=(const StereoPipeline&) = delete;
    StereoPipeline(StereoPipeline&&) = default;
    StereoPipeline& operator=(StereoPipeline&&) = default;

    /**
     * @brief 初始化流水线
     * @param camWidth  原始拼接图像宽度（例如640）
     * @param camHeight 原始拼接图像高度（例如480）
     * @param maxDisparity 最大视差
     */
    bool initialize(uint32_t camWidth, uint32_t camHeight, uint32_t maxDisparity);

    /// 设置左眼图像（压缩后尺寸，uint8_t 灰度）
    bool setLeftImage(const uint8_t* data);
    /// 设置右眼图像（压缩后尺寸，uint8_t 灰度）
    bool setRightImage(const uint8_t* data);

    /// 执行完整立体匹配流水线
    bool compute();

    /// 获取视差图（16位，输出缓冲区需预分配 w*h*sizeof(uint16_t)）
    bool getDisparityMap(uint16_t* output);

    /// 获取中间结果（调试）
    bool getIntermediateResult(uint32_t index, void* output, size_t size);

    /// 获取中心像素（160,240）的Census描述符（用于验证数据通路）
    bool getCenterCensus(uint32_t& leftLow, uint32_t& leftHigh,
                         uint32_t& rightLow, uint32_t& rightHigh);

    // 访问器
    uint32_t getCompressedWidth()  const { return m_compW; }
    uint32_t getCompressedHeight() const { return m_compH; }
    uint32_t getMaxDisparity()     const { return m_maxDisp; }
    bool     isInitialized()       const { return m_initialized; }

private:
    const VulkanContext& m_ctx;

    // 尺寸参数
    uint32_t m_origW = 0;   // 原始拼接宽度（640）
    uint32_t m_origH = 0;   // 原始拼接高度（480）
    uint32_t m_compW = 0;   // 压缩后单眼宽度（320）
    uint32_t m_compH = 0;   // 压缩后高度（480）
    uint32_t m_maxDisp = 64;
    bool m_initialized = false;

    // ------------------------------------------------------------
    // CPU 端图像缓存（32位扩展），用于拼接，避免GPU回读
    // ------------------------------------------------------------
    std::vector<uint32_t> m_leftCpu;
    std::vector<uint32_t> m_rightCpu;

    // ------------------------------------------------------------
    // GPU 缓冲区（全部使用 uint32_t 元素）
    // ------------------------------------------------------------
    std::unique_ptr<BufferManager> m_leftImgBuf;      // 压缩后单眼，w*h*4
    std::unique_ptr<BufferManager> m_rightImgBuf;     // 同上
    std::unique_ptr<BufferManager> m_stitchedBuf;     // 拼接图像，w*2 * h *4

    std::unique_ptr<BufferManager> m_leftCensusBuf;   // 左 Census，w*h *2*4
    std::unique_ptr<BufferManager> m_rightCensusBuf;  // 右 Census，w*h *2*4

    std::unique_ptr<BufferManager> m_costVolBuf;      // 代价立方体，w*h*maxDisp*4
    std::unique_ptr<BufferManager> m_disparityBuf;    // 视差图，w*h*4

    std::unique_ptr<BufferManager> m_tempBuf1;        // 临时（聚合代价）
    std::unique_ptr<BufferManager> m_tempBuf2;        // 临时（后处理）

    std::unique_ptr<BufferManager> m_paramsBuf;       // Uniform 参数，sizeof(PipelineParams)

    // 调试缓冲区（每个步骤独立）
    std::unique_ptr<BufferManager> m_censusDbgBuf;
    std::unique_ptr<BufferManager> m_costDbgBuf;
    std::unique_ptr<BufferManager> m_wtaDbgBuf;
    std::unique_ptr<BufferManager> m_postDbgBuf;

    // ------------------------------------------------------------
    // 描述符集布局（每个管线一个，手动创建）
    // ------------------------------------------------------------
    VkDescriptorSetLayout m_censusLayout   = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_costLayout     = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_aggregationLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_wtaLayout      = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_postLayout     = VK_NULL_HANDLE;

    // ------------------------------------------------------------
    // 计算管线
    // ------------------------------------------------------------
    std::unique_ptr<ComputePipeline> m_censusPipe;
    std::unique_ptr<ComputePipeline> m_costPipe;
    std::unique_ptr<ComputePipeline> m_aggregationPipe;
    std::unique_ptr<ComputePipeline> m_wtaPipe;
    std::unique_ptr<ComputePipeline> m_postPipe;

    // ------------------------------------------------------------
    // 私有方法
    // ------------------------------------------------------------
    bool createAllBuffers();
    bool createAllDescriptorSetLayouts();
    bool createAllPipelines();
    bool loadShader(ComputePipeline& pipe, const std::string& name);
    bool executePipeline(ComputePipeline& pipe, uint32_t gx, uint32_t gy);

    void expand8To32(const uint8_t* src, uint32_t* dst, size_t count);
    bool uploadAndStitch();   // 使用CPU缓存拼接

    // PipelineParams 必须与着色器的 Parameters 完全一致（std140，64字节）
    struct PipelineParams {
        uint32_t width;        // 偏移 0
        uint32_t height;       // 偏移 4
        uint32_t maxDisparity; // 偏移 8
        uint32_t windowSize;   // 偏移 12
        float uniquenessRatio; // 偏移 16
        float penaltyP1;       // 偏移 20
        float penaltyP2;       // 偏移 24
        uint32_t flags;        // 偏移 28
        uint32_t speckleWindow; // 偏移 32
        uint32_t speckleRange;  // 偏移 36
        uint32_t medianSize;    // 偏移 40
        uint32_t padding[3];    // 偏移 44,48,52
        uint32_t reserved1;     // 偏移 56
        uint32_t reserved2;     // 偏移 60
    };
    static_assert(sizeof(PipelineParams) == 64, "PipelineParams must be 64 bytes");
};

} // namespace vulkan
} // namespace stereo_depth
