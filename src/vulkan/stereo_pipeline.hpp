#pragma once

#include "vulkan/context.hpp"
#include "vulkan/buffer_manager.hpp"
#include "vulkan/compute_pipeline.hpp"
#include <memory>
#include <vector>

namespace stereo_depth {
namespace vulkan {

/**
 * @brief 立体匹配流水线，整合所有Vulkan组件
 * 
 * 负责管理立体匹配的整个计算流程：
 * 1. 图像输入（左右图像）
 * 2. 代价计算
 * 3. 视差优化
 * 4. 结果输出
 */
class StereoPipeline {
public:
    StereoPipeline() = delete;
    StereoPipeline(const VulkanContext& context);
    ~StereoPipeline();
    
    // 禁止拷贝
    StereoPipeline(const StereoPipeline&) = delete;
    StereoPipeline& operator=(const StereoPipeline&) = delete;
    
    // 允许移动
    StereoPipeline(StereoPipeline&& other) noexcept;
    StereoPipeline& operator=(StereoPipeline&& other) noexcept;
    
    /**
     * @brief 初始化流水线
     * @param imageWidth 图像宽度（压缩前的单眼宽度）
     * @param imageHeight 图像高度
     * @param maxDisparity 最大视差
     * @return 是否初始化成功
     */
    bool initialize(uint32_t imageWidth, uint32_t imageHeight, uint32_t maxDisparity);
    
    /**
     * @brief 设置左图像数据（压缩前）
     * @param data 图像数据指针（灰度图，8位/像素）
     * @return 是否设置成功
     */
    bool setLeftImage(const uint8_t* data);
    
    /**
     * @brief 设置右图像数据（压缩前）
     * @param data 图像数据指针（灰度图，8位/像素）
     * @return 是否设置成功
     */
    bool setRightImage(const uint8_t* data);
    
    /**
     * @brief 执行立体匹配计算
     * @return 是否计算成功
     */
    bool compute();
    
    /**
     * @brief 获取视差图（压缩后尺寸）
     * @param output 输出缓冲区，需要预分配compressedWidth * compressedHeight * sizeof(uint16_t)字节
     * @return 是否获取成功
     */
    bool getDisparityMap(uint16_t* output);
    
    /**
     * @brief 获取中间结果（用于调试）
     * @param bufferIndex 缓冲区索引
     * @param output 输出缓冲区
     * @param size 缓冲区大小
     * @return 是否获取成功
     */
    bool getIntermediateResult(uint32_t bufferIndex, void* output, size_t size);
    
    /**
     * @brief 获取压缩前图像宽度
     */
    uint32_t getOriginalImageWidth() const { return m_originalImageWidth; }
    
    /**
     * @brief 获取压缩前图像高度
     */
    uint32_t getOriginalImageHeight() const { return m_originalImageHeight; }
    
    /**
     * @brief 获取压缩后图像宽度
     */
    uint32_t getCompressedImageWidth() const { return m_compressedImageWidth; }
    
    /**
     * @brief 获取压缩后图像高度
     */
    uint32_t getCompressedImageHeight() const { return m_compressedImageHeight; }
    
    /**
     * @brief 获取最大视差
     */
    uint32_t getMaxDisparity() const { return m_maxDisparity; }
    
    /**
     * @brief 检查是否已初始化
     */
    bool isInitialized() const { return m_initialized; }
    
private:
    const VulkanContext& m_context;
    
    // 原始图像参数
    uint32_t m_originalImageWidth = 0;
    uint32_t m_originalImageHeight = 0;
    
    // 压缩后图像参数
    uint32_t m_compressedImageWidth = 0;
    uint32_t m_compressedImageHeight = 0;
    
    // 视差参数
    uint32_t m_maxDisparity = 64;
    bool m_initialized = false;
    
    // 原始输入缓冲区（压缩前）
    std::unique_ptr<BufferManager> m_leftImageBuffer;
    std::unique_ptr<BufferManager> m_rightImageBuffer;
    
    // 拼接图像缓冲区（压缩后，左右拼接）
    std::unique_ptr<BufferManager> m_stitchedImageBuffer;
    
    // Census变换输出缓冲区（双输出，压缩后尺寸）
    std::unique_ptr<BufferManager> m_leftCensusBuffer;
    std::unique_ptr<BufferManager> m_rightCensusBuffer;
    std::unique_ptr<BufferManager> m_censusDebugBuffer;
    
    // 中间结果缓冲区（压缩后尺寸）
    std::unique_ptr<BufferManager> m_costVolumeBuffer;
    std::unique_ptr<BufferManager> m_disparityBuffer;
    std::unique_ptr<BufferManager> m_tempBuffer1;
    std::unique_ptr<BufferManager> m_tempBuffer2;
    
    // 参数缓冲区
    std::unique_ptr<BufferManager> m_paramsBuffer;
    
    // 计算管线
    std::unique_ptr<ComputePipeline> m_censusPipeline;
    std::unique_ptr<ComputePipeline> m_costPipeline;
    std::unique_ptr<ComputePipeline> m_aggregationPipeline;
    std::unique_ptr<ComputePipeline> m_wtaPipeline;
    std::unique_ptr<ComputePipeline> m_postprocessPipeline;
    
    // 描述符集布局
    VkDescriptorSetLayout m_censusLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_costLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_aggregationLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_wtaLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_postprocessLayout = VK_NULL_HANDLE;
    
    void cleanup();
    
    /**
     * @brief 创建所有缓冲区
     */
    bool createBuffers();
    
    /**
     * @brief 创建所有计算管线
     */
    bool createPipelines();
    
    /**
     * @brief 加载着色器文件
     * @param pipeline 管线对象
     * @param shaderName 着色器文件名（不带路径）
     * @return 是否加载成功
     */
    bool loadShader(ComputePipeline& pipeline, const std::string& shaderName);
    
    /**
     * @brief 检查着色器文件是否存在
     * @param shaderName 着色器文件名
     * @return 是否存在
     */
    bool checkShaderExists(const std::string& shaderName);
    
    /**
     * @brief 创建描述符集布局
     */
    bool createDescriptorSetLayouts();
    
    /**
     * @brief 执行单步计算
     * @param pipeline 计算管线
     * @param groupCountX X方向工作组数量
     * @param groupCountY Y方向工作组数量
     */
    bool executePipelineStep(ComputePipeline& pipeline, 
                            uint32_t groupCountX, 
                            uint32_t groupCountY);
    
    /**
     * @brief 压缩并拼接左右图像
     * @return 是否成功
     * 
     * 将左右原始图像（各1080x720）压缩为540x720，
     * 然后拼接成一张1080x720的图像
     */
    bool compressAndStitchImages();
    
    /**
     * @brief 计算图像压缩平均值（简单算法）
     * @param src 源图像数据
     * @param srcWidth 源图像宽度
     * @param srcHeight 源图像高度
     * @param dstWidth 目标图像宽度
     * @param dstHeight 目标图像高度
     * @return 压缩后的图像数据
     */
    std::vector<uint8_t> compressImage(const uint8_t* src, 
                                      uint32_t srcWidth, uint32_t srcHeight,
                                      uint32_t dstWidth, uint32_t dstHeight);
    
    // 计算参数结构
    struct PipelineParams {
        uint32_t compressedImageWidth;   // 压缩后单眼宽度（对应着色器imageWidth）
        uint32_t compressedImageHeight;  // 压缩后高度（对应着色器imageHeight）
        uint32_t maxDisparity;           // 最大视差
        uint32_t windowSize;             // Census窗口大小
        
        float uniquenessRatio;           // 唯一性比率
        float penaltyP1;                 // 惩罚项P1
        float penaltyP2;                 // 惩罚项P2
        uint32_t flags;                  // 标志位
        
        uint32_t speckleWindow;          // 斑点窗口大小
        uint32_t speckleRange;           // 斑点范围
        uint32_t medianSize;             // 中值滤波大小
        uint32_t padding[3];             // 补齐到64字节对齐（着色器需要）
        
        // 额外字段使总大小达到64字节（着色器不访问这些）
        uint32_t reserved1;
        uint32_t reserved2;
    };
    
    static_assert(sizeof(PipelineParams) == 64, 
                  "PipelineParams must be 64 bytes");
};

} // namespace vulkan
} // namespace stereo_depth
