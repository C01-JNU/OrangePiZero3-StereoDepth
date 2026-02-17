#pragma once

#include "vulkan/context.hpp"
#include "vulkan/buffer_manager.hpp"
#include <vulkan/vulkan.h>
#include <memory>
#include <string>
#include <vector>

namespace stereo_depth {
namespace vulkan {

/**
 * @brief 计算管线管理器类，用于管理计算着色器管线
 */
class ComputePipeline {
public:
    ComputePipeline() = delete;
    ComputePipeline(const VulkanContext& context);
    ~ComputePipeline();
    
    // 禁止拷贝
    ComputePipeline(const ComputePipeline&) = delete;
    ComputePipeline& operator=(const ComputePipeline&) = delete;
    
    // 允许移动
    ComputePipeline(ComputePipeline&& other) noexcept;
    ComputePipeline& operator=(ComputePipeline&& other) noexcept;
    
    /**
     * @brief 从文件加载计算着色器
     * @param shaderPath 着色器文件路径（SPIR-V格式）
     * @return 是否加载成功
     */
    bool loadShaderFromFile(const std::string& shaderPath);
    
    /**
     * @brief 从内存加载计算着色器
     * @param shaderCode 着色器代码指针
     * @param codeSize 着色器代码大小（字节）
     * @return 是否加载成功
     */
    bool loadShaderFromMemory(const uint32_t* shaderCode, size_t codeSize);
    
    /**
     * @brief 设置描述符集布局
     * @param layout 描述符集布局
     */
    void setDescriptorSetLayout(VkDescriptorSetLayout layout);
    
    /**
     * @brief 创建计算管线
     * @param pushConstantSize 推送常量大小（字节），0表示不使用推送常量
     * @return 是否创建成功
     */
    bool createPipeline(size_t pushConstantSize = 0);
    
    /**
     * @brief 创建描述符集
     * @param buffers 缓冲区列表
     * @param bufferTypes 缓冲区类型列表（VK_DESCRIPTOR_TYPE_STORAGE_BUFFER等）
     * @return 是否创建成功
     */
    bool createDescriptorSet(const std::vector<VkBuffer>& buffers, 
                           const std::vector<VkDescriptorType>& bufferTypes);
    
    /**
     * @brief 记录计算命令
     * @param commandBuffer 命令缓冲区
     * @param groupCountX X方向工作组数量
     * @param groupCountY Y方向工作组数量
     * @param groupCountZ Z方向工作组数量
     * @param pushConstants 推送常量数据指针
     * @param pushConstantSize 推送常量大小
     */
    void recordCommands(VkCommandBuffer commandBuffer,
                       uint32_t groupCountX, 
                       uint32_t groupCountY = 1,
                       uint32_t groupCountZ = 1,
                       const void* pushConstants = nullptr,
                       size_t pushConstantSize = 0);
    
    /**
     * @brief 获取管线布局
     */
    VkPipelineLayout getPipelineLayout() const { return m_pipelineLayout; }
    
    /**
     * @brief 获取管线
     */
    VkPipeline getPipeline() const { return m_pipeline; }
    
    /**
     * @brief 获取描述符集
     */
    VkDescriptorSet getDescriptorSet() const { return m_descriptorSet; }
    
    /**
     * @brief 检查管线是否有效
     */
    bool isValid() const { return m_pipeline != VK_NULL_HANDLE; }
    
    /**
     * @brief 等待管线创建完成
     */
    void waitIdle() const { m_context.waitIdle(); }
    
private:
    const VulkanContext& m_context;
    VkShaderModule m_shaderModule = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet m_descriptorSet = VK_NULL_HANDLE;
    
    void cleanup();
    
    /**
     * @brief 创建着色器模块
     * @param code 着色器代码指针
     * @param codeSize 代码大小（字节）
     * @return 着色器模块，失败返回VK_NULL_HANDLE
     */
    VkShaderModule createShaderModule(const uint32_t* code, size_t codeSize);
    
    /**
     * @brief 创建描述符池
     * @param bufferTypes 缓冲区类型列表
     * @return 是否创建成功
     */
    bool createDescriptorPool(const std::vector<VkDescriptorType>& bufferTypes);
};

} // namespace vulkan
} // namespace stereo_depth
