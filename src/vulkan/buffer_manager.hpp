#pragma once

#include "vulkan/context.hpp"
#include <vulkan/vulkan.h>
#include <memory>
#include <vector>

namespace stereo_depth {
namespace vulkan {

/**
 * @brief 缓冲区管理器类，用于管理Vulkan缓冲区
 * 
 * 按照RAII原则管理缓冲区生命周期
 */
class BufferManager {
public:
    BufferManager() = delete;
    BufferManager(const VulkanContext& context);
    ~BufferManager();
    
    // 禁止拷贝
    BufferManager(const BufferManager&) = delete;
    BufferManager& operator=(const BufferManager&) = delete;
    
    // 允许移动
    BufferManager(BufferManager&& other) noexcept;
    BufferManager& operator=(BufferManager&& other) noexcept;
    
    /**
     * @brief 创建存储缓冲区
     * @param size 缓冲区大小（字节）
     * @param usage 额外使用标志
     * @return 是否创建成功
     */
    bool createStorageBuffer(VkDeviceSize size, VkBufferUsageFlags usage = 0);
    
    /**
     * @brief 创建Uniform缓冲区
     * @param size 缓冲区大小（字节）
     * @return 是否创建成功
     */
    bool createUniformBuffer(VkDeviceSize size);
    
    /**
     * @brief 创建暂存缓冲区（用于CPU到GPU数据传输）
     * @param size 缓冲区大小（字节）
     * @return 是否创建成功
     */
    bool createStagingBuffer(VkDeviceSize size);
    
    /**
     * @brief 获取缓冲区句柄
     */
    VkBuffer getBuffer() const { return m_buffer; }
    
    /**
     * @brief 获取缓冲区内存
     */
    VkDeviceMemory getMemory() const { return m_memory; }
    
    /**
     * @brief 获取缓冲区大小
     */
    VkDeviceSize getSize() const { return m_size; }
    
    /**
     * @brief 映射缓冲区内存到CPU地址空间
     * @return 映射后的指针，失败返回nullptr
     */
    void* map();
    
    /**
     * @brief 取消映射缓冲区内存
     */
    void unmap();
    
    /**
     * @brief 复制数据到缓冲区
     * @param data 源数据指针
     * @param size 数据大小（字节）
     * @param offset 缓冲区偏移量
     * @return 是否复制成功
     */
    bool copyToBuffer(const void* data, VkDeviceSize size, VkDeviceSize offset = 0);
    
    /**
     * @brief 从缓冲区复制数据
     * @param data 目标数据指针
     * @param size 数据大小（字节）
     * @param offset 缓冲区偏移量
     * @return 是否复制成功
     */
    bool copyFromBuffer(void* data, VkDeviceSize size, VkDeviceSize offset = 0);
    
    /**
     * @brief 清空缓冲区（填充0）
     * @param size 要清空的大小，0表示整个缓冲区
     */
    void clear(VkDeviceSize size = 0);
    
    /**
     * @brief 检查缓冲区是否有效
     */
    bool isValid() const { return m_buffer != VK_NULL_HANDLE; }
    
private:
    const VulkanContext& m_context;
    VkBuffer m_buffer = VK_NULL_HANDLE;
    VkDeviceMemory m_memory = VK_NULL_HANDLE;
    VkDeviceSize m_size = 0;
    VkBufferUsageFlags m_usage = 0;
    VkMemoryPropertyFlags m_memoryProperties = 0;
    void* m_mapped = nullptr;
    
    void cleanup();
};

/**
 * @brief 描述符集布局构建器，用于创建描述符集布局
 */
class DescriptorSetLayoutBuilder {
public:
    DescriptorSetLayoutBuilder(const VulkanContext& context);
    ~DescriptorSetLayoutBuilder();
    
    // 禁止拷贝，允许移动
    DescriptorSetLayoutBuilder(const DescriptorSetLayoutBuilder&) = delete;
    DescriptorSetLayoutBuilder& operator=(const DescriptorSetLayoutBuilder&) = delete;
    DescriptorSetLayoutBuilder(DescriptorSetLayoutBuilder&&) = default;
    DescriptorSetLayoutBuilder& operator=(DescriptorSetLayoutBuilder&&) = default;
    
    /**
     * @brief 添加存储缓冲区绑定
     * @param binding 绑定编号
     * @param count 描述符数量（数组大小）
     * @param stages 使用着色器阶段
     */
    DescriptorSetLayoutBuilder& addStorageBuffer(
        uint32_t binding, 
        uint32_t count = 1,
        VkShaderStageFlags stages = VK_SHADER_STAGE_COMPUTE_BIT
    );
    
    /**
     * @brief 添加Uniform缓冲区绑定
     * @param binding 绑定编号
     * @param count 描述符数量（数组大小）
     * @param stages 使用着色器阶段
     */
    DescriptorSetLayoutBuilder& addUniformBuffer(
        uint32_t binding, 
        uint32_t count = 1,
        VkShaderStageFlags stages = VK_SHADER_STAGE_COMPUTE_BIT
    );
    
    /**
     * @brief 构建描述符集布局
     * @return 创建的描述符集布局，失败返回VK_NULL_HANDLE
     */
    VkDescriptorSetLayout build();
    
private:
    const VulkanContext& m_context;
    std::vector<VkDescriptorSetLayoutBinding> m_bindings;
};

} // namespace vulkan
} // namespace stereo_depth
