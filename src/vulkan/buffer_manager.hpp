#pragma once

#include "vulkan/context.hpp"
#include <vulkan/vulkan.h>
#include <memory>
#include <vector>

namespace stereo_depth {
namespace vulkan {

/**
 * @brief 缓冲区管理器，封装 Vulkan 缓冲区和设备内存
 *        支持设备本地（DEVICE_LOCAL）和主机可见（HOST_VISIBLE）两种类型
 */
class BufferManager {
public:
    explicit BufferManager(const VulkanContext& context);
    ~BufferManager();

    // 禁止拷贝，允许移动
    BufferManager(const BufferManager&) = delete;
    BufferManager& operator=(const BufferManager&) = delete;
    BufferManager(BufferManager&& other) noexcept;
    BufferManager& operator=(BufferManager&& other) noexcept;

    // ---------- 设备本地缓冲区（GPU 专用，性能最优）----------
    bool createDeviceLocalBuffer(VkDeviceSize size, VkBufferUsageFlags usage);
    
    // ---------- 主机可见缓冲区（用于上传/下载暂存）----------
    bool createStagingBuffer(VkDeviceSize size, VkBufferUsageFlags usage);

    // ---------- Uniform 缓冲区（主机可见，用于小数据量参数）----------
    bool createUniformBuffer(VkDeviceSize size);

    // ---------- 数据上传/下载（通过暂存缓冲区）----------
    // 从 CPU 内存拷贝到设备本地缓冲区（阻塞，等待完成）
    bool copyToDevice(const void* data, VkDeviceSize size);
    
    // 从设备本地缓冲区拷贝到 CPU 内存（阻塞，等待完成）
    bool copyFromDevice(void* data, VkDeviceSize size);

    // ---------- 直接映射访问（仅对 HOST_VISIBLE 缓冲区有效）----------
    bool copyToBuffer(const void* data, VkDeviceSize size);
    bool copyFromBuffer(void* data, VkDeviceSize size);

    // 获取 Vulkan 对象
    VkBuffer getBuffer() const { return buffer_; }
    VkDeviceMemory getMemory() const { return memory_; }
    VkDeviceSize getSize() const { return size_; }
    bool isValid() const { return buffer_ != VK_NULL_HANDLE && memory_ != VK_NULL_HANDLE; }

    // 显式释放资源
    void release();

private:
    const VulkanContext& m_ctx;
    VkDevice m_device;

    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkDeviceSize size_ = 0;
    bool is_host_visible_ = false;   // 标记是否为 HOST_VISIBLE，用于映射

    void cleanup();
};

} // namespace vulkan
} // namespace stereo_depth
