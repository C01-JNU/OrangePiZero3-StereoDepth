#include "vulkan/buffer_manager.hpp"
#include "utils/logger.hpp"
#include <cstring>

namespace stereo_depth {
namespace vulkan {

BufferManager::BufferManager(const VulkanContext& context)
    : m_ctx(context)
    , m_device(context.getDevice()) {
}

BufferManager::~BufferManager() {
    cleanup();
}

BufferManager::BufferManager(BufferManager&& other) noexcept
    : m_ctx(other.m_ctx)
    , m_device(other.m_device)
    , buffer_(other.buffer_)
    , memory_(other.memory_)
    , size_(other.size_)
    , is_host_visible_(other.is_host_visible_) {
    other.buffer_ = VK_NULL_HANDLE;
    other.memory_ = VK_NULL_HANDLE;
    other.size_ = 0;
}

BufferManager& BufferManager::operator=(BufferManager&& other) noexcept {
    if (this != &other) {
        cleanup();
        m_device = other.m_device;
        buffer_ = other.buffer_;
        memory_ = other.memory_;
        size_ = other.size_;
        is_host_visible_ = other.is_host_visible_;
        other.buffer_ = VK_NULL_HANDLE;
        other.memory_ = VK_NULL_HANDLE;
        other.size_ = 0;
    }
    return *this;
}

bool BufferManager::createDeviceLocalBuffer(VkDeviceSize size, VkBufferUsageFlags usage) {
    if (isValid()) {
        LOG_WARN("缓冲区已存在，将先释放旧资源");
        cleanup();
    }

    // 设备本地缓冲区，不需要 HOST_VISIBLE
    if (!m_ctx.createDeviceLocalBuffer(size, usage, buffer_, memory_)) {
        LOG_ERROR("创建设备本地缓冲区失败");
        return false;
    }

    size_ = size;
    is_host_visible_ = false;
    LOG_DEBUG("创建设备本地缓冲区: {} 字节, 用途 {}", size, usage);
    return true;
}

bool BufferManager::createStagingBuffer(VkDeviceSize size, VkBufferUsageFlags usage) {
    if (isValid()) {
        cleanup();
    }

    if (!m_ctx.createStagingBuffer(size, usage, buffer_, memory_)) {
        LOG_ERROR("创建暂存缓冲区失败");
        return false;
    }

    size_ = size;
    is_host_visible_ = true;
    LOG_DEBUG("创建暂存缓冲区: {} 字节, 用途 {}", size, usage);
    return true;
}

bool BufferManager::createUniformBuffer(VkDeviceSize size) {
    if (isValid()) {
        cleanup();
    }

    // Uniform 缓冲区通常使用 HOST_VISIBLE | HOST_COHERENT
    if (!m_ctx.createBuffer(size,
                            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                            buffer_,
                            memory_)) {
        LOG_ERROR("创建 Uniform 缓冲区失败");
        return false;
    }

    size_ = size;
    is_host_visible_ = true;
    LOG_DEBUG("创建 Uniform 缓冲区: {} 字节", size);
    return true;
}

bool BufferManager::copyToDevice(const void* data, VkDeviceSize size) {
    if (!isValid() || size > size_) {
        LOG_ERROR("缓冲区无效或数据过大");
        return false;
    }

    if (is_host_visible_) {
        // 如果本身就是 HOST_VISIBLE，直接映射拷贝（回退，但不应使用）
        return copyToBuffer(data, size);
    }

    // 创建临时暂存缓冲区
    BufferManager staging(m_ctx);
    if (!staging.createStagingBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT)) {
        LOG_ERROR("创建暂存缓冲区失败");
        return false;
    }

    // 将数据拷贝到暂存缓冲区
    if (!staging.copyToBuffer(data, size)) {
        LOG_ERROR("拷贝数据到暂存缓冲区失败");
        return false;
    }

    // 执行 GPU 复制：暂存缓冲区 -> 设备本地缓冲区
    m_ctx.copyBuffer(staging.getBuffer(), buffer_, size);
    return true;
}

bool BufferManager::copyFromDevice(void* data, VkDeviceSize size) {
    if (!isValid() || size > size_) {
        LOG_ERROR("缓冲区无效或数据过大");
        return false;
    }

    if (is_host_visible_) {
        return copyFromBuffer(data, size);
    }

    BufferManager staging(m_ctx);
    if (!staging.createStagingBuffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT)) {
        LOG_ERROR("创建暂存缓冲区失败");
        return false;
    }

    // 执行 GPU 复制：设备本地缓冲区 -> 暂存缓冲区
    m_ctx.copyBuffer(buffer_, staging.getBuffer(), size);

    // 从暂存缓冲区映射读取
    return staging.copyFromBuffer(data, size);
}

bool BufferManager::copyToBuffer(const void* data, VkDeviceSize size) {
    if (!isValid() || size > size_ || !is_host_visible_) {
        LOG_ERROR("缓冲区无效、数据过大或非主机可见");
        return false;
    }

    void* mapped = nullptr;
    VkResult result = vkMapMemory(m_device, memory_, 0, size, 0, &mapped);
    if (result != VK_SUCCESS) {
        LOG_ERROR("映射内存失败: {}", result);
        return false;
    }

    memcpy(mapped, data, static_cast<size_t>(size));

    // 如果不使用 HOST_COHERENT，需要手动刷新，但我们创建时已指定 COHERENT
    vkUnmapMemory(m_device, memory_);
    return true;
}

bool BufferManager::copyFromBuffer(void* data, VkDeviceSize size) {
    if (!isValid() || size > size_ || !is_host_visible_) {
        LOG_ERROR("缓冲区无效、数据过大或非主机可见");
        return false;
    }

    void* mapped = nullptr;
    VkResult result = vkMapMemory(m_device, memory_, 0, size, 0, &mapped);
    if (result != VK_SUCCESS) {
        LOG_ERROR("映射内存失败: {}", result);
        return false;
    }

    memcpy(data, mapped, static_cast<size_t>(size));
    vkUnmapMemory(m_device, memory_);
    return true;
}

void BufferManager::release() {
    cleanup();
}

void BufferManager::cleanup() {
    if (memory_ != VK_NULL_HANDLE) {
        vkFreeMemory(m_device, memory_, nullptr);
        memory_ = VK_NULL_HANDLE;
    }
    if (buffer_ != VK_NULL_HANDLE) {
        vkDestroyBuffer(m_device, buffer_, nullptr);
        buffer_ = VK_NULL_HANDLE;
    }
    size_ = 0;
    is_host_visible_ = false;
}

} // namespace vulkan
} // namespace stereo_depth
