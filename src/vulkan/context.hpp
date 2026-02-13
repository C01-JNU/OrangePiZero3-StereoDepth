#pragma once

#include <vulkan/vulkan.h>
#include <memory>
#include <string>
#include <vector>
#include <functional>

namespace stereo_depth {
namespace vulkan {

class VulkanContext {
public:
    VulkanContext();
    ~VulkanContext();
    
    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;
    
    bool initialize(bool enable_validation = false);
    
    // 获取 Vulkan 对象
    VkInstance getInstance() const { return instance_; }
    VkPhysicalDevice getPhysicalDevice() const { return physical_device_; }
    VkDevice getDevice() const { return device_; }
    VkQueue getComputeQueue() const { return compute_queue_; }
    uint32_t getComputeQueueFamilyIndex() const { return compute_queue_family_index_; }
    
    // 工具函数
    VkCommandPool createCommandPool(VkCommandPoolCreateFlags flags = 0) const;
    VkCommandBuffer createCommandBuffer(VkCommandPool pool) const;
    
    void waitIdle() const;
    
    std::string getDeviceName() const;
    std::string getVulkanVersion() const;
    
    bool checkDeviceExtensionSupport(const std::vector<const char*>& extensions) const;
    
    // ---------- 缓冲区创建接口（内存类型显式指定）----------
    // 通用缓冲区创建（需指定内存属性）
    bool createBuffer(VkDeviceSize size,
                      VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags properties,
                      VkBuffer& buffer,
                      VkDeviceMemory& memory) const;
    
    // 设备本地缓冲区（GPU 专用，高性能）
    inline bool createDeviceLocalBuffer(VkDeviceSize size,
                                        VkBufferUsageFlags usage,
                                        VkBuffer& buffer,
                                        VkDeviceMemory& memory) const {
        return createBuffer(size, usage,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            buffer, memory);
    }
    
    // 主机可见缓冲区（用于暂存上传/下载）
    inline bool createStagingBuffer(VkDeviceSize size,
                                    VkBufferUsageFlags usage,
                                    VkBuffer& buffer,
                                    VkDeviceMemory& memory) const {
        return createBuffer(size,
            usage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            buffer, memory);
    }
    
    // 复制缓冲区（设备本地 ↔ 设备本地 或 暂存 ↔ 设备本地）
    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size,
                    VkCommandPool pool = VK_NULL_HANDLE) const;
    
private:
    VkInstance instance_;
    VkDebugUtilsMessengerEXT debug_messenger_;
    VkPhysicalDevice physical_device_;
    VkDevice device_;
    VkQueue compute_queue_;
    uint32_t compute_queue_family_index_;
    
    bool createInstance(bool enable_validation);
    bool selectPhysicalDevice();
    bool createLogicalDevice();
    bool checkValidationLayerSupport(const std::vector<const char*>& layers);
    void setupDebugMessenger(bool enable_validation);
    uint32_t findComputeQueueFamily() const;
    bool isDeviceSuitable(VkPhysicalDevice device) const;
    VkPhysicalDeviceProperties getPhysicalDeviceProperties() const;
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) const;
};

} // namespace vulkan
} // namespace stereo_depth
