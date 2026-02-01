#pragma once

#include <vulkan/vulkan.h>
#include <memory>
#include <string>
#include <vector>
#include <functional>

namespace stereo_depth {
namespace vulkan {

// Vulkan上下文类，使用Vulkan C API，暂时不使用VMA
class VulkanContext {
public:
    VulkanContext();
    ~VulkanContext();
    
    // 禁止拷贝
    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;
    
    // 初始化Vulkan环境
    bool initialize(bool enable_validation = false);
    
    // 获取Vulkan对象
    VkInstance getInstance() const { return instance_; }
    VkPhysicalDevice getPhysicalDevice() const { return physical_device_; }
    VkDevice getDevice() const { return device_; }
    VkQueue getComputeQueue() const { return compute_queue_; }
    uint32_t getComputeQueueFamilyIndex() const { return compute_queue_family_index_; }
    
    // 工具函数
    VkCommandPool createCommandPool(VkCommandPoolCreateFlags flags = 0) const;
    VkCommandBuffer createCommandBuffer(VkCommandPool pool) const;
    
    // 等待设备空闲
    void waitIdle() const;
    
    // 获取设备信息
    std::string getDeviceName() const;
    std::string getVulkanVersion() const;
    
    // 检查扩展支持
    bool checkDeviceExtensionSupport(const std::vector<const char*>& extensions) const;
    
    // 创建缓冲区（不使用VMA的简单实现）
    bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, 
                      VkBuffer& buffer, VkDeviceMemory& bufferMemory) const;
    
    // 复制缓冲区
    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) const;
    
private:
    // Vulkan实例
    VkInstance instance_;
    VkDebugUtilsMessengerEXT debug_messenger_;
    
    // 物理设备
    VkPhysicalDevice physical_device_;
    
    // 逻辑设备
    VkDevice device_;
    
    // 队列
    VkQueue compute_queue_;
    uint32_t compute_queue_family_index_;
    
    // 初始化步骤
    bool createInstance(bool enable_validation);
    bool selectPhysicalDevice();
    bool createLogicalDevice();
    
    // 验证层支持
    bool checkValidationLayerSupport(const std::vector<const char*>& layers);
    void setupDebugMessenger(bool enable_validation);
    
    // 工具函数
    uint32_t findComputeQueueFamily() const;
    bool isDeviceSuitable(VkPhysicalDevice device) const;
    
    // 获取物理设备属性
    VkPhysicalDeviceProperties getPhysicalDeviceProperties() const;
    
    // 内存辅助函数
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) const;
};

} // namespace vulkan
} // namespace stereo_depth
