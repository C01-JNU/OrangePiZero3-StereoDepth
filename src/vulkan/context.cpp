#include "vulkan/context.hpp"
#include "utils/logger.hpp"
#include <stdexcept>
#include <set>
#include <cstring>
#include <sstream>

namespace stereo_depth {
namespace vulkan {

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {
    
    if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        LOG_WARN("Vulkan验证层: {}", pCallbackData->pMessage);
    } else {
        LOG_DEBUG("Vulkan验证层: {}", pCallbackData->pMessage);
    }
    return VK_FALSE;
}

VulkanContext::VulkanContext() 
    : instance_(VK_NULL_HANDLE)
    , debug_messenger_(VK_NULL_HANDLE)
    , physical_device_(VK_NULL_HANDLE)
    , device_(VK_NULL_HANDLE)
    , compute_queue_(VK_NULL_HANDLE)
    , compute_queue_family_index_(0) {
}

VulkanContext::~VulkanContext() {
    if (device_) {
        vkDestroyDevice(device_, nullptr);
    }
    if (instance_) {
        if (debug_messenger_) {
            auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT");
            if (func) func(instance_, debug_messenger_, nullptr);
        }
        vkDestroyInstance(instance_, nullptr);
    }
}

bool VulkanContext::initialize(bool enable_validation) {
    LOG_INFO("正在初始化Vulkan上下文...");
    
    #ifdef __arm__
    if (enable_validation) {
        LOG_WARN("由于ARM Mali-G31驱动限制，验证层已禁用");
        enable_validation = false;
    }
    #endif
    
    try {
        if (!createInstance(enable_validation)) return false;
        if (!selectPhysicalDevice()) return false;
        if (!createLogicalDevice()) return false;
        
        LOG_INFO("Vulkan上下文初始化成功");
        LOG_INFO("设备: {}", getDeviceName());
        LOG_INFO("Vulkan版本: {}", getVulkanVersion());
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("Vulkan初始化异常: {}", e.what());
        return false;
    }
}

bool VulkanContext::createInstance(bool enable_validation) {
    VkApplicationInfo app_info = {};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "OrangePiZero3-StereoDepth";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "StereoDepth";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_0;
    
    VkInstanceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;
    create_info.enabledExtensionCount = 0;
    create_info.ppEnabledExtensionNames = nullptr;
    create_info.enabledLayerCount = 0;
    create_info.ppEnabledLayerNames = nullptr;
    
    VkResult result = vkCreateInstance(&create_info, nullptr, &instance_);
    if (result != VK_SUCCESS) {
        LOG_ERROR("创建Vulkan实例失败: {}", result);
        return false;
    }
    LOG_DEBUG("Vulkan实例创建成功");
    return true;
}

bool VulkanContext::selectPhysicalDevice() {
    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(instance_, &device_count, nullptr);
    if (device_count == 0) {
        LOG_ERROR("未找到支持Vulkan的设备");
        return false;
    }
    
    std::vector<VkPhysicalDevice> devices(device_count);
    vkEnumeratePhysicalDevices(instance_, &device_count, devices.data());
    
    for (const auto& device : devices) {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(device, &properties);
        LOG_DEBUG("检查设备: {} (Vulkan {}.{}.{})", 
                 properties.deviceName,
                 VK_VERSION_MAJOR(properties.apiVersion),
                 VK_VERSION_MINOR(properties.apiVersion),
                 VK_VERSION_PATCH(properties.apiVersion));
        
        if (strstr(properties.deviceName, "Mali-G31") != nullptr) {
            physical_device_ = device;
            LOG_INFO("选择Mali-G31 GPU");
            return true;
        }
    }
    
    for (const auto& device : devices) {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(device, &properties);
        if (strstr(properties.deviceName, "llvmpipe") != nullptr) {
            physical_device_ = device;
            LOG_INFO("选择llvmpipe（软件渲染器）作为回退方案");
            return true;
        }
    }
    
    if (!devices.empty()) {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(devices[0], &properties);
        physical_device_ = devices[0];
        LOG_INFO("选择第一个可用设备: {}", properties.deviceName);
        return true;
    }
    
    LOG_ERROR("未找到合适的物理设备");
    return false;
}

bool VulkanContext::createLogicalDevice() {
    compute_queue_family_index_ = findComputeQueueFamily();
    if (compute_queue_family_index_ == static_cast<uint32_t>(-1)) {
        LOG_ERROR("未找到计算队列族");
        return false;
    }
    
    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_create_info = {};
    queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_info.queueFamilyIndex = compute_queue_family_index_;
    queue_create_info.queueCount = 1;
    queue_create_info.pQueuePriorities = &queue_priority;
    
    VkPhysicalDeviceFeatures enabledFeatures = {};
    
    VkDeviceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    create_info.queueCreateInfoCount = 1;
    create_info.pQueueCreateInfos = &queue_create_info;
    create_info.pEnabledFeatures = &enabledFeatures;
    create_info.enabledExtensionCount = 0;
    create_info.ppEnabledExtensionNames = nullptr;
    
    VkResult result = vkCreateDevice(physical_device_, &create_info, nullptr, &device_);
    if (result != VK_SUCCESS) {
        LOG_ERROR("创建逻辑设备失败: {}", result);
        return false;
    }
    
    vkGetDeviceQueue(device_, compute_queue_family_index_, 0, &compute_queue_);
    LOG_DEBUG("逻辑设备创建成功");
    return true;
}

bool VulkanContext::checkValidationLayerSupport(const std::vector<const char*>& layers) {
    uint32_t layer_count;
    vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
    std::vector<VkLayerProperties> available_layers(layer_count);
    vkEnumerateInstanceLayerProperties(&layer_count, available_layers.data());
    for (const char* layer_name : layers) {
        bool found = false;
        for (const auto& layer : available_layers) {
            if (strcmp(layer_name, layer.layerName) == 0) {
                found = true;
                break;
            }
        }
        if (!found) return false;
    }
    return true;
}

void VulkanContext::setupDebugMessenger(bool enable_validation) {
    // PanVK 不支持，留空
}

uint32_t VulkanContext::findComputeQueueFamily() const {
    if (!physical_device_) return static_cast<uint32_t>(-1);
    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_family_count, nullptr);
    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_family_count, queue_families.data());
    for (uint32_t i = 0; i < queue_family_count; i++) {
        if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            return i;
        }
    }
    return static_cast<uint32_t>(-1);
}

bool VulkanContext::isDeviceSuitable(VkPhysicalDevice device) const {
    const_cast<VulkanContext*>(this)->physical_device_ = device;
    return findComputeQueueFamily() != static_cast<uint32_t>(-1);
}

bool VulkanContext::checkDeviceExtensionSupport(const std::vector<const char*>& extensions) const {
    return true;
}

VkCommandPool VulkanContext::createCommandPool(VkCommandPoolCreateFlags flags) const {
    if (!device_) return VK_NULL_HANDLE;
    VkCommandPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = compute_queue_family_index_;
    pool_info.flags = flags;
    VkCommandPool command_pool;
    VkResult result = vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool);
    if (result != VK_SUCCESS) {
        LOG_ERROR("创建命令池失败: {}", result);
        return VK_NULL_HANDLE;
    }
    return command_pool;
}

VkCommandBuffer VulkanContext::createCommandBuffer(VkCommandPool pool) const {
    if (!device_ || !pool) return VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;
    VkCommandBuffer command_buffer;
    VkResult result = vkAllocateCommandBuffers(device_, &alloc_info, &command_buffer);
    if (result != VK_SUCCESS) {
        LOG_ERROR("分配命令缓冲区失败: {}", result);
        return VK_NULL_HANDLE;
    }
    return command_buffer;
}

void VulkanContext::waitIdle() const {
    if (device_) vkDeviceWaitIdle(device_);
}

VkPhysicalDeviceProperties VulkanContext::getPhysicalDeviceProperties() const {
    VkPhysicalDeviceProperties properties;
    if (physical_device_) vkGetPhysicalDeviceProperties(physical_device_, &properties);
    return properties;
}

std::string VulkanContext::getDeviceName() const {
    if (!physical_device_) return "未知设备";
    auto properties = getPhysicalDeviceProperties();
    return properties.deviceName;
}

std::string VulkanContext::getVulkanVersion() const {
    if (!physical_device_) return "未知版本";
    auto properties = getPhysicalDeviceProperties();
    return std::to_string(VK_VERSION_MAJOR(properties.apiVersion)) + "." +
           std::to_string(VK_VERSION_MINOR(properties.apiVersion)) + "." +
           std::to_string(VK_VERSION_PATCH(properties.apiVersion));
}

uint32_t VulkanContext::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) const {
    if (!physical_device_) throw std::runtime_error("物理设备未初始化");
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physical_device_, &memProperties);
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && 
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("无法找到合适的内存类型");
}

bool VulkanContext::createBuffer(VkDeviceSize size,
                                 VkBufferUsageFlags usage,
                                 VkMemoryPropertyFlags properties,
                                 VkBuffer& buffer,
                                 VkDeviceMemory& memory) const {
    if (!device_) {
        LOG_ERROR("设备未初始化");
        return false;
    }
    
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    if (vkCreateBuffer(device_, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        LOG_ERROR("创建缓冲区失败");
        return false;
    }
    
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device_, buffer, &memRequirements);
    
    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    
    try {
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);
    } catch (const std::exception& e) {
        vkDestroyBuffer(device_, buffer, nullptr);
        LOG_ERROR("查找内存类型失败: {}", e.what());
        return false;
    }
    
    if (vkAllocateMemory(device_, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
        vkDestroyBuffer(device_, buffer, nullptr);
        LOG_ERROR("分配缓冲区内存失败");
        return false;
    }
    
    if (vkBindBufferMemory(device_, buffer, memory, 0) != VK_SUCCESS) {
        vkDestroyBuffer(device_, buffer, nullptr);
        vkFreeMemory(device_, memory, nullptr);
        LOG_ERROR("绑定缓冲区内存失败");
        return false;
    }
    
    return true;
}

void VulkanContext::copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size,
                               VkCommandPool pool) const {
    if (!device_ || !srcBuffer || !dstBuffer) {
        LOG_ERROR("设备或缓冲区无效，无法复制缓冲区");
        return;
    }
    
    bool need_clean_pool = false;
    if (pool == VK_NULL_HANDLE) {
        pool = createCommandPool();
        need_clean_pool = true;
    }
    if (!pool) {
        LOG_ERROR("创建命令池失败");
        return;
    }
    
    VkCommandBuffer commandBuffer = createCommandBuffer(pool);
    if (!commandBuffer) {
        if (need_clean_pool) vkDestroyCommandPool(device_, pool, nullptr);
        return;
    }
    
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    
    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        LOG_ERROR("开始命令缓冲区失败");
        if (need_clean_pool) vkDestroyCommandPool(device_, pool, nullptr);
        return;
    }
    
    VkBufferCopy copyRegion = {};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = 0;
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);
    
    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        LOG_ERROR("结束命令缓冲区失败");
        if (need_clean_pool) vkDestroyCommandPool(device_, pool, nullptr);
        return;
    }
    
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    
    if (vkQueueSubmit(compute_queue_, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
        LOG_ERROR("提交命令缓冲区失败");
        if (need_clean_pool) vkDestroyCommandPool(device_, pool, nullptr);
        return;
    }
    
    vkQueueWaitIdle(compute_queue_);
    
    vkFreeCommandBuffers(device_, pool, 1, &commandBuffer);
    if (need_clean_pool) vkDestroyCommandPool(device_, pool, nullptr);
}

} // namespace vulkan
} // namespace stereo_depth
