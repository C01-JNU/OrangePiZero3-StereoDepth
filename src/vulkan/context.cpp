#include "vulkan/context.hpp"
#include "utils/logger.hpp"
#include <stdexcept>
#include <set>
#include <cstring>
#include <sstream>

namespace stereo_depth {
namespace vulkan {

// 验证层回调函数
static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {
    
    if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        LOG_WARN("Vulkan Validation: {}", pCallbackData->pMessage);
    } else {
        LOG_DEBUG("Vulkan Validation: {}", pCallbackData->pMessage);
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
        // 销毁调试消息
        if (debug_messenger_) {
            auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT");
            if (func != nullptr) {
                func(instance_, debug_messenger_, nullptr);
            }
        }
        
        vkDestroyInstance(instance_, nullptr);
    }
}

bool VulkanContext::initialize(bool enable_validation) {
    LOG_INFO("Initializing Vulkan context (without VMA)...");
    
    // 根据环境判断是否需要禁用验证层（针对Mali-G31）
    #ifdef __arm__
    if (enable_validation) {
        LOG_WARN("Validation layers are disabled on ARM Mali-G31 due to driver limitations");
        enable_validation = false;
    }
    #endif
    
    // 为PanVK驱动设置更保守的初始化
    LOG_INFO("使用保守模式初始化Vulkan（针对PanVK驱动）...");
    
    try {
        LOG_DEBUG("Step 1: Creating Vulkan instance (minimal setup)...");
        if (!createInstance(enable_validation)) {
            LOG_ERROR("Failed to create Vulkan instance");
            return false;
        }
        
        LOG_DEBUG("Step 2: Selecting physical device...");
        if (!selectPhysicalDevice()) {
            LOG_ERROR("Failed to select physical device");
            return false;
        }
        
        LOG_DEBUG("Step 3: Creating logical device (minimal features)...");
        if (!createLogicalDevice()) {
            LOG_ERROR("Failed to create logical device");
            return false;
        }
        
        LOG_INFO("Vulkan context initialized successfully");
        LOG_INFO("Device: {}", getDeviceName());
        LOG_INFO("Vulkan Version: {}", getVulkanVersion());
        
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("Exception during Vulkan initialization: {}", e.what());
        return false;
    } catch (...) {
        LOG_ERROR("Unknown exception during Vulkan initialization");
        return false;
    }
}

bool VulkanContext::createInstance(bool enable_validation) {
    LOG_DEBUG("Creating Vulkan instance with minimal setup...");
    
    // 应用信息 - 使用Vulkan 1.0
    VkApplicationInfo app_info = {};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "OrangePiZero3-StereoDepth";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "StereoDepth";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_0; // 使用1.0兼容Mali-G31
    
    // 实例创建信息 - 最小化设置
    VkInstanceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;
    
    // PanVK可能不需要任何扩展
    create_info.enabledExtensionCount = 0;
    create_info.ppEnabledExtensionNames = nullptr;
    
    // 禁用所有验证层
    create_info.enabledLayerCount = 0;
    create_info.ppEnabledLayerNames = nullptr;
    
    LOG_DEBUG("Creating Vulkan instance with API version 1.0.0, no extensions, no layers");
    
    // 创建实例
    VkResult result = vkCreateInstance(&create_info, nullptr, &instance_);
    if (result != VK_SUCCESS) {
        LOG_ERROR("Failed to create Vulkan instance: {}", result);
        
        // 提供更有用的错误信息
        switch (result) {
            case VK_ERROR_OUT_OF_HOST_MEMORY:
                LOG_ERROR("Out of host memory");
                break;
            case VK_ERROR_OUT_OF_DEVICE_MEMORY:
                LOG_ERROR("Out of device memory");
                break;
            case VK_ERROR_INITIALIZATION_FAILED:
                LOG_ERROR("Initialization failed - Vulkan may not be supported");
                break;
            case VK_ERROR_LAYER_NOT_PRESENT:
                LOG_ERROR("Requested layer not present");
                break;
            case VK_ERROR_EXTENSION_NOT_PRESENT:
                LOG_ERROR("Requested extension not present");
                break;
            case VK_ERROR_INCOMPATIBLE_DRIVER:
                LOG_ERROR("Incompatible driver - may need to update Vulkan drivers");
                break;
            default:
                LOG_ERROR("Unknown error code: {}", result);
                break;
        }
        
        return false;
    }
    
    LOG_DEBUG("Vulkan instance created successfully");
    return true;
}

bool VulkanContext::selectPhysicalDevice() {
    LOG_DEBUG("Selecting physical device...");
    
    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(instance_, &device_count, nullptr);
    
    if (device_count == 0) {
        LOG_ERROR("No Vulkan-capable devices found");
        return false;
    }
    
    std::vector<VkPhysicalDevice> devices(device_count);
    vkEnumeratePhysicalDevices(instance_, &device_count, devices.data());
    
    LOG_DEBUG("Found {} physical device(s)", device_count);
    
    // 首先尝试选择Mali-G31
    for (const auto& device : devices) {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(device, &properties);
        
        LOG_DEBUG("Checking device: {} (Vulkan {}.{}.{})", 
                 properties.deviceName,
                 VK_VERSION_MAJOR(properties.apiVersion),
                 VK_VERSION_MINOR(properties.apiVersion),
                 VK_VERSION_PATCH(properties.apiVersion));
        
        // 优先选择Mali-G31
        if (strstr(properties.deviceName, "Mali-G31") != nullptr) {
            physical_device_ = device;
            LOG_INFO("Selected Mali-G31 GPU");
            return true;
        }
    }
    
    // 如果没有Mali-G31，尝试选择llvmpipe（软件渲染）
    for (const auto& device : devices) {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(device, &properties);
        
        if (strstr(properties.deviceName, "llvmpipe") != nullptr) {
            physical_device_ = device;
            LOG_INFO("Selected llvmpipe (software renderer) as fallback");
            return true;
        }
    }
    
    // 最后选择第一个设备
    if (!devices.empty()) {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(devices[0], &properties);
        physical_device_ = devices[0];
        LOG_INFO("Selected first available device: {}", properties.deviceName);
        return true;
    }
    
    LOG_ERROR("No suitable physical device found");
    return false;
}

bool VulkanContext::createLogicalDevice() {
    LOG_DEBUG("Creating logical device with minimal features...");
    
    compute_queue_family_index_ = findComputeQueueFamily();
    if (compute_queue_family_index_ == static_cast<uint32_t>(-1)) {
        LOG_ERROR("No compute queue family found");
        return false;
    }
    
    // 队列创建信息
    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_create_info = {};
    queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_info.queueFamilyIndex = compute_queue_family_index_;
    queue_create_info.queueCount = 1;
    queue_create_info.pQueuePriorities = &queue_priority;
    
    // 设备特性 - 最小化
    VkPhysicalDeviceFeatures device_features = {};
    
    // 对于PanVK，可能不需要任何扩展
    std::vector<const char*> device_extensions;
    
    LOG_DEBUG("Creating logical device with {} extensions", device_extensions.size());
    
    // 创建设备
    VkDeviceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    create_info.queueCreateInfoCount = 1;
    create_info.pQueueCreateInfos = &queue_create_info;
    create_info.pEnabledFeatures = &device_features;
    create_info.enabledExtensionCount = static_cast<uint32_t>(device_extensions.size());
    create_info.ppEnabledExtensionNames = device_extensions.empty() ? nullptr : device_extensions.data();
    
    VkResult result = vkCreateDevice(physical_device_, &create_info, nullptr, &device_);
    if (result != VK_SUCCESS) {
        LOG_ERROR("Failed to create logical device: {}", result);
        return false;
    }
    
    // 获取计算队列
    vkGetDeviceQueue(device_, compute_queue_family_index_, 0, &compute_queue_);
    
    LOG_DEBUG("Logical device created successfully");
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
        if (!found) {
            return false;
        }
    }
    
    return true;
}

void VulkanContext::setupDebugMessenger(bool enable_validation) {
    // PanVK不支持调试消息，跳过
    LOG_DEBUG("Debug messenger disabled for PanVK");
}

uint32_t VulkanContext::findComputeQueueFamily() const {
    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_family_count, nullptr);
    
    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_family_count, queue_families.data());
    
    for (uint32_t i = 0; i < queue_family_count; i++) {
        if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            LOG_DEBUG("Found compute queue family at index {} with {} queues", 
                     i, queue_families[i].queueCount);
            return i;
        }
    }
    
    return static_cast<uint32_t>(-1);
}

bool VulkanContext::isDeviceSuitable(VkPhysicalDevice device) const {
    // 简化检查：只要有计算队列就行
    // 保存当前设备以便findComputeQueueFamily使用
    const_cast<VulkanContext*>(this)->physical_device_ = device;
    return findComputeQueueFamily() != static_cast<uint32_t>(-1);
}

bool VulkanContext::checkDeviceExtensionSupport(const std::vector<const char*>& extensions) const {
    // 对于PanVK，我们可能不需要检查扩展，因为很多可能不支持
    return true; // 简化处理
}

VkCommandPool VulkanContext::createCommandPool(VkCommandPoolCreateFlags flags) const {
    if (!device_) {
        LOG_ERROR("Device not initialized");
        return VK_NULL_HANDLE;
    }
    
    VkCommandPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = compute_queue_family_index_;
    pool_info.flags = flags;
    
    VkCommandPool command_pool;
    VkResult result = vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool);
    
    if (result != VK_SUCCESS) {
        LOG_ERROR("Failed to create command pool: {}", result);
        return VK_NULL_HANDLE;
    }
    
    return command_pool;
}

VkCommandBuffer VulkanContext::createCommandBuffer(VkCommandPool pool) const {
    if (!device_ || !pool) {
        LOG_ERROR("Device or command pool not initialized");
        return VK_NULL_HANDLE;
    }
    
    VkCommandBufferAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;
    
    VkCommandBuffer command_buffer;
    VkResult result = vkAllocateCommandBuffers(device_, &alloc_info, &command_buffer);
    
    if (result != VK_SUCCESS) {
        LOG_ERROR("Failed to allocate command buffers: {}", result);
        return VK_NULL_HANDLE;
    }
    
    return command_buffer;
}

void VulkanContext::waitIdle() const {
    if (device_) {
        vkDeviceWaitIdle(device_);
    }
}

VkPhysicalDeviceProperties VulkanContext::getPhysicalDeviceProperties() const {
    VkPhysicalDeviceProperties properties;
    if (physical_device_) {
        vkGetPhysicalDeviceProperties(physical_device_, &properties);
    }
    return properties;
}

std::string VulkanContext::getDeviceName() const {
    if (!physical_device_) return "Unknown";
    
    auto properties = getPhysicalDeviceProperties();
    return properties.deviceName;
}

std::string VulkanContext::getVulkanVersion() const {
    if (!physical_device_) return "Unknown";
    
    auto properties = getPhysicalDeviceProperties();
    return std::to_string(VK_VERSION_MAJOR(properties.apiVersion)) + "." +
           std::to_string(VK_VERSION_MINOR(properties.apiVersion)) + "." +
           std::to_string(VK_VERSION_PATCH(properties.apiVersion));
}

uint32_t VulkanContext::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) const {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physical_device_, &memProperties);
    
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && 
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    
    throw std::runtime_error("failed to find suitable memory type!");
}

bool VulkanContext::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, 
                                VkBuffer& buffer, VkDeviceMemory& bufferMemory) const {
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    if (vkCreateBuffer(device_, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        LOG_ERROR("Failed to create buffer");
        return false;
    }
    
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device_, buffer, &memRequirements);
    
    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);
    
    if (vkAllocateMemory(device_, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
        vkDestroyBuffer(device_, buffer, nullptr);
        LOG_ERROR("Failed to allocate buffer memory");
        return false;
    }
    
    vkBindBufferMemory(device_, buffer, bufferMemory, 0);
    return true;
}

void VulkanContext::copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) const {
    VkCommandPool commandPool = createCommandPool();
    if (!commandPool) {
        LOG_ERROR("Failed to create command pool for buffer copy");
        return;
    }
    
    VkCommandBuffer commandBuffer = createCommandBuffer(commandPool);
    if (!commandBuffer) {
        LOG_ERROR("Failed to create command buffer for buffer copy");
        vkDestroyCommandPool(device_, commandPool, nullptr);
        return;
    }
    
    // 开始命令缓冲区
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    
    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    
    // 复制缓冲区
    VkBufferCopy copyRegion = {};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = 0;
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);
    
    vkEndCommandBuffer(commandBuffer);
    
    // 提交命令缓冲区
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    
    vkQueueSubmit(compute_queue_, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(compute_queue_);
    
    // 清理
    vkFreeCommandBuffers(device_, commandPool, 1, &commandBuffer);
    vkDestroyCommandPool(device_, commandPool, nullptr);
}

} // namespace vulkan
} // namespace stereo_depth
