#include "vulkan/buffer_manager.hpp"
#include "utils/logger.hpp"
#include <cstring>
#include <stdexcept>

namespace stereo_depth {
namespace vulkan {

// BufferManager实现
BufferManager::BufferManager(const VulkanContext& context)
    : m_context(context) {
}

BufferManager::~BufferManager() {
    cleanup();
}

BufferManager::BufferManager(BufferManager&& other) noexcept
    : m_context(other.m_context)
    , m_buffer(other.m_buffer)
    , m_memory(other.m_memory)
    , m_size(other.m_size)
    , m_usage(other.m_usage)
    , m_memoryProperties(other.m_memoryProperties)
    , m_mapped(other.m_mapped) {
    other.m_buffer = VK_NULL_HANDLE;
    other.m_memory = VK_NULL_HANDLE;
    other.m_mapped = nullptr;
    other.m_size = 0;
}

BufferManager& BufferManager::operator=(BufferManager&& other) noexcept {
    if (this != &other) {
        cleanup();
        m_buffer = other.m_buffer;
        m_memory = other.m_memory;
        m_size = other.m_size;
        m_usage = other.m_usage;
        m_memoryProperties = other.m_memoryProperties;
        m_mapped = other.m_mapped;
        
        other.m_buffer = VK_NULL_HANDLE;
        other.m_memory = VK_NULL_HANDLE;
        other.m_mapped = nullptr;
        other.m_size = 0;
    }
    return *this;
}

bool BufferManager::createStorageBuffer(VkDeviceSize size, VkBufferUsageFlags usage) {
    if (isValid()) {
        LOG_WARN("Buffer already exists, cleaning up first");
        cleanup();
    }
    
    m_size = size;
    m_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | usage;
    m_memoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | 
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    
    if (!m_context.createBuffer(m_size, m_usage, m_memoryProperties, m_buffer, m_memory)) {
        LOG_ERROR("Failed to create storage buffer of size {}", m_size);
        cleanup();
        return false;
    }
    
    LOG_DEBUG("Created storage buffer: size={} bytes, usage={:#x}", 
              m_size, m_usage);
    return true;
}

bool BufferManager::createUniformBuffer(VkDeviceSize size) {
    if (isValid()) {
        LOG_WARN("Buffer already exists, cleaning up first");
        cleanup();
    }
    
    m_size = size;
    m_usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    m_memoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | 
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    
    if (!m_context.createBuffer(m_size, m_usage, m_memoryProperties, m_buffer, m_memory)) {
        LOG_ERROR("Failed to create uniform buffer of size {}", m_size);
        cleanup();
        return false;
    }
    
    LOG_DEBUG("Created uniform buffer: size={} bytes", m_size);
    return true;
}

bool BufferManager::createStagingBuffer(VkDeviceSize size) {
    if (isValid()) {
        LOG_WARN("Buffer already exists, cleaning up first");
        cleanup();
    }
    
    m_size = size;
    m_usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    m_memoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | 
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    
    if (!m_context.createBuffer(m_size, m_usage, m_memoryProperties, m_buffer, m_memory)) {
        LOG_ERROR("Failed to create staging buffer of size {}", m_size);
        cleanup();
        return false;
    }
    
    LOG_DEBUG("Created staging buffer: size={} bytes", m_size);
    return true;
}

void* BufferManager::map() {
    if (!isValid()) {
        LOG_ERROR("Cannot map invalid buffer");
        return nullptr;
    }
    
    if (m_mapped != nullptr) {
        LOG_WARN("Buffer already mapped");
        return m_mapped;
    }
    
    VkDevice device = m_context.getDevice();
    VkResult result = vkMapMemory(device, m_memory, 0, m_size, 0, &m_mapped);
    
    if (result != VK_SUCCESS) {
        LOG_ERROR("Failed to map buffer memory: {}", result);
        m_mapped = nullptr;
        return nullptr;
    }
    
    return m_mapped;
}

void BufferManager::unmap() {
    if (!isValid() || m_mapped == nullptr) {
        return;
    }
    
    VkDevice device = m_context.getDevice();
    vkUnmapMemory(device, m_memory);
    m_mapped = nullptr;
}

bool BufferManager::copyToBuffer(const void* data, VkDeviceSize size, VkDeviceSize offset) {
    if (!isValid()) {
        LOG_ERROR("Cannot copy to invalid buffer");
        return false;
    }
    
    if (size == 0) {
        size = m_size;
    }
    
    if (offset + size > m_size) {
        LOG_ERROR("Copy exceeds buffer bounds: offset={}, size={}, buffer size={}", 
                  offset, size, m_size);
        return false;
    }
    
    void* mapped = map();
    if (mapped == nullptr) {
        return false;
    }
    
    std::memcpy(static_cast<char*>(mapped) + offset, data, size);
    unmap();
    
    return true;
}

bool BufferManager::copyFromBuffer(void* data, VkDeviceSize size, VkDeviceSize offset) {
    if (!isValid()) {
        LOG_ERROR("Cannot copy from invalid buffer");
        return false;
    }
    
    if (size == 0) {
        size = m_size;
    }
    
    if (offset + size > m_size) {
        LOG_ERROR("Copy exceeds buffer bounds: offset={}, size={}, buffer size={}", 
                  offset, size, m_size);
        return false;
    }
    
    void* mapped = map();
    if (mapped == nullptr) {
        return false;
    }
    
    std::memcpy(data, static_cast<char*>(mapped) + offset, size);
    unmap();
    
    return true;
}

void BufferManager::clear(VkDeviceSize size) {
    if (!isValid()) {
        return;
    }
    
    if (size == 0) {
        size = m_size;
    }
    
    void* mapped = map();
    if (mapped) {
        std::memset(mapped, 0, size);
        unmap();
    }
}

void BufferManager::cleanup() {
    if (!isValid()) {
        return;
    }
    
    VkDevice device = m_context.getDevice();
    
    if (m_mapped != nullptr) {
        unmap();
    }
    
    if (m_buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, m_buffer, nullptr);
        m_buffer = VK_NULL_HANDLE;
    }
    
    if (m_memory != VK_NULL_HANDLE) {
        vkFreeMemory(device, m_memory, nullptr);
        m_memory = VK_NULL_HANDLE;
    }
    
    m_size = 0;
    m_usage = 0;
    m_memoryProperties = 0;
}

// DescriptorSetLayoutBuilder实现
DescriptorSetLayoutBuilder::DescriptorSetLayoutBuilder(const VulkanContext& context)
    : m_context(context) {
}

DescriptorSetLayoutBuilder::~DescriptorSetLayoutBuilder() = default;

DescriptorSetLayoutBuilder& DescriptorSetLayoutBuilder::addStorageBuffer(
    uint32_t binding, 
    uint32_t count,
    VkShaderStageFlags stages) {
    
    VkDescriptorSetLayoutBinding layoutBinding = {};
    layoutBinding.binding = binding;
    layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    layoutBinding.descriptorCount = count;
    layoutBinding.stageFlags = stages;
    layoutBinding.pImmutableSamplers = nullptr;
    
    m_bindings.push_back(layoutBinding);
    return *this;
}

DescriptorSetLayoutBuilder& DescriptorSetLayoutBuilder::addUniformBuffer(
    uint32_t binding, 
    uint32_t count,
    VkShaderStageFlags stages) {
    
    VkDescriptorSetLayoutBinding layoutBinding = {};
    layoutBinding.binding = binding;
    layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    layoutBinding.descriptorCount = count;
    layoutBinding.stageFlags = stages;
    layoutBinding.pImmutableSamplers = nullptr;
    
    m_bindings.push_back(layoutBinding);
    return *this;
}

VkDescriptorSetLayout DescriptorSetLayoutBuilder::build() {
    if (m_bindings.empty()) {
        LOG_WARN("No bindings specified for descriptor set layout");
        return VK_NULL_HANDLE;
    }
    
    VkDevice device = m_context.getDevice();
    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(m_bindings.size());
    layoutInfo.pBindings = m_bindings.data();
    
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    VkResult result = vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &layout);
    
    if (result != VK_SUCCESS) {
        LOG_ERROR("Failed to create descriptor set layout: {}", result);
        return VK_NULL_HANDLE;
    }
    
    LOG_DEBUG("Created descriptor set layout with {} bindings", m_bindings.size());
    return layout;
}

} // namespace vulkan
} // namespace stereo_depth
