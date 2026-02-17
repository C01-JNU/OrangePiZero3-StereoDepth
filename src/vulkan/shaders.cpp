#include "vulkan/shaders.hpp"
#include "utils/logger.hpp"
#include <fstream>
#include <filesystem>
#include <cstring>

namespace stereo_depth {
namespace vulkan {

bool ShaderManager::loadShader(const std::string& name, const std::string& filePath) {
    std::ifstream file(filePath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        LOG_ERROR("Failed to open shader file: {}", filePath);
        return false;
    }
    
    size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));
    
    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
    file.close();
    
    return loadShader(name, buffer.data(), fileSize);
}

bool ShaderManager::loadShader(const std::string& name, const uint32_t* data, size_t size) {
    if (size == 0 || size % 4 != 0) {
        LOG_ERROR("Invalid shader size for {}: {} bytes", name, size);
        return false;
    }
    
    ShaderData shaderData;
    shaderData.size = size;
    shaderData.code.resize(size / sizeof(uint32_t));
    std::memcpy(shaderData.code.data(), data, size);
    
    m_shaders[name] = std::move(shaderData);
    LOG_DEBUG("Loaded shader '{}': {} bytes", name, size);
    
    return true;
}

const uint32_t* ShaderManager::getShaderCode(const std::string& name) const {
    auto it = m_shaders.find(name);
    if (it == m_shaders.end()) {
        return nullptr;
    }
    return it->second.code.data();
}

size_t ShaderManager::getShaderSize(const std::string& name) const {
    auto it = m_shaders.find(name);
    if (it == m_shaders.end()) {
        return 0;
    }
    return it->second.size;
}

bool ShaderManager::hasShader(const std::string& name) const {
    return m_shaders.find(name) != m_shaders.end();
}

void ShaderManager::clear() {
    m_shaders.clear();
    LOG_DEBUG("Cleared all shaders");
}

} // namespace vulkan
} // namespace stereo_depth
