#include "utils/config.hpp"
#include "utils/logger.hpp"
#include <fstream>
#include <sstream>
#include <type_traits>
#include <filesystem>
#include <algorithm>

namespace stereo_depth {
namespace utils {

bool Config::loadFromFile(const std::string& filepath) {
    try {
        // 检查文件是否存在
        if (!std::filesystem::exists(filepath)) {
            LOG_ERROR("Config file does not exist: {}", filepath);
            return false;
        }
        
        yaml_root_ = YAML::LoadFile(filepath);
        config_map_.clear();
        parseYamlNode(yaml_root_);
        LOG_INFO("Loaded config from: {}", filepath);
        return true;
    } catch (const YAML::Exception& e) {
        LOG_ERROR("Failed to load config file {}: {}", filepath, e.what());
        return false;
    } catch (const std::exception& e) {
        LOG_ERROR("Error loading config file {}: {}", filepath, e.what());
        return false;
    }
}

bool Config::loadFromString(const std::string& yaml_str) {
    try {
        yaml_root_ = YAML::Load(yaml_str);
        config_map_.clear();
        parseYamlNode(yaml_root_);
        LOG_INFO("Loaded config from string");
        return true;
    } catch (const YAML::Exception& e) {
        LOG_ERROR("Failed to load config from string: {}", e.what());
        return false;
    }
}

bool Config::saveToFile(const std::string& filepath) {
    try {
        // 创建目录（如果不存在）
        std::filesystem::path path(filepath);
        std::filesystem::create_directories(path.parent_path());
        
        std::ofstream fout(filepath);
        if (!fout.is_open()) {
            LOG_ERROR("Failed to open file for writing: {}", filepath);
            return false;
        }
        
        // 这里需要将config_map_转换回YAML
        // 简化实现：直接保存原始的YAML节点
        fout << yaml_root_;
        fout.close();
        
        LOG_INFO("Saved config to: {}", filepath);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to save config to file {}: {}", filepath, e.what());
        return false;
    }
}

bool Config::has(const std::string& key) const {
    return config_map_.find(key) != config_map_.end();
}

std::vector<std::string> Config::getKeys() const {
    std::vector<std::string> keys;
    keys.reserve(config_map_.size());
    
    for (const auto& pair : config_map_) {
        keys.push_back(pair.first);
    }
    
    return keys;
}

void Config::merge(const Config& other) {
    for (const auto& pair : other.config_map_) {
        config_map_[pair.first] = pair.second;
    }
}

void Config::parseYamlNode(const YAML::Node& node, const std::string& prefix) {
    if (node.IsMap()) {
        for (const auto& pair : node) {
            std::string key = pair.first.as<std::string>();
            std::string full_key = prefix.empty() ? key : prefix + "." + key;
            
            if (pair.second.IsScalar()) {
                setFromYaml(full_key, pair.second);
            } else if (pair.second.IsSequence() || pair.second.IsMap()) {
                parseYamlNode(pair.second, full_key);
            }
        }
    } else if (node.IsSequence()) {
        // 序列处理：将数组展开为带索引的键
        for (size_t i = 0; i < node.size(); ++i) {
            std::string full_key = prefix + "[" + std::to_string(i) + "]";
            if (node[i].IsScalar()) {
                setFromYaml(full_key, node[i]);
            } else {
                parseYamlNode(node[i], full_key);
            }
        }
    } else if (node.IsScalar()) {
        setFromYaml(prefix, node);
    }
}

void Config::setFromYaml(const std::string& key, const YAML::Node& node) {
    try {
        if (node.IsNull()) {
            return;
        }
        
        // 尝试解析为不同类型
        try {
            int int_val = node.as<int>();
            // 对于非负整数，同时存储为int和unsigned int
            config_map_[key] = Value(int_val);
            // 对于正数，也存储一个unsigned int版本
            if (int_val >= 0) {
                std::string uint_key = key + ".uint";
                unsigned int uint_val = static_cast<unsigned int>(int_val);
                config_map_[uint_key] = Value(uint_val);
            }
            return;
        } catch (...) {}
        
        try {
            float float_val = node.as<float>();
            config_map_[key] = Value(float_val);
            return;
        } catch (...) {}
        
        try {
            double double_val = node.as<double>();
            config_map_[key] = Value(double_val);
            return;
        } catch (...) {}
        
        try {
            bool bool_val = node.as<bool>();
            config_map_[key] = Value(bool_val);
            return;
        } catch (...) {}
        
        // 默认作为字符串
        std::string str_val = node.as<std::string>();
        config_map_[key] = Value(str_val);
        
    } catch (const std::exception& e) {
        LOG_WARN("Failed to parse config key {}: {}", key, e.what());
    }
}

// 类型转换辅助函数
template<typename T>
T Config::convertValue(const Value& var) {
    // 使用访问者模式处理类型转换
    struct ValueVisitor {
        T operator()(int val) const {
            if constexpr (std::is_same_v<T, std::string>) {
                return std::to_string(val);
            } else if constexpr (std::is_same_v<T, unsigned int>) {
                if (val >= 0) {
                    return static_cast<unsigned int>(val);
                } else {
                    throw std::bad_variant_access();
                }
            } else {
                return static_cast<T>(val);
            }
        }
        
        T operator()(unsigned int val) const {
            if constexpr (std::is_same_v<T, std::string>) {
                return std::to_string(val);
            } else if constexpr (std::is_same_v<T, int>) {
                if (val <= static_cast<unsigned int>(std::numeric_limits<int>::max())) {
                    return static_cast<int>(val);
                } else {
                    throw std::bad_variant_access();
                }
            } else {
                return static_cast<T>(val);
            }
        }
        
        T operator()(float val) const {
            if constexpr (std::is_same_v<T, std::string>) {
                return std::to_string(val);
            } else {
                return static_cast<T>(val);
            }
        }
        
        T operator()(double val) const {
            if constexpr (std::is_same_v<T, std::string>) {
                return std::to_string(val);
            } else {
                return static_cast<T>(val);
            }
        }
        
        T operator()(bool val) const {
            if constexpr (std::is_same_v<T, std::string>) {
                return val ? "true" : "false";
            } else {
                return static_cast<T>(val);
            }
        }
        
        T operator()(const std::string& val) const {
            if constexpr (std::is_same_v<T, std::string>) {
                return val;
            } else if constexpr (std::is_same_v<T, bool>) {
                // 尝试解析字符串为布尔值
                std::string lower_val = val;
                std::transform(lower_val.begin(), lower_val.end(), lower_val.begin(), ::tolower);
                if (lower_val == "true" || lower_val == "yes" || lower_val == "1") {
                    return true;
                } else if (lower_val == "false" || lower_val == "no" || lower_val == "0") {
                    return false;
                } else {
                    throw std::bad_variant_access();
                }
            } else if constexpr (std::is_integral_v<T> || std::is_floating_point_v<T>) {
                // 尝试解析字符串为数值
                std::stringstream ss(val);
                T result;
                ss >> result;
                if (ss.fail()) {
                    throw std::bad_variant_access();
                }
                return result;
            } else {
                throw std::bad_variant_access();
            }
        }
    };
    
    return std::visit(ValueVisitor{}, var);
}

// 模板方法实现
template<typename T>
T Config::get(const std::string& key, const T& default_value) const {
    auto it = config_map_.find(key);
    if (it == config_map_.end()) {
        LOG_DEBUG("Config key not found, using default: {}", key);
        return default_value;
    }
    
    try {
        return convertValue<T>(it->second);
    } catch (const std::exception& e) {
        LOG_WARN("Failed to convert config value for key {}: {}", key, e.what());
        return default_value;
    }
}

template<typename T>
void Config::set(const std::string& key, const T& value) {
    config_map_[key] = Value(value);
}

// 显式实例化常用类型
template int Config::get<int>(const std::string&, const int&) const;
template unsigned int Config::get<unsigned int>(const std::string&, const unsigned int&) const;
template float Config::get<float>(const std::string&, const float&) const;
template double Config::get<double>(const std::string&, const double&) const;
template bool Config::get<bool>(const std::string&, const bool&) const;
template std::string Config::get<std::string>(const std::string&, const std::string&) const;

template void Config::set<int>(const std::string&, const int&);
template void Config::set<unsigned int>(const std::string&, const unsigned int&);
template void Config::set<float>(const std::string&, const float&);
template void Config::set<double>(const std::string&, const double&);
template void Config::set<bool>(const std::string&, const bool&);
template void Config::set<std::string>(const std::string&, const std::string&);

// ConfigManager实现
ConfigManager& ConfigManager::getInstance() {
    static ConfigManager instance;
    return instance;
}

bool ConfigManager::loadGlobalConfig(const std::string& filepath) {
    config_path_ = filepath;
    bool success = config_.loadFromFile(filepath);
    
    if (success) {
        LOG_INFO("Global config loaded successfully from: {}", filepath);
        
        // 打印加载的配置摘要
        auto keys = config_.getKeys();
        LOG_DEBUG("Loaded {} config entries", keys.size());
        
        // 输出一些重要配置
        try {
            std::string mode = config_.get<std::string>("system.mode", "gpu");
            int debug_level = config_.get<int>("system.debug_level", 2);
            bool gpu_enabled = config_.get<bool>("gpu.enabled", true);
            
            LOG_INFO("System mode: {}", mode);
            LOG_INFO("Debug level: {}", debug_level);
            LOG_INFO("GPU enabled: {}", gpu_enabled);
        } catch (...) {
            LOG_WARN("Failed to parse some config values");
        }
    } else {
        LOG_ERROR("Failed to load global config from: {}", filepath);
    }
    
    return success;
}

bool ConfigManager::reload() {
    if (config_path_.empty()) {
        LOG_ERROR("No config path set for reload");
        return false;
    }
    
    LOG_INFO("Reloading config from: {}", config_path_);
    return config_.loadFromFile(config_path_);
}

} // namespace utils
} // namespace stereo_depth
