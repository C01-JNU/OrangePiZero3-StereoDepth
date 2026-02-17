#pragma once

#include <string>
#include <unordered_map>
#include <variant>
#include <memory>
#include <yaml-cpp/yaml.h>

namespace stereo_depth {
namespace utils {

class Config {
public:
    using Value = std::variant<int, unsigned int, float, double, bool, std::string>;
    
    Config() = default;
    ~Config() = default;
    
    // 从文件加载配置
    bool loadFromFile(const std::string& filepath);
    
    // 从字符串加载配置
    bool loadFromString(const std::string& yaml_str);
    
    // 保存配置到文件
    bool saveToFile(const std::string& filepath);
    
    // 获取配置值
    template<typename T>
    T get(const std::string& key, const T& default_value = T()) const;
    
    // 设置配置值
    template<typename T>
    void set(const std::string& key, const T& value);
    
    // 检查配置是否存在
    bool has(const std::string& key) const;
    
    // 获取所有配置键
    std::vector<std::string> getKeys() const;
    
    // 合并配置
    void merge(const Config& other);
    
private:
    std::unordered_map<std::string, Value> config_map_;
    YAML::Node yaml_root_;
    
    // 辅助函数
    void parseYamlNode(const YAML::Node& node, const std::string& prefix = "");
    void setFromYaml(const std::string& key, const YAML::Node& node);
    
    // 类型转换
    template<typename T>
    static T convertValue(const Value& var);
};

// 全局配置管理器
class ConfigManager {
public:
    static ConfigManager& getInstance();
    
    // 加载全局配置
    bool loadGlobalConfig(const std::string& filepath = "config/global_config.yaml");
    
    // 获取配置
    Config& getConfig() { return config_; }
    const Config& getConfig() const { return config_; }
    
    // 重新加载配置
    bool reload();
    
    // 获取配置路径
    std::string getConfigPath() const { return config_path_; }
    
private:
    ConfigManager() = default;
    ~ConfigManager() = default;
    
    Config config_;
    std::string config_path_;
};

} // namespace utils
} // namespace stereo_depth
