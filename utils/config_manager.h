#pragma once

/**
 * @file config_manager.h
 * @brief 配置文件管理器
 * @date 2026-01-18
 * @author C01-JNU
 */

#include <string>
#include <unordered_map>
#include <yaml-cpp/yaml.h>
#include "common_defines.h"

class ConfigManager {
public:
    // 获取单例实例
    static ConfigManager& getInstance() {
        static ConfigManager instance;
        return instance;
    }
    
    // 禁止拷贝和赋值
    ConfigManager(const ConfigManager&) = delete;
    ConfigManager& operator=(const ConfigManager&) = delete;
    
    /**
     * @brief 加载配置文件
     * @param config_path 配置文件路径
     * @return 是否加载成功
     */
    bool load(const std::string& config_path);
    
    /**
     * @brief 重新加载配置文件
     * @return 是否重新加载成功
     */
    bool reload();
    
    /**
     * @brief 保存当前配置到文件
     * @param config_path 配置文件路径
     * @return 是否保存成功
     */
    bool save(const std::string& config_path);
    
    /**
     * @brief 获取字符串配置值
     * @param key 配置键
     * @param default_value 默认值
     * @return 配置值
     */
    std::string getString(const std::string& key, const std::string& default_value = "");
    
    /**
     * @brief 获取整数配置值
     * @param key 配置键
     * @param default_value 默认值
     * @return 配置值
     */
    int getInt(const std::string& key, int default_value = 0);
    
    /**
     * @brief 获取浮点数配置值
     * @param key 配置键
     * @param default_value 默认值
     * @return 配置值
     */
    float getFloat(const std::string& key, float default_value = 0.0f);
    
    /**
     * @brief 获取布尔配置值
     * @param key 配置键
     * @param default_value 默认值
     * @return 配置值
     */
    bool getBool(const std::string& key, bool default_value = false);
    
    /**
     * @brief 设置配置值
     * @param key 配置键
     * @param value 配置值
     */
    void setString(const std::string& key, const std::string& value);
    void setInt(const std::string& key, int value);
    void setFloat(const std::string& key, float value);
    void setBool(const std::string& key, bool value);
    
    /**
     * @brief 获取原始YAML节点（高级用法）
     * @param key 配置键
     * @return YAML节点
     */
    YAML::Node getNode(const std::string& key);
    
    /**
     * @brief 检查配置键是否存在
     * @param key 配置键
     * @return 是否存在
     */
    bool hasKey(const std::string& key);
    
    /**
     * @brief 获取所有配置键
     * @return 配置键列表
     */
    std::vector<std::string> getKeys();
    
    /**
     * @brief 清空所有配置
     */
    void clear();
    
    /**
     * @brief 获取配置文件路径
     * @return 配置文件路径
     */
    std::string getConfigPath() const { return config_path_; }
    
    /**
     * @brief 检查配置文件是否已加载
     * @return 是否已加载
     */
    bool isLoaded() const { return loaded_; }

private:
    ConfigManager() = default;
    ~ConfigManager() = default;
    
    // 递归获取节点
    YAML::Node getNodeRecursive(const std::string& key);
    
    // 递归设置节点
    void setNodeRecursive(const std::string& key, const YAML::Node& value);
    
    std::string config_path_;
    YAML::Node root_node_;
    bool loaded_ = false;
    std::unordered_map<std::string, YAML::Node> cache_;
};
