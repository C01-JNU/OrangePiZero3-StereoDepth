/**
 * @file main.cpp
 * @brief OrangePiZero3-StereoDepth 主程序入口
 * @date 2026-01-18
 * @author C01-JNU
 */

#include <iostream>
#include <memory>
#include <csignal>
#include <atomic>
#include "common_defines.h"
#include "config_manager.h"
#include "logger.h"
#include "stereo_pipeline.h"
#include "camera_interface.h"
#include "profiler.h"

// 全局信号标志
std::atomic<bool> g_running{true};

/**
 * @brief 信号处理函数
 */
void signalHandler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        Logger::getInstance().log(LogLevel::INFO, "收到终止信号，正在退出...");
        g_running = false;
    }
}

/**
 * @brief 主函数
 */
int main(int argc, char** argv) {
    // 设置信号处理
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);
    
    // 初始化日志系统
    Logger& logger = Logger::getInstance();
    logger.setLogLevel(LogLevel::INFO);
    logger.enableConsoleOutput(true);
    logger.enableFileOutput(true, "logs/stereo_depth.log");
    
    LOG_INFO("==========================================");
    LOG_INFO("OrangePiZero3-StereoDepth 启动");
    LOG_INFO("版本: 1.0.0");
    LOG_INFO("日期: 2026-01-18");
    LOG_INFO("平台: %s", PLATFORM_ARM ? "ARM" : "x86");
    LOG_INFO("==========================================");
    
    try {
        // 加载配置文件
        LOG_INFO("正在加载配置文件...");
        ConfigManager& config = ConfigManager::getInstance();
        
        if (!config.load("config/global_config.yaml")) {
            LOG_ERROR("配置文件加载失败");
            return static_cast<int>(ErrorCode::ERROR_CONFIG_LOAD_FAILED);
        }
        
        // 获取运行模式
        std::string mode_str = config.getString("system.mode", "GPU_VULKAN");
        RunMode run_mode;
        
        if (mode_str == "CPU") {
            run_mode = RunMode::CPU;
            LOG_INFO("运行模式: CPU（备用算法）");
        } else if (mode_str == "GPU_VULKAN") {
            run_mode = RunMode::GPU_VULKAN;
            LOG_INFO("运行模式: GPU（Vulkan加速）");
        } else if (mode_str == "DEBUG") {
            run_mode = RunMode::DEBUG;
            LOG_INFO("运行模式: 调试模式");
        } else {
            run_mode = RunMode::GPU_VULKAN;
            LOG_WARN("未知模式 '%s'，使用默认 GPU_VULKAN 模式", mode_str.c_str());
        }
        
        // 获取调试级别
        std::string debug_level = config.getString("system.debug_level", "INFO");
        LogLevel log_level;
        
        if (debug_level == "TRACE") log_level = LogLevel::TRACE;
        else if (debug_level == "DEBUG") log_level = LogLevel::DEBUG;
        else if (debug_level == "WARN") log_level = LogLevel::WARN;
        else if (debug_level == "ERROR") log_level = LogLevel::ERROR;
        else if (debug_level == "FATAL") log_level = LogLevel::FATAL;
        else log_level = LogLevel::INFO;
        
        logger.setLogLevel(log_level);
        
        // 初始化性能分析器
        Profiler::getInstance().start();
        
        // 初始化立体视觉流水线
        LOG_INFO("初始化立体视觉流水线...");
        auto pipeline = std::make_unique<StereoPipeline>(run_mode);
        
        if (!pipeline->initialize()) {
            LOG_ERROR("立体视觉流水线初始化失败");
            return static_cast<int>(ErrorCode::ERROR_STEREO_MATCHING_FAILED);
        }
        
        // 初始化摄像头接口
        LOG_INFO("初始化摄像头接口...");
        auto camera = std::make_unique<CameraInterface>();
        
        // 尝试从摄像头捕获图像
        bool use_camera = camera->initialize();
        
        if (!use_camera) {
            LOG_WARN("摄像头初始化失败，将使用测试图像");
            
            // 加载测试图像
            std::string test_image_path = config.getString("paths.test_images", "images/test/");
            if (!pipeline->loadTestImages(test_image_path)) {
                LOG_ERROR("测试图像加载失败");
                return static_cast<int>(ErrorCode::ERROR_IMAGE_LOAD_FAILED);
            }
        }
        
        // 加载标定参数
        LOG_INFO("加载相机标定参数...");
        std::string calib_file = config.getString("calibration.calibration_file", 
                                                 "calibration_results/stereo_calibration.yml");
        
        if (!pipeline->loadCalibration(calib_file)) {
            LOG_WARN("标定文件加载失败，使用默认参数");
        }
        
        // 主循环
        LOG_INFO("进入主处理循环...");
        int frame_count = 0;
        double total_time = 0.0;
        
        while (g_running) {
            Profiler::getInstance().beginFrame();
            
            // 捕获或加载图像
            bool has_image = false;
            if (use_camera) {
                has_image = camera->captureFrame();
            } else {
                has_image = pipeline->loadNextTestImage();
            }
            
            if (!has_image) {
                LOG_WARN("无法获取图像，跳过本帧");
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            
            // 获取图像数据
            const auto& left_image = use_camera ? camera->getLeftImage() : pipeline->getCurrentLeftImage();
            const auto& right_image = use_camera ? camera->getRightImage() : pipeline->getCurrentRightImage();
            
            if (left_image.empty() || right_image.empty()) {
                LOG_WARN("获取的图像为空，跳过本帧");
                continue;
            }
            
            // 处理立体图像
            auto start_time = std::chrono::high_resolution_clock::now();
            
            bool success = pipeline->processStereoPair(left_image, right_image);
            
            auto end_time = std::chrono::high_resolution_clock::now();
            double process_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            
            if (!success) {
                LOG_ERROR("立体处理失败");
                continue;
            }
            
            // 获取结果
            auto disparity_map = pipeline->getDisparityMap();
            auto depth_map = pipeline->getDepthMap();
            
            // 更新统计数据
            frame_count++;
            total_time += process_time;
            
            // 输出处理信息
            if (frame_count % 10 == 0) {
                double avg_time = total_time / frame_count;
                double fps = 1000.0 / avg_time;
                
                LOG_INFO("帧 %d: 处理时间 %.2fms, 平均 %.2fms (%.1f FPS)", 
                        frame_count, process_time, avg_time, fps);
                
                // 输出性能分析信息
                if (config.getBool("system.enable_profiling", false)) {
                    Profiler::getInstance().printStats();
                }
            }
            
            // 保存输出图像
            if (config.getBool("output.save.disparity_map", true)) {
                std::string output_path = config.getString("paths.output_images", "images/output/");
                pipeline->saveDisparityMap(output_path + "disparity_" + std::to_string(frame_count) + ".png");
            }
            
            // 检查是否需要退出
            if (!use_camera && !pipeline->hasMoreTestImages()) {
                LOG_INFO("所有测试图像处理完成");
                break;
            }
            
            Profiler::getInstance().endFrame();
            
            // 控制帧率
            if (use_camera) {
                int target_fps = config.getInt("camera.fps", 30);
                if (target_fps > 0) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1000 / target_fps));
                }
            }
        }
        
        // 输出最终统计
        if (frame_count > 0) {
            double avg_time = total_time / frame_count;
            LOG_INFO("==========================================");
            LOG_INFO("处理完成");
            LOG_INFO("总帧数: %d", frame_count);
            LOG_INFO("总时间: %.2fms", total_time);
            LOG_INFO("平均处理时间: %.2fms", avg_time);
            LOG_INFO("平均帧率: %.1f FPS", 1000.0 / avg_time);
            LOG_INFO("==========================================");
        }
        
        // 清理资源
        LOG_INFO("清理资源...");
        pipeline->cleanup();
        
        if (use_camera) {
            camera->release();
        }
        
        Profiler::getInstance().stop();
        
    } catch (const std::exception& e) {
        LOG_ERROR("未捕获的异常: %s", e.what());
        return static_cast<int>(ErrorCode::ERROR_SYSTEM);
    }
    
    LOG_INFO("程序正常退出");
    return static_cast<int>(ErrorCode::SUCCESS);
}
