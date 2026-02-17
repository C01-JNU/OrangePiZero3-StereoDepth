#include "calibration/stereo_calibrator.hpp"
#include "utils/config.hpp"
#include "utils/logger.hpp"

#include <iostream>
#include <cstdlib>

int main(int argc, char* argv[]) {
    std::cout << "==================================================\n";
    std::cout << "OrangePiZero3-StereoDepth 立体相机标定工具\n";
    std::cout << "==================================================\n\n";
    
    // 初始化日志系统
    try {
        stereo_depth::utils::Logger::initialize("stereo_calibrator", spdlog::level::info);
    } catch (const std::exception& e) {
        std::cerr << "错误: 日志系统初始化失败: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    
    LOG_INFO("启动立体标定工具");
    
    // 尝试加载配置文件以获取图像尺寸
    std::string configPath = "config/global_config.yaml";
    auto& configManager = stereo_depth::utils::ConfigManager::getInstance();
    
    if (!configManager.loadGlobalConfig(configPath)) {
        std::cerr << "错误: 无法加载配置文件: " << configPath << std::endl;
        LOG_ERROR("无法加载配置文件: {}", configPath);
        return EXIT_FAILURE;
    }
    
    // 从配置读取图像尺寸
    const auto& config = configManager.getConfig();
    int imageWidth = 0;
    int imageHeight = 0;
    
    try {
        imageWidth = config.get<int>("stereo.image_width", 0);
        imageHeight = config.get<int>("stereo.image_height", 0);
        
        if (imageWidth <= 0 || imageHeight <= 0) {
            // 尝试其他可能的键
            imageWidth = config.get<int>("camera.width", 0);
            imageHeight = config.get<int>("camera.height", 0);
        }
    } catch (const std::exception& e) {
        LOG_WARN("读取图像尺寸时出错: {}", e.what());
    }
    
    std::cout << "开始立体标定...\n";
    std::cout << "请确保:\n";
    std::cout << "  1. 标定图像已放在 images/calibration/ 目录中\n";
    std::cout << "  2. 图像命名格式: *_left.jpg, *_right.jpg\n";
    
    // 动态显示图像尺寸
    if (imageWidth > 0 && imageHeight > 0) {
        std::cout << "  3. 图像尺寸: " << imageWidth << "x" << imageHeight << " (单眼)\n";
    } else {
        std::cout << "  3. 图像尺寸: 与配置文件中 stereo.image_width 和 stereo.image_height 一致\n";
    }
    
    std::cout << "  4. 至少需要10对有效图像\n\n";
    
    // 创建标定器
    stereo_depth::calibration::StereoCalibrator calibrator;
    
    // 执行标定
    if (calibrator.calibrate()) {
        std::cout << "\n✅ 标定成功！\n";
        std::cout << "  • RMS误差: " << calibrator.getCalibrationError() << "\n";
        std::cout << "  • 使用图像: " << calibrator.getImagesUsed() << " 对\n";
        std::cout << "  • 标定文件: calibration_results/stereo_calibration.yml\n";
        std::cout << "  • 标定报告: calibration_results/calibration_report.txt\n";
        std::cout << "  • 验证图像: calibration_results/rectification_validation.jpg\n";
        return EXIT_SUCCESS;
    } else {
        std::cout << "\n❌ 标定失败！\n";
        std::cout << "  请检查:\n";
        std::cout << "  • 标定图像是否存在且可读\n";
        std::cout << "  • 图像命名是否正确\n";
        std::cout << "  • 棋盘格参数是否正确\n";
        return EXIT_FAILURE;
    }
}
