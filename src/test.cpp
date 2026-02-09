#include "vulkan/context.hpp"
#include "vulkan/buffer_manager.hpp"
#include "vulkan/compute_pipeline.hpp"
#include "vulkan/stereo_pipeline.hpp"
#include "utils/logger.hpp"
#include "utils/config.hpp"
#include <iostream>
#include <vector>
#include <cstdint>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <sys/stat.h>
#include <filesystem>

namespace fs = std::filesystem;

/**
 * @brief 检查文件是否存在
 * @param filename 文件名
 * @return 文件是否存在
 */
bool fileExists(const std::string& filename) {
    struct stat buffer;
    return (stat(filename.c_str(), &buffer) == 0);
}

/**
 * @brief 确保目录存在，如果不存在则创建
 * @param dir 目录路径
 * @return 是否成功
 */
bool ensureDirectoryExists(const std::string& dir) {
    try {
        if (!fs::exists(dir)) {
            if (!fs::create_directories(dir)) {
                LOG_ERROR("无法创建目录: {}", dir);
                return false;
            }
            LOG_INFO("创建目录: {}", dir);
        }
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("创建目录时发生异常: {}", e.what());
        return false;
    }
}

/**
 * @brief 加载拼接图像并分割为左右眼图像
 * @param imagePath 拼接图像路径
 * @param leftImage 输出左眼图像
 * @param rightImage 输出右眼图像
 * @param config 配置对象
 * @return 是否成功
 */
bool loadAndSplitStereoImage(const std::string& imagePath, 
                           cv::Mat& leftImage, cv::Mat& rightImage,
                           const stereo_depth::utils::Config& config) {
    try {
        // 从配置读取图像尺寸
        uint32_t originalWidth = config.get<uint32_t>("camera.width");
        uint32_t originalHeight = config.get<uint32_t>("camera.height");
        uint32_t compressedWidth = config.get<uint32_t>("stereo.image_width");
        uint32_t compressedHeight = config.get<uint32_t>("stereo.image_height");
        
        LOG_INFO("加载拼接图像: {}", imagePath);
        LOG_INFO("原始图像尺寸: {}x{}", originalWidth, originalHeight);
        LOG_INFO("压缩后尺寸: {}x{}", compressedWidth, compressedHeight);
        
        // 验证压缩比例
        if (originalWidth % 2 != 0) {
            LOG_ERROR("原始图像宽度必须是偶数，当前宽度: {}", originalWidth);
            return false;
        }
        
        if (originalWidth / 2 != compressedWidth) {
            LOG_ERROR("压缩宽度不匹配: 原始宽度/2 = {}, 配置宽度 = {}", 
                     originalWidth / 2, compressedWidth);
            return false;
        }
        
        // 加载拼接图像
        cv::Mat stitchedImage = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
        if (stitchedImage.empty()) {
            LOG_ERROR("无法加载图像: {}", imagePath);
            return false;
        }
        
        // 验证图像尺寸
        if (static_cast<uint32_t>(stitchedImage.cols) != originalWidth || 
            static_cast<uint32_t>(stitchedImage.rows) != originalHeight) {
            LOG_ERROR("图像尺寸不匹配: 期望 {}x{}, 实际 {}x{}", 
                     originalWidth, originalHeight, 
                     stitchedImage.cols, stitchedImage.rows);
            return false;
        }
        
        LOG_INFO("✅ 图像加载成功: {}x{}", stitchedImage.cols, stitchedImage.rows);
        
        // 分割图像（左半部分：左眼，右半部分：右眼）
        cv::Rect leftRect(0, 0, compressedWidth, compressedHeight);
        cv::Rect rightRect(compressedWidth, 0, compressedWidth, compressedHeight);
        
        leftImage = stitchedImage(leftRect).clone();
        rightImage = stitchedImage(rightRect).clone();
        
        LOG_INFO("✅ 图像分割完成:");
        LOG_INFO("  左眼图像: {}x{}", leftImage.cols, leftImage.rows);
        LOG_INFO("  右眼图像: {}x{}", rightImage.cols, rightImage.rows);
        
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("加载和分割图像时发生异常: {}", e.what());
        return false;
    }
}

/**
 * @brief 保存视差图
 * @param disparityData 视差数据
 * @param width 图像宽度
 * @param height 图像高度
 * @param outputPath 输出路径
 * @param config 配置对象
 * @return 是否成功
 */
bool saveDisparityMap(const uint16_t* disparityData, 
                     uint32_t width, uint32_t height,
                     const std::string& outputPath,
                     const stereo_depth::utils::Config& config) {
    try {
        // 创建输出目录
        if (!ensureDirectoryExists(fs::path(outputPath).parent_path().string())) {
            return false;
        }
        
        // 将16位视差图转换为8位用于显示
        cv::Mat disparity16(height, width, CV_16UC1, (void*)disparityData);
        cv::Mat disparity8;
        
        // 从配置获取最大视差用于归一化
        uint32_t maxDisparity = config.get<uint32_t>("stereo.max_disparity");
        
        // 计算实际的最大视差值
        double minVal, maxVal;
        cv::minMaxLoc(disparity16, &minVal, &maxVal);
        
        LOG_INFO("视差图统计: 最小值={}, 最大值={}, 配置最大视差={}", 
                 minVal, maxVal, maxDisparity);
        
        // 使用配置的最大视差进行归一化，以便可视化
        if (maxVal > 0) {
            double scale = 255.0 / std::min(static_cast<double>(maxDisparity), maxVal);
            disparity16.convertTo(disparity8, CV_8UC1, scale);
        } else {
            // 如果全是0，创建一个空图像
            disparity8 = cv::Mat(height, width, CV_8UC1, cv::Scalar(0));
        }
        
        // 应用颜色映射以增强可视化
        cv::Mat colorDisparity;
        cv::applyColorMap(disparity8, colorDisparity, cv::COLORMAP_JET);
        
        // 保存图像
        std::string outputFormat = config.get<std::string>("output.output_format");
        std::string fullPath = outputPath + "." + outputFormat;
        
        if (cv::imwrite(fullPath, colorDisparity)) {
            LOG_INFO("✅ 视差图保存成功: {}", fullPath);
            
            // 同时保存原始16位视差图用于后续处理
            std::string rawPath = outputPath + "_raw.bin";
            std::ofstream rawFile(rawPath, std::ios::binary);
            if (rawFile) {
                rawFile.write(reinterpret_cast<const char*>(disparityData), 
                              width * height * sizeof(uint16_t));
                rawFile.close();
                LOG_INFO("✅ 原始视差数据保存成功: {}", rawPath);
            }
            return true;
        } else {
            LOG_ERROR("保存视差图失败: {}", fullPath);
            return false;
        }
        
    } catch (const std::exception& e) {
        LOG_ERROR("保存视差图时发生异常: {}", e.what());
        return false;
    }
}

/**
 * @brief 查找测试图像文件
 * @param testImageDir 测试图像目录
 * @return 找到的图像文件路径
 */
std::vector<std::string> findTestImages(const std::string& testImageDir) {
    std::vector<std::string> imagePaths;
    
    try {
        if (!fs::exists(testImageDir)) {
            LOG_ERROR("测试图像目录不存在: {}", testImageDir);
            return imagePaths;
        }
        
        LOG_INFO("扫描目录: {}", testImageDir);
        
        for (const auto& entry : fs::directory_iterator(testImageDir)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                
                if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                    imagePaths.push_back(entry.path().string());
                }
            }
        }
        
        std::sort(imagePaths.begin(), imagePaths.end());
        
        LOG_INFO("找到 {} 个图像文件", imagePaths.size());
        
    } catch (const std::exception& e) {
        LOG_ERROR("查找测试图像时发生异常: {}", e.what());
    }
    
    return imagePaths;
}

/**
 * @brief 验证配置参数
 * @param config 配置对象
 * @return 是否验证成功
 */
bool validateConfig(const stereo_depth::utils::Config& config) {
    try {
        LOG_INFO("验证配置参数...");
        
        // 验证必要的配置项
        std::vector<std::string> requiredKeys = {
            "camera.width",
            "camera.height", 
            "stereo.image_width",
            "stereo.image_height",
            "stereo.max_disparity",
            "output.test_image_dir",
            "output.output_dir",
            "output.output_format"
        };
        
        for (const auto& key : requiredKeys) {
            if (!config.has(key)) {
                LOG_ERROR("缺少必需的配置项: {}", key);
                return false;
            }
        }
        
        // 验证图像尺寸
        uint32_t cameraWidth = config.get<uint32_t>("camera.width");
        uint32_t cameraHeight = config.get<uint32_t>("camera.height");
        uint32_t stereoWidth = config.get<uint32_t>("stereo.image_width");
        uint32_t stereoHeight = config.get<uint32_t>("stereo.image_height");
        uint32_t maxDisparity = config.get<uint32_t>("stereo.max_disparity");
        
        LOG_INFO("配置参数:");
        LOG_INFO("  相机尺寸: {}x{}", cameraWidth, cameraHeight);
        LOG_INFO("  立体图像尺寸: {}x{}", stereoWidth, stereoHeight);
        LOG_INFO("  最大视差: {}", maxDisparity);
        
        // 验证压缩比例
        if (cameraWidth % 2 != 0) {
            LOG_ERROR("相机宽度必须是偶数: {}", cameraWidth);
            return false;
        }
        
        if (cameraWidth / 2 != stereoWidth) {
            LOG_ERROR("立体图像宽度必须等于相机宽度的一半: {} != {}/2", 
                     stereoWidth, cameraWidth);
            return false;
        }
        
        if (cameraHeight != stereoHeight) {
            LOG_ERROR("立体图像高度必须等于相机高度: {} != {}", 
                     stereoHeight, cameraHeight);
            return false;
        }
        
        if (maxDisparity == 0) {
            LOG_ERROR("最大视差不能为0");
            return false;
        }
        
        LOG_INFO("✅ 配置验证成功");
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("配置验证异常: {}", e.what());
        return false;
    }
}

/**
 * @brief 测试程序主函数
 * 
 * 测试Vulkan框架和立体匹配流水线：
 * 1. 加载配置参数
 * 2. 从images/test读取测试图像
 * 3. 使用立体匹配流水线计算视差图
 * 4. 将结果保存到images/output
 * 
 * 更新日期：2026年2月9日（完全使用配置文件参数）
 */
int main(int argc, char* argv[]) {
    using namespace stereo_depth;
    
    // 记录总开始时间
    auto totalStartTime = std::chrono::high_resolution_clock::now();
    
    // 初始化日志系统
    utils::Logger::initialize("test", spdlog::level::info);
    
    LOG_INFO("=== OrangePiZero3 立体深度 GPU 系统测试 ===");
    LOG_INFO("开始时间: 2026年2月9日");
    LOG_INFO("----------------------------------------");
    
    // 加载配置文件
    LOG_INFO("\n[1/6] 加载配置参数");
    LOG_INFO("------------------------");
    
    utils::Config config;
    std::string configPath = "config/global_config.yaml";
    
    if (!config.loadFromFile(configPath)) {
        LOG_ERROR("无法加载配置文件: {}", configPath);
        LOG_ERROR("请确保配置文件存在并格式正确");
        return EXIT_FAILURE;
    }
    
    LOG_INFO("✅ 配置加载成功: {}", configPath);
    
    // 设置日志级别
    int debugLevel = config.get<int>("system.debug_level", 2);
    spdlog::level::level_enum logLevel = static_cast<spdlog::level::level_enum>(
        std::min(std::max(debugLevel, 0), 5));
    utils::Logger::setLevel(logLevel);
    LOG_INFO("日志级别设置为: {}", spdlog::level::to_string_view(logLevel));
    
    // 验证配置参数
    if (!validateConfig(config)) {
        LOG_ERROR("配置验证失败");
        return EXIT_FAILURE;
    }
    
    // 从配置读取路径
    std::string testImageDir = config.get<std::string>("output.test_image_dir");
    std::string outputDir = config.get<std::string>("output.output_dir");
    
    LOG_INFO("测试图像目录: {}", testImageDir);
    LOG_INFO("输出目录: {}", outputDir);
    
    // 设置Mali-G31所需的环境变量（必须在Vulkan初始化之前）
    setenv("PAN_I_WANT_A_BROKEN_VULKAN_DRIVER", "1", 1);
    LOG_INFO("设置环境变量: PAN_I_WANT_A_BROKEN_VULKAN_DRIVER=1");
    
    // 查找测试图像
    LOG_INFO("\n[2/6] 查找测试图像");
    LOG_INFO("------------------------");
    
    std::vector<std::string> testImages = findTestImages(testImageDir);
    if (testImages.empty()) {
        LOG_ERROR("未找到测试图像，请将图像放入 {} 目录", testImageDir);
        LOG_ERROR("图像格式应为: {}x{} JPG，左半部分{}x{}为左眼，右半部分为右眼",
                 config.get<uint32_t>("camera.width"),
                 config.get<uint32_t>("camera.height"),
                 config.get<uint32_t>("stereo.image_width"),
                 config.get<uint32_t>("stereo.image_height"));
        return EXIT_FAILURE;
    }
    
    LOG_INFO("找到 {} 个测试图像:", testImages.size());
    for (size_t i = 0; i < std::min(testImages.size(), size_t(5)); ++i) {
        LOG_INFO("  {}. {}", i + 1, fs::path(testImages[i]).filename().string());
    }
    if (testImages.size() > 5) {
        LOG_INFO("  ... 和其他 {} 个图像", testImages.size() - 5);
    }
    
    try {
        // 创建Vulkan上下文
        LOG_INFO("\n[3/6] 初始化Vulkan系统");
        LOG_INFO("------------------------");
        
        auto vulkanStartTime = std::chrono::high_resolution_clock::now();
        
        vulkan::VulkanContext vulkanContext;
        if (!vulkanContext.initialize(false)) {
            LOG_ERROR("初始化Vulkan上下文失败");
            return EXIT_FAILURE;
        }
        
        auto vulkanEndTime = std::chrono::high_resolution_clock::now();
        auto vulkanDuration = std::chrono::duration_cast<std::chrono::milliseconds>(
            vulkanEndTime - vulkanStartTime);
        
        LOG_INFO("✅ Vulkan系统初始化完成");
        LOG_INFO("  设备: {}", vulkanContext.getDeviceName());
        LOG_INFO("  Vulkan版本: {}", vulkanContext.getVulkanVersion());
        LOG_INFO("  初始化耗时: {} 毫秒", vulkanDuration.count());
        
        // 从配置读取立体匹配参数
        uint32_t imageWidth = config.get<uint32_t>("camera.width");
        uint32_t imageHeight = config.get<uint32_t>("camera.height");
        uint32_t maxDisparity = config.get<uint32_t>("stereo.max_disparity");
        
        // 创建立体匹配流水线
        LOG_INFO("\n[4/6] 创建立体匹配流水线");
        LOG_INFO("------------------------");
        
        LOG_INFO("流水线参数:");
        LOG_INFO("  输入图像尺寸: {}x{}", imageWidth, imageHeight);
        LOG_INFO("  最大视差: {}", maxDisparity);
        LOG_INFO("  算法: {}", config.get<std::string>("stereo.algorithm", "wta"));
        LOG_INFO("  Census窗口: {}x{}", 
                 config.get<uint32_t>("stereo.window_size", 9),
                 config.get<uint32_t>("stereo.window_size", 9));
        LOG_INFO("  唯一性比率: {}%", config.get<uint32_t>("stereo.uniqueness_ratio", 15));
        
        auto pipelineStartTime = std::chrono::high_resolution_clock::now();
        
        vulkan::StereoPipeline stereoPipeline(vulkanContext);
        if (!stereoPipeline.initialize(imageWidth, imageHeight, maxDisparity)) {
            LOG_ERROR("立体匹配流水线初始化失败");
            LOG_ERROR("请确保着色器已编译: cd build && cmake --build . --target shaders");
            return EXIT_FAILURE;
        }
        
        auto pipelineEndTime = std::chrono::high_resolution_clock::now();
        auto pipelineDuration = std::chrono::duration_cast<std::chrono::milliseconds>(
            pipelineEndTime - pipelineStartTime);
        
        LOG_INFO("✅ 立体匹配流水线初始化完成");
        LOG_INFO("  初始化耗时: {} 毫秒", pipelineDuration.count());
        
        // 处理每个测试图像
        LOG_INFO("\n[5/6] 处理测试图像");
        LOG_INFO("------------------------");
        
        int processedCount = 0;
        int successCount = 0;
        
        for (size_t imgIdx = 0; imgIdx < testImages.size(); ++imgIdx) {
            const auto& imagePath = testImages[imgIdx];
            std::string imageName = fs::path(imagePath).stem().string();
            
            LOG_INFO("\n处理图像 {}/{}: {}", 
                     imgIdx + 1, testImages.size(), fs::path(imagePath).filename().string());
            
            // 加载并分割图像
            cv::Mat leftImage, rightImage;
            if (!loadAndSplitStereoImage(imagePath, leftImage, rightImage, config)) {
                LOG_ERROR("❌ 图像加载失败: {}", imagePath);
                continue;
            }
            
            processedCount++;
            
            // 设置图像数据到流水线
            auto setImageStartTime = std::chrono::high_resolution_clock::now();
            
            if (!stereoPipeline.setLeftImage(leftImage.data)) {
                LOG_ERROR("❌ 设置左图像数据失败");
                continue;
            }
            
            if (!stereoPipeline.setRightImage(rightImage.data)) {
                LOG_ERROR("❌ 设置右图像数据失败");
                continue;
            }
            
            auto setImageEndTime = std::chrono::high_resolution_clock::now();
            auto setImageDuration = std::chrono::duration_cast<std::chrono::milliseconds>(
                setImageEndTime - setImageStartTime);
            
            LOG_INFO("✅ 图像数据设置完成，耗时: {} 毫秒", setImageDuration.count());
            
            // 执行立体匹配计算
            LOG_INFO("开始立体匹配计算...");
            auto computeStartTime = std::chrono::high_resolution_clock::now();
            
            if (!stereoPipeline.compute()) {
                LOG_ERROR("❌ 立体匹配计算失败");
                continue;
            }
            
            auto computeEndTime = std::chrono::high_resolution_clock::now();
            auto computeDuration = std::chrono::duration_cast<std::chrono::milliseconds>(
                computeEndTime - computeStartTime);
            
            LOG_INFO("✅ 立体匹配计算完成");
            LOG_INFO("  计算耗时: {} 毫秒", computeDuration.count());
            if (computeDuration.count() > 0) {
                LOG_INFO("  估计帧率: {:.1f} FPS", 1000.0 / computeDuration.count());
            }
            
            // 获取视差图
            uint32_t compressedWidth = config.get<uint32_t>("stereo.image_width");
            uint32_t compressedHeight = config.get<uint32_t>("stereo.image_height");
            size_t disparitySize = static_cast<size_t>(compressedWidth) * 
                                  static_cast<size_t>(compressedHeight);
            
            std::vector<uint16_t> disparityMap(disparitySize);
            if (!stereoPipeline.getDisparityMap(disparityMap.data())) {
                LOG_ERROR("❌ 获取视差图失败");
                continue;
            }
            
            LOG_INFO("✅ 视差图获取成功: {} 像素", disparitySize);
            
            // 保存视差图
            std::string outputPath = outputDir + "/" + imageName + "_disparity";
            if (!saveDisparityMap(disparityMap.data(), compressedWidth, compressedHeight, 
                                 outputPath, config)) {
                LOG_ERROR("❌ 保存视差图失败");
                continue;
            }
            
            successCount++;
            
            // 输出性能摘要
            LOG_INFO("📊 性能摘要:");
            LOG_INFO("  图像加载: {} 毫秒", setImageDuration.count());
            LOG_INFO("  立体匹配: {} 毫秒", computeDuration.count());
            LOG_INFO("  总处理时间: {} 毫秒", setImageDuration.count() + computeDuration.count());
            if ((setImageDuration.count() + computeDuration.count()) > 0) {
                LOG_INFO("  处理速度: {:.1f} FPS", 
                         1000.0 / (setImageDuration.count() + computeDuration.count()));
            }
        }
        
        // 系统清理
        LOG_INFO("\n[6/6] 系统清理");
        LOG_INFO("------------------------");
        
        // 等待设备空闲
        vulkanContext.waitIdle();
        
        // 记录总结束时间
        auto totalEndTime = std::chrono::high_resolution_clock::now();
        auto totalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(
            totalEndTime - totalStartTime);
        
        // 输出测试摘要
        LOG_INFO("\n=== 测试完成 ===");
        LOG_INFO("📊 测试摘要:");
        LOG_INFO("  总测试时间: {} 毫秒 ({:.1f} 秒)", 
                 totalDuration.count(), totalDuration.count() / 1000.0);
        LOG_INFO("  找到图像: {}", testImages.size());
        LOG_INFO("  处理图像: {}", processedCount);
        LOG_INFO("  成功处理: {}", successCount);
        LOG_INFO("  失败处理: {}", processedCount - successCount);
        
        if (processedCount > 0) {
            double successRate = (static_cast<double>(successCount) / processedCount) * 100.0;
            LOG_INFO("  成功率: {:.1f}%", successRate);
        }
        
        if (successCount > 0) {
            LOG_INFO("✅ 立体匹配流水线测试成功");
            LOG_INFO("  输出目录: {}", outputDir);
            LOG_INFO("  请检查 {} 目录中的结果图像", outputDir);
        } else {
            LOG_ERROR("❌ 立体匹配流水线测试失败，没有成功处理的图像");
            return EXIT_FAILURE;
        }
        
        LOG_INFO("\n🎯 下一步:");
        LOG_INFO("  1. 检查输出图像质量");
        LOG_INFO("  2. 调整算法参数以获得更好效果");
        LOG_INFO("  3. 集成相机标定参数");
        LOG_INFO("  4. 进行实时视频流测试");
        
        return EXIT_SUCCESS;
        
    } catch (const std::exception& e) {
        LOG_ERROR("测试程序发生异常: {}", e.what());
        return EXIT_FAILURE;
    } catch (...) {
        LOG_ERROR("测试程序发生未知异常");
        return EXIT_FAILURE;
    }
}
