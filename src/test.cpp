/**
 * @file test.cpp
 * @brief OrangePiZero3 立体深度 GPU 系统测试主程序
 * @author C01-JNU
 * @date 2026-02-11
 *
 * 功能：
 * 1. 加载全局配置 global_config.yaml
 * 2. 初始化 Vulkan 上下文与立体匹配流水线
 * 3. 从 images/test 目录读取所有拼接图像
 * 4. 执行 GPU 立体匹配，生成视差图
 * 5. 保存视差图及调试信息到 images/output
 */

#include "vulkan/context.hpp"
#include "vulkan/buffer_manager.hpp"
#include "vulkan/compute_pipeline.hpp"
#include "vulkan/stereo_pipeline.hpp"
#include "utils/logger.hpp"
#include "utils/config.hpp"

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <chrono>

namespace fs = std::filesystem;
using namespace stereo_depth;

// ------------------------------------------------------------
// 辅助函数（中文日志）
// ------------------------------------------------------------

/**
 * @brief 确保目录存在，不存在则创建
 */
static bool ensureDirectoryExists(const std::string& dir) {
    try {
        if (!fs::exists(dir)) {
            if (!fs::create_directories(dir)) {
                LOG_ERROR("❌ 无法创建目录: {}", dir);
                return false;
            }
            LOG_INFO("📁 创建目录: {}", dir);
        }
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("❌ 创建目录异常: {}", e.what());
        return false;
    }
}

/**
 * @brief 加载拼接图像并分割为左右眼（压缩后尺寸）
 * @param imagePath   拼接图像路径（640x480，左眼320x480，右眼320x480）
 * @param leftImg     输出左眼图像（CV_8U, 320x480）
 * @param rightImg    输出右眼图像（CV_8U, 320x480）
 * @param config      配置对象
 */
static bool loadAndSplitImage(const std::string& imagePath,
                              cv::Mat& leftImg, cv::Mat& rightImg,
                              const utils::Config& config) {
    uint32_t camW = config.get<uint32_t>("camera.width");      // 640
    uint32_t camH = config.get<uint32_t>("camera.height");     // 480
    uint32_t stereoW = config.get<uint32_t>("stereo.image_width");  // 320
    uint32_t stereoH = config.get<uint32_t>("stereo.image_height"); // 480

    cv::Mat full = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (full.empty()) {
        LOG_ERROR("❌ 无法读取图像: {}", imagePath);
        return false;
    }

    if (full.cols != (int)camW || full.rows != (int)camH) {
        LOG_ERROR("❌ 图像尺寸不匹配: 期望 {}x{}, 实际 {}x{}",
                  camW, camH, full.cols, full.rows);
        return false;
    }

    leftImg  = full(cv::Rect(0, 0, stereoW, stereoH)).clone();
    rightImg = full(cv::Rect(stereoW, 0, stereoW, stereoH)).clone();

    LOG_DEBUG("✅ 图像分割: 左 {}x{}，右 {}x{}",
              leftImg.cols, leftImg.rows, rightImg.cols, rightImg.rows);
    return true;
}

/**
 * @brief 保存视差图（16位原始数据 + 8位彩色可视化）
 * @param disparityData GPU回读的16位视差数据
 * @param width   视差图宽度（压缩后）
 * @param height  视差图高度
 * @param basePath 输出文件前缀（不含扩展名）
 * @param config  配置对象
 */
static bool saveDisparityMap(const uint16_t* disparityData,
                             uint32_t width, uint32_t height,
                             const std::string& basePath,
                             const utils::Config& config) {
    // 创建输出目录
    std::string outDir = fs::path(basePath).parent_path().string();
    if (!ensureDirectoryExists(outDir)) return false;

    // 保存原始16位数据
    std::string rawPath = basePath + "_raw.bin";
    std::ofstream rawFile(rawPath, std::ios::binary);
    if (rawFile) {
        rawFile.write(reinterpret_cast<const char*>(disparityData),
                      width * height * sizeof(uint16_t));
        rawFile.close();
        LOG_INFO("💾 原始视差数据保存: {}", rawPath);
    }

    // 转换为8位可视化图像
    cv::Mat disp16(height, width, CV_16UC1, const_cast<uint16_t*>(disparityData));
    cv::Mat disp8;
    double minVal, maxVal;
    cv::minMaxLoc(disp16, &minVal, &maxVal);
    uint32_t maxDisp = config.get<uint32_t>("stereo.max_disparity", 64);

    if (maxVal > 0) {
        double scale = 255.0 / std::min((double)maxDisp, maxVal);
        disp16.convertTo(disp8, CV_8UC1, scale);
    } else {
        disp8 = cv::Mat(height, width, CV_8UC1, cv::Scalar(0));
    }

    cv::Mat color;
    cv::applyColorMap(disp8, color, cv::COLORMAP_JET);
    std::string ext = config.get<std::string>("output.output_format", "png");
    std::string visPath = basePath + "." + ext;
    if (cv::imwrite(visPath, color)) {
        LOG_INFO("🖼️ 视差图保存: {}", visPath);
        LOG_INFO("📊 视差统计: 最小值={:.0f}, 最大值={:.0f}, 配置最大视差={}",
                 minVal, maxVal, maxDisp);
        return true;
    } else {
        LOG_ERROR("❌ 保存视差图失败: {}", visPath);
        return false;
    }
}

/**
 * @brief 查找目录中的所有图像文件（jpg/png/bmp）
 */
static std::vector<std::string> findTestImages(const std::string& dir) {
    std::vector<std::string> files;
    if (!fs::exists(dir)) {
        LOG_ERROR("❌ 测试图像目录不存在: {}", dir);
        return files;
    }

    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
            files.push_back(entry.path().string());
        }
    }
    std::sort(files.begin(), files.end());
    LOG_INFO("🔍 找到 {} 个测试图像", files.size());
    return files;
}

// ------------------------------------------------------------
// 主函数
// ------------------------------------------------------------
int main(int argc, char* argv[]) {
    auto totalStart = std::chrono::high_resolution_clock::now();

    // 1. 初始化日志（中文）
    utils::Logger::initialize("test", spdlog::level::info);
    LOG_INFO("========================================");
    LOG_INFO("   OrangePiZero3 立体深度 GPU 测试");
    LOG_INFO("========================================");

    // 2. 加载配置
    LOG_INFO("\n[1/6] 加载配置文件");
    utils::Config config;
    if (!config.loadFromFile("config/global_config.yaml")) {
        LOG_ERROR("❌ 无法加载配置文件: config/global_config.yaml");
        return EXIT_FAILURE;
    }

    // 设置日志级别
    int level = config.get<int>("system.debug_level", 2);
    spdlog::level::level_enum logLevel = static_cast<spdlog::level::level_enum>(
        std::clamp(level, 0, 5));
    utils::Logger::setLevel(logLevel);
    LOG_INFO("📋 日志级别: {}", spdlog::level::to_string_view(logLevel));

    // 3. 设置 Mali-G31 必需的环境变量
    setenv("PAN_I_WANT_A_BROKEN_VULKAN_DRIVER", "1", 1);
    LOG_INFO("🔧 环境变量: PAN_I_WANT_A_BROKEN_VULKAN_DRIVER=1");

    // 4. 查找测试图像
    LOG_INFO("\n[2/6] 扫描测试图像");
    std::string testDir = config.get<std::string>("output.test_image_dir", "images/test");
    std::vector<std::string> images = findTestImages(testDir);
    if (images.empty()) {
        LOG_ERROR("❌ 未找到测试图像，请检查目录: {}", testDir);
        return EXIT_FAILURE;
    }

    // 5. 初始化 Vulkan
    LOG_INFO("\n[3/6] 初始化 Vulkan");
    auto t0 = std::chrono::high_resolution_clock::now();
    vulkan::VulkanContext ctx;
    if (!ctx.initialize(false)) {
        LOG_ERROR("❌ Vulkan 上下文初始化失败");
        return EXIT_FAILURE;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    LOG_INFO("✅ Vulkan 初始化完成 ({} ms)", 
             std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());
    LOG_INFO("  设备: {}", ctx.getDeviceName());
    LOG_INFO("  Vulkan 版本: {}", ctx.getVulkanVersion());

    // 6. 创建立体匹配流水线
    LOG_INFO("\n[4/6] 创建立体匹配流水线");
    uint32_t camW   = config.get<uint32_t>("camera.width");
    uint32_t camH   = config.get<uint32_t>("camera.height");
    uint32_t maxDisp = config.get<uint32_t>("stereo.max_disparity", 64);

    vulkan::StereoPipeline pipeline(ctx);
    if (!pipeline.initialize(camW, camH, maxDisp)) {
        LOG_ERROR("❌ 立体匹配流水线初始化失败");
        return EXIT_FAILURE;
    }

    LOG_INFO("✅ 流水线初始化成功");
    LOG_INFO("  压缩后尺寸: {}x{}", pipeline.getCompressedWidth(),
             pipeline.getCompressedHeight());          // ✅ 修正
    LOG_INFO("  最大视差: {}", pipeline.getMaxDisparity());

    // 7. 批量处理图像
    LOG_INFO("\n[5/6] 批量处理测试图像");
    int success = 0, failed = 0;
    std::string outDir = config.get<std::string>("output.output_dir", "images/output");

    for (size_t i = 0; i < images.size(); ++i) {
        const std::string& path = images[i];
        std::string name = fs::path(path).stem().string();
        LOG_INFO("----------------------------------------");
        LOG_INFO("[{}/{}] 处理: {}", i + 1, images.size(), fs::path(path).filename().string());

        // 加载并分割图像
        cv::Mat left, right;
        if (!loadAndSplitImage(path, left, right, config)) {
            failed++;
            continue;
        }

        // 上传图像到 GPU
        auto uploadStart = std::chrono::high_resolution_clock::now();
        if (!pipeline.setLeftImage(left.data) || !pipeline.setRightImage(right.data)) {
            LOG_ERROR("❌ 上传图像数据失败");
            failed++;
            continue;
        }
        auto uploadEnd = std::chrono::high_resolution_clock::now();

        // 执行立体匹配
        auto compStart = std::chrono::high_resolution_clock::now();
        if (!pipeline.compute()) {
            LOG_ERROR("❌ 立体匹配计算失败");
            failed++;
            continue;
        }
        auto compEnd = std::chrono::high_resolution_clock::now();

        // ========== 读取调试信息（关键！）==========
        uint32_t censusDebug[8] = {0}, costDebug[8] = {0}, wtaDebug[8] = {0};
        pipeline.getIntermediateResult(5, censusDebug, sizeof(censusDebug));
        pipeline.getIntermediateResult(6, costDebug,   sizeof(costDebug));
        pipeline.getIntermediateResult(7, wtaDebug,    sizeof(wtaDebug));

        LOG_DEBUG("---- Census 调试 ----");
        LOG_DEBUG("  尺寸: {}x{}, 窗口: {}, 配置一致: {}",
                  censusDebug[0], censusDebug[1], censusDebug[2], censusDebug[3]);
        LOG_DEBUG("  左描述符: {:#010x}{:08x}", censusDebug[5], censusDebug[4]);
        LOG_DEBUG("  右描述符: {:#010x}{:08x}", censusDebug[7], censusDebug[6]);

        LOG_DEBUG("---- Cost 调试 ----");
        LOG_DEBUG("  尺寸: {}x{}, 最大视差: {}, 配置一致: {}",
                  costDebug[0], costDebug[1], costDebug[2], costDebug[3]);
        LOG_DEBUG("  左描述符: {:#010x}{:08x}", costDebug[5], costDebug[4]);
        LOG_DEBUG("  右描述符: {:#010x}{:08x}", costDebug[7], costDebug[6]);
        LOG_DEBUG("  第一个像素汉明距离 (d=0): {}", costDebug[7]);

        LOG_DEBUG("---- WTA 调试 ----");
        LOG_DEBUG("  最佳视差: {}, 最小代价: {}", wtaDebug[3], wtaDebug[4]);
        LOG_DEBUG("  次佳视差: {}, 次小代价: {}", wtaDebug[5], wtaDebug[6]);
        LOG_DEBUG("  唯一性: {}", wtaDebug[7] ? "是" : "否");

        // 获取视差图（16位）
        uint32_t w = pipeline.getCompressedWidth();    // ✅ 修正
        uint32_t h = pipeline.getCompressedHeight();   // ✅ 修正
        std::vector<uint16_t> disparity(w * h);
        if (!pipeline.getDisparityMap(disparity.data())) {
            LOG_ERROR("❌ 获取视差图失败");
            failed++;
            continue;
        }

        // 保存结果
        std::string outPath = outDir + "/" + name + "_disparity";
        if (saveDisparityMap(disparity.data(), w, h, outPath, config)) {
            success++;
            int uploadMs = std::chrono::duration_cast<std::chrono::milliseconds>(uploadEnd - uploadStart).count();
            int compMs   = std::chrono::duration_cast<std::chrono::milliseconds>(compEnd - compStart).count();
            LOG_INFO("⏱️ 上传: {} ms, 匹配: {} ms, 总计: {} ms, {:.1f} FPS",
                     uploadMs, compMs, uploadMs + compMs,
                     1000.0 / (uploadMs + compMs + 0.01));
        } else {
            failed++;
        }
    }

    // 8. 清理与汇总
    LOG_INFO("\n[6/6] 清理资源");
    ctx.waitIdle();

    auto totalEnd = std::chrono::high_resolution_clock::now();
    auto totalMs  = std::chrono::duration_cast<std::chrono::milliseconds>(totalEnd - totalStart).count();

    LOG_INFO("\n========================================");
    LOG_INFO("📊 测试汇总");
    LOG_INFO("========================================");
    LOG_INFO("  总运行时间: {} 秒", totalMs / 1000.0);
    LOG_INFO("  图像总数: {}", images.size());
    LOG_INFO("  成功: {}", success);
    LOG_INFO("  失败: {}", failed);
    LOG_INFO("  成功率: {:.1f}%", 100.0 * success / (success + failed + 0.01));
    LOG_INFO("========================================");
    LOG_INFO(success > 0 ? "✅ 测试通过！" : "❌ 测试失败！");
    LOG_INFO("   输出目录: {}", outDir);

    return success > 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
