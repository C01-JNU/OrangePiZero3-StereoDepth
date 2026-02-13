#include <iostream>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>

// GPU 模块
#include "vulkan/context.hpp"
#include "vulkan/stereo_pipeline.hpp"
// CPU 模块
#include "cpu_stereo/cpu_stereo_matcher.hpp"
// 通用工具
#include "utils/logger.hpp"
#include "utils/config.hpp"

using namespace stereo_depth;
using namespace stereo_depth::vulkan;
using namespace stereo_depth::cpu_stereo;

bool isImageFile(const std::string& filename) {
    std::string ext;
    size_t pos = filename.rfind('.');
    if (pos != std::string::npos) {
        ext = filename.substr(pos);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp";
    }
    return false;
}

std::vector<std::string> listImageFiles(const std::string& dir) {
    std::vector<std::string> files;
    DIR* dp = opendir(dir.c_str());
    if (!dp) {
        LOG_ERROR("无法打开目录: {}", dir);
        return files;
    }
    struct dirent* entry;
    while ((entry = readdir(dp))) {
        std::string name = entry->d_name;
        if (name == "." || name == "..") continue;
        if (isImageFile(name)) {
            files.push_back(dir + "/" + name);
        }
    }
    closedir(dp);
    std::sort(files.begin(), files.end());
    return files;
}

bool ensureDirectory(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) == 0) {
        if (S_ISDIR(st.st_mode)) return true;
        LOG_ERROR("路径存在但不是目录: {}", path);
        return false;
    }
    if (mkdir(path.c_str(), 0755) == 0) return true;
    LOG_ERROR("创建目录失败: {}", path);
    return false;
}

int main(int argc, char** argv) {
    utils::Logger::initialize("stereo", spdlog::level::info);
    LOG_INFO("=========================================");
    LOG_INFO("  立体匹配批量处理工具启动");
    LOG_INFO("=========================================");

    auto& cfg_mgr = utils::ConfigManager::getInstance();
    if (!cfg_mgr.loadGlobalConfig("config/global_config.yaml")) {
        LOG_ERROR("加载全局配置失败");
        return -1;
    }
    const auto& cfg = cfg_mgr.getConfig();

    std::string system_mode = cfg.get<std::string>("system.mode", "gpu");
    std::string test_dir = cfg.get<std::string>("output.test_image_dir", "images/test");
    std::string out_dir  = cfg.get<std::string>("output.output_dir", "images/output");
    uint32_t cam_width   = cfg.get<uint32_t>("camera.width", 640);
    uint32_t cam_height  = cfg.get<uint32_t>("camera.height", 480);
    uint32_t single_width = cam_width / 2;

    LOG_INFO("运行模式: {}", system_mode);
    LOG_INFO("测试图像目录: {}", test_dir);
    LOG_INFO("输出目录:     {}", out_dir);
    LOG_INFO("拼接图像尺寸: {}x{}", cam_width, cam_height);
    LOG_INFO("单眼尺寸:     {}x{}", single_width, cam_height);

    if (!ensureDirectory(out_dir)) {
        LOG_ERROR("无法创建输出目录");
        return -1;
    }

    std::vector<std::string> image_files = listImageFiles(test_dir);
    if (image_files.empty()) {
        LOG_ERROR("目录 {} 中没有找到图像文件", test_dir);
        return -1;
    }
    LOG_INFO("找到 {} 个图像文件", image_files.size());

    // ---------- 根据模式选择处理引擎 ----------
    if (system_mode == "cpu") {
        LOG_INFO("使用 CPU 立体匹配引擎");
        CpuStereoMatcher cpu_matcher;
        if (!cpu_matcher.initializeFromConfig()) {
            LOG_ERROR("CPU 匹配器初始化失败");
            return -1;
        }

        int success_count = 0;
        auto total_start = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < image_files.size(); ++i) {
            const std::string& img_path = image_files[i];
            LOG_INFO("[{}/{}] 处理: {}", i+1, image_files.size(), img_path);

            cv::Mat stitched = cv::imread(img_path, cv::IMREAD_COLOR);
            if (stitched.empty()) {
                LOG_WARN("  无法读取图像，跳过");
                continue;
            }

            cv::Mat resized;
            cv::resize(stitched, resized, cv::Size(cam_width, cam_height));

            cv::Mat gray;
            cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);
            cv::Mat left_img  = gray(cv::Rect(0, 0, single_width, cam_height)).clone();
            cv::Mat right_img = gray(cv::Rect(single_width, 0, single_width, cam_height)).clone();

            cv::Mat disp_16u = cpu_matcher.compute(left_img, right_img);
            double time_ms = cpu_matcher.getLastTimeMs();

            // 检查是否全黑
            double min_val, max_val;
            cv::minMaxLoc(disp_16u, &min_val, &max_val);
            LOG_DEBUG("  视差统计 - 最小值: {:.0f}, 最大值: {:.0f}", min_val, max_val);

            cv::Mat disp_8u;
            disp_16u.convertTo(disp_8u, CV_8U, 255.0 / (max_val > 0 ? max_val : 64.0));

            std::string base_name = img_path.substr(img_path.find_last_of('/') + 1);
            size_t dot_pos = base_name.rfind('.');
            if (dot_pos != std::string::npos) base_name = base_name.substr(0, dot_pos);
            std::string out_path = out_dir + "/" + base_name + "_disparity_cpu.png";

            cv::imwrite(out_path, disp_8u);
            LOG_INFO("  耗时: {:.2f} ms, 视差范围: {:.0f}-{:.0f}, 已保存: {}",
                     time_ms, min_val, max_val, out_path);
            success_count++;
        }

        auto total_end = std::chrono::high_resolution_clock::now();
        float total_sec = std::chrono::duration<float>(total_end - total_start).count();
        LOG_INFO("=========================================");
        LOG_INFO("CPU 批量处理完成");
        LOG_INFO("成功: {} / {} 帧", success_count, image_files.size());
        LOG_INFO("总耗时: {:.2f} 秒", total_sec);
        if (success_count > 0) {
            LOG_INFO("平均每帧: {:.2f} ms", total_sec * 1000.0 / success_count);
        }

    } else {
        // ========== GPU 模式（原有 Vulkan 流水线）==========
        LOG_INFO("使用 GPU Vulkan 立体匹配引擎");

        VulkanContext ctx;
        if (!ctx.initialize()) {
            LOG_ERROR("VulkanContext 初始化失败");
            return -1;
        }

        StereoPipeline pipeline(ctx);
        if (!pipeline.initialize()) {
            LOG_ERROR("流水线初始化失败");
            return -1;
        }

        int success_count = 0;
        auto total_start = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < image_files.size(); ++i) {
            const std::string& img_path = image_files[i];
            LOG_INFO("[{}/{}] 处理: {}", i+1, image_files.size(), img_path);

            cv::Mat stitched = cv::imread(img_path, cv::IMREAD_COLOR);
            if (stitched.empty()) {
                LOG_WARN("  无法读取图像，跳过");
                continue;
            }

            cv::Mat resized;
            cv::resize(stitched, resized, cv::Size(cam_width, cam_height));

            cv::Mat gray;
            cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);
            cv::Mat left_img  = gray(cv::Rect(0, 0, single_width, cam_height));
            cv::Mat right_img = gray(cv::Rect(single_width, 0, single_width, cam_height));

            pipeline.setLeftImage(left_img.data);
            pipeline.setRightImage(right_img.data);

            auto frame_start = std::chrono::high_resolution_clock::now();
            if (!pipeline.compute()) {
                LOG_WARN("  计算失败，跳过");
                continue;
            }
            auto frame_end = std::chrono::high_resolution_clock::now();
            float frame_ms = std::chrono::duration<float, std::milli>(frame_end - frame_start).count();

            std::vector<uint16_t> disparity(pipeline.getBaseWidth() * pipeline.getBaseHeight());
            if (!pipeline.getDisparityMap(disparity.data())) {
                LOG_WARN("  获取视差图失败，跳过");
                continue;
            }

            cv::Mat disp_map(pipeline.getBaseHeight(), pipeline.getBaseWidth(), CV_16UC1, disparity.data());
            double min_disp, max_disp;
            cv::minMaxLoc(disp_map, &min_disp, &max_disp);
            cv::Mat disp_8u;
            disp_map.convertTo(disp_8u, CV_8U, 255.0 / (max_disp > 0 ? max_disp : 64.0));

            std::string base_name = img_path.substr(img_path.find_last_of('/') + 1);
            size_t dot_pos = base_name.rfind('.');
            if (dot_pos != std::string::npos) base_name = base_name.substr(0, dot_pos);
            std::string out_path = out_dir + "/" + base_name + "_disparity_gpu.png";

            cv::imwrite(out_path, disp_8u);
            LOG_INFO("  耗时: {:.2f} ms, 视差范围: {:.0f}-{:.0f}, 已保存: {}",
                     frame_ms, min_disp, max_disp, out_path);
            success_count++;
        }

        auto total_end = std::chrono::high_resolution_clock::now();
        float total_sec = std::chrono::duration<float>(total_end - total_start).count();
        LOG_INFO("=========================================");
        LOG_INFO("GPU 批量处理完成");
        LOG_INFO("成功: {} / {} 帧", success_count, image_files.size());
        LOG_INFO("总耗时: {:.2f} 秒", total_sec);
        if (success_count > 0) {
            LOG_INFO("平均每帧: {:.2f} ms", total_sec * 1000.0 / success_count);
        }
    }

    LOG_INFO("=========================================");
    return 0;
}
