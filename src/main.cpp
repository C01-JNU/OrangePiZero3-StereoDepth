#include <iostream>
#include <chrono>
#include <thread>
#include <memory>
#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <unistd.h>

#include "utils/logger.hpp"
#include "utils/config.hpp"
#include "calibration/stereo_rectifier.hpp"
#include "calibration/calibration_loader.hpp"
#include "camera/camera_factory.h"

#if ENABLE_CPU
#include "cpu_stereo/cpu_stereo_matcher.hpp"
#endif

#if ENABLE_GPU
#include "vulkan/context.hpp"
#include "vulkan/stereo_pipeline.hpp"
#endif

using namespace stereo_depth::utils;
using namespace stereo_depth;

// 获取可执行文件所在目录
std::string getExeDir() {
    char result[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
    if (count != -1) {
        result[count] = '\0';
        std::filesystem::path exePath(result);
        return exePath.parent_path().string();
    }
    return ".";
}

int main() {
    Logger::initialize("stereo_depth", spdlog::level::info);
    LOG_INFO("=========================================");
    LOG_INFO("  OrangePiZero3-StereoDepth 实时处理程序");
    LOG_INFO("=========================================");

    // 获取可执行文件目录
    std::string exeDir = getExeDir();
    LOG_INFO("可执行文件目录: {}", exeDir);

    // 加载配置文件（与可执行文件同目录下的 config/global_config.yaml）
    std::string configPath = (std::filesystem::path(exeDir) / "config" / "global_config.yaml").string();
    LOG_INFO("配置文件路径: {}", configPath);

    auto& cfg_mgr = ConfigManager::getInstance();
    if (!cfg_mgr.loadGlobalConfig(configPath)) {
        LOG_ERROR("加载全局配置失败");
        return -1;
    }
    const auto& cfg = cfg_mgr.getConfig();

    std::string system_mode = cfg.get<std::string>("system.mode", "cpu");
    int target_fps = cfg.get<int>("performance.target_fps", 15);
    int cam_width = cfg.get<int>("camera.width", 640);
    int cam_height = cfg.get<int>("camera.height", 480);
    int single_width = cam_width / 2;
    std::string camera_driver = cfg.get<std::string>("camera.driver", "mock");

    // 初始化摄像头
    auto cam = camera::CameraFactory::create(camera_driver);
    if (!cam) {
        LOG_ERROR("创建摄像头失败");
        return -1;
    }
    if (!cam->init(cam_width, cam_height, target_fps)) {
        LOG_ERROR("摄像头初始化失败");
        return -1;
    }
    LOG_INFO("摄像头已打开: {} ({}x{})", cam->getName(), cam->getWidth(), cam->getHeight());

    // 初始化立体校正
    bool use_rectification = cfg.get<bool>("calibration.rectify_images", false);
    std::string calib_file = cfg.get<std::string>("calibration.calibration_file", "calibration_results/stereo_calibration.yml");
    std::string calibPath = (std::filesystem::path(exeDir) / "calibration_results" / "stereo_calibration.yml").string();
    std::unique_ptr<calibration::StereoRectifier> rectifier = nullptr;
    if (use_rectification) {
        calibration::CalibrationParams params;
        calibration::CalibrationLoader loader;
        if (loader.loadFromFile(calibPath, params)) {
            rectifier = std::make_unique<calibration::StereoRectifier>();
            if (rectifier->initialize(params, calibration::RectificationMode::SCALE_TO_FIT)) {
                LOG_INFO("立体校正器初始化成功");
            } else {
                LOG_ERROR("立体校正器初始化失败，将跳过校正");
                rectifier = nullptr;
            }
        } else {
            LOG_ERROR("加载标定文件失败，将跳过校正");
        }
    }

    // 初始化深度计算引擎
#if ENABLE_CPU
    cpu_stereo::CpuStereoMatcher cpu_matcher;
    if (!cpu_matcher.initializeFromConfig()) {
        LOG_ERROR("CPU 匹配器初始化失败");
        return -1;
    }
    LOG_INFO("CPU 立体匹配引擎已初始化");
#elif ENABLE_GPU
    vulkan::VulkanContext ctx;
    if (!ctx.initialize()) {
        LOG_ERROR("VulkanContext 初始化失败");
        return -1;
    }
    vulkan::StereoPipeline pipeline(ctx);
    if (!pipeline.initialize()) {
        LOG_ERROR("GPU 流水线初始化失败");
        return -1;
    }
    LOG_INFO("GPU 立体匹配引擎已初始化");
#else
    LOG_ERROR("未启用任何深度计算模块");
    return -1;
#endif

    // 帧率控制队列
    std::queue<std::pair<cv::Mat, cv::Mat>> frame_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::atomic<bool> running{true};

    // 采集线程
    std::thread capture_thread([&]() {
        while (running) {
            cv::Mat left, right;
            if (cam->grab(left, right)) {
                std::lock_guard<std::mutex> lock(queue_mutex);
                if (frame_queue.size() >= 2) {
                    frame_queue.pop();
                }
                frame_queue.emplace(left.clone(), right.clone());
                queue_cv.notify_one();
            }
        }
    });

    // 处理循环
    auto frame_duration = std::chrono::milliseconds(1000 / target_fps);
    auto next_frame_time = std::chrono::steady_clock::now();
    int frame_count = 0;
    auto fps_start = std::chrono::steady_clock::now();

    LOG_INFO("开始处理，目标帧率: {} FPS", target_fps);

    // 输出目录（与可执行文件同目录下的 images/output）
    std::string out_dir = (std::filesystem::path(exeDir) / "images" / "output").string();
    std::filesystem::create_directories(out_dir);

    while (running) {
        // 从队列取最新帧
        cv::Mat left, right;
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (frame_queue.empty()) {
                queue_cv.wait_for(lock, std::chrono::milliseconds(100));
                if (frame_queue.empty()) continue;
            }
            while (frame_queue.size() > 1) {
                frame_queue.pop();
            }
            auto& frame = frame_queue.front();
            left = frame.first;
            right = frame.second;
            frame_queue.pop();
        }

        auto proc_start = std::chrono::high_resolution_clock::now();

        // 立体校正
        cv::Mat left_rect, right_rect;
        if (rectifier) {
            if (!rectifier->rectifyPair(left, right, left_rect, right_rect)) {
                LOG_WARN("校正失败，跳过此帧");
                continue;
            }
        } else {
            left_rect = left;
            right_rect = right;
        }

        // 执行立体匹配
        cv::Mat disparity;
#if ENABLE_CPU
        disparity = cpu_matcher.compute(left_rect, right_rect);
#elif ENABLE_GPU
        pipeline.setLeftImage(left_rect.data);
        pipeline.setRightImage(right_rect.data);
        if (!pipeline.compute()) {
            LOG_WARN("GPU 计算失败");
            continue;
        }
        std::vector<uint16_t> disp_buf(pipeline.getBaseWidth() * pipeline.getBaseHeight());
        pipeline.getDisparityMap(disp_buf.data());
        disparity = cv::Mat(pipeline.getBaseHeight(), pipeline.getBaseWidth(), CV_16UC1, disp_buf.data()).clone();
#endif

        auto proc_end = std::chrono::high_resolution_clock::now();
        float proc_ms = std::chrono::duration<float, std::milli>(proc_end - proc_start).count();

        // 保存视差图
        std::string out_path = out_dir + "/disparity_" + std::to_string(frame_count) + ".png";
        cv::Mat disp_8u;
        double min_val, max_val;
        cv::minMaxLoc(disparity, &min_val, &max_val);
        disparity.convertTo(disp_8u, CV_8U, 255.0 / (max_val > 0 ? max_val : 64.0));
        cv::imwrite(out_path, disp_8u);

        // 统计帧率
        frame_count++;
        auto now = std::chrono::steady_clock::now();
        float elapsed_sec = std::chrono::duration<float>(now - fps_start).count();
        if (elapsed_sec >= 5.0f) {
            float fps = frame_count / elapsed_sec;
            LOG_INFO("实际处理帧率: {:.2f} FPS (目标 {} FPS)", fps, target_fps);
            frame_count = 0;
            fps_start = now;
        }

        LOG_DEBUG("帧处理耗时: {:.2f} ms", proc_ms);

        // 等待以维持目标帧率
        auto now_time = std::chrono::steady_clock::now();
        if (now_time < next_frame_time) {
            std::this_thread::sleep_until(next_frame_time);
        }
        next_frame_time += frame_duration;
    }

    capture_thread.join();
    LOG_INFO("程序结束");
    return 0;
}
