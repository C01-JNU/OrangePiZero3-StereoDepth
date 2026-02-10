#include "calibration/calibration_loader.hpp"
#include "calibration/stereo_rectifier.hpp"
#include "utils/logger.hpp"
#include "utils/config.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <sys/stat.h>
#include <dirent.h>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <fstream>
#include <cstdlib>
#include <cstring>

namespace {

// 辅助函数：检查文件是否存在
bool fileExists(const std::string& filename) {
    struct stat buffer;
    return (stat(filename.c_str(), &buffer) == 0);
}

// 创建目录（递归创建）
bool createDirectory(const std::string& path) {
    if (path.empty()) return false;
    
    // 如果目录已存在
    if (fileExists(path)) {
        struct stat st;
        if (stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
            return true;
        }
        return false;
    }
    
    // 递归创建父目录
    size_t pos = path.find_last_of('/');
    if (pos != std::string::npos) {
        std::string parent = path.substr(0, pos);
        if (!parent.empty() && !createDirectory(parent)) {
            return false;
        }
    }
    
    // 创建当前目录
    return (mkdir(path.c_str(), 0777) == 0);
}

// 获取目录中的文件列表（按模式过滤）
std::vector<std::string> getTestImages(const std::string& directory) {
    std::vector<std::string> files;
    
    // 首先检查目录是否存在
    if (!fileExists(directory)) {
        LOG_ERROR("目录不存在: {}", directory);
        return files;
    }
    
    // 使用 opendir/readdir 遍历目录
    DIR* dir = opendir(directory.c_str());
    if (!dir) {
        LOG_ERROR("无法打开目录: {}", directory);
        return files;
    }
    
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;
        
        // 跳过 . 和 ..
        if (filename == "." || filename == "..") {
            continue;
        }
        
        // 检查是否是文件
        std::string full_path = directory + "/" + filename;
        struct stat st;
        if (stat(full_path.c_str(), &st) != 0) {
            continue; // 无法获取文件状态
        }
        
        // 检查是否是普通文件
        if (!S_ISREG(st.st_mode)) {
            continue; // 不是普通文件
        }
        
        // 检查文件扩展名
        size_t dot_pos = filename.find_last_of('.');
        if (dot_pos == std::string::npos) {
            continue; // 没有扩展名
        }
        
        std::string ext = filename.substr(dot_pos);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        
        // 支持常见的图像扩展名
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
            files.push_back(full_path);
        }
    }
    
    closedir(dir);
    
    // 按文件名排序
    std::sort(files.begin(), files.end());
    
    LOG_INFO("在目录 {} 中找到 {} 个图像文件", directory, files.size());
    return files;
}

// 从完整路径中提取文件名（不含扩展名）
std::string getFileNameWithoutExtension(const std::string& path) {
    size_t last_slash = path.find_last_of("/\\");
    std::string filename = (last_slash != std::string::npos) ? 
                          path.substr(last_slash + 1) : path;
    
    size_t dot_pos = filename.find_last_of('.');
    if (dot_pos != std::string::npos) {
        filename = filename.substr(0, dot_pos);
    }
    
    return filename;
}

// 分割拼接图像为左右眼 - 添加详细调试信息
bool splitStereoImage(const cv::Mat& stitched_image, 
                     cv::Mat& left_image, 
                     cv::Mat& right_image) {
    LOG_DEBUG("开始分割拼接图像: {}x{}, 类型: {}", 
              stitched_image.cols, stitched_image.rows, stitched_image.type());
    
    if (stitched_image.cols % 2 != 0) {
        LOG_ERROR("拼接图像宽度必须是偶数，实际宽度: {}", stitched_image.cols);
        return false;
    }
    
    int single_width = stitched_image.cols / 2;
    int height = stitched_image.rows;
    
    LOG_DEBUG("分割参数: 总宽={}, 单眼宽={}, 高={}", 
              stitched_image.cols, single_width, height);
    
    // 检查图像数据
    LOG_DEBUG("图像统计数据:");
    double min_val, max_val;
    cv::minMaxLoc(stitched_image, &min_val, &max_val);
    LOG_DEBUG("  像素值范围: {:.1f} - {:.1f}", min_val, max_val);
    
    // 左半部分 (0 → single_width-1)
    cv::Rect left_rect(0, 0, single_width, height);
    left_image = stitched_image(left_rect).clone();
    
    // 右半部分 (single_width → end)
    cv::Rect right_rect(single_width, 0, single_width, height);
    right_image = stitched_image(right_rect).clone();
    
    LOG_DEBUG("分割完成: 左眼 {}x{}, 右眼 {}x{}", 
              left_image.cols, left_image.rows, 
              right_image.cols, right_image.rows);
    
    // 检查分割后的图像
    double left_min, left_max, right_min, right_max;
    cv::minMaxLoc(left_image, &left_min, &left_max);
    cv::minMaxLoc(right_image, &right_min, &right_max);
    LOG_DEBUG("左眼像素范围: {:.1f} - {:.1f}", left_min, left_max);
    LOG_DEBUG("右眼像素范围: {:.1f} - {:.1f}", right_min, right_max);
    
    // 保存分割后的图像用于调试
    static bool debug_enabled = false;
    static int debug_count = 0;
    
    if (debug_enabled && debug_count < 3) {  // 只保存前3张用于调试
        std::string debug_dir = "debug_images";
        createDirectory(debug_dir);
        
        std::string base_name = "split_" + std::to_string(debug_count);
        cv::imwrite(debug_dir + "/" + base_name + "_left.png", left_image);
        cv::imwrite(debug_dir + "/" + base_name + "_right.png", right_image);
        
        LOG_DEBUG("保存调试图像到: {}/{}_*.png", debug_dir, base_name);
        debug_count++;
    }
    
    return true;
}

// 验证图像尺寸
bool validateImageSize(const cv::Mat& image, const cv::Size& expected_size, 
                      const std::string& image_name) {
    if (image.cols != expected_size.width || image.rows != expected_size.height) {
        LOG_WARN("图像 '{}' 尺寸不匹配: 期望 {}x{}, 实际 {}x{}", 
                 image_name, expected_size.width, expected_size.height,
                 image.cols, image.rows);
        return false;
    }
    return true;
}

// 计算图像统计信息
void printImageStats(const cv::Mat& image, const std::string& name) {
    if (image.empty()) {
        LOG_WARN("{}: 图像为空", name);
        return;
    }
    
    double min_val, max_val;
    cv::minMaxLoc(image, &min_val, &max_val);
    
    cv::Scalar mean_val, stddev_val;
    cv::meanStdDev(image, mean_val, stddev_val);
    
    int non_zero = cv::countNonZero(image);
    int total_pixels = image.rows * image.cols;
    double non_zero_percent = (non_zero * 100.0) / total_pixels;
    
    LOG_DEBUG("{} 统计: {}x{}, 类型={}", name, image.cols, image.rows, image.type());
    LOG_DEBUG("  像素范围: {:.1f} - {:.1f}", min_val, max_val);
    LOG_DEBUG("  均值: {:.2f}, 标准差: {:.2f}", mean_val[0], stddev_val[0]);
    LOG_DEBUG("  非零像素: {}/{} ({:.1f}%)", non_zero, total_pixels, non_zero_percent);
}

} // 匿名命名空间

// 打印使用说明
void printUsage(const std::string& program_name) {
    std::cout << "立体校正测试程序\n";
    std::cout << "用法: " << program_name << " [选项]\n";
    std::cout << "选项:\n";
    std::cout << "  -c, --calibration <文件>  标定文件路径 (默认: calibration_results/stereo_calibration.yml)\n";
    std::cout << "  -i, --input <目录>        输入图像目录 (默认: images/test)\n";
    std::cout << "  -o, --output <目录>       输出图像目录 (默认: images/output)\n";
    std::cout << "  --crop                    裁剪到有效ROI区域\n";
    std::cout << "  --help                    显示此帮助信息\n";
    std::cout << "  --config <文件>           配置文件路径 (默认: config/global_config.yaml)\n";
    std::cout << "\n示例:\n";
    std::cout << "  " << program_name << " -c calibration.yml -i test_images -o output\n";
    std::cout << "  " << program_name << " --config my_config.yaml\n";
}

// 解析命令行参数
struct ProgramOptions {
    std::string calibration_file = "calibration_results/stereo_calibration.yml";
    std::string input_dir = "images/test";
    std::string output_dir = "images/output";  // 修改默认输出目录
    bool crop_to_valid_roi = false;
    bool use_config = false;
    std::string config_file = "config/global_config.yaml";
    bool debug_mode = false;  // 调试模式
};

ProgramOptions parseArguments(int argc, char** argv) {
    ProgramOptions options;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            exit(0);
        } else if (arg == "-c" || arg == "--calibration") {
            if (i + 1 < argc) {
                options.calibration_file = argv[++i];
            } else {
                std::cerr << "错误: --calibration 参数需要文件路径\n";
                exit(1);
            }
        } else if (arg == "-i" || arg == "--input") {
            if (i + 1 < argc) {
                options.input_dir = argv[++i];
            } else {
                std::cerr << "错误: --input 参数需要目录路径\n";
                exit(1);
            }
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) {
                options.output_dir = argv[++i];
            } else {
                std::cerr << "错误: --output 参数需要目录路径\n";
                exit(1);
            }
        } else if (arg == "--crop") {
            options.crop_to_valid_roi = true;
        } else if (arg == "--config") {
            if (i + 1 < argc) {
                options.config_file = argv[++i];
                options.use_config = true;
            } else {
                std::cerr << "错误: --config 参数需要文件路径\n";
                exit(1);
            }
        } else if (arg == "--debug") {
            options.debug_mode = true;
        } else {
            std::cerr << "未知参数: " << arg << "\n";
            printUsage(argv[0]);
            exit(1);
        }
    }
    
    return options;
}

// 从配置文件读取选项
void loadOptionsFromConfig(ProgramOptions& options) {
    try {
        auto& config_manager = stereo_depth::utils::ConfigManager::getInstance();
        
        // 根据 stereo_calibrator_main.cpp 的用法，应该是 loadGlobalConfig
        if (!config_manager.loadGlobalConfig(options.config_file)) {
            LOG_WARN("无法加载配置文件 {}, 使用命令行参数", options.config_file);
            return;
        }
        
        const auto& config = config_manager.getConfig();
        
        // 从配置文件读取标定文件路径
        std::string calibration_from_config = config.get<std::string>("calibration.calibration_file", "");
        if (!calibration_from_config.empty()) {
            options.calibration_file = calibration_from_config;
            LOG_INFO("从配置读取标定文件: {}", options.calibration_file);
        }
        
        // 从配置文件读取测试图像目录
        std::string test_dir_from_config = config.get<std::string>("output.test_dir", "");
        if (!test_dir_from_config.empty()) {
            options.input_dir = test_dir_from_config;
            LOG_INFO("从配置读取测试目录: {}", options.input_dir);
        }
        
        // 从配置文件读取输出目录
        std::string output_dir_from_config = config.get<std::string>("output.output_dir", "");
        if (output_dir_from_config.empty()) {
            // 尝试其他可能的键名
            output_dir_from_config = config.get<std::string>("output.dir", "");
        }
        
        if (!output_dir_from_config.empty()) {
            options.output_dir = output_dir_from_config;
            LOG_INFO("从配置读取输出目录: {}", options.output_dir);
        }
        
    } catch (const std::exception& e) {
        LOG_WARN("读取配置时出错: {}, 使用命令行参数", e.what());
    }
}

int main(int argc, char** argv) {
    // 初始化日志 - 根据 stereo_calibrator_main.cpp 的用法
    try {
        stereo_depth::utils::Logger::initialize("test_calibration", spdlog::level::info);
    } catch (const std::exception& e) {
        std::cerr << "错误: 日志系统初始化失败: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    
    // 解析命令行参数
    ProgramOptions options = parseArguments(argc, argv);
    
    LOG_INFO("=== 立体校正测试程序 ===");
    LOG_INFO("程序启动: {}", argv[0]);
    LOG_INFO("输出目录已设置为: {}", options.output_dir);
    
    // 如果指定了配置文件，从中读取参数
    if (options.use_config) {
        LOG_INFO("使用配置文件: {}", options.config_file);
        loadOptionsFromConfig(options);
    }
    
    // 显示最终选项
    LOG_INFO("参数配置:");
    LOG_INFO("  标定文件: {}", options.calibration_file);
    LOG_INFO("  输入目录: {}", options.input_dir);
    LOG_INFO("  输出目录: {}", options.output_dir);
    LOG_INFO("  裁剪到有效ROI: {}", options.crop_to_valid_roi ? "是" : "否");
    LOG_INFO("  调试模式: {}", options.debug_mode ? "是" : "否");
    
    // 1. 验证输入文件/目录存在
    if (!fileExists(options.calibration_file)) {
        LOG_ERROR("标定文件不存在: {}", options.calibration_file);
        return 1;
    }
    
    if (!fileExists(options.input_dir)) {
        LOG_ERROR("输入目录不存在: {}", options.input_dir);
        return 1;
    }
    
    // 创建输出目录
    if (!createDirectory(options.output_dir)) {
        LOG_ERROR("无法创建输出目录: {}", options.output_dir);
        return 1;
    }
    
    // 2. 加载标定参数
    stereo_depth::calibration::CalibrationLoader loader;
    stereo_depth::calibration::CalibrationParams params;
    
    LOG_INFO("正在加载标定参数...");
    if (!loader.loadFromFile(options.calibration_file, params)) {
        LOG_ERROR("加载标定参数失败");
        return 1;
    }
    
    LOG_INFO("标定参数摘要:");
    LOG_INFO("  图像尺寸: {}x{}", params.image_size.width, params.image_size.height);
    LOG_INFO("  左相机内参: fx={:.1f}, fy={:.1f}, cx={:.1f}, cy={:.1f}", 
             params.camera_matrix_left.at<double>(0, 0),
             params.camera_matrix_left.at<double>(1, 1),
             params.camera_matrix_left.at<double>(0, 2),
             params.camera_matrix_left.at<double>(1, 2));
    LOG_INFO("  右相机内参: fx={:.1f}, fy={:.1f}, cx={:.1f}, cy={:.1f}", 
             params.camera_matrix_right.at<double>(0, 0),
             params.camera_matrix_right.at<double>(1, 1),
             params.camera_matrix_right.at<double>(0, 2),
             params.camera_matrix_right.at<double>(1, 2));
    
    // 3. 初始化校正器
    stereo_depth::calibration::StereoRectifier rectifier;
    LOG_INFO("正在初始化立体校正器...");
    if (!rectifier.initialize(params)) {
        LOG_ERROR("初始化立体校正器失败");
        return 1;
    }
    
    // 获取有效ROI信息
    auto [left_roi, right_roi] = rectifier.getValidROI();
    LOG_INFO("有效ROI信息:");
    LOG_INFO("  左眼: [{}, {}, {}, {}]", left_roi.x, left_roi.y, left_roi.width, left_roi.height);
    LOG_INFO("  右眼: [{}, {}, {}, {}]", right_roi.x, right_roi.y, right_roi.width, right_roi.height);
    
    // 4. 获取测试图像
    LOG_INFO("正在扫描测试图像...");
    auto test_images = getTestImages(options.input_dir);
    if (test_images.empty()) {
        LOG_ERROR("未找到测试图像");
        return 1;
    }
    
    LOG_INFO("找到 {} 张测试图像", test_images.size());
    
    // 5. 处理每张图像
    int processed_count = 0;
    int failed_count = 0;
    int resized_count = 0;
    
    auto total_start_time = std::chrono::high_resolution_clock::now();
    
    // 创建调试目录（如果启用调试模式）
    std::string debug_dir = "debug_images";
    if (options.debug_mode) {
        createDirectory(debug_dir);
        LOG_INFO("调试模式启用，中间图像将保存到: {}", debug_dir);
    }
    
    for (size_t i = 0; i < test_images.size(); ++i) {
        const auto& image_path = test_images[i];
        
        // 提取文件名
        std::string filename = getFileNameWithoutExtension(image_path);
        
        LOG_INFO("处理图像 {}/{}: {}", i + 1, test_images.size(), filename);
        
        try {
            // 读取拼接图像
            LOG_DEBUG("读取图像: {}", image_path);
            cv::Mat stitched_image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
            if (stitched_image.empty()) {
                LOG_ERROR("无法读取图像: {}", image_path);
                failed_count++;
                continue;
            }
            
            printImageStats(stitched_image, "原始拼接图像");
            
            // 保存原始拼接图像用于调试
            if (options.debug_mode) {
                cv::imwrite(debug_dir + "/" + filename + "_original.png", stitched_image);
            }
            
            // 分割为左右眼
            LOG_DEBUG("分割拼接图像...");
            cv::Mat left_raw, right_raw;
            if (!splitStereoImage(stitched_image, left_raw, right_raw)) {
                LOG_ERROR("分割图像失败: {}", image_path);
                failed_count++;
                continue;
            }
            
            printImageStats(left_raw, "左眼原始图像");
            printImageStats(right_raw, "右眼原始图像");
            
            // 保存分割后的图像用于调试
            if (options.debug_mode) {
                cv::imwrite(debug_dir + "/" + filename + "_left_raw.png", left_raw);
                cv::imwrite(debug_dir + "/" + filename + "_right_raw.png", right_raw);
                
                // 创建并排对比图
                cv::Mat side_by_side;
                cv::hconcat(left_raw, right_raw, side_by_side);
                cv::imwrite(debug_dir + "/" + filename + "_side_by_side.png", side_by_side);
            }
            
            // 检查图像尺寸是否匹配标定参数
            cv::Size expected_size = params.image_size;
            bool left_size_ok = validateImageSize(left_raw, expected_size, "左眼");
            bool right_size_ok = validateImageSize(right_raw, expected_size, "右眼");
            
            if (!left_size_ok || !right_size_ok) {
                // 调整图像尺寸到标定尺寸
                LOG_WARN("调整图像尺寸从 {}x{} 到 {}x{}", 
                         left_raw.cols, left_raw.rows, 
                         expected_size.width, expected_size.height);
                
                cv::resize(left_raw, left_raw, expected_size, 0, 0, cv::INTER_LINEAR);
                cv::resize(right_raw, right_raw, expected_size, 0, 0, cv::INTER_LINEAR);
                resized_count++;
                
                // 保存调整后的图像用于调试
                if (options.debug_mode) {
                    cv::imwrite(debug_dir + "/" + filename + "_left_resized.png", left_raw);
                    cv::imwrite(debug_dir + "/" + filename + "_right_resized.png", right_raw);
                }
            }
            
            // 执行立体校正
            LOG_DEBUG("执行立体校正...");
            cv::Mat left_rectified, right_rectified;
            
            auto start_time = std::chrono::high_resolution_clock::now();
            
            if (!rectifier.rectifyPair(left_raw, right_raw, 
                                       left_rectified, right_rectified,
                                       options.crop_to_valid_roi)) {
                LOG_ERROR("校正图像失败: {}", image_path);
                failed_count++;
                continue;
            }
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            LOG_DEBUG("校正耗时: {} ms", duration.count());
            
            // 输出校正后图像的统计信息
            printImageStats(left_rectified, "左眼校正后图像");
            printImageStats(right_rectified, "右眼校正后图像");
            
            // 保存校正后的图像用于调试
            if (options.debug_mode) {
                cv::imwrite(debug_dir + "/" + filename + "_left_rect.png", left_rectified);
                cv::imwrite(debug_dir + "/" + filename + "_right_rect.png", right_rectified);
                
                // 创建并排对比图
                cv::Mat rectified_side_by_side;
                cv::hconcat(left_rectified, right_rectified, rectified_side_by_side);
                cv::imwrite(debug_dir + "/" + filename + "_rectified_side_by_side.png", rectified_side_by_side);
                
                // 创建差异图（仅用于调试）
                cv::Mat diff;
                cv::absdiff(left_rectified, right_rectified, diff);
                cv::imwrite(debug_dir + "/" + filename + "_diff.png", diff);
                
                double diff_mean = cv::mean(diff)[0];
                LOG_DEBUG("左右图像差异均值: {:.2f}", diff_mean);
            }
            
            // 保存校正结果到输出目录
            std::string output_filename = filename + "_rectified";
            if (!stereo_depth::calibration::StereoRectifier::saveRectifiedImages(
                    left_rectified, right_rectified, options.output_dir, output_filename)) {
                LOG_ERROR("保存校正图像失败");
                failed_count++;
                continue;
            }
            
            processed_count++;
            
            // 每处理5张图像输出一次进度
            if ((i + 1) % 5 == 0) {
                LOG_INFO("进度: {}/{} 完成", i + 1, test_images.size());
            }
            
        } catch (const cv::Exception& e) {
            LOG_ERROR("OpenCV异常处理图像 {}: {}", image_path, e.what());
            LOG_ERROR("OpenCV错误代码: {}, 错误: {}", e.code, e.err.c_str());
            failed_count++;
        } catch (const std::exception& e) {
            LOG_ERROR("标准异常处理图像 {}: {}", image_path, e.what());
            failed_count++;
        }
    }
    
    auto total_end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        total_end_time - total_start_time);
    
    // 6. 输出统计信息
    LOG_INFO("=== 处理完成 ===");
    LOG_INFO("处理统计:");
    LOG_INFO("  总图像数: {}", test_images.size());
    LOG_INFO("  成功处理: {}", processed_count);
    LOG_INFO("  失败: {}", failed_count);
    LOG_INFO("  调整尺寸: {}", resized_count);
    
    if (processed_count > 0) {
        double avg_time = total_duration.count() / static_cast<double>(processed_count);
        double success_rate = (processed_count * 100.0) / test_images.size();
        
        LOG_INFO("性能统计:");
        LOG_INFO("  总耗时: {:.2f} 秒", total_duration.count() / 1000.0);
        LOG_INFO("  平均每对: {:.2f} 毫秒", avg_time);
        LOG_INFO("  成功率: {:.1f}%", success_rate);
        
        LOG_INFO("校正结果保存在: {}", options.output_dir);
        
        // 列出输出文件
        auto output_files = getTestImages(options.output_dir);
        if (!output_files.empty()) {
            LOG_INFO("输出文件列表 (前5个):");
            int show_count = std::min(5, static_cast<int>(output_files.size()));
            for (int i = 0; i < show_count; ++i) {
                std::string short_name = getFileNameWithoutExtension(output_files[i]);
                LOG_INFO("  {}", short_name);
            }
            if (output_files.size() > show_count) {
                LOG_INFO("  ... 还有 {} 个文件", output_files.size() - show_count);
            }
        }
        
        if (options.debug_mode) {
            LOG_INFO("调试图像保存在: {}", debug_dir);
        }
        
    } else {
        LOG_ERROR("没有成功处理的图像");
    }
    
    // 7. 保存处理报告
    std::string report_file = options.output_dir + "/rectification_report.txt";
    std::ofstream report(report_file);
    if (report.is_open()) {
        report << "立体校正处理报告\n";
        report << "=================\n";
        report << "处理时间: " << __DATE__ << " " << __TIME__ << "\n";
        report << "标定文件: " << options.calibration_file << "\n";
        report << "输入目录: " << options.input_dir << "\n";
        report << "输出目录: " << options.output_dir << "\n";
        report << "裁剪到ROI: " << (options.crop_to_valid_roi ? "是" : "否") << "\n";
        report << "\n";
        report << "处理统计:\n";
        report << "  总图像数: " << test_images.size() << "\n";
        report << "  成功处理: " << processed_count << "\n";
        report << "  失败: " << failed_count << "\n";
        report << "  调整尺寸: " << resized_count << "\n";
        report << "\n";
        
        if (processed_count > 0) {
            double avg_time = total_duration.count() / static_cast<double>(processed_count);
            double success_rate = (processed_count * 100.0) / test_images.size();
            
            report << "性能统计:\n";
            report << "  总耗时: " << (total_duration.count() / 1000.0) << " 秒\n";
            report << "  平均每对: " << avg_time << " 毫秒\n";
            report << "  成功率: " << std::fixed << std::setprecision(1) << success_rate << "%\n";
        }
        
        report << "\n有效ROI信息:\n";
        report << "  左眼: [" << left_roi.x << ", " << left_roi.y << ", " 
               << left_roi.width << ", " << left_roi.height << "]\n";
        report << "  右眼: [" << right_roi.x << ", " << right_roi.y << ", " 
               << right_roi.width << ", " << right_roi.height << "]\n";
        
        report << "\n建议:\n";
        if (failed_count > 0) {
            report << "  1. 检查标定参数与图像尺寸是否匹配\n";
            report << "  2. 检查输入图像是否为正确的拼接格式\n";
            report << "  3. 使用 --debug 参数生成调试图像进行分析\n";
        }
        if (resized_count > 0) {
            report << "  注意: " << resized_count << " 张图像被调整了尺寸，请确保使用正确尺寸的图像\n";
        }
        
        report.close();
        LOG_INFO("处理报告已保存: {}", report_file);
    }
    
    return (failed_count == 0 && processed_count > 0) ? 0 : 1;
}
