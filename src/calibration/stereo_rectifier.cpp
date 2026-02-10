#include "calibration/stereo_rectifier.hpp"
#include "utils/logger.hpp"
#include <sys/stat.h>
#include <opencv2/imgproc.hpp>
#include <chrono>

namespace stereo_depth {
namespace calibration {

StereoRectifier::StereoRectifier(const std::string& calibration_file) {
    loadAndInitialize(calibration_file);
}

bool StereoRectifier::initialize(const CalibrationParams& params) {
    if (!params.isValid()) {
        LOG_ERROR("无效的标定参数，无法初始化校正器");
        return false;
    }
    
    m_params = params;
    m_image_size = params.image_size;
    
    LOG_INFO("初始化立体校正处理器");
    LOG_INFO("  图像尺寸: {}x{}", m_image_size.width, m_image_size.height);
    LOG_INFO("  基线长度: {:.2f} mm", m_params.baseline_meters * 1000.0);
    
    // 计算校正映射表
    if (!computeRectificationMaps()) {
        LOG_ERROR("计算校正映射表失败");
        return false;
    }
    
    // 计算有效区域
    computeValidROI();
    
    m_initialized = true;
    LOG_INFO("立体校正处理器初始化完成");
    
    return true;
}

bool StereoRectifier::loadAndInitialize(const std::string& calibration_file) {
    CalibrationParams params;
    if (!m_loader.loadFromFile(calibration_file, params)) {
        LOG_ERROR("加载标定文件失败: {}", calibration_file);
        return false;
    }
    
    return initialize(params);
}

bool StereoRectifier::rectifyPair(const cv::Mat& left_image, 
                                 const cv::Mat& right_image,
                                 cv::Mat& left_rectified,
                                 cv::Mat& right_rectified,
                                 bool crop_to_valid_roi) {
    if (!m_initialized) {
        LOG_ERROR("校正器未初始化");
        return false;
    }
    
    // 检查输入图像尺寸
    if (left_image.size() != m_image_size || right_image.size() != m_image_size) {
        LOG_ERROR("输入图像尺寸不匹配: 期望 {}x{}, 实际左={}x{}, 右={}x{}",
                 m_image_size.width, m_image_size.height,
                 left_image.cols, left_image.rows,
                 right_image.cols, right_image.rows);
        return false;
    }
    
    // 检查图像类型（应该是单通道灰度图）
    if (left_image.type() != CV_8UC1 || right_image.type() != CV_8UC1) {
        LOG_ERROR("输入图像必须是单通道灰度图 (CV_8UC1)");
        return false;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // 执行校正
        cv::remap(left_image, left_rectified, m_left_map1, m_left_map2, 
                 cv::INTER_LINEAR, cv::BORDER_CONSTANT);
        cv::remap(right_image, right_rectified, m_right_map1, m_right_map2, 
                 cv::INTER_LINEAR, cv::BORDER_CONSTANT);
        
        // 如果要求裁剪到有效区域
        if (crop_to_valid_roi) {
            left_rectified = left_rectified(m_valid_roi_left);
            right_rectified = right_rectified(m_valid_roi_right);
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time);
        
        LOG_DEBUG("立体校正完成: {:.2f} ms", duration.count() / 1000.0);
        
        return true;
        
    } catch (const cv::Exception& e) {
        LOG_ERROR("OpenCV校正异常: {}", e.what());
        return false;
    } catch (const std::exception& e) {
        LOG_ERROR("标准校正异常: {}", e.what());
        return false;
    }
}

bool StereoRectifier::rectifyBatch(const std::vector<std::pair<cv::Mat, cv::Mat>>& image_pairs,
                                 std::vector<std::pair<cv::Mat, cv::Mat>>& rectified_pairs,
                                 bool crop_to_valid_roi) {
    if (!m_initialized) {
        LOG_ERROR("校正器未初始化");
        return false;
    }
    
    if (image_pairs.empty()) {
        LOG_WARN("输入图像对列表为空");
        return true; // 空列表视为成功
    }
    
    LOG_INFO("批量校正 {} 对图像", image_pairs.size());
    
    auto total_start = std::chrono::high_resolution_clock::now();
    rectified_pairs.clear();
    rectified_pairs.reserve(image_pairs.size());
    
    bool all_success = true;
    size_t success_count = 0;
    
    for (size_t i = 0; i < image_pairs.size(); ++i) {
        const auto& [left, right] = image_pairs[i];
        
        cv::Mat left_rect, right_rect;
        if (rectifyPair(left, right, left_rect, right_rect, crop_to_valid_roi)) {
            rectified_pairs.emplace_back(left_rect, right_rect);
            success_count++;
            
            if ((i + 1) % 10 == 0) {
                LOG_DEBUG("已处理 {}/{} 对图像", i + 1, image_pairs.size());
            }
        } else {
            LOG_ERROR("第 {} 对图像校正失败", i + 1);
            all_success = false;
            // 添加空图像对保持索引一致
            rectified_pairs.emplace_back(cv::Mat(), cv::Mat());
        }
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        total_end - total_start);
    
    double avg_time = total_duration.count() / static_cast<double>(success_count);
    LOG_INFO("批量校正完成: {}/{} 成功, 总耗时: {:.2f} ms, 平均: {:.2f} ms/对",
             success_count, image_pairs.size(), total_duration.count(), avg_time);
    
    return all_success;
}

bool StereoRectifier::computeRectificationMaps() {
    LOG_INFO("计算立体校正映射表...");
    
    try {
        // 左相机校正映射
        cv::initUndistortRectifyMap(
            m_params.camera_matrix_left,
            m_params.dist_coeffs_left,
            m_params.rectification_left,    // R1
            m_params.projection_left,        // P1
            m_image_size,
            CV_32FC1,                        // map1类型
            m_left_map1,
            m_left_map2
        );
        
        // 右相机校正映射
        cv::initUndistortRectifyMap(
            m_params.camera_matrix_right,
            m_params.dist_coeffs_right,
            m_params.rectification_right,    // R2
            m_params.projection_right,       // P2
            m_image_size,
            CV_32FC1,                        // map1类型
            m_right_map1,
            m_right_map2
        );
        
        LOG_INFO("校正映射表计算完成");
        LOG_DEBUG("左映射表尺寸: {}x{} (type={})", 
                 m_left_map1.cols, m_left_map1.rows, m_left_map1.type());
        LOG_DEBUG("右映射表尺寸: {}x{} (type={})", 
                 m_right_map1.cols, m_right_map1.rows, m_right_map1.type());
        
        return true;
        
    } catch (const cv::Exception& e) {
        LOG_ERROR("计算校正映射表异常: {}", e.what());
        return false;
    } catch (const std::exception& e) {
        LOG_ERROR("计算校正映射表标准异常: {}", e.what());
        return false;
    }
}

void StereoRectifier::computeValidROI() {
    LOG_INFO("计算有效区域ROI...");
    
    // 创建一个空白图像来测试有效区域
    cv::Mat test_image = cv::Mat::zeros(m_image_size, CV_8UC1);
    cv::Mat rectified_left, rectified_right;
    
    // 校正测试图像
    if (!rectifyPair(test_image, test_image, rectified_left, rectified_right, false)) {
        LOG_WARN("无法计算有效区域ROI，使用全图");
        m_valid_roi_left = cv::Rect(0, 0, m_image_size.width, m_image_size.height);
        m_valid_roi_right = cv::Rect(0, 0, m_image_size.width, m_image_size.height);
        return;
    }
    
    // 找到非零区域（校正后图像有效部分）
    cv::Mat left_mask = (rectified_left > 0);
    cv::Mat right_mask = (rectified_right > 0);
    
    std::vector<cv::Point> left_points, right_points;
    cv::findNonZero(left_mask, left_points);
    cv::findNonZero(right_mask, right_points);
    
    if (left_points.empty() || right_points.empty()) {
        LOG_WARN("校正后图像完全无效，使用全图");
        m_valid_roi_left = cv::Rect(0, 0, m_image_size.width, m_image_size.height);
        m_valid_roi_right = cv::Rect(0, 0, m_image_size.width, m_image_size.height);
        return;
    }
    
    // 计算边界框
    m_valid_roi_left = cv::boundingRect(left_points);
    m_valid_roi_right = cv::boundingRect(right_points);
    
    // 确保两个ROI大小一致（取交集）
    cv::Rect common_roi = m_valid_roi_left & m_valid_roi_right;
    
    if (common_roi.area() > 0) {
        m_valid_roi_left = common_roi;
        m_valid_roi_right = common_roi;
    }
    
    LOG_INFO("有效区域ROI:");
    LOG_INFO("  左眼: x={}, y={}, {}x{} (原图{:.1f}%)",
             m_valid_roi_left.x, m_valid_roi_left.y,
             m_valid_roi_left.width, m_valid_roi_left.height,
             (m_valid_roi_left.area() * 100.0) / (m_image_size.area()));
    LOG_INFO("  右眼: x={}, y={}, {}x{} (原图{:.1f}%)",
             m_valid_roi_right.x, m_valid_roi_right.y,
             m_valid_roi_right.width, m_valid_roi_right.height,
             (m_valid_roi_right.area() * 100.0) / (m_image_size.area()));
}

void StereoRectifier::reset() {
    LOG_INFO("重置立体校正处理器");
    
    m_params = CalibrationParams();
    m_image_size = cv::Size(0, 0);
    
    m_left_map1.release();
    m_left_map2.release();
    m_right_map1.release();
    m_right_map2.release();
    
    m_valid_roi_left = cv::Rect();
    m_valid_roi_right = cv::Rect();
    
    m_initialized = false;
}

bool StereoRectifier::saveRectifiedImages(const cv::Mat& left_rectified,
                                         const cv::Mat& right_rectified,
                                         const std::string& output_dir,
                                         const std::string& filename) {
    // 检查目录是否存在
    struct stat st;
    if (stat(output_dir.c_str(), &st) != 0) {
        if (mkdir(output_dir.c_str(), 0777) != 0) {
            LOG_ERROR("无法创建输出目录: {}", output_dir);
            return false;
        }
    }
    
    try {
        std::string left_path = output_dir + "/" + filename + "_left.png";
        std::string right_path = output_dir + "/" + filename + "_right.png";
        
        if (!cv::imwrite(left_path, left_rectified)) {
            LOG_ERROR("无法保存左校正图像: {}", left_path);
            return false;
        }
        
        if (!cv::imwrite(right_path, right_rectified)) {
            LOG_ERROR("无法保存右校正图像: {}", right_path);
            return false;
        }
        
        LOG_INFO("校正图像已保存: {} 和 {}", left_path, right_path);
        return true;
        
    } catch (const cv::Exception& e) {
        LOG_ERROR("保存校正图像异常: {}", e.what());
        return false;
    } catch (const std::exception& e) {
        LOG_ERROR("保存校正图像标准异常: {}", e.what());
        return false;
    }
}

} // namespace calibration
} // namespace stereo_depth
