#pragma once

#include "calibration/calibration_loader.hpp"
#include <opencv2/opencv.hpp>
#include <string>
#include <memory>

namespace stereo_depth {
namespace calibration {

/**
 * @brief 立体校正处理器
 * 
 * 负责使用标定参数对双目图像进行立体校正
 */
class StereoRectifier {
public:
    StereoRectifier() = default;
    
    /**
     * @brief 构造函数，直接加载标定文件
     * @param calibration_file 标定文件路径
     */
    explicit StereoRectifier(const std::string& calibration_file);
    
    /**
     * @brief 使用标定参数初始化校正器
     * @param params 标定参数
     * @return 初始化成功返回true
     */
    bool initialize(const CalibrationParams& params);
    
    /**
     * @brief 从文件加载标定参数并初始化
     * @param calibration_file 标定文件路径
     * @return 初始化成功返回true
     */
    bool loadAndInitialize(const std::string& calibration_file);
    
    /**
     * @brief 校正一对立体图像
     * @param left_image 左眼原始图像 (单通道灰度图)
     * @param right_image 右眼原始图像 (单通道灰度图)
     * @param left_rectified 输出：校正后的左眼图像
     * @param right_rectified 输出：校正后的右眼图像
     * @param crop_to_valid_roi 是否裁剪到有效区域
     * @return 校正成功返回true
     */
    bool rectifyPair(const cv::Mat& left_image, 
                     const cv::Mat& right_image,
                     cv::Mat& left_rectified,
                     cv::Mat& right_rectified,
                     bool crop_to_valid_roi = false);
    
    /**
     * @brief 校正一批立体图像
     * @param image_pairs 输入图像对列表
     * @param rectified_pairs 输出校正后图像对列表
     * @param crop_to_valid_roi 是否裁剪到有效区域
     * @return 校正成功返回true
     */
    bool rectifyBatch(const std::vector<std::pair<cv::Mat, cv::Mat>>& image_pairs,
                     std::vector<std::pair<cv::Mat, cv::Mat>>& rectified_pairs,
                     bool crop_to_valid_roi = false);
    
    /**
     * @brief 获取标定参数
     * @return 标定参数引用
     */
    const CalibrationParams& getCalibrationParams() const { return m_params; }
    
    /**
     * @brief 获取左相机校正映射
     * @return 映射表对 (map1, map2)
     */
    std::pair<cv::Mat, cv::Mat> getLeftMaps() const { 
        return {m_left_map1, m_left_map2}; 
    }
    
    /**
     * @brief 获取右相机校正映射
     * @return 映射表对 (map1, map2)
     */
    std::pair<cv::Mat, cv::Mat> getRightMaps() const { 
        return {m_right_map1, m_right_map2}; 
    }
    
    /**
     * @brief 获取有效区域ROI
     * @return ROI对 (左眼ROI, 右眼ROI)
     */
    std::pair<cv::Rect, cv::Rect> getValidROI() const { 
        return {m_valid_roi_left, m_valid_roi_right}; 
    }
    
    /**
     * @brief 检查校正器是否已初始化
     * @return 已初始化返回true
     */
    bool isInitialized() const { return m_initialized; }
    
    /**
     * @brief 获取图像尺寸
     * @return 图像尺寸
     */
    cv::Size getImageSize() const { return m_image_size; }
    
    /**
     * @brief 重置校正器状态
     */
    void reset();
    
    /**
     * @brief 保存校正后的图像用于验证
     * @param left_rectified 校正后的左眼图像
     * @param right_rectified 校正后的右眼图像
     * @param output_dir 输出目录
     * @param filename 文件名前缀
     * @return 保存成功返回true
     */
    static bool saveRectifiedImages(const cv::Mat& left_rectified,
                                   const cv::Mat& right_rectified,
                                   const std::string& output_dir,
                                   const std::string& filename = "rectified");
    
private:
    /**
     * @brief 计算校正映射表
     * @return 计算成功返回true
     */
    bool computeRectificationMaps();
    
    /**
     * @brief 计算有效区域ROI
     */
    void computeValidROI();
    
    CalibrationParams m_params;            // 标定参数
    cv::Size m_image_size;                 // 图像尺寸
    
    // 校正映射表
    cv::Mat m_left_map1, m_left_map2;
    cv::Mat m_right_map1, m_right_map2;
    
    // 有效区域
    cv::Rect m_valid_roi_left;
    cv::Rect m_valid_roi_right;
    
    bool m_initialized = false;
    CalibrationLoader m_loader;            // 标定参数加载器
};

} // namespace calibration
} // namespace stereo_depth
