#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace stereo_depth {
namespace calibration {

/**
 * @brief 立体相机标定器
 * 
 * 针对已经左右分好的320×480图像进行标定
 * 所有参数从配置文件读取
 */
class StereoCalibrator {
public:
    StereoCalibrator() = default;
    ~StereoCalibrator() = default;
    
    // 禁止拷贝
    StereoCalibrator(const StereoCalibrator&) = delete;
    StereoCalibrator& operator=(const StereoCalibrator&) = delete;
    
    /**
     * @brief 执行立体标定
     * @return 是否标定成功
     */
    bool calibrate();
    
    /**
     * @brief 获取标定错误（RMS）
     */
    double getCalibrationError() const { return m_rmsError; }
    
    /**
     * @brief 获取使用的图像数量
     */
    int getImagesUsed() const { return m_imagesUsed; }
    
private:
    /**
     * @brief 加载配置参数
     */
    bool loadConfiguration();
    
    /**
     * @brief 查找标定图像对
     */
    std::vector<std::pair<std::string, std::string>> findCalibrationImagePairs();
    
    /**
     * @brief 检测棋盘格角点
     * @param image 输入图像
     * @param corners 输出的角点坐标
     * @return 是否检测成功
     */
    bool detectChessboardCorners(const cv::Mat& image, std::vector<cv::Point2f>& corners);
    
    /**
     * @brief 保存标定结果到YAML文件
     */
    bool saveCalibrationResults();
    
    /**
     * @brief 生成标定报告
     */
    bool generateCalibrationReport();
    
    /**
     * @brief 验证标定结果（生成校正后的图像）
     */
    bool validateCalibrationResults();
    
private:
    // 标定参数
    cv::Size m_boardSize;            // 棋盘格内角点数 (宽度, 高度)
    float m_squareSize = 0.0f;       // 棋盘格方块物理尺寸 (米)
    cv::Size m_imageSize;            // 图像尺寸 (宽度, 高度)
    
    // 标定结果
    cv::Mat m_cameraMatrixLeft;
    cv::Mat m_distCoeffsLeft;
    cv::Mat m_cameraMatrixRight;
    cv::Mat m_distCoeffsRight;
    cv::Mat m_rotationMatrix;
    cv::Mat m_translationVector;
    cv::Mat m_essentialMatrix;
    cv::Mat m_fundamentalMatrix;
    cv::Mat m_rectificationTransformLeft;
    cv::Mat m_rectificationTransformRight;
    cv::Mat m_projectionMatrixLeft;
    cv::Mat m_projectionMatrixRight;
    cv::Mat m_disparityToDepthMappingMatrix;
    
    // 校正映射
    cv::Mat m_leftMap1, m_leftMap2;
    cv::Mat m_rightMap1, m_rightMap2;
    
    double m_rmsError = 0.0;
    int m_imagesUsed = 0;
    
    // 路径配置
    std::string m_calibrationDir;
    std::string m_outputDir;
    std::string m_configFile;
    
    // 标定过程中收集的数据
    std::vector<std::vector<cv::Point3f>> m_objectPoints;
    std::vector<std::vector<cv::Point2f>> m_imagePointsLeft;
    std::vector<std::vector<cv::Point2f>> m_imagePointsRight;
};

} // namespace calibration
} // namespace stereo_depth
