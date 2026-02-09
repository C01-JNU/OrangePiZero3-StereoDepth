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
 * 针对已经左右分好的图像进行标定
 * 所有参数从配置文件读取
 * 支持多分组大小优化标定
 */
class StereoCalibrator {
public:
    StereoCalibrator() = default;
    ~StereoCalibrator() = default;
    
    // 禁止拷贝
    StereoCalibrator(const StereoCalibrator&) = delete;
    StereoCalibrator& operator=(const StereoCalibrator&) = delete;
    
    /**
     * @brief 执行立体标定（多分组优化）
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
     * @brief 执行单组标定
     * @param groupObjectPoints 物体点
     * @param groupImagePointsLeft 左图像点
     * @param groupImagePointsRight 右图像点
     * @param cameraMatrixLeft 输出左相机内参
     * @param distCoeffsLeft 输出左相机畸变
     * @param cameraMatrixRight 输出右相机内参
     * @param distCoeffsRight 输出右相机畸变
     * @param R 输出旋转矩阵
     * @param T 输出平移向量
     * @param E 输出本质矩阵
     * @param F 输出基础矩阵
     * @param rms 输出RMS误差
     * @return 是否标定成功
     */
    bool performGroupCalibration(
        const std::vector<std::vector<cv::Point3f>>& groupObjectPoints,
        const std::vector<std::vector<cv::Point2f>>& groupImagePointsLeft,
        const std::vector<std::vector<cv::Point2f>>& groupImagePointsRight,
        cv::Mat& cameraMatrixLeft, cv::Mat& distCoeffsLeft,
        cv::Mat& cameraMatrixRight, cv::Mat& distCoeffsRight,
        cv::Mat& R, cv::Mat& T, cv::Mat& E, cv::Mat& F,
        double& rms);
    
    /**
     * @brief 保存标定结果到YAML文件
     * @param bestGroupSize 最佳分组大小
     * @param bestGroupIndex 最佳组号
     * @param bestGroupIndices 最佳组图像索引
     * @return 是否保存成功
     */
    bool saveCalibrationResults(int bestGroupSize, int bestGroupIndex, 
                               const std::vector<int>& bestGroupIndices);
    
    /**
     * @brief 生成标定报告
     * @param bestGroupSize 最佳分组大小
     * @param bestGroupIndex 最佳组号
     * @param bestGroupIndices 最佳组图像索引
     * @return 是否生成成功
     */
    bool generateCalibrationReport(int bestGroupSize, int bestGroupIndex,
                                  const std::vector<int>& bestGroupIndices);
    
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
};

} // namespace calibration
} // namespace stereo_depth
