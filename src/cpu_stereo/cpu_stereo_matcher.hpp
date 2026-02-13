#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <memory>

namespace stereo_depth {
namespace cpu_stereo {

/**
 * @brief CPU 立体匹配器，支持多种算法模式
 * 
 * 算法模式（通过配置文件 stereo.algorithm 选择）：
 *   - "census_wta" : 自定义 Census 变换 + 汉明距离 + WTA（已完整实现）
 *   - "bm"         : OpenCV StereoBM
 *   - "sgbm"       : OpenCV StereoSGBM
 *   - "default"    : 回退到 sgbm
 */
class CpuStereoMatcher {
public:
    CpuStereoMatcher();
    ~CpuStereoMatcher();

    // 从配置管理器初始化参数
    bool initializeFromConfig();

    // 直接设置参数
    void setParameters(
        int numDisparities,      // 最大视差
        int blockSize,          // Census窗口大小 / BM窗口
        int uniquenessRatio,    // 唯一性比率（%）
        int p1, int p2,         // SGBM惩罚参数
        bool useMedianFilter,
        int medianFilterSize,
        const std::string& algorithm = "sgbm"
    );

    // 核心计算接口
    cv::Mat compute(const cv::Mat& left, const cv::Mat& right);

    // 获取视差图（16位无符号，无效视差标记为 maxDisparity）
    cv::Mat getDisparityMap() const { return disparity_16u_; }

    // 获取运行时间统计（毫秒）
    double getLastTimeMs() const { return last_time_ms_; }

private:
    // 算法模式枚举
    enum class Algorithm {
        CENSUS_WTA,
        BM,
        SGBM,
        UNKNOWN
    };

    // 参数
    int num_disparities_ = 64;          // 自定义 Census 使用的视差范围
    int num_disparities_opencv_ = 64;   // OpenCV 对齐后的视差范围（16的倍数）
    int block_size_ = 7;
    int uniqueness_ratio_ = 15;
    int p1_ = 8;
    int p2_ = 32;
    bool use_median_filter_ = true;
    int median_filter_size_ = 3;
    Algorithm algorithm_ = Algorithm::SGBM;

    // 结果
    cv::Mat disparity_16u_;
    double last_time_ms_ = 0.0;

    // 内部实现
    cv::Mat censusWTA(const cv::Mat& left, const cv::Mat& right);
    cv::Mat computeBM(const cv::Mat& left, const cv::Mat& right);
    cv::Mat computeSGBM(const cv::Mat& left, const cv::Mat& right);
    void applyMedianFilter(cv::Mat& disp, int size);
};

} // namespace cpu_stereo
} // namespace stereo_depth
