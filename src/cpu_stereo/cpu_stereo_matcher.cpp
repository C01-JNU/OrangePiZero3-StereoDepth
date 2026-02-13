#include "cpu_stereo/cpu_stereo_matcher.hpp"
#include "utils/config.hpp"
#include "utils/logger.hpp"
#include <chrono>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <bitset>

namespace stereo_depth {
namespace cpu_stereo {

CpuStereoMatcher::CpuStereoMatcher() {
}

CpuStereoMatcher::~CpuStereoMatcher() {
}

bool CpuStereoMatcher::initializeFromConfig() {
    auto& cfg = utils::ConfigManager::getInstance().getConfig();

    // 读取立体匹配参数
    int max_disp = cfg.get<int>("stereo.disparity_range", 64);
    // OpenCV BM/SGBM 要求视差数为16的倍数，自定义 Census 无此限制
    int max_disp_opencv = ((max_disp + 15) / 16) * 16;
    
    int win_size = cfg.get<int>("stereo.census_window", 7);
    // 窗口必须为奇数
    if (win_size % 2 == 0) win_size += 1;
    if (win_size < 3) win_size = 3;
    if (win_size > 15) win_size = 15;  // 太大速度极慢

    int uniqueness = cfg.get<int>("stereo.uniqueness_ratio", 15);
    float p1_f = cfg.get<float>("stereo.penalty_p1", 8.0f);
    float p2_f = cfg.get<float>("stereo.penalty_p2", 32.0f);
    bool use_median = cfg.get<bool>("stereo.use_median_filter", true);
    int median_size = cfg.get<int>("stereo.median_filter_size", 3);
    if (median_size % 2 == 0) median_size += 1;
    if (median_size < 3) median_size = 3;
    if (median_size > 9) median_size = 9;
    
    std::string algo = cfg.get<std::string>("stereo.algorithm", "sgbm");

    // 保存参数（自定义 Census 使用原始 max_disp，OpenCV 使用对齐后的）
    num_disparities_ = max_disp;
    num_disparities_opencv_ = max_disp_opencv;
    block_size_ = win_size;
    uniqueness_ratio_ = uniqueness;
    p1_ = static_cast<int>(p1_f * 8);   // SGBM 典型转换
    p2_ = static_cast<int>(p2_f * 4);
    use_median_filter_ = use_median;
    median_filter_size_ = median_size;

    // 解析算法字符串
    std::string algo_lower = algo;
    std::transform(algo_lower.begin(), algo_lower.end(), algo_lower.begin(), ::tolower);
    if (algo_lower == "census_wta" || algo_lower == "census") {
        algorithm_ = Algorithm::CENSUS_WTA;
    } else if (algo_lower == "bm") {
        algorithm_ = Algorithm::BM;
    } else if (algo_lower == "sgbm") {
        algorithm_ = Algorithm::SGBM;
    } else {
        LOG_WARN("未知算法 '{}'，使用 SGBM 作为默认", algo);
        algorithm_ = Algorithm::SGBM;
    }

    LOG_INFO("CPU立体匹配器初始化完成，算法: {}, 视差范围: {}, 窗口: {}", 
             algo, num_disparities_, block_size_);
    return true;
}

void CpuStereoMatcher::setParameters(
    int numDisparities,
    int blockSize,
    int uniquenessRatio,
    int p1, int p2,
    bool useMedianFilter,
    int medianFilterSize,
    const std::string& algorithm)
{
    num_disparities_ = numDisparities;
    num_disparities_opencv_ = ((numDisparities + 15) / 16) * 16;
    block_size_ = (blockSize % 2 == 0) ? blockSize + 1 : blockSize;
    if (block_size_ < 3) block_size_ = 3;
    if (block_size_ > 15) block_size_ = 15;

    uniqueness_ratio_ = uniquenessRatio;
    p1_ = p1;
    p2_ = p2;
    use_median_filter_ = useMedianFilter;
    median_filter_size_ = (medianFilterSize % 2 == 0) ? medianFilterSize + 1 : medianFilterSize;
    if (median_filter_size_ < 3) median_filter_size_ = 3;
    if (median_filter_size_ > 9) median_filter_size_ = 9;

    std::string algo_lower = algorithm;
    std::transform(algo_lower.begin(), algo_lower.end(), algo_lower.begin(), ::tolower);
    if (algo_lower == "census_wta" || algo_lower == "census") {
        algorithm_ = Algorithm::CENSUS_WTA;
    } else if (algo_lower == "bm") {
        algorithm_ = Algorithm::BM;
    } else {
        algorithm_ = Algorithm::SGBM;
    }
}

cv::Mat CpuStereoMatcher::compute(const cv::Mat& left, const cv::Mat& right) {
    auto start = std::chrono::high_resolution_clock::now();

    cv::Mat disp;
    switch (algorithm_) {
        case Algorithm::CENSUS_WTA:
            disp = censusWTA(left, right);
            break;
        case Algorithm::BM:
            disp = computeBM(left, right);
            break;
        case Algorithm::SGBM:
            disp = computeSGBM(left, right);
            break;
        default:
            disp = computeSGBM(left, right);
            break;
    }

    // 中值滤波后处理（如果启用）
    if (use_median_filter_ && !disp.empty()) {
        applyMedianFilter(disp, median_filter_size_);
    }

    auto end = std::chrono::high_resolution_clock::now();
    last_time_ms_ = std::chrono::duration<double, std::milli>(end - start).count();

    // 转换为 16位无符号，无效像素标记为 max_disparity
    disp.convertTo(disparity_16u_, CV_16U);
    return disparity_16u_;
}

// ---------- 自定义 Census + WTA（完整实现）----------
cv::Mat CpuStereoMatcher::censusWTA(const cv::Mat& left, const cv::Mat& right) {
    CV_Assert(left.type() == CV_8U && right.type() == CV_8U);
    CV_Assert(left.size() == right.size());

    int w = left.cols, h = left.rows;
    int radius = block_size_ / 2;
    int max_disp = num_disparities_;

    // 初始化左右描述符矩阵（每个像素存储一个 uint64_t）
    cv::Mat left_desc(h, w, CV_64FC1);   // 用 double 存储 uint64_t，CV_64U 不存在，强制转换指针
    cv::Mat right_desc(h, w, CV_64FC1);
    
    // 计算左图 Census
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            uint64_t code = 0;
            int bit = 0;
            uint8_t center = left.at<uint8_t>(y, x);
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    int ny = y + dy;
                    int nx = x + dx;
                    if (ny >= 0 && ny < h && nx >= 0 && nx < w) {
                        uint8_t neighbor = left.at<uint8_t>(ny, nx);
                        if (neighbor > center) {
                            code |= (1ULL << bit);
                        }
                    }
                    ++bit;
                    if (bit >= 64) goto fill_left; // 最多64位
                }
            }
            fill_left:
            left_desc.at<double>(y, x) = static_cast<double>(code);
        }
    }

    // 计算右图 Census
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            uint64_t code = 0;
            int bit = 0;
            uint8_t center = right.at<uint8_t>(y, x);
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    int ny = y + dy;
                    int nx = x + dx;
                    if (ny >= 0 && ny < h && nx >= 0 && nx < w) {
                        uint8_t neighbor = right.at<uint8_t>(ny, nx);
                        if (neighbor > center) {
                            code |= (1ULL << bit);
                        }
                    }
                    ++bit;
                    if (bit >= 64) goto fill_right;
                }
            }
            fill_right:
            right_desc.at<double>(y, x) = static_cast<double>(code);
        }
    }

    // 汉明距离函数（使用内置 popcnt）
    auto hamming = [](uint64_t a, uint64_t b) -> int {
        return __builtin_popcountll(a ^ b);
    };

    // WTA：对每个左像素，在视差范围内扫描右图
    cv::Mat disp16(h, w, CV_16S, cv::Scalar::all(max_disp)); // 初始化为无效值
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            uint64_t left_val = static_cast<uint64_t>(left_desc.at<double>(y, x));
            int best_disp = 0;
            int best_cost = 64; // 最大汉明距离
            int second_cost = 64;

            for (int d = 0; d < max_disp; ++d) {
                int xr = x - d;
                if (xr < 0) break;
                uint64_t right_val = static_cast<uint64_t>(right_desc.at<double>(y, xr));
                int cost = hamming(left_val, right_val);
                if (cost < best_cost) {
                    second_cost = best_cost;
                    best_cost = cost;
                    best_disp = d;
                } else if (cost < second_cost) {
                    second_cost = cost;
                }
            }

            // 唯一性检查
            if (best_cost < 64 && best_cost > 0) {
                float ratio = static_cast<float>(second_cost) / static_cast<float>(best_cost);
                if (ratio >= (1.0f + uniqueness_ratio_ / 100.0f)) {
                    disp16.at<short>(y, x) = static_cast<short>(best_disp);
                }
            }
        }
    }

    // 转换到 16位无符号（CV_16U），与 OpenCV 接口一致
    cv::Mat disp16u;
    disp16.convertTo(disp16u, CV_16U);
    return disp16u;
}

// ---------- OpenCV StereoBM（优化参数）----------
cv::Mat CpuStereoMatcher::computeBM(const cv::Mat& left, const cv::Mat& right) {
    cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create(num_disparities_opencv_, block_size_);
    bm->setMinDisparity(0);
    bm->setNumDisparities(num_disparities_opencv_);
    bm->setBlockSize(block_size_);
    bm->setUniquenessRatio(uniqueness_ratio_);
    bm->setSpeckleWindowSize(100);    // 可从配置读取
    bm->setSpeckleRange(32);
    bm->setDisp12MaxDiff(1);
    bm->setPreFilterCap(31);
    bm->setPreFilterSize(9);
    bm->setPreFilterType(cv::StereoBM::PREFILTER_NORMALIZED_RESPONSE);
    bm->setTextureThreshold(10);
    bm->setSmallerBlockSize(0);

    cv::Mat disp16;
    bm->compute(left, right, disp16);
    
    // 将无效值（-16）等标记为 max_disparity
    cv::Mat disp16u;
    disp16.convertTo(disp16u, CV_16U);
    return disp16u;
}

// ---------- OpenCV StereoSGBM（优化参数）----------
cv::Mat CpuStereoMatcher::computeSGBM(const cv::Mat& left, const cv::Mat& right) {
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(
        0,                          // minDisparity
        num_disparities_opencv_,    // numDisparities
        block_size_,               // blockSize
        p1_,                       // P1
        p2_,                       // P2
        1,                         // disp12MaxDiff
        uniqueness_ratio_,         // preFilterCap? 实际上是 uniquenessRatio，SGBM构造参数顺序特殊
        0,                         // 这里 preFilterCap 被省略，我们手动 set
        100,                       // speckleWindowSize
        32,                        // speckleRange
        cv::StereoSGBM::MODE_SGBM_3WAY
    );
    sgbm->setPreFilterCap(31);      // 手动设置
    sgbm->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);
    
    cv::Mat disp16;
    sgbm->compute(left, right, disp16);
    
    cv::Mat disp16u;
    disp16.convertTo(disp16u, CV_16U);
    return disp16u;
}

void CpuStereoMatcher::applyMedianFilter(cv::Mat& disp, int size) {
    cv::medianBlur(disp, disp, size);
}

} // namespace cpu_stereo
} // namespace stereo_depth
