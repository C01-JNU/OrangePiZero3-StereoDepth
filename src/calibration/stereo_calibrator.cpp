#include "calibration/stereo_calibrator.hpp"
#include "utils/config.hpp"
#include "utils/logger.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <random>
#include <set>
#include <chrono>
#include <numeric>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#include <cmath>

namespace stereo_depth {
namespace calibration {

// 辅助函数：检查文件是否存在
bool fileExists(const std::string& path) {
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

// 辅助函数：创建目录
bool createDirectory(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        if (mkdir(path.c_str(), 0777) != 0) {
            return false;
        }
    }
    return true;
}

// 将矩阵转换为YAML格式字符串
std::string matrixToYaml(const cv::Mat& mat, const std::string& name) {
    std::stringstream ss;
    ss << name << ": !!opencv-matrix" << std::endl;
    ss << "   rows: " << mat.rows << std::endl;
    ss << "   cols: " << mat.cols << std::endl;
    ss << "   dt: d" << std::endl;
    ss << "   data: [";
    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            ss << std::setprecision(17) << mat.at<double>(i, j);
            if (!(i == mat.rows-1 && j == mat.cols-1)) {
                ss << ", ";
            }
        }
    }
    ss << "]" << std::endl;
    return ss.str();
}

// 计算组合数 C(n, k) - 辅助函数
static long long calculateCombination(int n, int k) {
    if (k > n) return 0;
    if (k * 2 > n) k = n - k;
    if (k == 0) return 1;
    
    long long result = n;
    for (int i = 2; i <= k; ++i) {
        result *= (n - i + 1);
        result /= i;
    }
    return result;
}

// 加权随机选择 - 辅助函数
static std::vector<int> weightedRandomSelection(int n, int k, 
                                                const std::vector<int>& usageCount,
                                                std::mt19937& rng,
                                                std::set<std::vector<int>>& triedCombinations) {
    std::vector<int> combination;
    
    // 如果k接近n，使用所有图像
    if (k >= n - 2) {
        for (int i = 0; i < n; i++) {
            combination.push_back(i);
        }
        return combination;
    }
    
    // 尝试生成新组合的最大次数
    int maxAttempts = 1000;
    for (int attempt = 0; attempt < maxAttempts; attempt++) {
        combination.clear();
        
        // 计算权重：使用次数越少，权重越高
        std::vector<double> weights(n);
        double totalWeight = 0.0;
        
        for (int i = 0; i < n; i++) {
            // 使用次数为0时权重为1，否则为 1/(1+usageCount)
            weights[i] = 1.0 / (1.0 + usageCount[i]);
            totalWeight += weights[i];
        }
        
        // 准备候选索引
        std::vector<int> candidates(n);
        std::iota(candidates.begin(), candidates.end(), 0);
        
        // 使用加权随机选择k个图像
        for (int i = 0; i < k; i++) {
            // 生成0到totalWeight之间的随机数
            std::uniform_real_distribution<double> dist(0.0, totalWeight);
            double randomValue = dist(rng);
            
            // 根据权重选择
            double cumulative = 0.0;
            for (int j = 0; j < (int)candidates.size(); j++) {
                cumulative += weights[candidates[j]];
                if (cumulative >= randomValue) {
                    // 选中的图像
                    combination.push_back(candidates[j]);
                    
                    // 从候选列表中移除已选中的图像
                    totalWeight -= weights[candidates[j]];
                    // 注意：这里应该更新weights和candidates向量
                    // 但为了简化，我们使用更直接的方法
                    weights.erase(weights.begin() + j);
                    candidates.erase(candidates.begin() + j);
                    break;
                }
            }
        }
        
        std::sort(combination.begin(), combination.end());
        
        // 检查是否已经尝试过
        if (triedCombinations.find(combination) == triedCombinations.end()) {
            triedCombinations.insert(combination);
            return combination;
        }
    }
    
    // 无法生成新组合，返回空
    return {};
}

// 格式化向量为字符串 - 辅助函数
static std::string formatVector(const std::vector<int>& vec) {
    std::stringstream ss;
    for (size_t i = 0; i < vec.size(); i++) {
        ss << vec[i] + 1; // 转换为1-based索引
        if (i < vec.size() - 1) {
            ss << ", ";
        }
    }
    return ss.str();
}

bool StereoCalibrator::calibrate() {
    LOG_INFO("开始立体相机标定");
    
    // 1. 加载配置
    if (!loadConfiguration()) {
        LOG_ERROR("加载配置失败");
        return false;
    }
    
    LOG_INFO("标定参数:");
    LOG_INFO("  棋盘格尺寸: {}x{} 内角点", m_boardSize.width, m_boardSize.height);
    LOG_INFO("  方块尺寸: {:.3f} 米", m_squareSize);
    LOG_INFO("  图像尺寸: {}x{}", m_imageSize.width, m_imageSize.height);
    LOG_INFO("  标定图像目录: {}", m_calibrationDir);
    LOG_INFO("  输出目录: {}", m_outputDir);
    
    // 2. 查找标定图像对
    auto imagePairs = findCalibrationImagePairs();
    LOG_INFO("找到 {} 对标定图像", imagePairs.size());
    
    if (imagePairs.size() < 5) {
        LOG_ERROR("标定图像不足，至少需要5对，当前只有 {} 对", imagePairs.size());
        return false;
    }
    
    // 3. 准备物体点（棋盘格物理坐标）
    std::vector<cv::Point3f> objectPattern;
    for (int i = 0; i < m_boardSize.height; ++i) {
        for (int j = 0; j < m_boardSize.width; ++j) {
            objectPattern.push_back(cv::Point3f(j * m_squareSize, i * m_squareSize, 0.0f));
        }
    }
    
    // 4. 处理每对标定图像，收集角点数据
    std::vector<std::vector<cv::Point2f>> allImagePointsLeft;
    std::vector<std::vector<cv::Point2f>> allImagePointsRight;
    std::vector<int> validImageIndices;
    
    int totalPairs = static_cast<int>(imagePairs.size());
    int validPairs = 0;
    
    for (int i = 0; i < totalPairs; ++i) {
        const auto& [leftPath, rightPath] = imagePairs[i];
        
        // 读取左右图像
        cv::Mat leftImage = cv::imread(leftPath, cv::IMREAD_GRAYSCALE);
        cv::Mat rightImage = cv::imread(rightPath, cv::IMREAD_GRAYSCALE);
        
        if (leftImage.empty() || rightImage.empty()) {
            LOG_WARN("无法读取图像对 {}: {}, {}", i, leftPath, rightPath);
            continue;
        }
        
        if (leftImage.size() != m_imageSize || rightImage.size() != m_imageSize) {
            LOG_WARN("图像尺寸不匹配: 期望 {}x{}, 实际 {}x{}", 
                     m_imageSize.width, m_imageSize.height, 
                     leftImage.cols, leftImage.rows);
            continue;
        }
        
        // 检测角点
        std::vector<cv::Point2f> leftCorners, rightCorners;
        bool leftFound = detectChessboardCorners(leftImage, leftCorners);
        bool rightFound = detectChessboardCorners(rightImage, rightCorners);
        
        if (leftFound && rightFound) {
            // 精化角点位置
            cv::cornerSubPix(leftImage, leftCorners, cv::Size(11, 11), cv::Size(-1, -1),
                           cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01));
            cv::cornerSubPix(rightImage, rightCorners, cv::Size(11, 11), cv::Size(-1, -1),
                           cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01));
            
            // 保存数据
            allImagePointsLeft.push_back(leftCorners);
            allImagePointsRight.push_back(rightCorners);
            validImageIndices.push_back(i);
            validPairs++;
            
            LOG_DEBUG("图像对 {}: 检测到 {}x{} 个角点", 
                     validPairs, m_boardSize.width, m_boardSize.height);
        } else {
            LOG_WARN("图像对 {}: 角点检测失败", i);
        }
        
        // 每处理10对输出一次进度
        if ((i + 1) % 10 == 0) {
            LOG_INFO("已处理 {}/{} 对图像，有效 {} 对", 
                    i + 1, totalPairs, validPairs);
        }
    }
    
    LOG_INFO("角点检测完成，有效图像对: {}/{}", validPairs, totalPairs);
    
    if (validPairs < 5) {
        LOG_ERROR("有效图像对不足，至少需要5对，当前只有 {} 对", validPairs);
        return false;
    }
    
    // 5. 单分组大小优化标定（使用0.2比例）
    LOG_INFO("开始单分组大小优化标定...");
    
    // 准备物体点
    std::vector<std::vector<cv::Point3f>> allObjectPoints(validPairs, objectPattern);
    int validImageCount = validPairs;
    
    // 计算分组大小：使用0.2比例（1/5）
    int groupSize = 0;
    
    // 特殊情况处理
    if (validImageCount < 7) {
        // 小于7张时，使用全部图像
        groupSize = validImageCount;
        LOG_INFO("有效图像较少（{}对），直接使用全部图像进行标定", validImageCount);
    } else {
        // 使用0.2比例
        groupSize = static_cast<int>(std::round(validImageCount * 0.2));
        
        // 确保分组大小在合理范围内
        if (groupSize < 5) {
            LOG_INFO("计算的分组大小 {} 小于最小值5，使用最小值5", groupSize);
            groupSize = 5;
        }
        if (groupSize > validImageCount) {
            LOG_INFO("计算的分组大小 {} 超过总图像数，使用全部图像", validImageCount);
            groupSize = validImageCount;
        }
    }
    
    LOG_INFO("使用分组大小: {} (总图像数的 {:.1f}%)", 
            groupSize, (groupSize * 100.0 / validImageCount));
    
    // 存储最佳结果
    double bestRms = 1e10;
    cv::Mat bestCameraMatrixLeft, bestDistCoeffsLeft;
    cv::Mat bestCameraMatrixRight, bestDistCoeffsRight;
    cv::Mat bestR, bestT, bestE, bestF;
    int bestGroupSize = -1;
    int bestGroupIndex = -1;
    std::vector<int> bestGroupIndices;
    
    // 特殊情况：如果分组大小等于总图像数，直接使用所有图像
    if (groupSize == validImageCount) {
        LOG_INFO("使用全部 {} 对图像进行标定", validImageCount);
        
        // 提取所有图像数据
        std::vector<std::vector<cv::Point3f>> groupObjectPoints = allObjectPoints;
        std::vector<std::vector<cv::Point2f>> groupImagePointsLeft = allImagePointsLeft;
        std::vector<std::vector<cv::Point2f>> groupImagePointsRight = allImagePointsRight;
        
        // 执行标定
        cv::Mat cameraMatrixLeft, distCoeffsLeft;
        cv::Mat cameraMatrixRight, distCoeffsRight;
        cv::Mat R, T, E, F;
        double rms = 0.0;
        
        bool success = performGroupCalibration(groupObjectPoints, groupImagePointsLeft,
                                             groupImagePointsRight, cameraMatrixLeft, distCoeffsLeft,
                                             cameraMatrixRight, distCoeffsRight, R, T, E, F, rms);
        
        if (success) {
            bestRms = rms;
            bestCameraMatrixLeft = cameraMatrixLeft.clone();
            bestDistCoeffsLeft = distCoeffsLeft.clone();
            bestCameraMatrixRight = cameraMatrixRight.clone();
            bestDistCoeffsRight = distCoeffsRight.clone();
            bestR = R.clone();
            bestT = T.clone();
            bestE = E.clone();
            bestF = F.clone();
            bestGroupSize = groupSize;
            bestGroupIndex = 1;
            bestGroupIndices = validImageIndices;
            
            LOG_INFO("标定成功! RMS = {:.6f}", rms);
        } else {
            LOG_ERROR("使用全部图像标定失败");
            return false;
        }
    } else {
        LOG_WARN("注意：加权随机分组标定将尝试多种组合，可能需要较长时间！");
        
        // 计算组合数
        long long totalCombinations = calculateCombination(validImageCount, groupSize);
        LOG_INFO("理论组合数: C({},{}) = {}", validImageCount, groupSize, 
                (totalCombinations > 10000) ? ">10000" : std::to_string(totalCombinations));
        
        // 限制实际尝试的组合数
        int maxTrials = std::min(50, (int)std::min(totalCombinations, 100LL));
        int actualTrials = std::min(maxTrials, validImageCount * 2);
        
        LOG_INFO("实际尝试组合数: {} (受限于时间)", actualTrials);
        
        // 设置随机数生成器
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::mt19937 rng(seed);
        
        // 存储已经尝试过的组合，避免重复
        std::set<std::vector<int>> triedCombinations;
        
        // 记录每张图像的使用次数（用于加权随机）
        std::vector<int> imageUsageCount(validImageCount, 0);
        
        // 对每组进行标定
        for (int trial = 0; trial < actualTrials; trial++) {
            // 使用加权随机选择生成组合
            std::vector<int> combination = weightedRandomSelection(
                validImageCount, groupSize, imageUsageCount, rng, triedCombinations);
            
            // 如果无法生成新组合，提前结束
            if (combination.empty()) {
                LOG_INFO("无法生成新的组合，提前结束（已尝试 {} 组）", trial);
                break;
            }
            
            LOG_INFO("=== 标定第 {} 组 (分组大小 {}) ===", trial + 1, groupSize);
            LOG_INFO("使用图像索引: [{}]", formatVector(combination));
            
            // 更新图像使用次数
            for (int idx : combination) {
                imageUsageCount[idx]++;
            }
            
            // 提取当前组的标定数据
            std::vector<std::vector<cv::Point3f>> groupObjectPoints;
            std::vector<std::vector<cv::Point2f>> groupImagePointsLeft;
            std::vector<std::vector<cv::Point2f>> groupImagePointsRight;
            
            for (int idx : combination) {
                groupObjectPoints.push_back(allObjectPoints[idx]);
                groupImagePointsLeft.push_back(allImagePointsLeft[idx]);
                groupImagePointsRight.push_back(allImagePointsRight[idx]);
            }
            
            // 执行标定
            cv::Mat cameraMatrixLeft, distCoeffsLeft;
            cv::Mat cameraMatrixRight, distCoeffsRight;
            cv::Mat R, T, E, F;
            double rms = 0.0;
            
            bool success = performGroupCalibration(groupObjectPoints, groupImagePointsLeft,
                                                 groupImagePointsRight, cameraMatrixLeft, distCoeffsLeft,
                                                 cameraMatrixRight, distCoeffsRight, R, T, E, F, rms);
            
            if (success) {
                if (rms < bestRms) {
                    bestRms = rms;
                    bestCameraMatrixLeft = cameraMatrixLeft.clone();
                    bestDistCoeffsLeft = distCoeffsLeft.clone();
                    bestCameraMatrixRight = cameraMatrixRight.clone();
                    bestDistCoeffsRight = distCoeffsRight.clone();
                    bestR = R.clone();
                    bestT = T.clone();
                    bestE = E.clone();
                    bestF = F.clone();
                    bestGroupSize = groupSize;
                    bestGroupIndex = trial + 1;
                    
                    // 保存最佳组的图像索引（使用原始图像索引）
                    bestGroupIndices.clear();
                    for (int idx : combination) {
                        bestGroupIndices.push_back(validImageIndices[idx]);
                    }
                    
                    LOG_INFO("  -> 更新最佳结果! RMS = {:.6f}", rms);
                }
                
                // 输出进度
                if ((trial + 1) % 10 == 0 || trial + 1 == actualTrials) {
                    LOG_INFO("进度: {}/{} 组，当前最佳RMS: {:.6f}", 
                            trial + 1, actualTrials, bestRms);
                    
                    // 输出使用次数统计
                    int minUsage = *std::min_element(imageUsageCount.begin(), imageUsageCount.end());
                    int maxUsage = *std::max_element(imageUsageCount.begin(), imageUsageCount.end());
                    double avgUsage = std::accumulate(imageUsageCount.begin(), imageUsageCount.end(), 0.0) / validImageCount;
                    
                    LOG_DEBUG("图像使用统计 - 最小: {}, 最大: {}, 平均: {:.2f}", 
                             minUsage, maxUsage, avgUsage);
                }
            } else {
                LOG_WARN("标定失败，组合: {}", formatVector(combination));
            }
            
            // 每5组休息一下，避免过热
            if ((trial + 1) % 5 == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
        
        // 输出最终使用统计
        LOG_INFO("=== 图像使用统计 ===");
        int minUsage = *std::min_element(imageUsageCount.begin(), imageUsageCount.end());
        int maxUsage = *std::max_element(imageUsageCount.begin(), imageUsageCount.end());
        double avgUsage = std::accumulate(imageUsageCount.begin(), imageUsageCount.end(), 0.0) / validImageCount;
        
        LOG_INFO("总尝试组数: {}", std::min(actualTrials, (int)triedCombinations.size()));
        LOG_INFO("最小使用次数: {}", minUsage);
        LOG_INFO("最大使用次数: {}", maxUsage);
        LOG_INFO("平均使用次数: {:.2f}", avgUsage);
        
        // 输出使用分布（每10张图像显示一次）
        LOG_INFO("使用次数分布:");
        for (int i = 0; i < validImageCount; i++) {
            if ((i + 1) % 10 == 0 || i == validImageCount - 1) {
                LOG_INFO("图像 {:3d}: {} 次", i + 1, imageUsageCount[i]);
            }
        }
    }
    
    if (bestGroupSize == -1) {
        LOG_ERROR("所有分组标定均失败");
        return false;
    }
    
    LOG_INFO("=== 最佳结果 ===");
    LOG_INFO("最佳RMS误差: {:.6f}", bestRms);
    LOG_INFO("来自分组大小: {}, 组号: {}", bestGroupSize, bestGroupIndex);
    LOG_INFO("使用图像范围: {} - {}", bestGroupIndices.front() + 1, bestGroupIndices.back() + 1);
    LOG_INFO("使用图像数量: {}", bestGroupIndices.size());
    
    // 6. 使用最佳结果计算立体校正参数
    LOG_INFO("计算立体校正参数...");
    cv::Rect validRoi[2];
    cv::stereoRectify(
        bestCameraMatrixLeft, bestDistCoeffsLeft,
        bestCameraMatrixRight, bestDistCoeffsRight,
        m_imageSize,
        bestR, bestT,
        m_rectificationTransformLeft, m_rectificationTransformRight,
        m_projectionMatrixLeft, m_projectionMatrixRight,
        m_disparityToDepthMappingMatrix,
        cv::CALIB_ZERO_DISPARITY, 1, m_imageSize,
        &validRoi[0], &validRoi[1]
    );
    
    // 保存最佳结果到成员变量
    m_cameraMatrixLeft = bestCameraMatrixLeft;
    m_distCoeffsLeft = bestDistCoeffsLeft;
    m_cameraMatrixRight = bestCameraMatrixRight;
    m_distCoeffsRight = bestDistCoeffsRight;
    m_rotationMatrix = bestR;
    m_translationVector = bestT;
    m_essentialMatrix = bestE;
    m_fundamentalMatrix = bestF;
    m_rmsError = bestRms;
    m_imagesUsed = static_cast<int>(bestGroupIndices.size());
    
    // 7. 计算校正映射
    cv::initUndistortRectifyMap(
        m_cameraMatrixLeft, m_distCoeffsLeft,
        m_rectificationTransformLeft, m_projectionMatrixLeft,
        m_imageSize, CV_32FC1, m_leftMap1, m_leftMap2
    );
    
    cv::initUndistortRectifyMap(
        m_cameraMatrixRight, m_distCoeffsRight,
        m_rectificationTransformRight, m_projectionMatrixRight,
        m_imageSize, CV_32FC1, m_rightMap1, m_rightMap2
    );
    
    LOG_INFO("校正参数计算完成");
    
    // 8. 保存标定结果
    if (!saveCalibrationResults(bestGroupSize, bestGroupIndex, bestGroupIndices)) {
        LOG_ERROR("保存标定结果失败");
        return false;
    }
    
    // 9. 生成标定报告
    if (!generateCalibrationReport(bestGroupSize, bestGroupIndex, bestGroupIndices)) {
        LOG_WARN("生成标定报告失败");
    }
    
    // 10. 验证标定结果（可选）
    LOG_INFO("验证标定结果...");
    if (!validateCalibrationResults()) {
        LOG_WARN("标定结果验证失败");
    }
    
    LOG_INFO("立体标定完成!");
    LOG_INFO("  RMS误差: {:.6f}", m_rmsError);
    LOG_INFO("  使用图像: {} 对", m_imagesUsed);
    LOG_INFO("  基线长度: {:.4f} 米", cv::norm(m_translationVector));
    LOG_INFO("  分组大小: {}, 组号: {}", bestGroupSize, bestGroupIndex);
    LOG_INFO("  标定文件: {}/stereo_calibration.yml", m_outputDir);
    
    return true;
}

bool StereoCalibrator::performGroupCalibration(
    const std::vector<std::vector<cv::Point3f>>& groupObjectPoints,
    const std::vector<std::vector<cv::Point2f>>& groupImagePointsLeft,
    const std::vector<std::vector<cv::Point2f>>& groupImagePointsRight,
    cv::Mat& cameraMatrixLeft, cv::Mat& distCoeffsLeft,
    cv::Mat& cameraMatrixRight, cv::Mat& distCoeffsRight,
    cv::Mat& R, cv::Mat& T, cv::Mat& E, cv::Mat& F,
    double& rms) {
    
    try {
        if (groupObjectPoints.size() < 5) {
            LOG_WARN("  - 跳过，图像数量不足: {} (需要至少5张)", groupObjectPoints.size());
            return false;
        }
        
        // 初始化相机矩阵和畸变系数 - 使用完整5个系数
        cameraMatrixLeft = cv::Mat::eye(3, 3, CV_64F);
        cameraMatrixRight = cv::Mat::eye(3, 3, CV_64F);
        distCoeffsLeft = cv::Mat::zeros(5, 1, CV_64F);   // 使用完整5个畸变系数
        distCoeffsRight = cv::Mat::zeros(5, 1, CV_64F);  // 使用完整5个畸变系数
        
        // 单目标定 - 使用完整参数
        std::vector<cv::Mat> rvecsLeft, tvecsLeft;
        double rmsLeft = cv::calibrateCamera(groupObjectPoints, groupImagePointsLeft, m_imageSize, 
                                           cameraMatrixLeft, distCoeffsLeft, rvecsLeft, tvecsLeft);
        
        std::vector<cv::Mat> rvecsRight, tvecsRight;
        double rmsRight = cv::calibrateCamera(groupObjectPoints, groupImagePointsRight, m_imageSize, 
                                            cameraMatrixRight, distCoeffsRight, rvecsRight, tvecsRight);
        
        LOG_DEBUG("  - 单目标定RMS: 左={:.6f}, 右={:.6f}", rmsLeft, rmsRight);
        
        // 双目标定 - 使用完整参数，自动优化
        int stereoFlags = cv::CALIB_USE_INTRINSIC_GUESS;  // 使用初始猜测，但不固定任何参数
        
        rms = cv::stereoCalibrate(groupObjectPoints, groupImagePointsLeft, groupImagePointsRight,
                                 cameraMatrixLeft, distCoeffsLeft,
                                 cameraMatrixRight, distCoeffsRight,
                                 m_imageSize, R, T, E, F,
                                 stereoFlags,
                                 cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 200, 1e-6));
        
        LOG_DEBUG("  - 双目标定RMS: {:.6f}", rms);
        
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("  - 标定失败: {}", e.what());
        return false;
    }
}

bool StereoCalibrator::loadConfiguration() {
    try {
        auto& configManager = utils::ConfigManager::getInstance();
        const auto& config = configManager.getConfig();
        
        // 获取配置文件路径用于调试
        std::string configPath = configManager.getConfigPath();
        LOG_DEBUG("配置文件路径: {}", configPath);
        
        // 从calibration节读取棋盘格参数
        int boardWidth = 0;
        int boardHeight = 0;
        
        // 方法1：尝试读取数组索引语法
        try {
            boardWidth = config.get<int>("calibration.chessboard_size[0]", 0);
            boardHeight = config.get<int>("calibration.chessboard_size[1]", 0);
            LOG_DEBUG("尝试读取数组索引: [0]={}, [1]={}", boardWidth, boardHeight);
        } catch (...) {
            LOG_DEBUG("数组索引语法读取失败");
        }
        
        // 方法2：尝试读取完整的配置路径
        if (boardWidth == 0 || boardHeight == 0) {
            // 列出所有配置键来调试
            try {
                auto keys = config.getKeys();
                LOG_DEBUG("可用的配置键总数: {}", keys.size());
                for (const auto& key : keys) {
                    if (key.find("chessboard") != std::string::npos || 
                        key.find("board") != std::string::npos) {
                        LOG_DEBUG("找到相关键: {} = {}", key, config.get<std::string>(key, ""));
                    }
                }
            } catch (...) {}
            
            // 尝试直接读取配置值
            try {
                // 从 calibration.chessboard_size 读取
                std::string chessboardStr = config.get<std::string>("calibration.chessboard_size", "");
                if (!chessboardStr.empty()) {
                    LOG_DEBUG("读取到棋盘格字符串: {}", chessboardStr);
                    // 尝试解析格式如 "[9, 6]"
                    if (chessboardStr.find("[") != std::string::npos && 
                        chessboardStr.find("]") != std::string::npos) {
                        // 简单解析
                        chessboardStr.erase(0, chessboardStr.find("[") + 1);
                        chessboardStr.erase(chessboardStr.find("]"));
                        std::replace(chessboardStr.begin(), chessboardStr.end(), ',', ' ');
                        std::stringstream ss(chessboardStr);
                        ss >> boardWidth >> boardHeight;
                        LOG_DEBUG("解析后: {}x{}", boardWidth, boardHeight);
                    }
                }
            } catch (...) {}
        }
        
        // 方法3：使用硬编码默认值（仅在调试阶段）
        if (boardWidth == 0 || boardHeight == 0) {
            LOG_WARN("无法从配置读取棋盘格尺寸，使用默认值: 9x6");
            boardWidth = 9;
            boardHeight = 6;
        }
        
        m_boardSize = cv::Size(boardWidth, boardHeight);
        
        // 方格物理尺寸（米）
        m_squareSize = config.get<float>("calibration.square_size", 0.0f);
        
        // 尝试不同的键名
        if (m_squareSize <= 0) {
            // 尝试读取字符串然后转换
            try {
                std::string squareSizeStr = config.get<std::string>("calibration.square_size", "0");
                if (squareSizeStr != "0") {
                    m_squareSize = std::stof(squareSizeStr);
                    LOG_DEBUG("从字符串转换方格尺寸: {} -> {}", squareSizeStr, m_squareSize);
                }
            } catch (...) {}
            
            // 尝试其他可能的键
            if (m_squareSize <= 0) {
                m_squareSize = config.get<float>("square_size", 0.0f);
            }
        }
        
        if (m_squareSize <= 0) {
            LOG_ERROR("无效的方格尺寸: {}", m_squareSize);
            
            // 列出所有配置键帮助调试
            try {
                auto keys = config.getKeys();
                LOG_INFO("所有配置键:");
                for (const auto& key : keys) {
                    if (key.find("calibration") == 0 || key.find(".calibration") != std::string::npos ||
                        key.find("square") != std::string::npos) {
                        std::string value = config.get<std::string>(key, "");
                        LOG_INFO("  {} = {}", key, value);
                    }
                }
            } catch (...) {}
            
            return false;
        }
        
        // 从stereo节读取图像尺寸
        int imageWidth = config.get<int>("stereo.image_width", 0);
        int imageHeight = config.get<int>("stereo.image_height", 0);
        
        if (imageWidth <= 0 || imageHeight <= 0) {
            LOG_ERROR("无效的图像尺寸: {}x{}", imageWidth, imageHeight);
            return false;
        }
        m_imageSize = cv::Size(imageWidth, imageHeight);
        
        // 从output节读取路径
        m_calibrationDir = config.get<std::string>("output.calibration_dir", "");
        
        // 从calibration节读取标定文件路径，并提取目录
        std::string calibrationFile = config.get<std::string>("calibration.calibration_file", "");
        if (calibrationFile.empty()) {
            LOG_ERROR("未指定标定文件路径");
            return false;
        }
        
        // 提取目录路径（从文件路径中）
        size_t lastSlash = calibrationFile.find_last_of("/");
        if (lastSlash != std::string::npos) {
            m_outputDir = calibrationFile.substr(0, lastSlash);
        } else {
            m_outputDir = ".";
        }
        
        if (m_calibrationDir.empty()) {
            LOG_ERROR("未指定标定图像目录");
            return false;
        }
        
        // 确保目录存在
        if (!createDirectory(m_outputDir)) {
            LOG_ERROR("无法创建输出目录: {}", m_outputDir);
            return false;
        }
        
        LOG_DEBUG("配置加载完成:");
        LOG_DEBUG("  棋盘格: {}x{}", m_boardSize.width, m_boardSize.height);
        LOG_DEBUG("  方格尺寸: {} 米", m_squareSize);
        LOG_DEBUG("  图像尺寸: {}x{}", m_imageSize.width, m_imageSize.height);
        LOG_DEBUG("  标定目录: {}", m_calibrationDir);
        LOG_DEBUG("  输出目录: {}", m_outputDir);
        
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("加载配置时发生异常: {}", e.what());
        return false;
    }
}

std::vector<std::pair<std::string, std::string>> StereoCalibrator::findCalibrationImagePairs() {
    std::vector<std::pair<std::string, std::string>> pairs;
    
    try {
        if (!fileExists(m_calibrationDir)) {
            LOG_ERROR("标定图像目录不存在: {}", m_calibrationDir);
            return pairs;
        }
        
        // 使用 opendir/readdir 遍历目录
        DIR* dir = opendir(m_calibrationDir.c_str());
        if (!dir) {
            LOG_ERROR("无法打开标定图像目录: {}", m_calibrationDir);
            return pairs;
        }
        
        std::vector<std::string> leftImages;
        
        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
            std::string filename = entry->d_name;
            
            // 跳过 . 和 ..
            if (filename == "." || filename == "..") {
                continue;
            }
            
            // 查找左眼图像
            if (filename.find("_left.") != std::string::npos) {
                std::string fullPath = m_calibrationDir + "/" + filename;
                leftImages.push_back(fullPath);
            }
        }
        closedir(dir);
        
        // 按文件名排序
        std::sort(leftImages.begin(), leftImages.end());
        
        // 为每个左眼图像查找对应的右眼图像
        for (const auto& leftPath : leftImages) {
            // 从完整路径中提取文件名
            size_t lastSlash = leftPath.find_last_of("/");
            std::string leftFilename = (lastSlash != std::string::npos) ? 
                                       leftPath.substr(lastSlash + 1) : leftPath;
            
            // 构建右眼图像文件名
            size_t pos = leftFilename.find("_left.");
            if (pos == std::string::npos) continue;
            
            std::string baseName = leftFilename.substr(0, pos);
            size_t dotPos = leftFilename.find_last_of(".");
            std::string extension = (dotPos != std::string::npos) ? 
                                    leftFilename.substr(dotPos) : "";
            
            std::string rightFilename = baseName + "_right" + extension;
            
            // 构建完整路径
            std::string rightPath;
            if (lastSlash != std::string::npos) {
                rightPath = leftPath.substr(0, lastSlash + 1) + rightFilename;
            } else {
                rightPath = rightFilename;
            }
            
            if (fileExists(rightPath)) {
                pairs.emplace_back(leftPath, rightPath);
                LOG_DEBUG("找到图像对: {} -> {}", leftFilename, rightFilename);
            } else {
                LOG_WARN("找不到对应的右眼图像: {}", rightFilename);
            }
        }
        
        LOG_INFO("在目录 {} 中找到 {} 对标定图像", m_calibrationDir, pairs.size());
        
    } catch (const std::exception& e) {
        LOG_ERROR("查找标定图像对时发生异常: {}", e.what());
    }
    
    return pairs;
}

bool StereoCalibrator::detectChessboardCorners(const cv::Mat& image, std::vector<cv::Point2f>& corners) {
    if (image.empty()) {
        LOG_WARN("图像为空");
        return false;
    }
    
    bool found = cv::findChessboardCorners(image, m_boardSize, corners,
                                         cv::CALIB_CB_ADAPTIVE_THRESH + 
                                         cv::CALIB_CB_NORMALIZE_IMAGE +
                                         cv::CALIB_CB_FAST_CHECK);
    
    return found;
}

bool StereoCalibrator::saveCalibrationResults(int bestGroupSize, int bestGroupIndex, 
                                             const std::vector<int>& bestGroupIndices) {
    std::string outputFile = m_outputDir + "/stereo_calibration.yml";
    
    try {
        cv::FileStorage fs(outputFile, cv::FileStorage::WRITE);
        if (!fs.isOpened()) {
            LOG_ERROR("无法创建标定文件: {}", outputFile);
            return false;
        }
        
        // 写入标定时间（使用安全的键名）
        std::string dateStr = __DATE__;
        std::string timeStr = __TIME__;
        std::string dateTime = dateStr + "_" + timeStr;
        std::replace(dateTime.begin(), dateTime.end(), ' ', '_');
        std::replace(dateTime.begin(), dateTime.end(), ':', '_');
        
        fs << "calibration_date" << dateTime;
        fs << "image_width" << m_imageSize.width;
        fs << "image_height" << m_imageSize.height;
        fs << "board_width" << m_boardSize.width;
        fs << "board_height" << m_boardSize.height;
        fs << "square_size" << m_squareSize;
        fs << "images_used" << m_imagesUsed;
        fs << "best_group_size" << bestGroupSize;
        fs << "best_group_index" << bestGroupIndex;
        
        // 写入标定参数
        fs << "camera_matrix_left" << m_cameraMatrixLeft;
        fs << "distortion_coefficients_left" << m_distCoeffsLeft;
        fs << "camera_matrix_right" << m_cameraMatrixRight;
        fs << "distortion_coefficients_right" << m_distCoeffsRight;
        fs << "rotation_matrix" << m_rotationMatrix;
        fs << "translation_vector" << m_translationVector;
        fs << "essential_matrix" << m_essentialMatrix;
        fs << "fundamental_matrix" << m_fundamentalMatrix;
        fs << "rectification_transform_left" << m_rectificationTransformLeft;
        fs << "rectification_transform_right" << m_rectificationTransformRight;
        fs << "projection_matrix_left" << m_projectionMatrixLeft;
        fs << "projection_matrix_right" << m_projectionMatrixRight;
        fs << "disparity_to_depth_mapping_matrix" << m_disparityToDepthMappingMatrix;
        
        // 写入误差和附加信息
        fs << "rms_error" << m_rmsError;
        fs << "baseline_meters" << cv::norm(m_translationVector);
        
        // 写入使用的图像索引
        fs << "used_image_indices" << "[";
        for (int idx : bestGroupIndices) {
            fs << idx + 1; // 保存为1-based索引
        }
        fs << "]";
        
        fs.release();
        
        LOG_INFO("标定结果已保存到: {}", outputFile);
        return true;
        
    } catch (const cv::Exception& e) {
        LOG_ERROR("OpenCV保存标定结果时发生异常: {}", e.what());
        LOG_ERROR("错误代码: {}", e.code);
        LOG_ERROR("请检查键名是否合法（必须以字母或下划线开头）");
        return false;
    } catch (const std::exception& e) {
        LOG_ERROR("保存标定结果时发生异常: {}", e.what());
        return false;
    }
}

bool StereoCalibrator::generateCalibrationReport(int bestGroupSize, int bestGroupIndex,
                                                const std::vector<int>& bestGroupIndices) {
    std::string reportFile = m_outputDir + "/calibration_report.txt";
    
    try {
        std::ofstream report(reportFile);
        if (!report.is_open()) {
            LOG_ERROR("无法创建标定报告: {}", reportFile);
            return false;
        }
        
        report << "OrangePiZero3-StereoDepth 立体相机标定报告\n";
        report << "==========================================\n\n";
        
        report << "标定时间: " << __DATE__ << " " << __TIME__ << "\n\n";
        
        report << "1. 标定参数\n";
        report << "   图像尺寸: " << m_imageSize.width << "x" << m_imageSize.height << "\n";
        report << "   棋盘格尺寸: " << m_boardSize.width << "x" << m_boardSize.height << " 内角点\n";
        report << "   方格物理尺寸: " << m_squareSize << " 米\n";
        report << "   总有效图像: " << bestGroupIndices.size() << " 对\n\n";
        
        report << "2. 多分组优化结果\n";
        report << "   最佳分组大小: " << bestGroupSize << "\n";
        report << "   最佳组号: " << bestGroupIndex << "\n";
        report << "   使用图像范围: " << bestGroupIndices.front() + 1 << " - " << bestGroupIndices.back() + 1 << "\n";
        report << "   使用图像数量: " << bestGroupIndices.size() << " 对\n\n";
        
        report << "3. 标定结果\n";
        report << "   RMS误差: " << std::fixed << std::setprecision(6) << m_rmsError << "\n";
        report << "   基线长度: " << cv::norm(m_translationVector) << " 米\n\n";
        
        report << "4. 左相机内参矩阵\n";
        for (int i = 0; i < 3; ++i) {
            report << "   [";
            for (int j = 0; j < 3; ++j) {
                report << std::setw(12) << std::setprecision(6) << std::fixed 
                      << m_cameraMatrixLeft.at<double>(i, j);
                if (j < 2) report << ", ";
            }
            report << "]\n";
        }
        report << "\n";
        
        report << "5. 右相机内参矩阵\n";
        for (int i = 0; i < 3; ++i) {
            report << "   [";
            for (int j = 0; j < 3; ++j) {
                report << std::setw(12) << std::setprecision(6) << std::fixed 
                      << m_cameraMatrixRight.at<double>(i, j);
                if (j < 2) report << ", ";
            }
            report << "]\n";
        }
        report << "\n";
        
        report << "6. 立体参数\n";
        report << "   旋转矩阵 R:\n";
        for (int i = 0; i < 3; ++i) {
            report << "   [";
            for (int j = 0; j < 3; ++j) {
                report << std::setw(12) << std::setprecision(6) << std::fixed 
                      << m_rotationMatrix.at<double>(i, j);
                if (j < 2) report << ", ";
            }
            report << "]\n";
        }
        
        report << "\n   平移向量 T (米):\n";
        report << "   [";
        for (int i = 0; i < 3; ++i) {
            report << std::setw(12) << std::setprecision(6) << std::fixed 
                  << m_translationVector.at<double>(i, 0);
            if (i < 2) report << ", ";
        }
        report << "]\n\n";
        
        report << "7. 文件位置\n";
        report << "   标定参数: " << m_outputDir << "/stereo_calibration.yml\n";
        report << "   标定图像: " << m_calibrationDir << "\n";
        report << "   生成时间: " << __DATE__ << " " << __TIME__ << "\n";
        
        report << "\n8. 使用图像列表\n";
        report << "   分组大小: " << bestGroupSize << ", 组号: " << bestGroupIndex << "\n";
        report << "   RMS误差: " << m_rmsError << "\n";
        report << "   图像范围: " << bestGroupIndices.front() + 1 << " - " << bestGroupIndices.back() + 1 << "\n";
        report << "   图像数量: " << bestGroupIndices.size() << "\n\n";
        
        // 每行显示5个图像编号，便于阅读
        for (size_t i = 0; i < bestGroupIndices.size(); i++) {
            report << "图像对 " << std::setw(3) << bestGroupIndices[i] + 1;
            if ((i + 1) % 5 == 0 || i == bestGroupIndices.size() - 1) {
                report << "\n";
            } else {
                report << ", ";
            }
        }
        
        report.close();
        
        LOG_INFO("标定报告已保存到: {}", reportFile);
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("生成标定报告时发生异常: {}", e.what());
        return false;
    }
}

bool StereoCalibrator::validateCalibrationResults() {
    // 从已处理的图像对中取第一对进行验证
    auto imagePairs = findCalibrationImagePairs();
    if (imagePairs.empty()) {
        LOG_WARN("没有找到图像对进行验证");
        return false;
    }
    
    // 只验证第一对
    const auto& [leftPath, rightPath] = imagePairs[0];
    cv::Mat leftImage = cv::imread(leftPath, cv::IMREAD_COLOR);
    cv::Mat rightImage = cv::imread(rightPath, cv::IMREAD_COLOR);
    
    if (leftImage.empty() || rightImage.empty()) {
        LOG_WARN("无法读取验证图像");
        return false;
    }
    
    // 转换为灰度图（校正映射可以在彩色图上工作）
    cv::Mat leftGray, rightGray;
    if (leftImage.channels() == 3) {
        cv::cvtColor(leftImage, leftGray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(rightImage, rightGray, cv::COLOR_BGR2GRAY);
    } else {
        leftGray = leftImage;
        rightGray = rightImage;
    }
    
    // 应用校正
    cv::Mat leftRectified, rightRectified;
    cv::remap(leftGray, leftRectified, m_leftMap1, m_leftMap2, cv::INTER_LINEAR);
    cv::remap(rightGray, rightRectified, m_rightMap1, m_rightMap2, cv::INTER_LINEAR);
    
    // 创建验证图像（并排显示）
    cv::Mat validationImage;
    cv::hconcat(leftRectified, rightRectified, validationImage);
    
    // 在验证图像上画水平线，用于检查校正质量
    int lineSpacing = validationImage.rows / 10;
    for (int i = 1; i < 10; ++i) {
        cv::line(validationImage, 
                cv::Point(0, i * lineSpacing),
                cv::Point(validationImage.cols - 1, i * lineSpacing),
                cv::Scalar(255, 0, 0), 1);  // 蓝色线条
    }
    
    // 保存验证图像
    std::string validationPath = m_outputDir + "/rectification_validation.jpg";
    if (cv::imwrite(validationPath, validationImage)) {
        LOG_INFO("校正验证图像已保存到: {}", validationPath);
        
        // 额外保存一个彩色版本（如果原始是彩色）
        if (leftImage.channels() == 3) {
            cv::Mat leftRectifiedColor, rightRectifiedColor;
            cv::remap(leftImage, leftRectifiedColor, m_leftMap1, m_leftMap2, cv::INTER_LINEAR);
            cv::remap(rightImage, rightRectifiedColor, m_rightMap1, m_rightMap2, cv::INTER_LINEAR);
            
            cv::Mat validationColor;
            cv::hconcat(leftRectifiedColor, rightRectifiedColor, validationColor);
            
            std::string validationColorPath = m_outputDir + "/rectification_validation_color.jpg";
            cv::imwrite(validationColorPath, validationColor);
            LOG_INFO("彩色校正验证图像已保存到: {}", validationColorPath);
        }
        
        return true;
    } else {
        LOG_WARN("无法保存验证图像");
        return false;
    }
}

} // namespace calibration
} // namespace stereo_depth
