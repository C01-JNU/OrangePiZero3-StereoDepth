#pragma once

/**
 * @file common_defines.h
 * @brief 项目通用定义和常量
 * @date 2026-01-18
 * @author C01-JNU
 */

#include <cstdint>
#include <string>
#include <vector>

// ==============================
// 1. 错误码定义
// ==============================
enum class ErrorCode : int32_t {
    SUCCESS = 0,
    
    // 配置错误
    ERROR_CONFIG_LOAD_FAILED = -100,
    ERROR_CONFIG_PARSE_FAILED = -101,
    
    // Vulkan错误
    ERROR_VULKAN_INIT_FAILED = -200,
    ERROR_VULKAN_DEVICE_NOT_FOUND = -201,
    ERROR_VULKAN_SHADER_COMPILATION = -202,
    ERROR_VULKAN_BUFFER_CREATION = -203,
    ERROR_VULKAN_PIPELINE_CREATION = -204,
    
    // 摄像头错误
    ERROR_CAMERA_INIT_FAILED = -300,
    ERROR_CAMERA_NOT_FOUND = -301,
    ERROR_CAMERA_CAPTURE_FAILED = -302,
    
    // 标定错误
    ERROR_CALIBRATION_FILE_NOT_FOUND = -400,
    ERROR_CALIBRATION_DATA_INVALID = -401,
    
    // 图像处理错误
    ERROR_IMAGE_LOAD_FAILED = -500,
    ERROR_IMAGE_SIZE_MISMATCH = -501,
    
    // 计算错误
    ERROR_STEREO_MATCHING_FAILED = -600,
    ERROR_DEPTH_CALCULATION_FAILED = -601,
    
    // 内存错误
    ERROR_MEMORY_ALLOCATION = -700,
    
    // 系统错误
    ERROR_SYSTEM = -999
};

// ==============================
// 2. 运行模式定义
// ==============================
enum class RunMode : uint8_t {
    CPU = 0,
    GPU_VULKAN = 1,
    DEBUG = 2,
    BENCHMARK = 3
};

// ==============================
// 3. 日志级别定义
// ==============================
enum class LogLevel : uint8_t {
    TRACE = 0,
    DEBUG = 1,
    INFO = 2,
    WARN = 3,
    ERROR = 4,
    FATAL = 5
};

// ==============================
// 4. 图像格式定义
// ==============================
enum class ImageFormat : uint8_t {
    GRAY8 = 0,
    RGB8 = 1,
    BGR8 = 2,
    RGBA8 = 3,
    BGRA8 = 4,
    YUYV = 5,
    NV12 = 6
};

// ==============================
// 5. Census变换类型
// ==============================
enum class CensusType : uint8_t {
    STANDARD = 0,
    MODIFIED = 1,
    ADAPTIVE = 2
};

// ==============================
// 6. 代价计算方法
// ==============================
enum class CostMethod : uint8_t {
    HAMMING = 0,
    CENSUS = 1,
    SAD = 2,
    SSD = 3,
    NCC = 4
};

// ==============================
// 7. 后处理方法
// ==============================
enum class PostProcessMethod : uint8_t {
    NONE = 0,
    MEDIAN_FILTER = 1,
    BILATERAL_FILTER = 2,
    CONSISTENCY_CHECK = 3,
    HOLE_FILLING = 4,
    SPEECKLE_FILTER = 5
};

// ==============================
// 8. 常量定义
// ==============================
namespace Constants {
    // 最大视差
    constexpr int MAX_DISPARITY = 128;
    
    // 默认图像尺寸
    constexpr int DEFAULT_WIDTH = 640;
    constexpr int DEFAULT_HEIGHT = 480;
    
    // 数学常量
    constexpr float PI = 3.14159265358979323846f;
    constexpr float DEG_TO_RAD = PI / 180.0f;
    constexpr float RAD_TO_DEG = 180.0f / PI;
    
    // 性能常量
    constexpr int MAX_PYRAMID_LEVELS = 5;
    constexpr int MIN_IMAGE_SIZE = 64;
    
    // Vulkan常量
    constexpr int VULKAN_WORKGROUP_SIZE_X = 16;
    constexpr int VULKAN_WORKGROUP_SIZE_Y = 16;
}

// ==============================
// 9. 数据结构
// ==============================
struct ImageSize {
    int width;
    int height;
    
    ImageSize(int w = Constants::DEFAULT_WIDTH, int h = Constants::DEFAULT_HEIGHT)
        : width(w), height(h) {}
    
    size_t pixelCount() const { return static_cast<size_t>(width) * height; }
    float aspectRatio() const { return static_cast<float>(width) / height; }
};

struct StereoCameraParams {
    float focal_length;      // 焦距（像素）
    float baseline;          // 基线距离（米）
    float cx, cy;           // 主点坐标
    float k1, k2, p1, p2;   // 畸变参数
    
    StereoCameraParams() 
        : focal_length(0.0f), baseline(0.0f), 
          cx(0.0f), cy(0.0f),
          k1(0.0f), k2(0.0f), p1(0.0f), p2(0.0f) {}
};

struct CensusConfig {
    int window_size;
    int threshold;
    bool use_center_pixel;
    CensusType type;
    
    CensusConfig()
        : window_size(9), threshold(25),
          use_center_pixel(true), type(CensusType::STANDARD) {}
};

struct WTAConfig {
    int uniqueness_ratio;
    bool lr_consistency_check;
    float consistency_threshold;
    bool subpixel_refinement;
    
    WTAConfig()
        : uniqueness_ratio(15),
          lr_consistency_check(true),
          consistency_threshold(1.0f),
          subpixel_refinement(true) {}
};

struct PostProcessConfig {
    int median_filter_size;
    bool speckle_filter_enabled;
    int speckle_window_size;
    int speckle_range;
    bool edge_preserving;
    float edge_threshold;
    
    PostProcessConfig()
        : median_filter_size(3),
          speckle_filter_enabled(true),
          speckle_window_size(100),
          speckle_range(32),
          edge_preserving(true),
          edge_threshold(5.0f) {}
};

// ==============================
// 10. 性能优化宏
// ==============================
#if defined(__GNUC__) || defined(__clang__)
    #define LIKELY(x)   __builtin_expect(!!(x), 1)
    #define UNLIKELY(x) __builtin_expect(!!(x), 0)
    #define FORCE_INLINE inline __attribute__((always_inline))
#else
    #define LIKELY(x)   (x)
    #define UNLIKELY(x) (x)
    #define FORCE_INLINE inline
#endif

// 平台检测
#ifdef __arm__ || __aarch64__
    #define PLATFORM_ARM 1
    #ifdef __ARM_NEON
        #define HAS_NEON 1
    #else
        #define HAS_NEON 0
    #endif
#else
    #define PLATFORM_ARM 0
    #define HAS_NEON 0
#endif

// 断言宏（只在调试版本启用）
#ifdef NDEBUG
    #define ASSERT(expr) ((void)0)
#else
    #define ASSERT(expr) \
        do { \
            if (!(expr)) { \
                fprintf(stderr, "Assertion failed: %s, file %s, line %d\n", \
                        #expr, __FILE__, __LINE__); \
                std::abort(); \
            } \
        } while(0)
#endif

// 不抛异常保证
#define NOEXCEPT noexcept
