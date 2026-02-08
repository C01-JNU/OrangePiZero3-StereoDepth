#!/usr/bin/env python3
"""
统一版本配置生成脚本
读取config/global_config.yaml，生成一致的着色器参数文件
最后更新：2026年2月8日
"""

import yaml
import os
import sys

def load_config(config_path):
    """加载YAML配置文件"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"错误: 无法加载配置文件 {config_path}: {e}")
        return None

def generate_shader_params(config):
    """从配置生成着色器参数"""
    
    # 提取参数
    stereo = config.get('stereo', {})
    gpu = config.get('gpu', {})
    camera = config.get('camera', {})
    
    # 计算参数值
    image_width = camera.get('width', 640) // 2  # 拼接后单眼宽度
    image_height = camera.get('height', 480)
    max_disparity = stereo.get('max_disparity', 64)
    min_disparity = stereo.get('min_disparity', 0)
    
    # 计算标志位
    flags = 0
    if stereo.get('use_census', True):
        flags |= 1 << 0
    if stereo.get('enable_postprocessing', True):
        flags |= 1 << 1
    if stereo.get('use_median_filter', True):
        flags |= 1 << 2
    
    # 返回参数字典
    return {
        'image_width': image_width,
        'image_height': image_height,
        'max_disparity': max_disparity,
        'min_disparity': min_disparity,
        
        'window_size': stereo.get('window_size', 9),
        'aggregation_window': stereo.get('cost_aggregation_window', 5),
        'uniqueness_ratio': stereo.get('uniqueness_ratio', 15) / 100.0,
        'penalty_p1': stereo.get('penalty_p1', 8.0),
        
        'penalty_p2': stereo.get('penalty_p2', 32.0),
        'flags': flags,
        'speckle_window': stereo.get('speckle_window_size', 100),
        'speckle_range': stereo.get('speckle_range', 32),
        
        'median_size': stereo.get('median_filter_size', 3),
        'workgroup_x': gpu.get('shader_workgroup_size', [16, 16])[0],
        'workgroup_y': gpu.get('shader_workgroup_size', [16, 16])[1]
    }

def generate_glsl_params(params, output_path):
    """生成GLSL参数文件"""
    
    content = f"""// 自动生成的着色器参数
// 来源: config/global_config.yaml
// 最后生成: 2026-02-08
// 注意: 此文件用于编译时插入着色器，不要手动修改

// 图像参数
#define IMAGE_WIDTH {params['image_width']}
#define IMAGE_HEIGHT {params['image_height']}
#define MAX_DISPARITY {params['max_disparity']}
#define MIN_DISPARITY {params['min_disparity']}

// 算法参数
#define WINDOW_SIZE {params['window_size']}
#define AGGREGATION_WINDOW {params['aggregation_window']}
#define UNIQUENESS_RATIO {params['uniqueness_ratio']}
#define PENALTY_P1 {params['penalty_p1']}
#define PENALTY_P2 {params['penalty_p2']}

// 标志位参数
#define FLAGS {params['flags']}
#define SPECKLE_WINDOW {params['speckle_window']}
#define SPECKLE_RANGE {params['speckle_range']}
#define MEDIAN_SIZE {params['median_size']}

// 工作组参数
#define WORKGROUP_X {params['workgroup_x']}
#define WORKGROUP_Y {params['workgroup_y']}

// 标志位定义
#define FLAG_USE_CENSUS (1 << 0)
#define FLAG_USE_POSTPROCESSING (1 << 1)
#define FLAG_USE_MEDIAN_FILTER (1 << 2)
#define IS_FLAG_SET(flag) ((FLAGS & flag) != 0)

// 辅助宏
#define GET_PIXEL_INDEX(x, y) ((y) * IMAGE_WIDTH + (x))
#define IS_VALID_PIXEL(x, y) ((x) >= 0 && (x) < IMAGE_WIDTH && (y) >= 0 && (y) < IMAGE_HEIGHT)

// Uniform缓冲区结构体（必须与C++端的PipelineParams完全匹配）
#define UNIFORM_PARAMS_STRUCT \\
    uint imageWidth;        \\n\\
    uint imageHeight;       \\n\\
    uint maxDisparity;      \\n\\
    uint windowSize;        \\n\\
    \\n\\
    float uniquenessRatio;  \\n\\
    float penaltyP1;        \\n\\
    float penaltyP2;        \\n\\
    uint flags;             \\n\\
    \\n\\
    uint speckleWindow;     \\n\\
    uint speckleRange;      \\n\\
    uint medianSize;        \\n\\
    uint padding[3];
"""
    
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(content)
        print(f"生成GLSL参数文件: {output_path}")
        return True
    except Exception as e:
        print(f"错误: 无法写入文件 {output_path}: {e}")
        return False

def generate_cpp_header(params, output_path):
    """生成C++参数头文件"""
    
    content = f"""#pragma once
#include <cstdint>

namespace stereo_depth {{
namespace vulkan {{

// 着色器编译时常量（从配置生成）
// 最后生成: 2026-02-08
// 注意: 此文件用于C++代码中的编译时常量，不要手动修改
struct ShaderParameters {{
    // 图像参数
    static constexpr uint32_t IMAGE_WIDTH = {params['image_width']};
    static constexpr uint32_t IMAGE_HEIGHT = {params['image_height']};
    static constexpr uint32_t MAX_DISPARITY = {params['max_disparity']};
    static constexpr uint32_t MIN_DISPARITY = {params['min_disparity']};
    
    // 算法参数
    static constexpr uint32_t WINDOW_SIZE = {params['window_size']};
    static constexpr uint32_t AGGREGATION_WINDOW = {params['aggregation_window']};
    static constexpr float UNIQUENESS_RATIO = {params['uniqueness_ratio']}f;
    static constexpr float PENALTY_P1 = {params['penalty_p1']}f;
    static constexpr float PENALTY_P2 = {params['penalty_p2']}f;
    
    // 标志位参数
    static constexpr uint32_t FLAGS = {params['flags']};
    static constexpr uint32_t SPECKLE_WINDOW = {params['speckle_window']};
    static constexpr uint32_t SPECKLE_RANGE = {params['speckle_range']};
    static constexpr uint32_t MEDIAN_SIZE = {params['median_size']};
    
    // 工作组参数
    static constexpr uint32_t WORKGROUP_X = {params['workgroup_x']};
    static constexpr uint32_t WORKGROUP_Y = {params['workgroup_y']};
    
    // 标志位定义
    static constexpr uint32_t FLAG_USE_CENSUS = 1 << 0;
    static constexpr uint32_t FLAG_USE_POSTPROCESSING = 1 << 1;
    static constexpr uint32_t FLAG_USE_MEDIAN_FILTER = 1 << 2;
    
    // Uniform缓冲区结构体（必须与GLSL端的Parameters完全匹配）
    // 总大小: 16字节对齐
    struct PipelineParams {{
        uint32_t imageWidth;        // 偏移: 0
        uint32_t imageHeight;       // 偏移: 4
        uint32_t maxDisparity;      // 偏移: 8
        uint32_t windowSize;        // 偏移: 12
                                    // 16字节对齐 ✅
        
        float uniquenessRatio;      // 偏移: 16
        float penaltyP1;            // 偏移: 20
        float penaltyP2;            // 偏移: 24
        uint32_t flags;             // 偏移: 28
                                    // 32字节对齐 ✅
        
        uint32_t speckleWindow;     // 偏移: 32
        uint32_t speckleRange;      // 偏移: 36
        uint32_t medianSize;        // 偏移: 40
        uint32_t padding[3];        // 偏移: 44, 48, 52
                                    // 56字节对齐 ✅
        
        // 总大小: 56字节
        
        // 静态验证
        static_assert(sizeof(PipelineParams) == 56, "PipelineParams size must be 56 bytes");
        static_assert(offsetof(PipelineParams, uniquenessRatio) == 16, "uniquenessRatio offset mismatch");
        static_assert(offsetof(PipelineParams, speckleWindow) == 32, "speckleWindow offset mismatch");
        
        // 构造函数
        PipelineParams() {{
            imageWidth = IMAGE_WIDTH;
            imageHeight = IMAGE_HEIGHT;
            maxDisparity = MAX_DISPARITY;
            windowSize = WINDOW_SIZE;
            
            uniquenessRatio = UNIQUENESS_RATIO;
            penaltyP1 = PENALTY_P1;
            penaltyP2 = PENALTY_P2;
            flags = FLAGS;
            
            speckleWindow = SPECKLE_WINDOW;
            speckleRange = SPECKLE_RANGE;
            medianSize = MEDIAN_SIZE;
            padding[0] = padding[1] = padding[2] = 0;
        }}
        
        // 打印参数（用于调试）
        void print() const {{
            printf("PipelineParams:\\n");
            printf("  图像: %u x %u\\n", imageWidth, imageHeight);
            printf("  视差: %u\\n", maxDisparity);
            printf("  窗口: %u\\n", windowSize);
            printf("  唯一性: %.2f\\n", uniquenessRatio);
            printf("  惩罚: P1=%.1f, P2=%.1f\\n", penaltyP1, penaltyP2);
            printf("  标志位: 0x%08X\\n", flags);
            printf("  斑点窗口: %u, 范围: %u\\n", speckleWindow, speckleRange);
            printf("  中值滤波: %u\\n", medianSize);
            printf("  总大小: %zu 字节\\n", sizeof(PipelineParams));
        }}
    }};
    
    // 验证参数
    static bool validate() {{
        return IMAGE_WIDTH > 0 && IMAGE_HEIGHT > 0 && MAX_DISPARITY > 0;
    }}
    
    // 打印编译时常量（用于调试）
    static void print() {{
        printf("Shader Parameters (编译时常量):\\n");
        printf("  图像尺寸: %u x %u\\n", IMAGE_WIDTH, IMAGE_HEIGHT);
        printf("  视差范围: %u-%u\\n", MIN_DISPARITY, MAX_DISPARITY);
        printf("  窗口大小: %u, 聚合窗口: %u\\n", WINDOW_SIZE, AGGREGATION_WINDOW);
        printf("  唯一性比率: %.2f, P1: %.1f, P2: %.1f\\n", 
               UNIQUENESS_RATIO, PENALTY_P1, PENALTY_P2);
        printf("  标志位: 0x%08X\\n", FLAGS);
        printf("  工作组: %u x %u\\n", WORKGROUP_X, WORKGROUP_Y);
    }}
}};

}} // namespace vulkan
}} // namespace stereo_depth
"""
    
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(content)
        print(f"生成C++参数头文件: {output_path}")
        return True
    except Exception as e:
        print(f"错误: 无法写入文件 {output_path}: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("用法: python3 generate_shader_config.py <config_file>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    if not os.path.exists(config_path):
        print(f"错误: 配置文件不存在: {config_path}")
        sys.exit(1)
    
    config = load_config(config_path)
    if config is None:
        sys.exit(1)
    
    # 生成参数
    params = generate_shader_params(config)
    
    # 输出目录
    output_dir = "src/vulkan/generated"
    
    # 生成文件
    success = True
    success &= generate_glsl_params(params, f"{output_dir}/shader_params.glsl")
    success &= generate_cpp_header(params, f"{output_dir}/shader_params.hpp")
    
    if success:
        print("\n✅ 配置生成完成")
        print(f"输出目录: {output_dir}")
        print(f"关键参数:")
        print(f"  图像尺寸: {params['image_width']}x{params['image_height']}")
        print(f"  视差范围: {params['min_disparity']}-{params['max_disparity']}")
        print(f"  窗口大小: {params['window_size']}")
        print(f"  标志位: 0x{params['flags']:08X}")
        print(f"  工作组: {params['workgroup_x']}x{params['workgroup_y']}")
        print(f"  Uniform缓冲区大小: 56 字节")
    else:
        print("\n❌ 生成过程中出现错误")
        sys.exit(1)

if __name__ == "__main__":
    main()
