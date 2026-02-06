#!/usr/bin/env python3
"""
简化版配置生成脚本
读取config/global_config.yaml，生成着色器参数文件
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
        'min_disparity': stereo.get('min_disparity', 0),
        'window_size': stereo.get('window_size', 9),
        'aggregation_window': stereo.get('cost_aggregation_window', 5),
        'uniqueness_ratio': stereo.get('uniqueness_ratio', 15) / 100.0,
        'penalty_p1': stereo.get('penalty_p1', 8.0),
        'penalty_p2': stereo.get('penalty_p2', 32.0),
        'workgroup_x': gpu.get('shader_workgroup_size', [16, 16])[0],
        'workgroup_y': gpu.get('shader_workgroup_size', [16, 16])[1],
        'flags': flags,
        'speckle_window': stereo.get('speckle_window_size', 100),
        'speckle_range': stereo.get('speckle_range', 32),
        'median_size': stereo.get('median_filter_size', 3)
    }

def generate_glsl_params(params, output_path):
    """生成GLSL参数文件"""
    
    content = f"""// 自动生成的着色器参数
// 来源: config/global_config.yaml

#define IMAGE_WIDTH {params['image_width']}
#define IMAGE_HEIGHT {params['image_height']}
#define MAX_DISPARITY {params['max_disparity']}
#define MIN_DISPARITY {params['min_disparity']}

#define WINDOW_SIZE {params['window_size']}
#define AGGREGATION_WINDOW {params['aggregation_window']}
#define UNIQUENESS_RATIO {params['uniqueness_ratio']}
#define PENALTY_P1 {params['penalty_p1']}
#define PENALTY_P2 {params['penalty_p2']}

#define WORKGROUP_X {params['workgroup_x']}
#define WORKGROUP_Y {params['workgroup_y']}
#define FLAGS {params['flags']}

#define SPECKLE_WINDOW {params['speckle_window']}
#define SPECKLE_RANGE {params['speckle_range']}
#define MEDIAN_SIZE {params['median_size']}

// 标志位定义
#define FLAG_USE_CENSUS (1 << 0)
#define FLAG_USE_POSTPROCESSING (1 << 1)
#define FLAG_USE_MEDIAN_FILTER (1 << 2)
#define IS_FLAG_SET(flag) ((FLAGS & flag) != 0)

// 辅助宏
#define GET_PIXEL_INDEX(x, y) ((y) * IMAGE_WIDTH + (x))
#define IS_VALID_PIXEL(x, y) ((x) >= 0 && (x) < IMAGE_WIDTH && (y) >= 0 && (y) < IMAGE_HEIGHT)
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

// 着色器参数（从配置生成）
struct ShaderParameters {{
    static constexpr uint32_t IMAGE_WIDTH = {params['image_width']};
    static constexpr uint32_t IMAGE_HEIGHT = {params['image_height']};
    static constexpr uint32_t MAX_DISPARITY = {params['max_disparity']};
    static constexpr uint32_t MIN_DISPARITY = {params['min_disparity']};
    
    static constexpr uint32_t WINDOW_SIZE = {params['window_size']};
    static constexpr uint32_t AGGREGATION_WINDOW = {params['aggregation_window']};
    static constexpr float UNIQUENESS_RATIO = {params['uniqueness_ratio']}f;
    static constexpr float PENALTY_P1 = {params['penalty_p1']}f;
    static constexpr float PENALTY_P2 = {params['penalty_p2']}f;
    
    static constexpr uint32_t WORKGROUP_X = {params['workgroup_x']};
    static constexpr uint32_t WORKGROUP_Y = {params['workgroup_y']};
    static constexpr uint32_t FLAGS = {params['flags']};
    
    static constexpr uint32_t SPECKLE_WINDOW = {params['speckle_window']};
    static constexpr uint32_t SPECKLE_RANGE = {params['speckle_range']};
    static constexpr uint32_t MEDIAN_SIZE = {params['median_size']};
    
    // 标志位定义
    static constexpr uint32_t FLAG_USE_CENSUS = 1 << 0;
    static constexpr uint32_t FLAG_USE_POSTPROCESSING = 1 << 1;
    static constexpr uint32_t FLAG_USE_MEDIAN_FILTER = 1 << 2;
    
    // 验证参数
    static bool validate() {{
        return IMAGE_WIDTH > 0 && IMAGE_HEIGHT > 0 && MAX_DISPARITY > 0;
    }}
    
    // 打印参数（用于调试）
    static void print() {{
        printf("Shader Parameters:\\n");
        printf("  Image: %u x %u\\n", IMAGE_WIDTH, IMAGE_HEIGHT);
        printf("  Disparity: %u-%u\\n", MIN_DISPARITY, MAX_DISPARITY);
        printf("  Window: %u, Aggregation: %u\\n", WINDOW_SIZE, AGGREGATION_WINDOW);
        printf("  Uniqueness: %.2f, P1: %.1f, P2: %.1f\\n", 
               UNIQUENESS_RATIO, PENALTY_P1, PENALTY_P2);
        printf("  Workgroup: %u x %u\\n", WORKGROUP_X, WORKGROUP_Y);
        printf("  Flags: 0x%08X\\n", FLAGS);
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
        print("\n配置生成完成")
        print(f"输出目录: {output_dir}")
        print(f"关键参数:")
        print(f"  图像尺寸: {params['image_width']}x{params['image_height']}")
        print(f"  视差范围: {params['min_disparity']}-{params['max_disparity']}")
        print(f"  工作组: {params['workgroup_x']}x{params['workgroup_y']}")
        print(f"  标志位: 0x{params['flags']:08X}")
    else:
        print("\n生成过程中出现错误")
        sys.exit(1)

if __name__ == "__main__":
    main()
