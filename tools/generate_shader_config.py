#!/usr/bin/env python3
import yaml
import os
import sys
import argparse

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def compute_layer_params(config, level):
    base_w = config['camera']['width'] // 2
    base_h = config['camera']['height']
    scale = config.get('pyramid', {}).get('scale_factor', 0.5)
    w, h = base_w, base_h
    for _ in range(level):
        w = max(1, int(round(w * scale))) & ~1
        h = max(1, int(round(h * scale))) & ~1
    stereo = config.get('stereo', {})
    params = {
        'width': w,
        'height': h,
        'max_disparity': stereo.get('disparity_range', 64),
        'min_disparity': stereo.get('min_disparity', 0),
        'window_size': stereo.get('census_window', 7),
        'aggregation_window': stereo.get('cost_aggregation_window', 5),
        'uniqueness_ratio': stereo.get('uniqueness_ratio', 15) / 100.0,
        'penalty_p1': stereo.get('penalty_p1', 8.0),
        'penalty_p2': stereo.get('penalty_p2', 32.0),
        'speckle_window': stereo.get('speckle_window_size', 100),
        'speckle_range': stereo.get('speckle_range', 32),
        'median_size': stereo.get('median_filter_size', 3),
        'workgroup_x': config.get('gpu', {}).get('shader_workgroup_size', [16,16])[0],
        'workgroup_y': config.get('gpu', {}).get('shader_workgroup_size', [16,16])[1],
        'search_radius': config.get('pyramid', {}).get('search_radius', 8),  # 新增
    }
    flags = 0
    if stereo.get('use_census', True):          flags |= 1 << 0
    if stereo.get('enable_postprocessing', True): flags |= 1 << 1
    if stereo.get('use_median_filter', True):   flags |= 1 << 2
    params['flags'] = flags
    for ov in config.get('layer_overrides', []):
        if ov['level'] == level:
            if 'census_window' in ov:
                params['window_size'] = ov['census_window']
            if 'disparity_range' in ov:
                params['max_disparity'] = ov['disparity_range']
    return params

def generate_glsl_params(params, output_path):
    content = f"""// 自动生成的着色器参数（层参数）
#define IMAGE_WIDTH {params['width']}u
#define IMAGE_HEIGHT {params['height']}u
#define MAX_DISPARITY {params['max_disparity']}u
#define MIN_DISPARITY {params['min_disparity']}u
#define WINDOW_SIZE {params['window_size']}u
#define AGGREGATION_WINDOW {params['aggregation_window']}u
#define UNIQUENESS_RATIO {params['uniqueness_ratio']}f
#define PENALTY_P1 {params['penalty_p1']}f
#define PENALTY_P2 {params['penalty_p2']}f
#define FLAGS {params['flags']}u
#define SPECKLE_WINDOW {params['speckle_window']}u
#define SPECKLE_RANGE {params['speckle_range']}u
#define MEDIAN_SIZE {params['median_size']}u
#define WORKGROUP_X {params['workgroup_x']}u
#define WORKGROUP_Y {params['workgroup_y']}u
#define SEARCH_RADIUS {params['search_radius']}u   // 搜索半径
#define FLAG_USE_CENSUS         (1u << 0)
#define FLAG_USE_POSTPROCESSING (1u << 1)
#define FLAG_USE_MEDIAN_FILTER  (1u << 2)
#define IS_FLAG_SET(flag) ((FLAGS & (flag)) != 0u)
#define GET_PIXEL_INDEX(x, y) ((y) * IMAGE_WIDTH + (x))
#define IS_VALID_PIXEL(x, y) ((x) < IMAGE_WIDTH && (y) < IMAGE_HEIGHT)
// Uniform结构体（56字节，包含searchRadius）
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
    uint searchRadius;      \\n\\
    uint padding[2];        // 保持56字节（4+4+4+4+4+4+4+4+4+4+4+4+4+4 = 56）
"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(content)

def generate_cpp_header(params, output_path):
    content = f"""#pragma once
#include <cstdint>
#include <cstdio>
namespace stereo_depth::vulkan {{
struct ShaderParameters {{
    static constexpr uint32_t IMAGE_WIDTH = {params['width']}u;
    static constexpr uint32_t IMAGE_HEIGHT = {params['height']}u;
    static constexpr uint32_t MAX_DISPARITY = {params['max_disparity']}u;
    static constexpr uint32_t MIN_DISPARITY = {params['min_disparity']}u;
    static constexpr uint32_t WINDOW_SIZE = {params['window_size']}u;
    static constexpr uint32_t AGGREGATION_WINDOW = {params['aggregation_window']}u;
    static constexpr float UNIQUENESS_RATIO = {params['uniqueness_ratio']}f;
    static constexpr float PENALTY_P1 = {params['penalty_p1']}f;
    static constexpr float PENALTY_P2 = {params['penalty_p2']}f;
    static constexpr uint32_t FLAGS = {params['flags']}u;
    static constexpr uint32_t SPECKLE_WINDOW = {params['speckle_window']}u;
    static constexpr uint32_t SPECKLE_RANGE = {params['speckle_range']}u;
    static constexpr uint32_t MEDIAN_SIZE = {params['median_size']}u;
    static constexpr uint32_t WORKGROUP_X = {params['workgroup_x']}u;
    static constexpr uint32_t WORKGROUP_Y = {params['workgroup_y']}u;
    static constexpr uint32_t SEARCH_RADIUS = {params['search_radius']}u;
    static constexpr uint32_t FLAG_USE_CENSUS = 1u << 0;
    static constexpr uint32_t FLAG_USE_POSTPROCESSING = 1u << 1;
    static constexpr uint32_t FLAG_USE_MEDIAN_FILTER = 1u << 2;
    struct PipelineParams {{
        uint32_t imageWidth, imageHeight, maxDisparity, windowSize;
        float uniquenessRatio, penaltyP1, penaltyP2;
        uint32_t flags, speckleWindow, speckleRange, medianSize, searchRadius;
        uint32_t padding[2];
        PipelineParams() {{
            imageWidth = IMAGE_WIDTH; imageHeight = IMAGE_HEIGHT;
            maxDisparity = MAX_DISPARITY; windowSize = WINDOW_SIZE;
            uniquenessRatio = UNIQUENESS_RATIO; penaltyP1 = PENALTY_P1;
            penaltyP2 = PENALTY_P2; flags = FLAGS;
            speckleWindow = SPECKLE_WINDOW; speckleRange = SPECKLE_RANGE;
            medianSize = MEDIAN_SIZE; searchRadius = SEARCH_RADIUS;
            padding[0] = padding[1] = 0;
        }}
        void print() const {{
            printf("PipelineParams: %ux%u, maxDisp=%u, win=%u, flags=0x%x, radius=%u\\n",
                   imageWidth, imageHeight, maxDisparity, windowSize, flags, searchRadius);
        }}
    }};
    static_assert(sizeof(PipelineParams) == 56, "PipelineParams must be 56 bytes");
}};
}} // namespace stereo_depth::vulkan
"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(content)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--layer', type=int, required=True)
    parser.add_argument('--outdir', required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    params = compute_layer_params(config, args.layer)
    generate_glsl_params(params, os.path.join(args.outdir, 'shader_params.glsl'))
    generate_cpp_header(params, os.path.join(args.outdir, 'shader_params.hpp'))
    print(f"层 {args.layer} 配置生成完成")
if __name__ == '__main__':
    main()
