#!/usr/bin/env python3
import yaml
import os
import argparse
from generate_shader_config import compute_layer_params

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--outfile', required=True)
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    levels = config.get('pyramid', {}).get('levels', 1)
    wg_x = config.get('gpu', {}).get('shader_workgroup_size', [16,16])[0]
    wg_y = config.get('gpu', {}).get('shader_workgroup_size', [16,16])[1]
    level_params = []
    for l in range(levels):
        p = compute_layer_params(config, l)
        level_params.append(p)
    header = f"""#pragma once
#include <cstdint>
namespace stereo_depth::vulkan {{
constexpr uint32_t WORKGROUP_X = {wg_x}u;
constexpr uint32_t WORKGROUP_Y = {wg_y}u;
struct PyramidLevelParams {{
    uint32_t width; uint32_t height; uint32_t max_disparity; uint32_t window_size;
    uint32_t aggregation_window; float uniqueness_ratio; float penalty_p1; float penalty_p2;
    uint32_t flags; uint32_t speckle_window; uint32_t speckle_range; uint32_t median_size;
    uint32_t search_radius;   // 新增：搜索半径
}};
constexpr uint32_t PYRAMID_LEVEL_COUNT = {levels}u;
constexpr PyramidLevelParams PYRAMID_LEVELS[PYRAMID_LEVEL_COUNT] = {{
"""
    for p in level_params:
        header += f"""    {{{p['width']}u, {p['height']}u,
      {p['max_disparity']}u, {p['window_size']}u,
      {p['aggregation_window']}u, {p['uniqueness_ratio']}f,
      {p['penalty_p1']}f, {p['penalty_p2']}f,
      {p['flags']}u, {p['speckle_window']}u,
      {p['speckle_range']}u, {p['median_size']}u,
      {p['search_radius']}u}},
"""
    header += f"""}};
}} // namespace stereo_depth::vulkan
"""
    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    with open(args.outfile, 'w') as f:
        f.write(header)
    print(f"生成金字塔配置: {args.outfile}")
if __name__ == '__main__':
    main()
