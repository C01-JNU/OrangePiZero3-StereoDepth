#!/usr/bin/env python3
"""
合并着色器脚本
将shader_params.glsl的内容插入到着色器文件中
"""

import os
import sys

def merge_shader(shader_file, params_file, output_file):
    """将参数文件合并到着色器文件中"""
    
    try:
        # 读取参数文件
        with open(params_file, 'r') as f:
            params_content = f.read()
        
        # 读取着色器文件
        with open(shader_file, 'r') as f:
            shader_lines = f.readlines()
        
        # 查找#version行
        version_line_index = -1
        for i, line in enumerate(shader_lines):
            if line.strip().startswith('#version'):
                version_line_index = i
                break
        
        if version_line_index == -1:
            # 如果没有找到#version，添加默认的
            shader_lines.insert(0, '#version 450\n')
            version_line_index = 0
        
        # 在#version行后插入参数内容
        merged_lines = []
        for i, line in enumerate(shader_lines):
            merged_lines.append(line)
            if i == version_line_index:
                # 在#version行后插入参数内容
                merged_lines.append('\n// 自动插入的配置参数\n')
                merged_lines.append(params_content)
                merged_lines.append('\n')
        
        # 写入输出文件
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            f.writelines(merged_lines)
        
        print(f"合并完成: {os.path.basename(shader_file)} + {os.path.basename(params_file)} -> {os.path.basename(output_file)}")
        return True
        
    except Exception as e:
        print(f"合并失败: {e}")
        return False

def main():
    if len(sys.argv) < 4:
        print("用法: python3 merge_shader.py <shader_file> <params_file> <output_file>")
        sys.exit(1)
    
    shader_file = sys.argv[1]
    params_file = sys.argv[2]
    output_file = sys.argv[3]
    
    if not os.path.exists(shader_file):
        print(f"错误: 着色器文件不存在: {shader_file}")
        sys.exit(1)
    
    if not os.path.exists(params_file):
        print(f"错误: 参数文件不存在: {params_file}")
        sys.exit(1)
    
    if merge_shader(shader_file, params_file, output_file):
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
