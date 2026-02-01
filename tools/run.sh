#!/bin/bash

echo "=== 运行 OrangePiZero3-StereoDepth ==="
echo "当前目录: $(pwd)"
echo ""

# 设置Mali-G31环境变量
export PAN_I_WANT_A_BROKEN_VULKAN_DRIVER=1

# 检查可执行文件是否存在
if [ ! -f "build/stereo_depth" ]; then
    echo "错误: 可执行文件不存在！"
    echo "请先运行: ./tools/build.sh"
    exit 1
fi

# 检查配置文件是否存在
if [ ! -f "config/global_config.yaml" ]; then
    echo "警告: 配置文件不存在，使用默认配置"
    # 确保配置文件存在
    if [ ! -d "config" ]; then
        mkdir -p config
    fi
    cp ../config/global_config.yaml config/ 2>/dev/null || echo "无法复制配置文件"
fi

echo "运行程序..."
echo "======================================"
cd build
./stereo_depth
