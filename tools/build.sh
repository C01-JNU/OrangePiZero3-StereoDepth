#!/bin/bash

# OrangePiZero3-StereoDepth 构建脚本
# 最后更新: 2026年2月17日
set -e
echo "=== OrangePiZero3 StereoDepth 构建脚本 ==="
echo "当前目录: $(pwd)"
echo ""

# 检查配置文件
CONFIG_FILE="config/global_config.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件 $CONFIG_FILE 不存在！"
    exit 1
fi

# 清理之前的构建（可选，默认保留）
if [ "$1" == "clean" ]; then
    echo "清理构建目录..."
    rm -rf build
fi

# 创建构建目录
mkdir -p build
cd build

echo ""
echo "配置项目: cmake .. $CMAKE_OPTIONS"
cmake .. $CMAKE_OPTIONS

if [ $? -ne 0 ]; then
    echo ""
    echo "CMake 配置失败！"
    echo "可能的问题："
    echo "1. 缺少依赖库 (yaml-cpp, vulkan, opencv, spdlog)"
    echo "2. CMake 版本过低"
    exit 1
fi

echo ""
echo "编译项目..."
make -j$(nproc)

if [ $? -eq 0 ]; then
    echo ""
    echo "=== 构建成功！ ==="
    echo "可执行文件位于: build/bin/"
    echo ""
    echo "重要提示：运行 GPU 模式前请确保已设置环境变量："
    echo "export PAN_I_WANT_A_BROKEN_VULKAN_DRIVER=1"
else
    echo ""
    echo "=== 构建失败！ ==="
    exit 1
fi
