#!/bin/bash

# 设置Mali-G31环境变量
export PAN_I_WANT_A_BROKEN_VULKAN_DRIVER=1

echo "=== 构建 OrangePiZero3 StereoDepth (不使用VMA) ==="
echo "当前目录: $(pwd)"
echo ""

# 检查必要的目录
echo "检查目录结构..."
if [ ! -d "src/vulkan" ]; then
    echo "错误: src/vulkan 目录不存在"
    exit 1
fi

# 清理之前的构建
echo ""
if [ -d "build" ]; then
    echo "清理之前的构建..."
    rm -rf build/*
else
    echo "创建构建目录..."
    mkdir -p build
fi

cd build

echo ""
echo "配置项目..."
cmake .. -DCMAKE_BUILD_TYPE=Release

if [ $? -ne 0 ]; then
    echo "CMake配置失败！"
    echo "可能的问题："
    echo "1. 缺少依赖库 (yaml-cpp, vulkan, opencv, spdlog)"
    echo "2. CMake版本过低"
    exit 1
fi

echo ""
echo "编译项目..."
make -j$(nproc)

if [ $? -eq 0 ]; then
    echo ""
    echo "=== 构建成功！ ==="
    echo "重要提示：确保已设置环境变量："
    echo "export PAN_I_WANT_A_BROKEN_VULKAN_DRIVER=1"
else
    echo ""
    echo "=== 构建失败！ ==="
    exit 1
fi
