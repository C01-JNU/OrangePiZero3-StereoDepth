#!/bin/bash

# OrangePiZero3-StereoDepth 智能构建脚本
# 功能：根据 config/global_config.yaml 中的 system.mode 自动选择编译模块
#       直接执行 cmake .. && make 时默认编译全部模块
# 最后更新: 2026年2月13日

set -e

# 设置 Mali-G31 环境变量
export PAN_I_WANT_A_BROKEN_VULKAN_DRIVER=1

echo "=== OrangePiZero3 StereoDepth 智能构建脚本 ==="
echo "当前目录: $(pwd)"
echo ""

# 检查配置文件
CONFIG_FILE="config/global_config.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件 $CONFIG_FILE 不存在！"
    exit 1
fi

# 使用 python3 解析 YAML（要求 python3 和 pyyaml）
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 python3，无法解析配置文件"
    exit 1
fi

# 读取 system.mode，默认 gpu
SYSTEM_MODE=$(python3 -c "
import yaml, sys
try:
    with open('$CONFIG_FILE') as f:
        cfg = yaml.safe_load(f)
    mode = cfg.get('system', {}).get('mode', 'gpu')
    print(mode)
except Exception as e:
    print('gpu', file=sys.stderr)
    print('gpu')
" 2>/dev/null)

echo "配置文件 system.mode = $SYSTEM_MODE"

# 根据 mode 设置 CMake 选项
CMAKE_OPTIONS=""

if [ "$SYSTEM_MODE" = "gpu" ]; then
    echo "→ 仅编译 GPU 模块（禁用 CPU）"
    CMAKE_OPTIONS="-DENABLE_GPU=ON -DENABLE_CPU=OFF"
elif [ "$SYSTEM_MODE" = "cpu" ]; then
    echo "→ 仅编译 CPU 模块（禁用 GPU）"
    CMAKE_OPTIONS="-DENABLE_GPU=OFF -DENABLE_CPU=ON"
else
    echo "→ 未知模式，默认编译全部模块"
    CMAKE_OPTIONS="-DENABLE_GPU=ON -DENABLE_CPU=ON"
fi

# 检查必要的目录
if [ ! -d "src/vulkan" ] && [ "$SYSTEM_MODE" != "cpu" ]; then
    echo "警告: src/vulkan 目录不存在，GPU 模块可能无法编译"
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
