#!/bin/bash

# OrangePiZero3-StereoDepth 运行脚本
# 最后更新: 2026-01-18
# 作者: C01-JNU

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
BIN_DIR="$BUILD_DIR/bin"
EXECUTABLE="$BIN_DIR/stereo_depth_main"

echo "=========================================="
echo "OrangePiZero3-StereoDepth 启动"
echo "=========================================="

# 检查可执行文件是否存在
if [ ! -f "$EXECUTABLE" ]; then
    echo "错误: 找不到可执行文件 $EXECUTABLE"
    echo "请先运行: ./scripts/build.sh"
    exit 1
fi

# 创建必要的目录
mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$PROJECT_ROOT/images/output"
mkdir -p "$PROJECT_ROOT/images/calibration"
mkdir -p "$PROJECT_ROOT/calibration_results"

# 检查配置文件
CONFIG_FILE="$PROJECT_ROOT/config/global_config.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 找不到配置文件 $CONFIG_FILE"
    exit 1
fi

# 检查标定文件（如果存在则提示）
CALIB_FILE="$PROJECT_ROOT/calibration_results/stereo_calibration.yml"
if [ ! -f "$CALIB_FILE" ]; then
    echo "警告: 标定文件不存在: $CALIB_FILE"
    echo "请先运行相机标定程序"
    echo ""
    read -p "继续运行? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
fi

# 检查测试图像
TEST_IMAGE_DIR="$PROJECT_ROOT/images/test"
if [ ! -d "$TEST_IMAGE_DIR" ] || [ -z "$(ls -A "$TEST_IMAGE_DIR" 2>/dev/null)" ]; then
    echo "警告: 测试图像目录为空: $TEST_IMAGE_DIR"
    echo "请将测试图像放在此目录"
    echo ""
    read -p "继续运行? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
fi

# 检查当前平台
if [[ $(uname -m) == "aarch64" ]] || [[ $(uname -m) == "armv7l" ]]; then
    echo "检测到 ARM 平台，启用特定优化..."
    export MALI_HW_VERSION="G31"
    export VULKAN_ICD_FILENAMES="/usr/share/vulkan/icd.d/mali_icd.json"
else
    echo "检测到 x86 平台..."
fi

# 设置环境变量
export LD_LIBRARY_PATH="$BUILD_DIR/lib:$LD_LIBRARY_PATH"
export VK_LOADER_DEBUG=all  # Vulkan加载器调试信息

echo ""
echo "运行配置:"
echo "  配置文件: $CONFIG_FILE"
echo "  标定文件: $CALIB_FILE"
echo "  测试图像: $TEST_IMAGE_DIR"
echo "  输出目录: $PROJECT_ROOT/images/output"
echo "  日志文件: $PROJECT_ROOT/logs/stereo_depth.log"
echo ""

# 运行程序
cd "$PROJECT_ROOT"
"$EXECUTABLE"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "程序正常退出"
else
    echo "程序异常退出，代码: $EXIT_CODE"
fi
echo "输出图像保存到: $PROJECT_ROOT/images/output/"
echo "=========================================="
