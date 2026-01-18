#!/bin/bash

# OrangePiZero3-StereoDepth 构建脚本
# 最后更新: 2026-01-18
# 作者: C01-JNU

set -e  # 遇到错误立即退出

echo "=========================================="
echo "OrangePiZero3-StereoDepth 构建系统"
echo "=========================================="

# 参数解析
BUILD_TYPE="Release"
BUILD_DIR="build"
CLEAN_BUILD=false
ENABLE_TESTS=false
ENABLE_ROS2=false
JOBS=$(nproc)

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        -c|--clean)
            CLEAN_BUILD=true
            shift
            ;;
        -t|--tests)
            ENABLE_TESTS=true
            shift
            ;;
        -r|--ros2)
            ENABLE_ROS2=true
            shift
            ;;
        -j|--jobs)
            JOBS="$2"
            shift 2
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  -d, --debug     调试构建"
            echo "  -c, --clean     清理构建"
            echo "  -t, --tests     启用测试"
            echo "  -r, --ros2      启用ROS2支持"
            echo "  -j, --jobs      并行构建任务数"
            echo "  -h, --help      显示帮助"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            exit 1
            ;;
    esac
done

# 检查glslangValidator（着色器编译器）
if ! command -v glslangValidator &> /dev/null; then
    echo "错误: 找不到 glslangValidator"
    echo "请安装: sudo apt install glslang-tools"
    exit 1
fi

# 检查Vulkan开发包
if ! pkg-config --exists vulkan; then
    echo "错误: 找不到 Vulkan 开发包"
    echo "请安装: sudo apt install vulkan-tools libvulkan-dev"
    exit 1
fi

# 检查OpenCV
if ! pkg-config --exists opencv4; then
    if ! pkg-config --exists opencv; then
        echo "错误: 找不到 OpenCV 开发包"
        echo "请安装: sudo apt install libopencv-dev"
        exit 1
    fi
fi

# 检查yaml-cpp
if ! pkg-config --exists yaml-cpp; then
    echo "错误: 找不到 yaml-cpp 开发包"
    echo "请安装: sudo apt install libyaml-cpp-dev"
    exit 1
fi

echo "构建配置:"
echo "  构建类型: $BUILD_TYPE"
echo "  构建目录: $BUILD_DIR"
echo "  并行任务: $JOBS"
echo "  启用测试: $ENABLE_TESTS"
echo "  ROS2支持: $ENABLE_ROS2"

# 清理构建目录
if [ "$CLEAN_BUILD" = true ]; then
    echo "清理构建目录..."
    rm -rf $BUILD_DIR
fi

# 创建构建目录
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# 配置CMake
echo "配置CMake..."
CMAKE_OPTIONS="-DCMAKE_BUILD_TYPE=$BUILD_TYPE"
CMAKE_OPTIONS="$CMAKE_OPTIONS -DBUILD_TESTS=$ENABLE_TESTS"
CMAKE_OPTIONS="$CMAKE_OPTIONS -DWITH_ROS2=$ENABLE_ROS2"

cmake .. $CMAKE_OPTIONS

# 构建项目
echo "开始构建..."
make -j$JOBS

# 编译着色器
echo "编译着色器..."
make compile_shaders -j$JOBS

echo "=========================================="
echo "构建完成!"
echo "=========================================="
echo ""
echo "可执行文件位置:"
echo "  $(pwd)/bin/stereo_depth_main"
echo ""
echo "运行命令:"
echo "  cd $(pwd)/bin && ./stereo_depth_main"
echo ""
echo "或使用项目根目录的 run.sh 脚本"
echo "=========================================="
