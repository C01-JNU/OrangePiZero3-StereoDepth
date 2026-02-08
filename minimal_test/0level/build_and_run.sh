#!/bin/bash
# minimal_test_simple/build_and_run.sh

echo "=== 最简单的Vulkan测试 ==="
echo ""

# 0. 确保在正确的目录
cd "$(dirname "$0")"

# 1. 编译着色器
echo "1. 编译着色器..."
if command -v glslangValidator &> /dev/null; then
    glslangValidator -V minimal.comp -o minimal.comp.spv
    if [ $? -eq 0 ]; then
        echo "✅ 着色器编译成功"
    else
        echo "❌ 着色器编译失败"
        exit 1
    fi
else
    echo "❌ glslangValidator未安装"
    exit 1
fi

# 2. 编译C++程序
echo ""
echo "2. 编译C++程序..."
g++ -std=c++17 -I/usr/include/vulkan -I/usr/include -c simple_test.cpp -o simple_test.o
if [ $? -ne 0 ]; then
    echo "❌ 编译失败"
    exit 1
fi

# 3. 链接程序
echo ""
echo "3. 链接程序..."
g++ simple_test.o -lvulkan -o simple_test
if [ $? -ne 0 ]; then
    echo "❌ 链接失败"
    exit 1
fi

# 4. 运行测试
echo ""
echo "4. 运行测试..."
echo "========================================"
./simple_test
RESULT=$?
echo "========================================"

# 5. 清理中间文件
rm -f simple_test.o

if [ $RESULT -eq 0 ]; then
    echo ""
    echo "🎉 测试成功！PanVK驱动可以工作。"
    echo "问题出在项目环境的配置差异上。"
else
    echo ""
    echo "❌ 测试失败！PanVK驱动有问题。"
fi

exit $RESULT
