#!/bin/bash
# minimal_test/4level/build_and_run.sh

echo "=== 4level测试：存储缓冲区与描述符池 ==="
echo ""

cd "$(dirname "$0")"

# 1. 编译着色器
echo "1. 编译着色器..."
glslangValidator -V compute.comp -o compute.comp.spv
if [ $? -ne 0 ]; then
    echo "❌ 着色器编译失败"
    exit 1
fi
echo "✅ 着色器编译成功"

# 2. 编译程序
echo ""
echo "2. 编译程序..."
g++ -std=c++17 -I/usr/include/vulkan test.cpp -lvulkan -o test
if [ $? -ne 0 ]; then
    echo "❌ 编译失败"
    exit 1
fi
echo "✅ 程序编译成功"

# 3. 运行测试
echo ""
echo "3. 运行测试..."
echo "========================================"
./test
RESULT=$?
echo "========================================"

if [ $RESULT -eq 0 ]; then
    echo ""
    echo "✅ 4level测试通过！"
    echo "存储缓冲区、描述符池、描述符集创建成功"
    echo "继续5level测试..."
else
    echo ""
    echo "❌ 4level测试失败！"
    echo "问题在存储缓冲区/描述符池/描述符集创建阶段"
    echo ""
    echo "需要检查："
    echo "1. 内存类型是否合适"
    echo "2. 描述符池大小"
    echo "3. 缓冲区使用标志"
fi

exit $RESULT
