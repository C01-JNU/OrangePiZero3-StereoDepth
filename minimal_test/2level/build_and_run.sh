#!/bin/bash
# minimal_test/2level/build_and_run.sh

echo "=== 2level测试：多个描述符集绑定 ==="
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
    echo "✅ 2level测试通过！"
    echo "多个描述符集绑定创建成功"
    echo "继续3level测试..."
else
    echo ""
    echo "❌ 2level测试失败！"
    echo "问题在多个描述符集绑定创建阶段"
fi

exit $RESULT
