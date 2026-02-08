#!/bin/bash
echo "=== 5level测试：多阶段计算管线与多个缓冲区 ==="
echo "模拟项目中的立体匹配流水线：多个阶段、多个缓冲区、资源管理"
echo ""

# 1. 编译着色器
echo "1. 编译着色器..."
glslangValidator -V compute_stage1.comp -o compute_stage1.comp.spv
if [ $? -ne 0 ]; then
    echo "❌ 着色器1编译失败"
    exit 1
fi
echo "✅ 着色器1编译成功"

glslangValidator -V compute_stage2.comp -o compute_stage2.comp.spv
if [ $? -ne 0 ]; then
    echo "❌ 着色器2编译失败"
    exit 1
fi
echo "✅ 着色器2编译成功"

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
    echo "✅ 5level测试通过！"
    echo "多阶段计算管线、多个缓冲区、资源管理全部成功"
    echo "说明：PanVK驱动支持复杂的多阶段流水线"
else
    echo ""
    echo "❌ 5level测试失败！"
    echo "说明：问题可能出现在多阶段流水线或资源管理上"
fi

exit $RESULT
