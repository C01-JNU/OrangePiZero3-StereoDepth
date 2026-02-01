#!/bin/bash

echo "=== 测试Vulkan环境 ==="
echo ""

# 设置Mali-G31环境变量
export PAN_I_WANT_A_BROKEN_VULKAN_DRIVER=1

echo "1. 检查Vulkan安装..."
if command -v vulkaninfo >/dev/null 2>&1; then
    echo "Vulkan信息工具已安装"
    echo "运行 vulkaninfo --summary..."
    vulkaninfo --summary 2>/dev/null | head -20
else
    echo "警告: vulkaninfo 未安装"
    echo "尝试安装: sudo apt install vulkan-tools"
fi

echo ""
echo "2. 检查Vulkan设备..."
if command -v vulkaninfo >/dev/null 2>&1; then
    echo "可用的Vulkan设备:"
    vulkaninfo --summary 2>/dev/null | grep "deviceName" | head -5
fi

echo ""
echo "3. 检查环境变量..."
echo "PAN_I_WANT_A_BROKEN_VULKAN_DRIVER=$PAN_I_WANT_A_BROKEN_VULKAN_DRIVER"

echo ""
echo "4. 检查用户组..."
if groups | grep -q render; then
    echo "用户属于render组 ✓"
else
    echo "警告: 用户不属于render组"
    echo "可能需要运行: sudo usermod -aG render $USER"
fi

echo ""
echo "=== 测试完成 ==="
