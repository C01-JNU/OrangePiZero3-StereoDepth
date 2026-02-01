#!/bin/bash

echo "=== OrangePi Zero3 StereoDepth 故障排除 ==="
echo ""

echo "1. 检查Vulkan驱动..."
echo "----------------------------------------"
if dpkg -l | grep -q "mesa-vulkan-drivers"; then
    echo "✓ Vulkan驱动已安装"
else
    echo "✗ Vulkan驱动未安装"
    echo "  运行: sudo apt install mesa-vulkan-drivers"
fi

echo ""
echo "2. 检查环境变量..."
echo "----------------------------------------"
if [ -n "$PAN_I_WANT_A_BROKEN_VULKAN_DRIVER" ]; then
    echo "✓ PAN_I_WANT_A_BROKEN_VULKAN_DRIVER=$PAN_I_WANT_A_BROKEN_VULKAN_DRIVER"
else
    echo "✗ PAN_I_WANT_A_BROKEN_VULKAN_DRIVER未设置"
    echo "  添加到 ~/.bashrc: export PAN_I_WANT_A_BROKEN_VULKAN_DRIVER=1"
    echo "  然后运行: source ~/.bashrc"
fi

echo ""
echo "3. 检查用户组..."
echo "----------------------------------------"
if groups | grep -q render; then
    echo "✓ 用户属于render组"
else
    echo "✗ 用户不属于render组"
    echo "  运行: sudo usermod -aG render $USER"
    echo "  然后重新登录或重启"
fi

echo ""
echo "4. 测试Vulkan..."
echo "----------------------------------------"
if command -v vulkaninfo >/dev/null 2>&1; then
    echo "✓ vulkaninfo命令可用"
    echo "  运行测试: vulkaninfo --summary 2>/dev/null | head -10"
    vulkaninfo --summary 2>/dev/null | head -10
else
    echo "✗ vulkaninfo命令不可用"
    echo "  安装: sudo apt install vulkan-tools"
fi

echo ""
echo "5. 检查项目构建..."
echo "----------------------------------------"
if [ -f "build/stereo_depth" ]; then
    echo "✓ 项目已构建"
    echo "  可执行文件: $(ls -lh build/stereo_depth)"
else
    echo "✗ 项目未构建"
    echo "  运行: ./tools/build.sh"
fi

echo ""
echo "=== 故障排除完成 ==="
echo "如果仍有问题，请检查日志文件或重新构建项目"
