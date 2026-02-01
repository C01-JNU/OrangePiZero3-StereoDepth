#!/bin/bash

echo "=== OrangePi Zero3 Vulkan 诊断工具 ==="
echo ""

echo "1. 系统信息..."
echo "----------------------------------------"
echo "主机名: $(hostname)"
echo "内核版本: $(uname -r)"
echo "架构: $(uname -m)"
echo ""

echo "2. Vulkan驱动检查..."
echo "----------------------------------------"
echo "已安装的Vulkan包:"
dpkg -l | grep -E "vulkan|mesa" | awk '{print "  " $2 " (" $3 ")"}'

echo ""
echo "3. 设备权限检查..."
echo "----------------------------------------"
echo "DRI设备:"
ls -la /dev/dri/ 2>/dev/null || echo "未找到/dev/dri目录"

echo ""
echo "用户组:"
groups

echo ""
echo "4. 环境变量检查..."
echo "----------------------------------------"
echo "PAN_I_WANT_A_BROKEN_VULKAN_DRIVER: $PAN_I_WANT_A_BROKEN_VULKAN_DRIVER"
echo "VK_ICD_FILENAMES: $VK_ICD_FILENAMES"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

echo ""
echo "5. Vulkan ICD文件..."
echo "----------------------------------------"
if [ -d "/usr/share/vulkan/icd.d" ]; then
    echo "ICD目录内容:"
    ls -la /usr/share/vulkan/icd.d/
    echo ""
    echo "ICD文件内容:"
    for icd in /usr/share/vulkan/icd.d/*.json; do
        echo "=== $icd ==="
        cat "$icd" 2>/dev/null | head -5
        echo ""
    done
else
    echo "未找到Vulkan ICD目录"
fi

echo ""
echo "6. 测试Vulkan简单程序..."
echo "----------------------------------------"
cat > /tmp/test_vulkan.c << 'TESTCODE'
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    // 设置环境变量
    setenv("PAN_I_WANT_A_BROKEN_VULKAN_DRIVER", "1", 1);
    
    VkApplicationInfo appInfo = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "Test",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "Test",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_0,
    };
    
    VkInstanceCreateInfo createInfo = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &appInfo,
    };
    
    VkInstance instance;
    VkResult result = vkCreateInstance(&createInfo, NULL, &instance);
    
    if (result == VK_SUCCESS) {
        printf("✓ Vulkan实例创建成功\n");
        
        // 枚举物理设备
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, NULL);
        printf("找到 %u 个物理设备\n", deviceCount);
        
        if (deviceCount > 0) {
            VkPhysicalDevice* devices = malloc(deviceCount * sizeof(VkPhysicalDevice));
            vkEnumeratePhysicalDevices(instance, &deviceCount, devices);
            
            for (uint32_t i = 0; i < deviceCount; i++) {
                VkPhysicalDeviceProperties properties;
                vkGetPhysicalDeviceProperties(devices[i], &properties);
                printf("设备 %u: %s (Vulkan %d.%d.%d)\n", 
                       i, properties.deviceName,
                       VK_VERSION_MAJOR(properties.apiVersion),
                       VK_VERSION_MINOR(properties.apiVersion),
                       VK_VERSION_PATCH(properties.apiVersion));
            }
            
            free(devices);
        }
        
        vkDestroyInstance(instance, NULL);
        return 0;
    } else {
        printf("✗ Vulkan实例创建失败: %d\n", result);
        return 1;
    }
}
TESTCODE

echo "编译测试程序..."
gcc -o /tmp/test_vulkan /tmp/test_vulkan.c -lvulkan 2>&1

if [ $? -eq 0 ]; then
    echo "运行测试程序..."
    /tmp/test_vulkan
else
    echo "编译失败"
fi

echo ""
echo "7. 检查PanVK驱动..."
echo "----------------------------------------"
if lsmod | grep -q panfrost; then
    echo "✓ panfrost内核模块已加载"
    lsmod | grep panfrost
else
    echo "✗ panfrost内核模块未加载"
    echo "尝试加载: sudo modprobe panfrost"
fi

echo ""
echo "8. 内存和资源..."
echo "----------------------------------------"
echo "可用内存: $(free -h | awk '/^Mem:/ {print $4}')"
echo "GPU内存信息（如果可用）:"
if [ -f /sys/kernel/debug/dri/0/memory ]; then
    cat /sys/kernel/debug/dri/0/memory 2>/dev/null | head -20
else
    echo "GPU内存信息不可用"
fi

echo ""
echo "=== 诊断完成 ==="
