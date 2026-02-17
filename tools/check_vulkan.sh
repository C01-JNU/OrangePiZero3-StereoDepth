#!/bin/bash

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color
BOLD='\033[1m'
DIM='\033[2m'

# 分隔线
HR="${BLUE}────────────────────────────────────────────────────────────────────────${NC}"

# 打印带颜色的标题
print_header() {
    echo -e "\n${PURPLE}${BOLD}$1${NC}"
    echo -e "$HR"
}

# 打印带颜色的信息
print_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 进度指示器
spinner() {
    local pid=$1
    local delay=0.1
    local spinstr='|/-\'
    while [ "$(ps a | awk '{print $1}' | grep $pid)" ]; do
        local temp=${spinstr#?}
        printf " [%c]  " "$spinstr"
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
        printf "\b\b\b\b\b\b"
    done
    printf "    \b\b\b\b"
}

# 欢迎界面
clear
echo -e "${BLUE}${BOLD}"
echo "  ╔═══════════════════════════════════════════════════════╗"
echo "  ║                                                       ║"
echo "  ║         OrangePi Zero3 Vulkan 诊断工具                ║"
echo "  ║                Vulkan Diagnostic Tool                 ║"
echo "  ║                                                       ║"
echo "  ╚═══════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo -e "${DIM}版本: 2.0 | 适用于: OrangePi Zero3 | 作者: C01-JNU${NC}"
echo ""

# 询问是否继续
read -p "是否开始诊断Vulkan配置？(y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}诊断已取消。${NC}"
    exit 1
fi

echo -e "\n${GREEN}🚀 开始诊断...${NC}\n"

# 1. 系统信息
print_header "1. 系统信息"
echo -e "${WHITE}${BOLD}主机名:${NC} $(hostname)"
echo -e "${WHITE}${BOLD}内核版本:${NC} $(uname -r)"
echo -e "${WHITE}${BOLD}架构:${NC} $(uname -m)"
echo -e "${WHITE}${BOLD}系统时间:${NC} $(date)"
echo -e "${WHITE}${BOLD}运行时间:${NC} $(uptime -p)"

# 2. Vulkan驱动检查
print_header "2. Vulkan驱动检查"
echo -e "${WHITE}${BOLD}已安装的Vulkan/Mesa包:${NC}"
if dpkg -l | grep -E "vulkan|mesa" &>/dev/null; then
    dpkg -l | grep -E "vulkan|mesa" | while read line; do
        pkg=$(echo "$line" | awk '{print $2}')
        ver=$(echo "$line" | awk '{print $3}')
        status=$(echo "$line" | awk '{print $1}')
        if [[ $status == "ii" ]]; then
            echo -e "  ${GREEN}✓${NC} $pkg ($ver)"
        else
            echo -e "  ${YELLOW}⚠${NC} $pkg ($ver)"
        fi
    done
else
    print_warning "未找到Vulkan或Mesa相关包"
fi

# 3. 设备权限检查
print_header "3. 设备权限检查"
echo -e "${WHITE}${BOLD}DRI设备文件:${NC}"
if [ -d "/dev/dri" ]; then
    ls -la /dev/dri/ | while read line; do
        if [[ $line == total* ]]; then
            continue
        fi
        # 检查权限
        if [[ $line == *"rw"*"rw"* ]]; then
            echo -e "  ${GREEN}✓${NC} $line"
        else
            echo -e "  ${YELLOW}⚠${NC} $line"
        fi
    done
else
    print_error "未找到/dev/dri目录"
fi

echo -e "\n${WHITE}${BOLD}当前用户组:${NC} $(groups)"
echo -e "${WHITE}${BOLD}当前用户:${NC} $(whoami)"

# 检查是否在video/render组
if groups | grep -q "video" && groups | grep -q "render"; then
    print_success "用户在video和render组中"
else
    print_warning "用户可能不在video/render组中"
    echo -e "${DIM}提示: 可以运行 'sudo usermod -aG video,render $(whoami)' 添加权限${NC}"
fi

# 4. 环境变量检查
print_header "4. 环境变量检查"
echo -e "${WHITE}${BOLD}PAN_I_WANT_A_BROKEN_VULKAN_DRIVER:${NC} ${GREEN}${PAN_I_WANT_A_BROKEN_VULKAN_DRIVER:-未设置}${NC}"
echo -e "${WHITE}${BOLD}VK_ICD_FILENAMES:${NC} ${CYAN}${VK_ICD_FILENAMES:-未设置}${NC}"
echo -e "${WHITE}${BOLD}LD_LIBRARY_PATH:${NC} ${CYAN}${LD_LIBRARY_PATH:-未设置}${NC}"

# 5. Vulkan ICD文件
print_header "5. Vulkan ICD文件"
ICD_DIRS="/usr/share/vulkan/icd.d /etc/vulkan/icd.d"
found_icd=0

for dir in $ICD_DIRS; do
    if [ -d "$dir" ]; then
        echo -e "${WHITE}${BOLD}ICD目录: $dir${NC}"
        count=$(ls -1 "$dir"/*.json 2>/dev/null | wc -l)
        if [ $count -gt 0 ]; then
            found_icd=1
            for icd in "$dir"/*.json; do
                echo -e "\n  ${GREEN}►${NC} $(basename "$icd")"
                # 提取关键信息
                lib_path=$(grep '"library_path"' "$icd" | cut -d'"' -f4 2>/dev/null)
                if [ -n "$lib_path" ]; then
                    echo -e "    ${DIM}库路径: $lib_path${NC}"
                    if [ -f "$lib_path" ]; then
                        echo -e "    ${GREEN}✓ 库文件存在${NC}"
                    else
                        echo -e "    ${RED}✗ 库文件不存在${NC}"
                    fi
                fi
            done
        fi
    fi
done

if [ $found_icd -eq 0 ]; then
    print_error "未找到Vulkan ICD文件"
fi

# 6. 测试Vulkan简单程序
print_header "6. Vulkan功能测试"
echo -e "${DIM}创建测试程序...${NC}"

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
        printf("SUCCESS: Vulkan实例创建成功\n");
        
        // 枚举物理设备
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, NULL);
        printf("INFO: 找到 %u 个物理设备\n", deviceCount);
        
        if (deviceCount > 0) {
            VkPhysicalDevice* devices = malloc(deviceCount * sizeof(VkPhysicalDevice));
            vkEnumeratePhysicalDevices(instance, &deviceCount, devices);
            
            for (uint32_t i = 0; i < deviceCount; i++) {
                VkPhysicalDeviceProperties properties;
                vkGetPhysicalDeviceProperties(devices[i], &properties);
                
                VkPhysicalDeviceFeatures features;
                vkGetPhysicalDeviceFeatures(devices[i], &features);
                
                printf("DEVICE[%u]:\n", i);
                printf("  名称: %s\n", properties.deviceName);
                printf("  类型: %d\n", properties.deviceType);
                printf("  Vulkan版本: %d.%d.%d\n", 
                       VK_VERSION_MAJOR(properties.apiVersion),
                       VK_VERSION_MINOR(properties.apiVersion),
                       VK_VERSION_PATCH(properties.apiVersion));
                printf("  驱动版本: %d.%d.%d\n",
                       VK_VERSION_MAJOR(properties.driverVersion),
                       VK_VERSION_MINOR(properties.driverVersion),
                       VK_VERSION_PATCH(properties.driverVersion));
                printf("  供应商ID: 0x%X\n", properties.vendorID);
                printf("  设备ID: 0x%X\n", properties.deviceID);
            }
            
            free(devices);
        }
        
        vkDestroyInstance(instance, NULL);
        return 0;
    } else {
        printf("ERROR: Vulkan实例创建失败 (错误码: %d)\n", result);
        return 1;
    }
}
TESTCODE

echo -e "${DIM}编译测试程序...${NC}"
if gcc -o /tmp/test_vulkan /tmp/test_vulkan.c -lvulkan 2>/dev/null; then
    print_success "编译成功"
    echo -e "\n${WHITE}${BOLD}运行测试...${NC}"
    echo "$HR"
    /tmp/test_vulkan | while read line; do
        if [[ $line == SUCCESS:* ]]; then
            echo -e "${GREEN}${line#SUCCESS: }${NC}"
        elif [[ $line == ERROR:* ]]; then
            echo -e "${RED}${line#ERROR: }${NC}"
        elif [[ $line == INFO:* ]]; then
            echo -e "${CYAN}${line#INFO: }${NC}"
        elif [[ $line == DEVICE* ]]; then
            echo -e "${WHITE}${BOLD}${line}${NC}"
        else
            echo -e "  ${line}"
        fi
    done
    echo "$HR"
else
    print_error "编译失败"
    echo -e "${DIM}编译错误信息:${NC}"
    gcc -o /tmp/test_vulkan /tmp/test_vulkan.c -lvulkan 2>&1 | sed 's/^/  /'
fi

# 7. 检查PanVK驱动
print_header "7. PanVK驱动状态"
if lsmod | grep -q panfrost; then
    print_success "panfrost内核模块已加载"
    echo -e "${DIM}模块详情:${NC}"
    lsmod | grep panfrost | sed 's/^/  /'
    
    # 检查版本
    if [ -f "/sys/module/panfrost/version" ]; then
        echo -e "${DIM}模块版本: $(cat /sys/module/panfrost/version)${NC}"
    fi
else
    print_error "panfrost内核模块未加载"
    echo -e "${YELLOW}尝试加载模块...${NC}"
    if sudo modprobe panfrost 2>/dev/null; then
        print_success "模块加载成功"
    else
        print_error "模块加载失败"
    fi
fi

# 8. 内存和资源
print_header "8. 系统资源"
echo -e "${WHITE}${BOLD}内存使用:${NC}"
free -h | sed 's/^/  /'

echo -e "\n${WHITE}${BOLD}GPU信息:${NC}"
if [ -f "/sys/kernel/debug/dri/0/name" ]; then
    echo -e "  设备名称: $(cat /sys/kernel/debug/dri/0/name)"
fi

if [ -f "/sys/kernel/debug/dri/0/memory" ]; then
    echo -e "\n${DIM}GPU内存统计:${NC}"
    cat /sys/kernel/debug/dri/0/memory 2>/dev/null | head -10 | sed 's/^/  /'
else
    echo -e "  ${DIM}GPU内存信息不可用${NC}"
fi

# 总结
print_header "诊断总结"
echo -e "${WHITE}${BOLD}诊断完成于:${NC} $(date)"
echo -e "${WHITE}${BOLD}总体状态:${NC}"

# 简单评估
if [ -f "/tmp/test_vulkan" ] && /tmp/test_vulkan &>/dev/null; then
    echo -e "  ${GREEN}✅ Vulkan工作正常${NC}"
else
    echo -e "  ${RED}❌ Vulkan可能有问题${NC}"
fi

if lsmod | grep -q panfrost; then
    echo -e "  ${GREEN}✅ PanFrost驱动已加载${NC}"
else
    echo -e "  ${RED}❌ PanFrost驱动未加载${NC}"
fi

echo -e "\n${WHITE}${BOLD}建议:${NC}"
echo -e "  1. 确保安装了正确的驱动: ${DIM}sudo apt install mesa-vulkan-drivers${NC}"
echo -e "  2. 添加用户到video/render组: ${DIM}sudo usermod -aG video,render $(whoami)${NC}"
echo -e "  3. 重启系统使更改生效"

echo -e "\n${GREEN}${BOLD}✨ 诊断完成！✨${NC}"
echo -e "${DIM}感谢使用OrangePi Zero3 Vulkan诊断工具${NC}\n"

# 清理临时文件
rm -f /tmp/test_vulkan.c /tmp/test_vulkan 2>/dev/null
