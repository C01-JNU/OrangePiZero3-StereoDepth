# OrangePiZero3-StereoDepth

本项目代码由 DeepSeek 编写。  
项目为香橙派 Zero3 的官方 Ubuntu 24.04 系统打造，支持 ROS2 Jazzy。

**代码仓库**  
- Gitee: [https://gitee.com/C01-JNU/orange-pi-zero3-stereo-depth.git](https://gitee.com/C01-JNU/orange-pi-zero3-stereo-depth.git)  
- GitHub: [https://github.com/C01-JNU/OrangePiZero3-StereoDepth.git](https://github.com/C01-JNU/OrangePiZero3-StereoDepth.git)

## 第三方依赖

```bash
sudo apt install libspdlog-dev
```

## 性能说明

目前实测，CPU 模式的 SGBM 算法处理 640×480 的双目拼接图像约需 **50ms/帧**，CPU 占用约 70%；而 GPU 模式需要约 **1700ms/帧**。香橙派 Zero3 的 GPU 性能有限，因此除 CPU SGBM 外，其他模式（如 GPU、自定义 Census）目前仅作实验性参考。配置文件默认使用 CPU 模式的 SGBM。

项目编译耗时约 **10 分钟**，请耐心等待。


## 部分默认配置

默认的拼接图像分辨率是 **640x480**

默认的运行帧率上限是 **10fps**

上述设置可以在 config/global_config.yaml 中找到和修改

---

## 使用方式

### 1. 下载项目
```bash
git clone https://gitee.com/C01-JNU/orange-pi-zero3-stereo-depth.git
cd OrangePiZero3-StereoDepth
```
或
```bash
git clone https://github.com/C01-JNU/OrangePiZero3-StereoDepth.git
cd OrangePiZero3-StereoDepth
```

### 2. 检查配置文件
项目包含两个配置文件：
- **主配置文件**：`config/global_config.yaml`（立体匹配参数、摄像头参数、标定路径、ROS2 话题等）
- **ROS2 节点控制文件**：`src/ros2_node/config/params.yaml`（深度图/点云发布开关）

可根据注释修改，**修改后需重新编译**。

### 3. 标定
项目默认提供了无畸变的标定文件 `calibration_results/stereo_calibration.yml`，对应 640×480 拼接图像，基线 40.5mm。如需重新标定，请参考下文【标定详细步骤】。

### 4. ROS2 编译与运行
在 ROS2 工作空间（例如 `~/camera/ros2_ws/src`）中执行：

```bash
colcon build --packages-select orangepizero3_stereodepth
```

CMake 将自动根据 `config/global_config.yaml` 中的 `system.mode`（`cpu`/`gpu`）和 `ros2.enabled` 决定编译选项。编译完成后：

```bash
source install/setup.bash
ros2 launch orangepizero3_stereodepth stereo_depth_launch.py
```

- **订阅话题**：`/camera/left/image_raw`、`/camera/right/image_raw`（默认）
- **发布话题**：
  - `/stereo/disparity`（视差图，始终发布）
  - `/stereo/depth`（深度图，需在 `params.yaml` 中开启）
  - `/stereo/points`（点云，需在 `params.yaml` 中开启）

所有话题名称可在 `global_config.yaml` 的 `ros2.topics` 中修改。

### 5. 在自有 C++ 项目中直接调用立体匹配引擎

#### 5.1 输入/输出接口说明
- **输入图像**：
  - 格式：单通道 8 位灰度图（`CV_8UC1`）
  - 尺寸：与配置文件 `camera.width` 和 `camera.height` 一致（例如 640×480 拼接图 ⇒ 单眼 320×480）。内部处理时会自动分割左右图，因此调用者只需提供原始拼接图像。
- **输出视差图**：
  - 格式：16 位无符号单通道（`CV_16UC1`），像素值即为视差值（单位：像素），无效点视差为 0
  - 尺寸：与输入单眼尺寸相同（例如 320×480）
- **标定校正**（可选）：
  - 如需校正，可使用 `StereoRectifier` 类对原始左右图像进行校正，校正后的图像尺寸仍为 320×480（`SCALE_TO_FIT` 模式）

#### 5.2 接口调用示例

**CPU 匹配器**（推荐用于生产环境）：
```cpp
#include "cpu_stereo/cpu_stereo_matcher.hpp"
#include "calibration/stereo_rectifier.hpp"

// 加载配置
stereo_depth::utils::ConfigManager::getInstance().loadGlobalConfig("path/to/global_config.yaml");

// 可选：初始化校正器
stereo_depth::calibration::StereoRectifier rectifier;
rectifier.loadAndInitialize("calibration_results/stereo_calibration.yml");

// 初始化匹配器
stereo_depth::cpu_stereo::CpuStereoMatcher matcher;
matcher.initializeFromConfig();   // 使用全局配置中的算法参数

// 处理每帧图像
cv::Mat left_raw, right_raw;      // 从摄像头或文件读取的左右图（CV_8U）
cv::Mat left_rect, right_rect;
if (rectifier.isInitialized()) {
    rectifier.rectifyPair(left_raw, right_raw, left_rect, right_rect);
} else {
    left_rect = left_raw;
    right_rect = right_raw;
}
cv::Mat disparity = matcher.compute(left_rect, right_rect);   // 返回 CV_16U
```

**GPU 匹配器**（实验性）：
```cpp
#include "vulkan/context.hpp"
#include "vulkan/stereo_pipeline.hpp"

stereo_depth::vulkan::VulkanContext ctx;
ctx.initialize();   // 需设置环境变量 PAN_I_WANT_A_BROKEN_VULKAN_DRIVER=1

stereo_depth::vulkan::StereoPipeline pipeline(ctx);
pipeline.initialize();
pipeline.setLeftImage(left_rect.data);   // left_rect.data 为 uchar* 灰度数据
pipeline.setRightImage(right_rect.data);
pipeline.compute();

std::vector<uint16_t> disp(width * height);
pipeline.getDisparityMap(disp.data());
cv::Mat disparity(height, width, CV_16UC1, disp.data());   // 转换为 cv::Mat
```

#### 5.3 构建与注意事项
- **链接依赖**：在 CMake 中需链接对应模块（`cpu_stereo`、`vulkan_stereo`）以及 `utils`、`calibration_utils`、`camera`（若使用模拟摄像头）。可参考项目根 `CMakeLists.txt` 中的目标链接方式。
- **性能性质**：所有 `compute()` 均为**同步阻塞调用**，单帧耗时约 80ms（CPU SGBM）或 400ms+（GPU Vulkan）。**调用者必须自行实现帧率控制**（如使用双缓冲或生产者‑消费者线程分离采集与计算），否则将阻塞摄像头采集流。项目提供的 `main.cpp` 中已内置帧率控制示例（基于定时器等待），可供参考。
- **配置驱动**：所有参数从 `config/global_config.yaml` 读取，修改后需重新编译项目（参数通过生成脚本嵌入二进制）。运行时需确保配置文件路径正确（可通过绝对路径或工作目录设置）。

---

## 配置文件说明

1. **主配置文件**：`config/global_config.yaml`  
   包含立体匹配算法参数、摄像头尺寸、标定文件路径、ROS2 话题名称等。

2. **ROS2 节点控制文件**：`src/ros2_node/config/params.yaml`  
   控制深度图（`publish_depth`）和点云（`publish_pointcloud`）的发布开关，默认关闭深度图节点，开启点云节点。

---

## 标定详细步骤

有两种方式引入标定参数：

### 方式一：使用现有标定文件
直接将符合 OpenCV 格式的标定文件重命名为 `stereo_calibration.yml`，放入 `calibration_results/` 文件夹覆盖默认文件。

### 方式二：拍摄图像并自行标定
1. **准备标定板**：打开项目根目录的 `标定板.jpg`，测量屏幕上方格的实际边长，将边长（米）填入 `config/global_config.yaml` 的 `square_size` 参数（默认 0.15 m）。
2. **拍摄图像**：将拍摄的双目图像（左右拼接）放入某个目录（如 `images/raw/`）。
3. **分割图像**：使用 `tools/split_stereo_images.sh` 自动分割左右图并放入标定目录。  
   ```bash
   ./tools/split_stereo_images.sh images/raw images/calibration 0
   ```
   默认标定图像目录为 `images/calibration`，可在配置文件中修改。
4. **编译项目**：在项目根目录运行 `./tools/build.sh`。
5. **执行标定**：运行标定程序  
   ```bash
   ./build/bin/stereo_calibrator
   ```
   标定完成后，检查生成的验证图像（位于 `calibration_results/`）是否符合预期。

---

## 注意事项

1. **配置修改后必须重新编译**  
   `config/global_config.yaml` 中的参数只在编译时应用。若修改了 GPU 相关设置，建议手动删除 `src/vulkan/spv` 和 `src/vulkan/generated` 文件夹，以强制重新生成着色器。

2. **输入图像格式要求**  
   双目图像必须为左右拼接格式：若总分辨率为 `A×B`，则单眼分辨率为 `(A/2)×B`。否则需要自行预处理。

3. **标定图像拍摄建议**  
   - 棋盘格保持水平
   - 多距离、多角度拍摄
   - 相机对准棋盘格中心

4. **标定程序运行位置**  
   目前 `bin/stereo_calibrator` 和 `bin/test_calibration` 必须在项目根目录运行，其他程序无此限制。
