# Nvidia Jetson

!!! note "引言"
    Nvidia Jetson是英伟达（Nvidia）推出的面向边缘AI计算的嵌入式硬件平台。与传统的嵌入式处理器不同，Jetson系列集成了Nvidia GPU，能够在低功耗条件下运行深度学习推理、计算机视觉和传感器融合等计算密集型任务。Jetson平台在自主机器人、无人机、自动驾驶和智能摄像头等领域得到广泛应用。

Jetson是英伟达推出的面向嵌入式计算的硬件平台。

## 1. Jetson型号

Jetson的主要型号系列包括：

**Xavier 系列（2018～2021年）**

- Jetson Nano
- Jetson TX1
- Jetson TX2
- Jetson Xavier（AGX Xavier）
- Jetson Xavier NX

**Orin 系列（2022年起）**

- Jetson Orin Nano
- Jetson Orin NX
- Jetson AGX Orin


## 2. Jetson各型号对比

### Xavier 系列对比

Jetson Xavier 系列各型号的对比如下：

| 硬件特性 | Jetson Nano | Jetson TX1 | Jetson TX2/TX2i | Jetson Xavier | Jetson Xavier NX |
|----------|-------------|-----------|----------------|--------------|----------------|
| CPU | 4核 ARM A57 @ 1.43 GHz | 4核 ARM Cortex-A57 @ 1.73 GHz | 4核 ARM Cortex-A57 @ 2 GHz + 2核 Denver2 @ 2 GHz | 8核 ARM Carmel v8.2 @ 2.26 GHz | 6核 NVIDIA Carmel ARM v8.2 |
| GPU | 128核 Maxwell @ 921 MHz | 256核 Maxwell @ 998 MHz | 256核 Pascal @ 1.3 GHz | 512核 Volta @ 1.37 GHz | 384核 NVIDIA Volta |
| 内存 | 4 GB LPDDR4，25.6 GB/s | 4 GB LPDDR4，25.6 GB/s | 8 GB 128位 LPDDR4，58.3 GB/s | 16 GB 256位 LPDDR4，137 GB/s | 8 GB 128位 LPDDR4x，51.2 GB/s |
| 存储 | MicroSD | 16 GB eMMC 5.1 | 32 GB eMMC 5.1 | 32 GB eMMC 5.1 | 16 GB eMMC 5.1 |
| Tensor 核心 | — | — | — | 64 | 48 |
| AI 算力 | 0.5 TOPS | — | 1.3 TOPS | 32 TOPS | 21 TOPS |
| 功耗 | 5W / 10W | 10W | 7.5W / 15W | 10W / 15W / 30W | 10W / 15W |
| USB | 4× USB 3.0 + Micro-USB 2.0 | 1× USB 3.0 + 1× USB 2.0 | 1× USB 3.0 + 1× USB 2.0 | 3× USB 3.1 + 4× USB 2.0 | — |
| PCIe | 4通道 Gen 2 | 5通道 Gen 2 | 5通道 Gen 2 | 16通道 Gen 4 | 1×1 + 1×4 Gen 3 |


### Jetson Orin 系列（2022年新品）

Orin 系列采用 NVIDIA Ampere 架构 GPU 和 ARM Cortex-A78AE CPU，相比 Xavier 系列性能大幅提升，尤其在深度学习推理算力方面有显著进步：

| 型号 | AI 算力 | CPU | GPU | 内存 | 功耗 | 定位 |
|------|---------|-----|-----|------|------|------|
| Jetson Orin Nano 4GB | 20 TOPS | 6核 ARM Cortex-A78AE | 512核 Ampere | 4 GB LPDDR5 | 7W～10W | 入门级，替代 Jetson Nano |
| Jetson Orin Nano 8GB | 20 TOPS | 6核 ARM Cortex-A78AE | 512核 Ampere | 8 GB LPDDR5 | 7W～15W | 入门级 |
| Jetson Orin NX 8GB | 70 TOPS | 6核 ARM Cortex-A78AE | 1024核 Ampere | 8 GB LPDDR5 | 10W～20W | 中端，替代 Xavier NX |
| Jetson Orin NX 16GB | 100 TOPS | 8核 ARM Cortex-A78AE | 1024核 Ampere | 16 GB LPDDR5 | 10W～25W | 中端 |
| Jetson AGX Orin 32GB | 200 TOPS | 12核 ARM Cortex-A78AE | 2048核 Ampere | 32 GB LPDDR5 | 15W～40W | 旗舰，替代 AGX Xavier |
| Jetson AGX Orin 64GB | 275 TOPS | 12核 ARM Cortex-A78AE | 2048核 Ampere | 64 GB LPDDR5 | 15W～60W | 旗舰最高配 |

相比 Xavier 系列，Orin 的主要改进有：

- **Ampere GPU**：支持第三代 Tensor Core，INT8 和 FP16 推理速度大幅提升
- **更大内存与更高带宽**：LPDDR5 内存带宽比 Xavier 提高 50%～100%
- **新增 DLA（Deep Learning Accelerator）**：两个专用推理加速器，可在不占用 GPU 的情况下运行网络推理
- **视频编解码能力大幅增强**：支持 AV1 硬件解码，可并行处理更多路视频流


## 3. JetPack SDK

JetPack是Nvidia为Jetson平台提供的官方软件开发套件（Software Development Kit），包含了开发Jetson应用所需的全部软件组件：

- **L4T（Linux for Tegra）**：基于Ubuntu的操作系统，针对Jetson硬件优化。
- **CUDA**：Nvidia的并行计算平台和编程模型，允许开发者利用GPU进行通用计算。
- **cuDNN**：CUDA深度神经网络库，提供了高度优化的深度学习基础运算（卷积、池化、归一化等）。
- **TensorRT**：高性能深度学习推理优化器和运行时库。
- **VisionWorks / VPI**：计算机视觉基础库。
- **Multimedia API**：视频编解码和图像处理接口。
- **开发工具**：包括CUDA编译器（nvcc）、调试器（cuda-gdb）和性能分析器（Nsight Systems）。

### JetPack 版本说明

| JetPack 版本 | 支持平台 | Ubuntu 版本 | CUDA 版本 | 状态 |
|-------------|---------|------------|----------|------|
| JetPack 4.x | Nano, TX1/TX2, Xavier | Ubuntu 18.04 | CUDA 10.2 | 维护中 |
| JetPack 5.x | Xavier, Orin | Ubuntu 20.04 | CUDA 11.4 | 当前主流 |
| JetPack 6.x | Orin | Ubuntu 22.04 | CUDA 12.x | 最新版本 |

### 刷机方法

- **SDK Manager（推荐）**：在 x86 Ubuntu 主机上安装 NVIDIA SDK Manager，通过 USB 线将 Jetson 进入恢复模式后一键刷机，同时安装所有 SDK 组件
- **Balena Etcher**：适用于带载板的开发套件（如 Jetson Nano 开发板），将 JetPack 镜像烧录到 MicroSD 卡或 NVMe SSD


### 关键组件详解

**TensorRT（优化推理引擎）**：将训练好的神经网络模型转换为高度优化的推理引擎。主要优化手段包括层融合（Layer Fusion）、权重量化（INT8/FP16）和内核自动调优（Kernel Auto-Tuning），在 Jetson 上可获得比原生 PyTorch 高出 2～10 倍的推理速度。

**VPI（Vision Programming Interface）**：NVIDIA 的硬件加速计算机视觉库，为图像滤波、特征检测、光流等算法提供统一接口，底层可调度 GPU、NVENC、VIC 等多种加速硬件。

**DeepStream（视频分析 SDK）**：基于 GStreamer 构建的端到端视频分析框架，支持多路视频流并行处理与 TensorRT 推理加速，适用于智能摄像头和多摄像头机器人感知系统。


## 4. CUDA与TensorRT

### CUDA

CUDA（Compute Unified Device Architecture）是Nvidia的并行计算平台。在Jetson上，CUDA允许开发者编写在GPU上运行的核函数（Kernel），实现大规模并行计算。对于机器人应用中的图像处理、点云处理和矩阵运算等任务，CUDA可以提供数倍到数十倍的性能提升。

### TensorRT

TensorRT是Nvidia的深度学习推理优化引擎。它的核心功能包括：

- **模型优化**：对训练好的深度学习模型进行层融合（Layer Fusion）、精度校准（Calibration）和内核自动调优（Kernel Auto-Tuning），显著提高推理速度。
- **精度模式**：支持FP32、FP16和INT8三种推理精度。在Jetson Xavier和Xavier NX上使用INT8模式可以获得最高的推理性能。
- **动态形状支持**：支持运行时动态调整输入张量的形状。
- **多框架支持**：可以导入TensorFlow、PyTorch（通过ONNX）和Caffe等框架训练的模型。

在机器人视觉应用中，TensorRT常用于加速目标检测（如YOLO、SSD）、语义分割和姿态估计等深度学习模型的推理。


## 5. DeepStream SDK

DeepStream是Nvidia提供的流媒体分析SDK，专为构建智能视频分析（Intelligent Video Analytics，IVA）应用而设计。其核心特点包括：

- **端到端的视频处理流水线**：从视频解码、预处理、深度学习推理到后处理和输出，全部通过GStreamer插件实现。
- **多流并行处理**：可以同时处理多路视频流，充分利用Jetson的GPU算力。
- **与TensorRT集成**：推理环节自动调用TensorRT，获得最优推理性能。
- **消息代理（Message Broker）**：支持将分析结果通过Kafka、MQTT等协议发送到云端或其他系统。

DeepStream在机器人领域可用于：多摄像头环境感知、实时目标追踪、异常行为检测等场景。


## 6. 环境搭建与配置

### 首次开机配置

Jetson 开发板首次启动后，按屏幕提示完成 Ubuntu 系统初始化（用户名、密码、时区、网络等）。

### 最大性能模式

```bash
# 解锁所有时钟到最高频率（等效于最大功耗模式）
sudo jetson_clocks

# 查询当前功耗模式
sudo nvpmodel -q

# 切换到最大性能模式（模式编号因型号而异，MAXN 通常为 0）
sudo nvpmodel -m 0
```

### 系统状态监控

`tegrastats` 是 Jetson 专用的系统监控工具，可实时查看 CPU/GPU 使用率、内存占用和温度：

```bash
# 每500毫秒刷新一次系统状态
tegrastats --interval 500
```

输出示例：

```
RAM 3456/7772MB (lfb 512x4MB) SWAP 0/3886MB CPU [45%@1420,38%@1420,...]
EMC_FREQ 5%@1600 GR3D_FREQ 72%@1109 AO@42.5C GPU@47C Tboard@39C
```

### 配置交换空间

Jetson 的内存容量有限，配置合理的交换空间（Swap）可以在内存紧张时避免进程崩溃：

```bash
# 创建 8GB 的交换文件
sudo fallocate -l 8G /var/swapfile

# 设置文件权限（交换文件不应对其他用户可读）
sudo chmod 600 /var/swapfile

# 格式化为交换区域
sudo mkswap /var/swapfile

# 立即启用交换
sudo swapon /var/swapfile

# 开机自动挂载（追加到 /etc/fstab）
echo '/var/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### 安装 PyTorch

NVIDIA 为 Jetson 提供了针对 ARM 架构和 CUDA 优化的 PyTorch 预编译包，不能直接通过 `pip install torch` 安装：

```bash
# 从 NVIDIA 官方下载 PyTorch wheel（根据 JetPack 版本选择对应版本）
# JetPack 5.x 对应 PyTorch 2.x
wget https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl

pip3 install torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
```


## 7. 深度学习推理部署

### PyTorch → ONNX → TensorRT 工作流

在 Jetson 上部署深度学习模型的标准流程是：在 PC 上训练模型，导出为 ONNX 格式，再在 Jetson 上使用 TensorRT 编译为优化后的推理引擎：

**第一步：导出 ONNX 模型（在训练 PC 上执行）**

```python
import torch
import torch.onnx

# 加载已训练好的模型
model = MyModel()
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

# 准备示例输入（用于指定输入形状）
dummy_input = torch.randn(1, 3, 640, 640)

# 导出为 ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}}  # 支持动态批大小
)
print("ONNX 导出成功")
```

**第二步：使用 trtexec 转换为 TensorRT 引擎（在 Jetson 上执行）**

```bash
# FP16 精度（速度与精度平衡，推荐）
trtexec --onnx=model.onnx \
        --saveEngine=model_fp16.engine \
        --fp16 \
        --workspace=2048

# INT8 精度（最高速度，需要校准数据集）
trtexec --onnx=model.onnx \
        --saveEngine=model_int8.engine \
        --int8 \
        --calib=calibration_cache.bin
```

**第三步：使用 TensorRT Python API 进行推理**

```python
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

# 加载序列化的 TensorRT 引擎
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
with open("model_fp16.engine", "rb") as f:
    serialized_engine = f.read()

runtime = trt.Runtime(TRT_LOGGER)
engine = runtime.deserialize_cuda_engine(serialized_engine)
context = engine.create_execution_context()

# 分配 GPU 内存
input_shape = (1, 3, 640, 640)
output_shape = (1, 1000)

d_input = cuda.mem_alloc(np.prod(input_shape) * np.dtype(np.float32).itemsize)
d_output = cuda.mem_alloc(np.prod(output_shape) * np.dtype(np.float32).itemsize)
bindings = [int(d_input), int(d_output)]

# 推理
input_data = np.random.randn(*input_shape).astype(np.float32)
cuda.memcpy_htod(d_input, input_data)
context.execute_v2(bindings)
output = np.empty(output_shape, dtype=np.float32)
cuda.memcpy_dtoh(output, d_output)
```

### 推理性能对比参考

以 YOLOv8n 目标检测模型（输入 640×640）为例：

| 平台 | 精度模式 | FPS（参考） |
|------|---------|-----------|
| Jetson Orin NX 16GB | TensorRT FP16 | ~120 fps |
| Jetson Xavier NX | TensorRT FP16 | ~60 fps |
| Jetson Nano | TensorRT FP16 | ~25 fps |
| Jetson Nano | PyTorch FP32 | ~5 fps |

> 注：实际性能因模型结构、批大小和功耗模式而有差异。

### NVIDIA TAO Toolkit

NVIDIA TAO（Train, Adapt, Optimize）Toolkit 提供了基于迁移学习的模型训练工具链，允许用户在小规模数据集上微调 NVIDIA 预训练模型（目标检测、分类、姿态估计等），并直接导出为 TensorRT 格式部署到 Jetson。


## 8. 机器人应用开发

### ROS 2 on Jetson

NVIDIA 官方支持在 Jetson 上运行 ROS 2（Humble、Iron、Jazzy），并提供了专为 Jetson 优化的 ROS 2 节点和示例。

```bash
# 安装 ROS 2 Humble（Ubuntu 20.04 + JetPack 5.x）
sudo apt update && sudo apt install -y ros-humble-desktop
source /opt/ros/humble/setup.bash

# 验证安装
ros2 run demo_nodes_cpp talker
```

### CSI 相机与 USB 相机配置

Jetson 开发板通常配备 CSI（Camera Serial Interface）接口，可直接连接树莓派摄像头模块或 NVIDIA 官方摄像头：

```bash
# 列出可用的视频设备
ls /dev/video*

# 使用 GStreamer 管道从 CSI 相机采集图像
gst-launch-1.0 nvarguscamerasrc ! \
    'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1' ! \
    nvvidconv flip-method=0 ! \
    'video/x-raw,width=960,height=540' ! \
    nvvidconv ! nvegltransform ! nveglglessink

# 在 Python 中通过 GStreamer 管道读取 CSI 相机
import cv2

gst_str = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
    "nvvidconv flip-method=0 ! "
    "video/x-raw, format=BGRx ! videoconvert ! "
    "video/x-raw, format=BGR ! appsink"
)
cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
```

### ZED 2 深度相机集成

Stereolabs ZED 2 相机在 Jetson 上有完整的官方支持，包括本地 SDK 和 ROS 2 驱动：

```bash
# 安装 ZED SDK（从官网下载对应 JetPack 版本的安装包）
chmod +x ZED_SDK_Tegra_L4T35.4_v4.1.run
./ZED_SDK_Tegra_L4T35.4_v4.1.run

# 安装 ROS 2 驱动
sudo apt install ros-humble-zed-ros2-wrapper
ros2 launch zed_wrapper zed2.launch.py
```

### 典型机器人计算栈

Jetson 在自主移动机器人中通常担任中央计算单元，其典型软件栈如下：

```
传感器层：深度相机 + 激光雷达 + IMU
    ↓
驱动层：ROS 2 驱动节点（realsense2_camera, sllidar_ros2 等）
    ↓
感知层：目标检测（TensorRT）、语义分割、深度估计
    ↓
定位层：视觉SLAM（ORB-SLAM3）、激光SLAM（Cartographer）
    ↓
导航层：ROS 2 Nav2（路径规划、避障、行为树）
    ↓
执行层：底盘控制（串口/CAN 通信）、机械臂控制（MoveIt2）
```


## 9. Jetson Nano 入门项目

### 使用 jetson-inference 库运行 YOLOv8

NVIDIA 提供了 `jetson-inference` 开源库，内置了多种预训练模型，可快速在 Jetson Nano 上进行目标检测：

```bash
# 克隆并构建 jetson-inference
git clone --recursive https://github.com/dusty-nv/jetson-inference
cd jetson-inference && mkdir build && cd build
cmake ../
make -j$(nproc)
sudo make install

# 运行实时目标检测（使用 USB 摄像头）
detectnet /dev/video0
```

### 自定义模型推理示例

```python
import jetson.inference
import jetson.utils

# 加载目标检测网络（可选 ssd-mobilenet-v2, detectnet-v2 等）
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

# 打开摄像头
camera = jetson.utils.videoSource("/dev/video0")
display = jetson.utils.videoOutput("display://0")

while display.IsStreaming():
    img = camera.Capture()
    detections = net.Detect(img)
    for det in detections:
        print(f"检测到: {net.GetClassDesc(det.ClassID)} "
              f"置信度: {det.Confidence:.2f} "
              f"位置: ({det.Left:.0f}, {det.Top:.0f})")
    display.Render(img)
    display.SetStatus(f"目标检测 | FPS: {net.GetNetworkFPS():.1f}")
```

### ROS 2 导航演示

在 Jetson Nano 上运行 TurtleBot3 仿真（需配合外部主机运行 Gazebo）：

```bash
# 启动 Nav2 导航栈
ros2 launch nav2_bringup navigation_launch.py \
    use_sim_time:=True \
    map:=/path/to/map.yaml
```


## 10. 功耗管理

### Jetson Orin NX 功耗模式

| 模式名称 | 功耗上限 | CPU 核心/频率 | GPU 频率 | DLA 频率 | 适用场景 |
|---------|---------|--------------|---------|---------|---------|
| MAXN | 25W | 8核 @ 2.0 GHz | 918 MHz | 1.6 GHz | 最大性能 |
| 10W | 10W | 4核 @ 1.5 GHz | 612 MHz | 1.1 GHz | 低功耗优先 |
| 15W | 15W | 6核 @ 1.8 GHz | 765 MHz | 1.4 GHz | 平衡模式 |
| 20W | 20W | 8核 @ 2.0 GHz | 765 MHz | 1.4 GHz | 性能优先 |

```bash
# 列出所有可用功耗模式
sudo nvpmodel --listmodes

# 查看当前模式
sudo nvpmodel -q --verbose

# 切换到 15W 平衡模式（模式编号因型号而异，请参考 --listmodes 输出）
sudo nvpmodel -m 2
```

### 电池供电机器人的功耗考虑

对于电池供电的移动机器人，功耗管理至关重要：

- 根据任务类型动态切换功耗模式：导航时使用 10W 模式，执行复杂感知任务时切换到 MAXN 模式
- 利用 DLA（深度学习加速器）运行常驻推理网络，释放 GPU 处理其他任务
- 关闭不必要的外设（如 HDMI 输出、USB 集线器）以节省功耗

### 散热管理

| 散热方案 | 适用场景 | 持续功耗上限 |
|---------|---------|------------|
| 被动散热（铝制散热片） | 轻载场景、低功耗模式 | 10W |
| 主动散热（5V 风扇） | 一般机器人应用 | 20W |
| 液冷 / 大型风冷 | 长时间满载推理 | 60W（AGX Orin） |

```bash
# 通过 tegrastats 实时监控温度
watch -n 1 "tegrastats | grep -o 'GPU@[0-9.]*C\|CPU@[0-9.]*C\|Tboard@[0-9.]*C'"
```


## 11. 与树莓派对比

| 特性 | Raspberry Pi 4 (4GB) | Raspberry Pi 5 (8GB) | Jetson Orin Nano 8GB | Jetson Orin NX 16GB |
|------|---------------------|---------------------|---------------------|---------------------|
| CPU | 4核 ARM Cortex-A72 @ 1.8 GHz | 4核 ARM Cortex-A76 @ 2.4 GHz | 6核 ARM Cortex-A78AE @ 1.5 GHz | 8核 ARM Cortex-A78AE @ 2.0 GHz |
| GPU | VideoCore VI（无 CUDA） | VideoCore VII（无 CUDA） | 512核 NVIDIA Ampere | 1024核 NVIDIA Ampere |
| AI 加速 | 无 | 无 | 20 TOPS | 70～100 TOPS |
| 内存 | 4 GB LPDDR4X | 8 GB LPDDR4X | 8 GB LPDDR5 | 16 GB LPDDR5 |
| 功耗 | 5W～15W | 5W～20W | 7W～15W | 10W～25W |
| 参考价格 | ¥300～500 | ¥400～700 | ¥800～1200 | ¥1500～2500 |
| CUDA 支持 | 否 | 否 | 是 | 是 |
| TensorRT | 否 | 否 | 是 | 是 |
| ROS 2 支持 | 社区支持 | 社区支持 | NVIDIA 官方支持 | NVIDIA 官方支持 |
| 深度学习推理 | 仅 CPU，较慢 | 仅 CPU，较慢 | GPU + DLA 加速 | GPU + DLA 加速 |
| 适用场景 | 教学、轻量控制任务 | 教学、中等计算任务 | 边缘 AI 机器人 | 复杂感知与自主导航 |

**选型建议**：
- 若仅需运行 ROS 2、简单控制算法和低速图像处理，树莓派 4/5 性价比更高
- 若需要运行深度学习模型（目标检测、语义分割、深度估计等），Jetson Orin Nano 起步为最低门槛选择
- 复杂自主导航（同时运行 SLAM、目标检测和路径规划）推荐 Jetson Orin NX 或 AGX Orin


## 12. 快速入门

以下是在Jetson平台上开始机器人开发的基本步骤：

1. **烧录系统镜像**：从Nvidia开发者网站下载JetPack SDK，使用SDK Manager将系统镜像烧录到Jetson模块。
2. **初始设置**：首次启动后完成Ubuntu系统的基本配置（用户名、密码、网络等）。
3. **安装开发工具**：JetPack已预装CUDA、cuDNN和TensorRT，可额外安装PyTorch、TensorFlow等深度学习框架。
4. **安装ROS**：根据需要安装ROS Noetic或ROS 2 Humble。
5. **连接传感器**：通过USB、CSI或I2C接口连接摄像头、激光雷达和IMU等传感器。
6. **运行示例**：JetPack附带了大量示例代码，包括CUDA示例、TensorRT推理示例和视觉处理示例。

---

## 参考资料

1. Benchmark comparison for Jetson Nano, TX1, TX2 and AGX Xavier, https://www.fastcompression.com/blog/jetson-benchmark-comparison.htm
2. Jetson Xavier NX, https://developer.nvidia.com/embedded/jetson-xavier-nx
3. [Nvidia JetPack SDK文档](https://developer.nvidia.com/embedded/jetpack)
4. [Nvidia Isaac SDK](https://developer.nvidia.com/isaac-sdk)
5. [Jetson开发者论坛](https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems/)
6. [Jetson Orin 系列产品页面](https://developer.nvidia.com/embedded/jetson-orin)
7. [jetson-inference 开源库](https://github.com/dusty-nv/jetson-inference)
8. [NVIDIA TAO Toolkit 文档](https://docs.nvidia.com/tao/tao-toolkit/)
