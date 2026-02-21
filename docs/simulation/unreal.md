# Unreal Engine

- 官方网站：https://www.unrealengine.com/
- AirSim项目：https://github.com/microsoft/AirSim
- 物理引擎：PhysX (内置)
- 许可：免费使用（收入超过一定阈值后收取版税）

!!! note "引言"
    Unreal Engine是Epic Games开发的大型游戏引擎，凭借其出色的视觉渲染能力，近年来被越来越多地应用于机器人仿真领域。游戏引擎级别的渲染质量使得Unreal Engine在需要逼真视觉效果的仿真场景中具有独特优势，特别是自动驾驶 (Autonomous Driving)、无人机 (UAV) 仿真以及基于视觉的机器人学习等方向。

## Unreal Engine 在机器人仿真中的价值

传统机器人仿真器（如Gazebo、CoppeliaSim）虽然在物理仿真方面表现优秀，但其渲染质量与真实世界存在明显差距 (Sim-to-Real Gap)。Unreal Engine的高保真渲染 (High-Fidelity Rendering) 能够生成接近照片级别的图像，这对于以下应用至关重要：

- 训练基于视觉的深度学习模型，减少仿真到真实的迁移差距
- 生成合成数据集 (Synthetic Dataset) 用于目标检测和语义分割
- 测试自动驾驶系统的感知模块在复杂光照和天气条件下的表现

## AirSim

AirSim (Aerial Informatics and Robotics Simulation) 是微软 (Microsoft) 基于Unreal Engine开发的开源机器人仿真平台，是Unreal Engine在机器人领域最具代表性的应用。AirSim支持无人机和自动驾驶车辆的仿真，主要特性包括：

- **无人机仿真**：支持多旋翼无人机的飞行动力学仿真，内置PX4和ArduPilot飞控固件接口
- **车辆仿真**：提供汽车动力学模型，支持自动驾驶算法的测试
- **传感器仿真**：模拟RGB相机、深度相机、激光雷达 (LiDAR)、IMU、GPS、气压计 (Barometer) 等传感器
- **天气系统**：支持动态天气变化，包括雨、雾、雪、风等
- **Python API / C++ API**：提供编程接口用于控制仿真中的车辆和无人机

AirSim虽已被微软标记为存档项目 (Archived)，但其技术思路和架构对后续的Unreal Engine机器人仿真项目产生了深远影响。

## 高保真渲染能力

Unreal Engine的渲染管线 (Rendering Pipeline) 为机器人仿真提供了以下关键能力：

- **全局光照 (Global Illumination)**：Lumen系统提供实时动态全局光照，准确模拟光线在场景中的传播和反射
- **材质系统 (Material System)**：基于物理的渲染 (PBR, Physically Based Rendering) 材质，真实还原金属、玻璃、水面等表面特性
- **粒子效果 (Particle Effects)**：Niagara粒子系统可以模拟灰尘、烟雾、雨雪等环境效果
- **动态天空和光照**：支持昼夜变化 (Day-Night Cycle)、动态云层和大气散射效果

## 传感器仿真

在Unreal Engine环境中可以实现多种传感器的高保真仿真：

- **相机 (Camera)**：利用引擎的渲染能力生成极其逼真的RGB图像，支持运动模糊 (Motion Blur)、景深 (Depth of Field)、镜头畸变 (Lens Distortion) 等真实相机效果
- **深度传感器 (Depth Sensor)**：通过渲染管线获取精确的逐像素深度数据
- **激光雷达 (LiDAR)**：使用光线投射 (Ray Casting) 模拟激光雷达的扫描过程
- **语义分割 (Semantic Segmentation)**：引擎可以为场景中的每个物体生成语义标签图

## Blueprint 可视化脚本

Unreal Engine提供Blueprint可视化脚本系统 (Visual Scripting System)，允许用户通过图形化节点编程实现逻辑控制，无需编写C++代码。在机器人仿真中，Blueprint可以用于：

- 定义仿真场景中物体的交互逻辑
- 配置传感器参数和数据输出
- 创建动态事件（如随机出现的行人、变化的交通信号灯）
- 控制仿真流程和数据记录

对于需要更高性能或更复杂逻辑的场景，用户可以使用C++ API直接与引擎底层交互。

## 自动驾驶仿真应用

Unreal Engine在自动驾驶仿真领域应用广泛，除AirSim外还有多个相关项目：

- **CARLA**：基于Unreal Engine构建的开源自动驾驶仿真器，由巴塞罗那计算机视觉中心 (CVC) 开发，支持丰富的城市场景和交通模拟
- 提供交通流仿真 (Traffic Simulation)、行人模拟 (Pedestrian Simulation) 和多种天气条件
- 支持传感器套件的灵活配置，生成用于感知算法训练的多模态数据

## 优势与局限

**优势：**

- 渲染质量业界领先，适合视觉相关的仿真任务
- 丰富的资产商城 (Asset Marketplace) 提供现成的三维模型和场景
- 社区庞大，开发资源丰富
- 跨平台支持

**局限：**

- 学习曲线陡峭，需要掌握Unreal Engine开发技能
- 物理仿真精度不如专业机器人仿真器
- 对硬件要求较高，需要较强的GPU支持
- 与ROS的集成不如Gazebo、Webots等原生支持

## AirSim / Cosys-AirSim

AirSim（Aerial Informatics and Robotics Simulation）是微软研究院基于 Unreal Engine 开发的开源机器人仿真平台，专为无人机（UAV）和自动驾驶车辆设计。2023年微软停止官方维护后，社区版 **Cosys-AirSim** 继续活跃开发。

### 特点

- **物理真实感**：基于 Unreal Engine 的 Chaos Physics，提供精确的飞行动力学仿真
- **传感器仿真**：RGB/深度/语义分割相机、激光雷达、IMU、气压计、GPS
- **多环境支持**：城市、森林、冬季等多套场景包（Environments）
- **ROS 2 集成**：通过 AirSim ROS2 Wrapper 桥接

### Python API

```python
import airsim
import numpy as np

# 连接到 AirSim 仿真器
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# 起飞
client.takeoffAsync().join()

# 移动到指定位置（北-东-下坐标系，单位米）
client.moveToPositionAsync(10, 0, -5, velocity=5).join()

# 获取多模态传感器数据
responses = client.simGetImages([
    airsim.ImageRequest("front_center", airsim.ImageType.Scene),       # RGB
    airsim.ImageRequest("front_center", airsim.ImageType.DepthPlanar), # 深度图
    airsim.ImageRequest("front_center", airsim.ImageType.Segmentation),# 语义分割
])

# 获取激光雷达点云
lidar_data = client.getLidarData(lidar_name="LidarSensor1")
points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)

# 获取 IMU 数据
imu_data = client.getImuData()
print(f"角速度: {imu_data.angular_velocity}")
print(f"线加速度: {imu_data.linear_acceleration}")

# 降落并断开连接
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)
```

### 与 ROS 2 集成

```bash
# 安装 AirSim ROS 2 Wrapper
cd ~/ros2_ws/src
git clone https://github.com/microsoft/AirSim-ROS2-Wrapper.git
cd .. && colcon build

# 启动桥接节点
ros2 launch airsim_ros_pkgs airsim_node.launch.py
```

主要发布的话题：

| 话题 | 消息类型 | 内容 |
|------|---------|------|
| `/airsim_node/drone_1/imu/imu` | `sensor_msgs/Imu` | IMU 数据 |
| `/airsim_node/drone_1/front_center/Scene` | `sensor_msgs/Image` | RGB 图像 |
| `/airsim_node/drone_1/lidar/LidarSensor1` | `sensor_msgs/PointCloud2` | 点云 |
| `/airsim_node/drone_1/odom_local_ned` | `nav_msgs/Odometry` | 里程计 |

## rclUE：ROS 2 原生集成插件

rclUE 是专为 Unreal Engine 设计的 ROS 2 原生客户端库插件（C++），无需外部桥接即可在 UE 蓝图和 C++ 中直接使用 ROS 2 API：

```cpp
// UE C++ 中直接发布 ROS 2 话题
#include "ROS2Publisher.h"
#include "Msgs/ROS2TwistMsg.h"

// 在 Actor 中创建发布者
UROS2Publisher* Publisher;

void AMyRobot::BeginPlay()
{
    Publisher = NewObject<UROS2Publisher>(this);
    Publisher->SetMessageType(UROS2TwistMsg::StaticClass());
    Publisher->SetTopicName(TEXT("/cmd_vel"));
    Publisher->Init();
}

// 发布速度命令
void AMyRobot::PublishVelocity(float LinearX, float AngularZ)
{
    UROS2TwistMsg* Msg = NewObject<UROS2TwistMsg>();
    Msg->SetLinearX(LinearX);
    Msg->SetAngularZ(AngularZ);
    Publisher->Publish(Msg);
}
```

## 高保真传感器仿真

Unreal Engine 的光线追踪（Ray Tracing）能力使其在高保真传感器仿真方面具有独特优势：

### 激光雷达仿真

UE 可通过光线投射（Raycast）精确模拟固态激光雷达和机械旋转激光雷达的测量原理：

- **多回波（Multiple Returns）**：模拟激光束穿透玻璃或植被时的多次返回
- **强度仿真（Intensity Simulation）**：根据材质反射率计算回波强度
- **大气效应**：模拟雨、雾对激光的散射效应

### 相机仿真

- **镜头畸变**：精确的径向和切向畸变模型
- **运动模糊**：物理正确的快门速度和运动模糊效果
- **HDR 与曝光**：自动曝光和 HDR 渲染，接近真实相机响应曲线
- **噪声模型**：Gaussian 噪声、热噪声和泊松噪声仿真

## NVIDIA Omniverse 与 Isaac Sim 的对比

| 特性 | Unreal Engine + AirSim | NVIDIA Isaac Sim |
|------|------------------------|-----------------|
| 渲染质量 | 极高（Lumen + Nanite） | 高（RTX 光线追踪） |
| 物理引擎 | Chaos Physics | PhysX 5 |
| ROS 2 支持 | 通过 rclUE / AirSim | 原生支持 |
| GPU 并行仿真 | 有限 | 支持（Warp） |
| 主要应用 | 无人机、自动驾驶 | 工业机器人、操作任务 |
| 许可证 | 商业授权（含免费版） | 商业授权（含免费版） |
| 学习曲线 | 较陡（需 UE 知识） | 中等 |

## 参考资料

- [Unreal Engine官方文档](https://docs.unrealengine.com/)
- [AirSim GitHub仓库](https://github.com/microsoft/AirSim)
- [CARLA仿真器](https://carla.org/)
- Shah, S., Dey, D., Lovett, C., & Kapoor, A. (2018). AirSim: High-fidelity visual and physical simulation for autonomous vehicles. *Field and Service Robotics*.
- Dosovitskiy, A., Ros, G., Codevilla, F., Lopez, A., & Koltun, V. (2017). CARLA: An open urban driving simulator. *Conference on Robot Learning*.
- [Cosys-AirSim GitHub](https://github.com/Cosys-Lab/Cosys-AirSim)
- [rclUE GitHub](https://github.com/rapyuta-robotics/rclUE)
- Shah, S., et al. (2018). AirSim: High-Fidelity Visual and Physical Simulation for Autonomous Vehicles. *FSR*.
