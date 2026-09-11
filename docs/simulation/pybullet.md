# PyBullet
![5e430036ce538f09f700003a](assets/0e7165a8018643ceb13a601fcb43f2ba.png)

- 官方网站：https://pybullet.org/
- GitHub：https://github.com/bulletphysics/bullet3
- 物理引擎：Bullet
- 许可：zlib 开源许可证
- 开源仿真环境

!!! note "引言"
    PyBullet基于Bullet物理引擎，是一款面向机器人仿真和强化学习研究的开源仿真平台。PyBullet和Python紧密结合，提供简洁直观的API接口，目前在强化学习 (Reinforcement Learning) 领域中被广泛应用。该环境可以结合主流深度学习框架实现RL训练，支持DQN、PPO、TRPO、DDPG等算法。

## Bullet 物理引擎

Bullet Physics是一款久经考验的开源物理引擎，最初由Erwin Coumans开发。Bullet不仅在机器人仿真领域有广泛应用，还被好莱坞电影特效和AAA级游戏广泛采用。其核心能力包括：

- **刚体动力学 (Rigid Body Dynamics)**：高效的刚体碰撞检测和动力学求解
- **软体仿真 (Soft Body Simulation)**：支持可变形物体的仿真
- **约束求解器 (Constraint Solver)**：支持多种关节类型和运动约束
- **碰撞检测 (Collision Detection)**：分层碰撞检测架构，包括宽阶段 (Broad Phase) 和窄阶段 (Narrow Phase)

## Python API

PyBullet的Python API设计简洁，降低了使用门槛。以加载一个URDF模型并运行仿真为例：

```python
import pybullet as p
import pybullet_data

# 连接物理服务器
physics_client = p.connect(p.GUI)

# 设置搜索路径和重力
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# 加载地面和机器人
plane_id = p.loadURDF("plane.urdf")
robot_id = p.loadURDF("r2d2.urdf", [0, 0, 0.5])

# 运行仿真
for _ in range(10000):
    p.stepSimulation()

p.disconnect()
```

PyBullet提供两种连接模式：`p.GUI` 模式带有三维可视化窗口，适合调试和演示；`p.DIRECT` 模式不创建窗口，适合无头服务器上的批量训练。

## URDF 模型支持

PyBullet原生支持URDF (Unified Robot Description Format) 格式的机器人模型。URDF文件定义了机器人的连杆 (Link)、关节 (Joint)、碰撞体 (Collision Geometry) 和视觉外观 (Visual Geometry)。此外，PyBullet也支持加载SDF和MJCF格式的模型文件。

PyBullet自带了多种预置机器人模型，通过 `pybullet_data` 包提供：

- **KUKA iiwa**：七自由度工业机械臂
- **Franka Panda**：七自由度协作机械臂
- **Minitaur**：四足机器人
- **Humanoid**：人形机器人
- **R2D2**：演示用双足机器人

## 强化学习环境

PyBullet通过 `pybullet-gym` 和 `PyBullet Gymperium` 等项目提供了与OpenAI Gym（现为Gymnasium）兼容的强化学习环境。这些环境覆盖了多种经典控制任务：

- 四足机器人行走 (Locomotion)
- 机械臂抓取 (Grasping)
- 平衡控制 (Balance Control)
- 导航任务 (Navigation)

由于PyBullet完全免费且开源，它成为了MuJoCo在开源之前最受欢迎的替代方案。许多研究论文使用PyBullet环境来验证强化学习算法的有效性。

## 渲染能力

PyBullet提供两种渲染方式：

- **OpenGL渲染**：默认的实时渲染方式，在GUI模式下提供交互式三维视图
- **TinyRenderer**：内置的软件渲染器 (Software Renderer)，不依赖GPU，可在无显示设备的服务器环境中生成RGB图像和深度图

通过 `getCameraImage` 函数，用户可以获取仿真场景的RGB图像、深度图 (Depth Map) 和语义分割图 (Segmentation Mask)，这些数据可直接用于视觉强化学习 (Visual RL) 和计算机视觉任务的训练。

## 关键功能

除了基础的动力学仿真，PyBullet还提供以下关键功能：

- **逆运动学求解 (Inverse Kinematics)**：内置IK求解器，支持快速计算关节角度
- **逆动力学求解 (Inverse Dynamics)**：计算实现目标加速度所需的关节力矩
- **运动规划 (Motion Planning)**：可与OMPL等外部规划库集成
- **虚拟现实支持 (VR Support)**：支持通过VR设备进行遥操作 (Teleoperation)
- **多体仿真 (Multi-Body Simulation)**：支持同一场景中加载和仿真多个机器人


## 安装与配置

### 安装依赖

使用 pip 安装 PyBullet 及常用的强化学习配套库：

```python
pip install pybullet
pip install stable-baselines3
pip install gymnasium
```

如需安装特定版本或在 conda 环境中使用，建议先创建独立的虚拟环境：

```python
conda create -n pybullet_env python=3.10
conda activate pybullet_env
pip install pybullet stable-baselines3 gymnasium
```

### GUI 模式与 DIRECT 模式

PyBullet 在连接物理服务器时支持两种主要模式：

**GUI 模式**：启动带有 OpenGL 可视化窗口的仿真环境，适合调试机器人行为、观察仿真效果和制作演示视频。

```python
import pybullet as p
physics_client = p.connect(p.GUI)
```

**DIRECT 模式**：不创建任何图形窗口，仿真在后台运行，速度更快，适合在无头服务器（无显示设备）上进行大规模强化学习训练。

```python
physics_client = p.connect(p.DIRECT)
```

在训练阶段推荐使用 DIRECT 模式以获得最高的仿真吞吐量；在调试和最终演示阶段切换至 GUI 模式。


## 与 MuJoCo 对比

| 对比项目 | PyBullet | MuJoCo |
| --- | --- | --- |
| **开源协议** | zlib 开源，完全免费 | 2022 年开源（Apache 2.0） |
| **仿真性能** | 中等，适合大多数任务 | 较高，尤其是关节动力学 |
| **接触建模精度** | 基于罚函数法，较为粗糙 | 基于凸优化的精确接触建模 |
| **强化学习生态** | 丰富，有大量开源环境 | 丰富，MuJoCo Menagerie 持续扩展 |
| **安装难度** | 简单，`pip install pybullet` | 简单（2022 年后），`pip install mujoco` |
| **URDF 支持** | 原生支持 | 通过工具链转换 |
| **Python API** | 功能完整，文档较分散 | 功能完整，官方文档详细 |
| **社区活跃度** | 维护趋于稳定，更新较少 | 活跃，DeepMind 持续维护 |
| **适用场景** | 快速原型、强化学习入门 | 高精度动力学研究、生产级训练 |

**选择建议**：对于入门学习和快速验证，PyBullet 上手简单，资料丰富，是良好的起点。对于追求动力学精度的机器人研究，尤其是接触密集型任务（抓取、灵巧手操控），MuJoCo 的接触建模优势更为明显。目前两者均已免费开源，可根据具体需求灵活选择。


## 实践技巧

### 仿真频率与时间步长

PyBullet 的默认仿真频率为 240 Hz，对应时间步长 1/240 秒。可通过 `setTimeStep` 修改：

```python
p.setTimeStep(1.0 / 240.0)   # 默认值，适合大多数任务
p.setTimeStep(1.0 / 1000.0)  # 更高频率，适合接触密集型任务（但更慢）
p.setTimeStep(1.0 / 60.0)    # 更低频率，仿真更快但精度降低
```

实时仿真模式下，仿真时钟与真实时钟同步：

```python
p.setRealTimeSimulation(1)   # 开启实时仿真（GUI 调试用）
p.setRealTimeSimulation(0)   # 关闭实时仿真（训练时使用，速度最快）
```

强化学习训练时，始终关闭实时仿真，由代码手动调用 `stepSimulation`，以获得最高的仿真吞吐量。

### 调试技巧

**可视化辅助线**：在 GUI 模式下可绘制辅助线段、文字，方便调试：

```python
# 绘制坐标轴（红色 X，绿色 Y，蓝色 Z）
origin = [0, 0, 0]
p.addUserDebugLine(origin, [0.3, 0, 0], [1, 0, 0], 2)
p.addUserDebugLine(origin, [0, 0.3, 0], [0, 1, 0], 2)
p.addUserDebugLine(origin, [0, 0, 0.3], [0, 0, 1], 2)

# 在三维空间中显示文字
p.addUserDebugText(
    text="Target",
    textPosition=target_pos,
    textColorRGB=[1, 0, 0],
    textSize=1.5
)
```

**使用 GUI 滑块进行交互调试**：

```python
# 创建滑块控件
slider_id = p.addUserDebugParameter("joint_0", -3.14, 3.14, 0.0)

while True:
    value = p.readUserDebugParameter(slider_id)
    p.setJointMotorControl2(robot_id, 0, p.POSITION_CONTROL,
                            targetPosition=value, force=500)
    p.stepSimulation()
```

**性能优化**：

- 训练时使用 `p.DIRECT` 模式，避免渲染开销
- 使用 `SubprocVecEnv` 并行多个仿真实例
- 适当降低仿真频率（如使用 1/60 s 时间步）以提高采样效率
- 避免在每个步骤中调用 `getContactPoints` 等查询函数，除非任务确实需要


## 本章内容导览

PyBullet 章节按「引擎与安装 → 控制与动力学 → 强化学习 → 感知与多机 → 速查」的顺序组织：

| 页面 | 主要内容 |
|------|---------|
| [PyBullet](pybullet.md) | Bullet 引擎、Python API 概览、URDF 支持、安装配置、与 MuJoCo 对比 |
| [控制与动力学](pybullet-control-dynamics.md) | 关节控制模式、PD 控制、逆运动学与逆动力学、接触力与约束 |
| [强化学习工作流](pybullet-rl-workflows.md) | Gymnasium 接口、自定义环境、奖励设计、并行训练、域随机化 |
| [视觉与多机器人仿真](pybullet-vision-multirobot.md) | 相机采集、深度与分割图、多机器人场景、性能优化 |
| [API 与资源速查](pybullet-full-reference.md) | 常用 API 速查、模型库、学习资源 |
| [MuJoCo](mujoco.md) | 接触建模更精确的替代仿真器 |
| [Gazebo](gazebo.md) | 与 ROS 深度集成的仿真器 |


## 参考资料

- [PyBullet快速入门指南](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/)
- [Bullet Physics GitHub仓库](https://github.com/bulletphysics/bullet3)
- [pybullet_data 预置模型](https://github.com/bulletphysics/bullet3/tree/master/data)
- Coumans, E., & Bai, Y. (2016). PyBullet, a Python module for physics simulation for games, robotics and machine learning.
- [Stable-Baselines3 官方文档](https://stable-baselines3.readthedocs.io/)
- [Gymnasium 官方文档](https://gymnasium.farama.org/)
