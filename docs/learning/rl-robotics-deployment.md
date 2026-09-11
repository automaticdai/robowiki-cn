# 强化学习机器人部署

!!! note "引言"
    将强化学习策略从仿真环境部署到真实机器人（Sim-to-Real Transfer）是 RL 在机器人领域落地的关键挑战。仿真与现实之间的差距（Reality Gap）可能导致在仿真中表现良好的策略在真实世界中完全失效。本文介绍域随机化、系统辨识、安全约束等技术，以及 Jetson 部署和 ROS 2 集成的实践方法。


## Sim-to-Real 核心挑战

仿真与现实的差距主要体现在以下方面：

| 差异来源 | 具体表现 | 影响程度 |
|---------|---------|---------|
| 动力学模型 | 摩擦力、接触力、柔性体 | 高 |
| 传感器噪声 | 测量偏差、延迟、量化误差 | 中 |
| 执行器特性 | 响应延迟、死区、非线性 | 高 |
| 视觉差异 | 光照、纹理、背景 | 中（视觉策略） |
| 时间延迟 | 通信延迟、计算延迟 | 中 |


## 域随机化

域随机化（Domain Randomization, DR）通过在仿真训练时随机化环境参数，迫使策略学习对参数变化具有鲁棒性的行为。

### 物理参数随机化

```python
import numpy as np

class DomainRandomizer:
    """域随机化器：在每个 episode 开始时随机化物理参数"""

    def __init__(self):
        self.param_ranges = {
            'friction': (0.3, 1.5),       # 摩擦系数
            'mass_scale': (0.8, 1.2),     # 质量缩放因子
            'damping': (0.1, 0.5),        # 关节阻尼
            'actuator_delay': (0, 3),     # 执行器延迟（帧数）
            'obs_noise_std': (0.0, 0.05), # 观测噪声标准差
        }

    def sample(self):
        params = {}
        for key, (low, high) in self.param_ranges.items():
            params[key] = np.random.uniform(low, high)
        return params

    def apply(self, env, params):
        env.model.geom_friction[:] *= params['friction']
        env.model.body_mass[:] *= params['mass_scale']
        env.model.dof_damping[:] = params['damping']
```

### 视觉随机化

对于基于视觉的策略，需要随机化：

- 光照方向、强度和颜色
- 物体纹理和背景
- 相机位置和内参
- 图像噪声和模糊


### 自适应域随机化（ADR）

OpenAI 在 Rubik's Cube 项目中提出了自动域随机化（Automatic Domain Randomization）：

1. 初始使用较窄的参数范围
2. 监控策略在当前范围内的成功率
3. 成功率超过阈值时自动扩大范围
4. 逐步增加随机化难度直到策略足够鲁棒


## 系统辨识

系统辨识（System Identification）通过测量真实系统的行为来精确标定仿真参数，缩小 Sim-to-Real Gap。

### 常用方法

1. **参数辨识**：通过最小二乘法等方法拟合动力学参数

$$
\theta^* = \arg\min_\theta \sum_{t=1}^{T} \|x_{real}(t) - x_{sim}(t; \theta)\|^2
$$

2. **贝叶斯优化**：在参数空间中搜索使仿真行为最接近真实行为的参数

3. **神经网络残差模型**：学习一个残差网络来补偿仿真与真实之间的差异

$$
\hat{x}_{t+1} = f_{sim}(x_t, a_t; \theta) + f_{res}(x_t, a_t; \phi)
$$


## 安全约束

### 动作空间约束

在真实部署时限制动作空间，防止危险行为：

安全动作包装器的核心逻辑是裁剪动作到力矩安全范围，并在关节接近极限位置时限制运动方向（接近下限时禁止负力矩，接近上限时禁止正力矩）。

部署时还应添加独立的安全监控层，检测关节位置/速度/力矩超限、碰撞力阈值、外力异常（人机协作场景），并设置紧急停止触发机制。


## 策略蒸馏

策略蒸馏（Policy Distillation）将大型教师策略压缩为轻量级学生策略，便于在边缘设备上部署：

$$
L_{distill} = \mathbb{E}_s \left[D_{KL}\left(\pi_{teacher}(\cdot|s) \| \pi_{student}(\cdot|s)\right)\right]
$$

### 蒸馏策略

| 方法 | 描述 | 优点 |
|------|------|------|
| 行为克隆蒸馏 | 学生模仿教师的动作输出 | 简单高效 |
| 特征蒸馏 | 学生学习教师的中间特征 | 迁移能力更强 |
| 多教师蒸馏 | 从多个专家策略蒸馏到一个通用策略 | 覆盖多种场景 |


## Jetson 部署

NVIDIA Jetson 系列是机器人边缘计算的主流平台。

### 模型优化与部署

```python
import torch
from stable_baselines3 import PPO

# 1. 加载训练好的模型
model = PPO.load("ppo_robot")

# 2. 导出为 ONNX 格式
dummy_obs = torch.randn(1, model.observation_space.shape[0])
torch.onnx.export(
    model.policy,
    dummy_obs,
    "policy.onnx",
    input_names=["observation"],
    output_names=["action"],
    opset_version=11
)

# 3. 使用 TensorRT 优化（在 Jetson 上执行）
# trtexec --onnx=policy.onnx --saveEngine=policy.engine --fp16
```

### Jetson 平台对比

| 平台 | GPU 算力 | 内存 | 功耗 | 适用场景 |
|------|---------|------|------|---------|
| Jetson Nano | 472 GFLOPS | 4 GB | 5-10 W | 简单策略推理 |
| Jetson Xavier NX | 21 TOPS | 8 GB | 10-20 W | 中等复杂度策略 |
| Jetson Orin | 275 TOPS | 32 GB | 15-60 W | 复杂视觉策略 |

### 推理优化技巧

- 使用 FP16 或 INT8 量化减少计算量
- 使用 TensorRT 优化推理引擎
- 控制频率匹配（策略推理频率与控制频率对齐）
- 异步推理：传感器数据采集与策略推理并行


## ROS 2 集成

### RL 策略节点

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np
import onnxruntime as ort

class RLPolicyNode(Node):
    def __init__(self):
        super().__init__('rl_policy_node')

        # 加载 ONNX 模型
        self.session = ort.InferenceSession("policy.onnx")
        self.input_name = self.session.get_inputs()[0].name

        # 订阅关节状态
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10
        )

        # 发布控制指令
        self.cmd_pub = self.create_publisher(
            Float64MultiArray, '/joint_commands', 10
        )

        # 控制频率定时器
        self.timer = self.create_timer(0.02, self.control_loop)  # 50 Hz
        self.latest_obs = None

    def joint_callback(self, msg):
        self.latest_obs = np.array(
            list(msg.position) + list(msg.velocity), dtype=np.float32
        )

    def control_loop(self):
        if self.latest_obs is None:
            return
        obs = self.latest_obs.reshape(1, -1)
        action = self.session.run(None, {self.input_name: obs})[0]

        cmd = Float64MultiArray()
        cmd.data = action.flatten().tolist()
        self.cmd_pub.publish(cmd)
```


## 部署案例

### 四足机器人运动控制

典型部署流程：

1. 在 Isaac Gym 中使用 PPO 训练运动策略（约 1 亿步，数小时）
2. 域随机化：质量、摩擦力、推力扰动、传感器噪声
3. 导出 ONNX 模型，TensorRT 优化
4. Jetson Orin 上以 50 Hz 运行策略推理
5. 通过 EtherCAT 或 CAN 总线发送关节指令

### 机械臂抓取

1. 在 MuJoCo / Isaac Sim 中训练视觉-运动策略
2. 使用随机纹理和光照进行视觉域随机化
3. 策略蒸馏为轻量 CNN + MLP 网络
4. ROS 2 节点接收 RGBD 图像，输出末端执行器位姿


## Isaac Lab 机器人训练示例

NVIDIA Isaac Lab 是专为大规模机器人强化学习设计的 GPU 加速仿真框架，支持数千个并行环境同步运行，大幅缩短训练时间。

### 环境类结构

Isaac Lab 中的任务环境继承自 `DirectRLEnv` 或 `ManagerBasedRLEnv`，需要实现以下核心方法：

```python
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
import torch

class HumanoidLocomotionEnv(DirectRLEnv):
    cfg: DirectRLEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # 初始化关节目标、奖励缓冲区等
        self._joint_dof_idx, _ = self.robot.find_joints(".*")
        self.action_scale = 0.5

    def _get_observations(self) -> dict:
        # 收集观测：关节位置、速度、基座姿态、IMU数据
        obs = torch.cat([
            self.robot.data.joint_pos[:, self._joint_dof_idx],      # 关节位置
            self.robot.data.joint_vel[:, self._joint_dof_idx],      # 关节速度
            self.robot.data.root_lin_vel_b,                          # 基座线速度（机体系）
            self.robot.data.root_ang_vel_b,                          # 基座角速度
            self.robot.data.projected_gravity_b,                     # 投影重力方向
            self.commands[:, :3],                                    # 速度指令
        ], dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # 组合多项奖励
        alive_reward    = 1.0 * (~self.reset_terminated).float()
        vel_tracking    = self._reward_velocity_tracking()
        energy_penalty  = self._penalty_energy()
        contact_penalty = self._penalty_contact_forces()
        return alive_reward + vel_tracking + energy_penalty + contact_penalty
```

### 观测空间设计

运动控制任务的观测通常包含以下分量：

| 观测分量 | 维度 | 说明 |
|----------|------|------|
| 关节位置 | \(n_{dof}\) | 各关节当前角度（减去默认角度） |
| 关节速度 | \(n_{dof}\) | 各关节角速度 |
| 基座线速度 | 3 | 机体坐标系下的 \(v_x, v_y, v_z\) |
| 基座角速度 | 3 | 机体坐标系下的滚转/俯仰/偏航角速度 |
| 投影重力 | 3 | 重力向量在机体系的投影，隐式编码姿态 |
| 速度指令 | 3 | 目标前向速度、侧向速度、偏航速率 |
| 上一步动作 | \(n_{dof}\) | 提供动作历史信息，有助于平滑控制 |

### 奖励函数定义

```python
def _reward_velocity_tracking(self) -> torch.Tensor:
    # 跟踪目标线速度：使用指数核函数
    lin_vel_error = torch.sum(
        (self.commands[:, :2] - self.robot.data.root_lin_vel_b[:, :2]) ** 2, dim=1
    )
    return torch.exp(-lin_vel_error / 0.25) * 1.0

def _penalty_energy(self) -> torch.Tensor:
    # 能量惩罚：抑制关节扭矩过大
    return -torch.sum(
        torch.abs(self.robot.data.applied_torque[:, self._joint_dof_idx]), dim=1
    ) * 0.0002

def _penalty_contact_forces(self) -> torch.Tensor:
    # 碰撞惩罚：防止非预期肢体接触地面
    net_contact = torch.norm(
        self.contact_sensor.data.net_forces_w[:, self.undesired_contact_body_ids, :], dim=-1
    )
    return -torch.sum((net_contact > 1.0).float(), dim=1) * 0.1
```

### 域随机化参数

域随机化 (Domain Randomization) 是 Sim-to-Real 迁移的关键，Isaac Lab 通过配置类统一管理：

```python
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg, UniformNoiseCfg

randomization_cfg = dict(
    # 物理参数随机化
    physics_material=dict(
        static_friction=(0.6, 1.2),
        dynamic_friction=(0.4, 0.9),
        restitution=(0.0, 0.1),
    ),
    # 关节参数随机化
    joint_stiffness=(0.8, 1.2),   # 相对额定值的倍率范围
    joint_damping=(0.8, 1.2),
    # 观测噪声
    obs_noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.02),
    # 外力扰动
    push_robot=dict(interval_s=5.0, magnitude=(0.0, 1.0)),
)
```

### 训练启动与真机部署

```bash
# 在 Isaac Lab 根目录下启动训练（4096 并行环境）
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Velocity-Rough-Anymal-C-v0 \
    --num_envs=4096 \
    --headless

# 导出策略为 ONNX 格式用于真机推理
python scripts/reinforcement_learning/rsl_rl/export_onnx.py \
    --task=Isaac-Velocity-Rough-Anymal-C-v0 \
    --checkpoint=logs/rsl_rl/anymal_c_rough/model_5000.pt
```

训练完成后，将导出的 ONNX 模型部署到机器人的实时控制器上，通常以 400–1000 Hz 的频率运行策略推理。


## 参考资料

- Tobin J, et al. Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World. *IROS*, 2017.
- Peng X B, et al. Sim-to-Real Transfer of Robotic Control with Dynamics Randomization. *ICRA*, 2018.
- OpenAI. Solving Rubik's Cube with a Robot Hand. 2019.
- NVIDIA Jetson 开发文档：<https://developer.nvidia.com/embedded-computing>
- Rudin N, et al. Learning to Walk in Minutes Using Massively Parallel Deep RL. *CoRL*, 2022.
