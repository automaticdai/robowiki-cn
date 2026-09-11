# CoppeliaSim Lua 脚本与强化学习

!!! note "引言"
    CoppeliaSim 的一大特点是「脚本内嵌于场景」：每个对象都可以挂载 Lua 子脚本（Child Script），在仿真回调中直接读写关节、传感器与信号，无需外部进程即可实现完整的控制逻辑。这种设计使场景本身就是可运行的程序，但也要求开发者理解其回调时序与非阻塞约定。另一方面，当 CoppeliaSim 作为强化学习环境使用时，需要把仿真切换到步进模式并包装为 Gymnasium 接口，才能与主流训练框架对接。本页面介绍这两部分内容。


## 强化学习应用

### CoppeliaSim 作为 Gymnasium 环境

CoppeliaSim 通过 Python ZeroMQ API 提供步进控制能力，天然适合作为强化学习环境的后端仿真器。以下展示如何将 CoppeliaSim 封装为 Gymnasium 兼容接口：

```python
import gymnasium as gym
import numpy as np
import coppeliasim_zmqremoteapi_client as zmq_client


class CoppeliaSimEnv(gym.Env):
    """将 CoppeliaSim 仿真封装为 Gymnasium 标准环境"""

    def __init__(self):
        super().__init__()
        self.client = zmq_client.RemoteAPIClient()
        self.sim = self.client.require('sim')
        self.client.setStepping(True)

        # 定义动作空间：关节速度（6维连续）
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )
        # 定义观测空间：关节角度 + 末端位姿（6 + 7 = 13维）
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32
        )

        # 获取关节句柄
        self.joint_handles = [
            self.sim.getObject(f'/robot/joint{i}') for i in range(1, 7)
        ]
        self.tip_handle = self.sim.getObject('/robot/tip')
        self.target_handle = self.sim.getObject('/robot/target')

    def reset(self, seed=None, options=None):
        self.sim.stopSimulation()
        self.sim.startSimulation()
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        # 执行动作：设置关节速度
        for i, h in enumerate(self.joint_handles):
            self.sim.setJointTargetVelocity(h, float(action[i]))

        # 推进仿真一步
        self.client.step()

        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._is_done()
        return obs, reward, terminated, False, {}

    def _get_obs(self):
        joint_angles = np.array([
            self.sim.getJointPosition(h) for h in self.joint_handles
        ], dtype=np.float32)
        tip_pose = np.array(
            self.sim.getObjectPose(self.tip_handle, self.sim.handle_world),
            dtype=np.float32
        )
        return np.concatenate([joint_angles, tip_pose])

    def _compute_reward(self):
        tip_pos = self.sim.getObjectPosition(
            self.tip_handle, self.sim.handle_world
        )
        target_pos = self.sim.getObjectPosition(
            self.target_handle, self.sim.handle_world
        )
        dist = np.linalg.norm(np.array(tip_pos) - np.array(target_pos))
        return -dist  # 奖励为负距离

    def _is_done(self):
        tip_pos = self.sim.getObjectPosition(
            self.tip_handle, self.sim.handle_world
        )
        target_pos = self.sim.getObjectPosition(
            self.target_handle, self.sim.handle_world
        )
        dist = np.linalg.norm(np.array(tip_pos) - np.array(target_pos))
        return dist < 0.02  # 距离目标 2cm 以内视为成功

    def close(self):
        self.sim.stopSimulation()
```

### 无头仿真（Headless Simulation）

在服务器或无显示器的环境中进行强化学习训练时，需要以无头模式运行 CoppeliaSim：

```bash
# 方法一：使用虚拟显示（Xvfb）
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99

# 启动 CoppeliaSim（无 GUI 渲染，但保留物理仿真）
./coppeliaSim.sh -h scene.ttt &

# 然后运行 Python 训练脚本
python3 train_rl.py
```

```bash
# 方法二：直接使用无头模式参数
./coppeliaSim.sh -h -s scene.ttt
# -h: headless 模式，不创建 GUI 窗口
# -s: 自动加载指定场景
```

### 与其他仿真器的对比（强化学习场景）

| 特性 | CoppeliaSim | Gazebo | PyBullet |
|------|-------------|--------|----------|
| 物理精度 | 高（多引擎可选） | 高 | 中（偏快速） |
| Python API 完整度 | 高（覆盖全部仿真功能） | 中 | 高 |
| 步进控制 | 原生支持 | 支持 | 原生支持 |
| 无头模式 | 支持（需 Xvfb 或 -h 参数） | 支持 | 原生支持（无 GUI 依赖） |
| 仿真速度 | 中（与实时绑定） | 中 | 快（可超实时） |
| 模型格式 | URDF、SDF、VRML | SDF、URDF | URDF |
| 授权 | Edu 免费，Pro 收费 | 完全开源 | 完全开源 |
| 适合场景 | 复杂传感器仿真、工业机械臂 | ROS 集成、移动机器人 | 快速原型、接触丰富任务 |


## Lua 脚本基础

CoppeliaSim 的内置脚本语言是 Lua，每个仿真对象都可以附加 Lua 子脚本（Child Script）来控制其行为。

### 脚本类型

| 脚本类型 | 运行方式 | 适用场景 |
|---------|---------|---------|
| 非线程子脚本（Non-threaded） | 在主仿真循环中同步调用 | 绝大多数控制逻辑 |
| 线程子脚本（Threaded） | 在独立线程中运行 | 需要 `sleep` 或阻塞等待的场景 |
| 主脚本（Main Script） | 管理整个仿真生命周期 | 高级用户定制仿真循环 |

### 常用 Lua API 函数

```lua
-- 对象操作
handle = sim.getObject('/robot')                          -- 获取对象句柄
pos = sim.getObjectPosition(handle, sim.handle_world)    -- 获取位置 [x, y, z]
sim.setObjectPosition(handle, pos, sim.handle_world)     -- 设置位置

-- 关节操作
angle = sim.getJointPosition(joint_handle)               -- 读取关节角度（弧度）
sim.setJointTargetPosition(joint_handle, math.pi / 2)   -- 设置目标位置
sim.setJointTargetVelocity(joint_handle, 1.0)           -- 设置目标速度（rad/s）

-- 传感器读取
result, dist = sim.readProximitySensor(sensor_handle)   -- 读取接近传感器
result, img, resX, resY = sim.getVisionSensorImg(cam_handle)  -- 读取相机图像

-- 仿真时间
t = sim.getSimulationTime()                              -- 获取当前仿真时间（秒）
```

### 简单差速移动机器人 Lua 控制示例

```lua
-- 非线程子脚本：附加在移动机器人对象上
function sysCall_init()
    -- 获取左右轮电机句柄
    left_motor  = sim.getObject('./LeftMotor')
    right_motor = sim.getObject('./RightMotor')

    -- 获取接近传感器句柄（用于障碍物检测）
    front_sensor = sim.getObject('./FrontSensor')

    -- 初始速度设置（rad/s）
    max_speed = 3.0
    start_time = sim.getSimulationTime()
end

function sysCall_actuation()
    -- 读取前方接近传感器
    local detected, dist = sim.readProximitySensor(front_sensor)

    if detected and dist < 0.5 then
        -- 检测到障碍物，原地左转
        sim.setJointTargetVelocity(left_motor,  -max_speed)
        sim.setJointTargetVelocity(right_motor,  max_speed)
    else
        -- 直行
        sim.setJointTargetVelocity(left_motor,  max_speed)
        sim.setJointTargetVelocity(right_motor, max_speed)
    end
end

function sysCall_sensing()
    -- 此回调在每个仿真步的感知阶段执行
    -- 可在此记录传感器数据用于后续分析
end

function sysCall_cleanup()
    -- 仿真结束时的清理工作
    sim.setJointTargetVelocity(left_motor,  0)
    sim.setJointTargetVelocity(right_motor, 0)
end
```


## 性能优化

在进行大规模强化学习训练或长时间仿真时，CoppeliaSim 的性能优化至关重要。

### 减少碰撞网格的多边形数量

物理引擎进行碰撞检测时使用专用的简化网格（Collision Mesh），而非渲染网格（Visual Mesh）。应将碰撞网格的多边形数量控制在最小必要范围：

- 在场景编辑器中，为每个对象的碰撞形状选择"凸包（Convex Hull）"或"包围盒（Bounding Box）"而非精确网格
- 对于机器人连杆，通常使用圆柱体或长方体近似碰撞形状即可满足仿真需求

### 物理引擎选择

| 物理引擎 | 速度 | 精度 | 适用场景 |
|---------|------|------|---------|
| ODE | 快 | 中 | 移动机器人、一般场景（默认推荐） |
| Bullet | 快 | 中 | 柔体、软体仿真 |
| Newton | 中 | 高 | 需要高精度动力学的场景 |
| Vortex | 慢 | 最高 | 工业级精密仿真（需授权） |

```lua
-- 在 Lua 脚本中切换物理引擎
sim.setInt32Parameter(sim.intparam_dynamic_engine, sim.physics_ode)
-- 可选值: sim.physics_ode / sim.physics_bullet / sim.physics_newton / sim.physics_vortex
```

### 无头训练时关闭渲染

在强化学习训练中，渲染是主要的性能瓶颈之一。在无头模式下，默认渲染会被跳过，但若使用视觉传感器仿真相机观测，仍会触发渲染。可按需降低视觉传感器的分辨率：

```lua
-- 在 Lua 脚本中降低视觉传感器分辨率
local cam_handle = sim.getObject('./Camera')
sim.setVisionSensorResolution(cam_handle, 64, 64)  -- 训练时使用低分辨率
```

### 加速仿真（实时系数）

CoppeliaSim 默认以实时速度运行，可以在场景设置中调整仿真时间步长和实时系数：

```lua
-- 设置仿真时间步长（秒）
sim.setFloatParameter(sim.floatparam_simulation_time_step, 0.05)

-- 在步进模式下，仿真速度不受实时限制，完全由 Python 的 client.step() 调用速率决定
```


## 参考资料

- [CoppeliaSim 官方文档：Lua 脚本](https://www.coppeliarobotics.com/helpFiles/en/scripts.htm)
- [Gymnasium 文档](https://gymnasium.farama.org/)
- [CoppeliaSim 总览](vrep.md)
- [远程 API 与 ROS 2 集成](vrep-remote-api-ros2.md)
- [性能优化与调试](vrep-performance-debugging.md)
- [强化学习](../learning/rl.md)
