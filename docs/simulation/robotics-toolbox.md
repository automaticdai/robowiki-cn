# MATLAB Robotics Toolbox
![5e430036ce538f09f700003a](assets/4a1ffd6b2aa24f0dab6442665d7b9469.png)

- 官方网站：https://uk.mathworks.com/products/robotics.html
- Peter Corke Robotics Toolbox：https://petercorke.com/toolboxes/robotics-toolbox/
- 许可：MATLAB (商业许可) / Peter Corke 版本 (MIT 开源)

!!! note "引言"
    MATLAB机器人工具箱是机器人学教育和研究中广泛使用的软件工具。实际上存在两个不同但相关的工具箱：一是MathWorks官方发布的Robotics System Toolbox，从MATLAB 2013版本开始引入；二是由Peter Corke教授开发的开源Robotics Toolbox for MATLAB (RTB)。两者都为机器人建模、仿真和控制提供了丰富的函数库。


## Peter Corke 的 Robotics Toolbox

Peter Corke教授 (Queensland University of Technology, 昆士兰科技大学) 开发的Robotics Toolbox (RTB) 是机器人学教育领域的经典工具，其历史可追溯至1990年代。该工具箱与其著名教材《Robotics, Vision and Control》配套使用，被全球数百所大学的机器人学课程采用。

RTB的核心功能包括：

- 齐次变换 (Homogeneous Transformation) 与旋转表示
- DH参数 (Denavit-Hartenberg Parameters) 建模
- 正运动学 (Forward Kinematics) 与逆运动学 (Inverse Kinematics)
- 雅可比矩阵 (Jacobian Matrix) 计算
- 轨迹规划 (Trajectory Planning)
- 动力学建模 (Dynamics Modeling)

Peter Corke还发布了Python版本的工具箱 **Robotics Toolbox for Python**（`roboticstoolbox-python`），使得非MATLAB用户也能使用相同的功能。


## MathWorks Robotics System Toolbox

MathWorks官方的Robotics System Toolbox提供了更工程化的功能集：

- **机器人建模**：支持通过URDF导入机器人模型，自动构建刚体树 (Rigid Body Tree) 结构
- **运动规划 (Motion Planning)**：集成了RRT、PRM等路径规划算法，以及基于优化的轨迹规划方法
- **定位与建图 (Localization and Mapping)**：提供SLAM算法、粒子滤波 (Particle Filter) 等功能
- **感知处理**：支持点云 (Point Cloud) 处理、障碍物检测等
- **ROS集成**：提供ROS和ROS 2接口，使得MATLAB代码和Simulink可以与ROS系统通信


## DH 参数建模

DH参数 (Denavit-Hartenberg Parameters) 是描述串联机械臂运动学结构的标准方法。每个关节由四个参数定义：连杆长度 (a)、连杆扭角 (alpha)、连杆偏距 (d) 和关节角 (theta)。使用Peter Corke的RTB建模示例：

```matlab
% 定义PUMA 560机械臂的DH参数
L(1) = Link('d', 0,     'a', 0,      'alpha', pi/2);
L(2) = Link('d', 0,     'a', 0.4318, 'alpha', 0);
L(3) = Link('d', 0.15,  'a', 0.0203, 'alpha', -pi/2);
L(4) = Link('d', 0.4318,'a', 0,      'alpha', pi/2);
L(5) = Link('d', 0,     'a', 0,      'alpha', -pi/2);
L(6) = Link('d', 0,     'a', 0,      'alpha', 0);

robot = SerialLink(L, 'name', 'PUMA 560');
robot.plot([0 0 0 0 0 0]);  % 可视化零位姿态
```


## 轨迹规划 (Trajectory Planning)

工具箱提供了多种轨迹生成方法：

- **关节空间轨迹 (Joint-Space Trajectory)**：使用 `jtraj` 函数在关节空间中生成平滑的多项式轨迹
- **笛卡尔空间轨迹 (Cartesian-Space Trajectory)**：使用 `ctraj` 函数在操作空间中生成直线或弧线轨迹
- **梯形速度曲线 (Trapezoidal Velocity Profile)**：使用 `tpoly` 或 `lspb` 函数生成具有加速-匀速-减速阶段的轨迹

轨迹规划结合逆运动学求解，可以实现机械臂末端执行器 (End Effector) 沿指定路径运动。


## 可视化功能

MATLAB环境下的可视化是工具箱的一大优势：

- **机器人模型可视化**：通过 `plot` 函数显示机器人的三维模型和关节运动动画
- **工作空间分析 (Workspace Analysis)**：可视化机械臂的可达工作空间
- **轨迹可视化**：在三维空间中显示末端执行器的运动轨迹
- **Simulink集成**：通过Simulink模块实现机器人控制系统的图形化建模和仿真


## 与ROS的集成

Robotics System Toolbox提供了ROS的接口，使得MATLAB代码和Simulink可以和ROS很好的结合。具体功能包括：

- 从MATLAB连接到ROS Master，订阅和发布ROS话题
- 调用ROS服务和动作
- 在Simulink中使用ROS消息类型进行控制回路设计
- 支持将MATLAB/Simulink算法生成ROS节点并部署

这一集成使得研究人员可以在MATLAB中进行算法开发和验证，然后将成熟的算法部署到基于ROS的真实机器人系统中。


## 优势与局限

**优势：**

- MATLAB语言易学，矩阵运算能力强，适合算法原型开发
- 可视化和绘图功能丰富
- 与Simulink集成，支持控制系统设计和验证
- Peter Corke版本配套教材，教育价值高

**局限：**

- MathWorks版本需要购买商业许可，成本较高
- 不适合大规模三维场景仿真或实时物理仿真
- 执行效率不如C++编写的仿真器


## 安装指南

### Python 版本安装

使用 pip 安装 Robotics Toolbox for Python 及其空间数学依赖库：

```bash
# Python版本
pip install roboticstoolbox-python spatialmath-python

# 可选依赖
pip install matplotlib sympy qpsolvers
```

其中 `spatialmath-python` 提供 SE3、SO3、四元数等空间数学对象；`matplotlib` 用于二维绘图；`sympy` 支持符号运算；`qpsolvers` 用于基于二次规划的逆运动学求解。


## Python版Robotics Toolbox详解

`roboticstoolbox-python` 是Peter Corke工具箱的Python重新实现，提供与MATLAB版本几乎相同的API。内置了多种常用机器人模型，包括：

- **Panda**：Franka Emika Panda，7自由度协作机械臂
- **UR5**：Universal Robots UR5，6自由度工业机械臂
- **Puma560**：经典PUMA 560，6自由度机械臂（用于教学示例）
- **KR5**：KUKA KR5，6自由度工业机械臂

以下示例展示了正运动学、逆运动学与雅可比矩阵的基本用法：

```python
import roboticstoolbox as rtb
import numpy as np

# 加载 Panda 机器人模型
robot = rtb.models.Panda()
print(robot)  # 显示DH参数和关节信息

# 正运动学 (Forward Kinematics)
q = robot.qr  # 就绪位形 (ready configuration)
T = robot.fkine(q)  # SE3齐次变换矩阵
print(T)

# 逆运动学 (Inverse Kinematics, Levenberg-Marquardt)
sol = robot.ikine_LM(T)
if sol.success:
    print(f"IK解: {sol.q}")

# 雅可比矩阵 (Jacobian Matrix)
J = robot.jacobe(q)  # 末端坐标系中的雅可比
```

`robot.qr` 是每个内置模型预设的就绪位形，方便快速测试。`fkine` 返回 `SE3` 对象，可直接用于位姿运算。`ikine_LM` 使用Levenberg-Marquardt数值迭代法求解逆运动学，`sol.success` 指示是否收敛。


## Swift浏览器可视化

Swift是 `roboticstoolbox-python` 配套的三维可视化后端，基于Web浏览器渲染，无需安装独立GUI应用，在服务器无头模式 (headless) 下也可运行。Swift将场景渲染为交互式WebGL页面，自动在默认浏览器中打开。

```python
# 浏览器可视化：显示单帧位形
robot.plot(q, backend='swift')

# 动画轨迹：在浏览器中播放关节空间轨迹
q_target = np.array([0.1, -0.5, 0.3, -1.2, 0.2, 1.0, 0])
traj = rtb.jtraj(robot.qr, q_target, 50)
robot.plot(traj.q, backend='swift')
```

此外，Swift还支持通过 `env.step()` 接口进行步进仿真，可集成到控制循环中实时显示机器人运动过程。


## 轨迹规划代码示例

工具箱提供了完整的轨迹规划函数集，覆盖关节空间和笛卡尔空间两类方法：

```python
from spatialmath import SE3
import roboticstoolbox as rtb
import numpy as np

# 关节空间轨迹 (Joint-Space Trajectory)
q0 = robot.qr
qf = np.array([0.1, -0.5, 0.3, -1.2, 0.2, 1.0, 0])
traj = rtb.jtraj(q0, qf, 100)  # 100个时间步
# traj.q  形状为 (100, 7)，每行是一个关节位形
# traj.qd 关节速度，traj.qdd 关节加速度

# 笛卡尔轨迹 (Cartesian Trajectory)
T0 = robot.fkine(q0)
T1 = T0 * SE3.Tx(0.1)  # x方向移动0.1m
ctraj = rtb.ctraj(T0, T1, 50)
# ctraj 是 SE3 数组，长度为50

# 梯形速度曲线 (Trapezoidal Velocity Profile)
s, sd, sdd = rtb.trapezoidal(0, 1, 50)
# s：位置插值参数 (0→1)
# sd：速度，sdd：加速度
```

`jtraj` 使用五次多项式插值，保证两端速度和加速度均为零，运动平滑。`ctraj` 在 SE3 流形上进行插值，确保末端执行器在笛卡尔空间中匀速运动。梯形速度曲线适用于需要限制加速度的精确定位任务。


## spatialmath空间数学库

`spatialmath-python` 是独立的空间数学库，提供 SO(3)、SE(3)、单位四元数等李群对象，支持链式运算和插值：

```python
from spatialmath import SE3, SO3, UnitQuaternion
import numpy as np

# 创建变换：绕各轴的平移和旋转
T = SE3.Tx(0.5) * SE3.RPY([0, 0, np.pi/4])
R = SO3.Rz(np.pi/4)
q = UnitQuaternion.Rx(0.5)

# 变换组合：平移后绕Y轴旋转30度
T_combined = SE3.Trans(0.1, 0.2, 0.3) * SE3.Ry(np.pi/6)

# 插值：在两个位姿之间取中间值
T0 = SE3.Trans(0, 0, 0)
T1 = SE3.Trans(1, 0, 0) * SE3.Rz(np.pi/2)
T_interp = T0.interp(T1, s=0.5)  # 中间位姿

# 从旋转矩阵转四元数
q_from_R = UnitQuaternion(R)
print(q_from_R)  # 输出四元数分量
```

`SE3.RPY` 接受滚转-俯仰-偏航角 (Roll-Pitch-Yaw) 创建旋转，`interp` 方法在旋转部分使用球面线性插值 (SLERP)，在平移部分使用线性插值，保证插值路径的几何合理性。


## 动力学仿真

工具箱基于递归牛顿欧拉法 (Recursive Newton-Euler Algorithm, RNE) 实现逆动力学计算，也支持前向动力学仿真：

```python
# 质量矩阵 (Mass Matrix / Inertia Matrix)
# M(q) 为关节空间惯量矩阵，形状 (n, n)
M = robot.inertia(q)

# 递归牛顿欧拉法 (Recursive Newton-Euler, RNE)
# 给定位形、速度、加速度，计算所需关节力矩
qd  = np.zeros(robot.n)   # 关节速度
qdd = np.zeros(robot.n)   # 关节加速度
tau = robot.rne(q, qd, qdd)

# 重力载荷 (Gravity Load)
# 仅考虑重力时各关节需要的补偿力矩
tau_g = robot.gravload(q)

# 前向动力学仿真
# 给定力矩，计算关节加速度
qdd = robot.accel(q, qd, tau)
```

动力学模型是实现力矩控制 (Torque Control)、导纳控制 (Admittance Control) 和模型预测控制 (Model Predictive Control, MPC) 的基础。在实际部署时，可将 `rne` 计算出的前馈力矩与PD反馈控制结合，实现计算力矩控制 (Computed Torque Control)。


## 与Simulink集成（MATLAB版）

MathWorks Robotics System Toolbox 在 Simulink 环境中提供了完整的机器人仿真模块链：

- **Robotics System Toolbox模块库**：包含正运动学、逆运动学、雅可比矩阵等Simulink功能模块，可拖拽组合构建控制系统
- **URDF导入**：可将标准 URDF/SDF 文件导入Simulink，自动生成刚体树模型，用于多体动力学仿真
- **代码生成**：Simulink模型可通过Embedded Coder或MATLAB Coder生成C/C++代码，部署到嵌入式控制器或实时目标机 (Real-Time Target)
- **硬件在环测试 (Hardware-in-the-Loop, HIL)**：通过Simulink Real-Time与dSPACE等实时平台集成，实现控制算法的HIL测试，在部署到真实机器人前验证控制逻辑

这一工具链特别适合需要经过严格V型开发流程验证的工业机器人应用，从算法原型到嵌入式部署形成完整闭环。


## 适用场景对比

以下表格对比了三种主要工具在不同维度的特性：

| 维度 | Peter Corke Python版 | MathWorks MATLAB版 | Gazebo |
|---|---|---|---|
| **教学价值** | 高，配套教材完善 | 高，图形化交互强 | 中，学习曲线陡 |
| **开源/许可** | MIT 开源，免费 | 商业许可，成本高 | Apache 2.0，免费 |
| **算法研究** | 优秀，纯Python易扩展 | 优秀，符号运算支持强 | 一般，侧重场景仿真 |
| **传感器真实感** | 低，无真实物理仿真 | 低，无真实物理仿真 | 高，支持相机/激光等 |
| **物理精度** | 无（运动学模型） | 无（运动学模型） | 高（ODE/Bullet物理引擎） |
| **工程部署** | 一般，需自行集成 | 强，支持代码生成和HIL | 强，ROS原生集成 |
| **计算效率** | 高（纯Python，轻量） | 高（MATLAB JIT加速） | 低（三维物理仿真开销大） |
| **成本** | 免费 | 商业许可（学术折扣） | 免费 |
| **适用人群** | 学生、算法研究者 | 工程师、控制系统设计 | 全栈机器人开发者 |

**选型建议：**

- 学习机器人运动学/动力学基础理论，推荐 **Peter Corke Python版**，开源免费且与教材紧密结合。
- 需要控制系统设计、Simulink建模或代码生成部署的工程项目，选择 **MathWorks MATLAB版**。
- 需要高保真传感器仿真、三维环境交互或完整ROS系统测试，使用 **Gazebo**。


## 参考资料

- [MathWorks Robotics System Toolbox文档](https://www.mathworks.com/help/robotics/)
- [Peter Corke Robotics Toolbox for MATLAB](https://petercorke.com/toolboxes/robotics-toolbox/)
- [Robotics Toolbox for Python](https://github.com/petercorke/robotics-toolbox-python)
- [spatialmath-python文档](https://github.com/petercorke/spatialmath-python)
- [Swift可视化后端](https://github.com/jhavl/swift)
- Corke, P. (2017). *Robotics, Vision and Control: Fundamental Algorithms in MATLAB* (2nd ed.). Springer.
- Corke, P., & Haviland, J. (2021). Not your grandmother's toolbox – the Robotics Toolbox reinvented for Python. *IEEE International Conference on Robotics and Automation (ICRA)*.
