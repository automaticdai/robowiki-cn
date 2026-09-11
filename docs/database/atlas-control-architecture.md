# Atlas 控制架构

!!! note "引言"
    Atlas 之所以能完成后空翻、跑酷与跳箱等高动态动作，核心在于其分层的实时控制架构：底层由高带宽液压伺服闭环保证力矩跟踪精度，中层由全身控制（Whole-Body Control, WBC）在满足接触与关节约束的前提下求解全身力矩分配，上层则由基于模型预测控制（Model Predictive Control, MPC）的运动规划器生成质心轨迹与落足点。本页面拆解液压版 Atlas 的这三层架构及其关键算法。


## 液压版控制架构

液压版 Atlas 的控制体系是其实现复杂运动的核心，融合了多个层次的算法模块，构成了一套从感知到执行的完整控制栈（Control Stack）。

### 液压伺服系统

Atlas 液压版的每个关节由一个液压缸（Hydraulic Cylinder）和对应的电液伺服阀（Electrohydraulic Servo Valve）驱动。液压系统的优势在于功率密度（Power Density）极高——每单位质量可提供的峰值力矩远超同等质量的电机，这对于后空翻等需要短时爆发的高动态动作至关重要。

液压系统的主要挑战包括：

- **非线性动力学**：液压缸内的油液压缩性、伺服阀的非线性流量特性以及摩擦力使得精确力控制（Force Control）更为困难，需要额外的补偿算法。
- **温度敏感性**：液压油的粘度随温度显著变化，导致系统在冷启动和长时间运行后的动态特性不一致。
- **维护成本**：液压系统存在密封件磨损和液压油泄漏等长期维护问题，不适合高频率部署使用。

每个关节配备了位置传感器（Position Sensor）和力矩传感器（Torque Sensor），末端足部集成了六维力/力矩传感器（6-axis Force/Torque Sensor），用于实时检测与地面的接触力。惯性测量单元（Inertial Measurement Unit, IMU）安装于机器人躯干，以 1 kHz 的采样频率提供姿态（Orientation）与角速度（Angular Velocity）测量值，是状态估计器（State Estimator）的核心输入。

### 状态估计

在高动态运动中，精确的状态估计（State Estimation）是整个控制系统的基础。Atlas 的状态估计器融合了 IMU 测量值、关节编码器读数和足底力传感器数据，通过扩展卡尔曼滤波器（Extended Kalman Filter, EKF）估计机器人质心的位置、速度和姿态。

足底力传感器在接触状态检测中起关键作用：当足底力超过某一阈值时，系统判定该足处于接触状态，并将接触点的位置加入状态估计的约束集合（Constraint Set），从而提高位置估计精度。在腾空相（Flight Phase）期间，足底力为零，状态估计仅依赖 IMU 积分，误差会随时间累积，因此腾空相的持续时间必须尽可能短。

### 模型预测控制（MPC）层

Boston Dynamics 在液压版 Atlas 上采用了模型预测控制（Model Predictive Control, MPC）作为高层运动规划框架。MPC 在一个有限时域（Receding Horizon）内对机器人质心（Center of Mass, CoM）的运动轨迹进行在线优化，以线性倒立摆（Linear Inverted Pendulum, LIP）或质心动力学（Centroidal Dynamics）模型为基础，预测未来若干步的状态演化。

典型的 MPC 问题形式为：在给定当前状态和期望步态模式（Gait Pattern）的条件下，求解一个预测时域（例如未来 10 个控制步）内的最优质心轨迹和接触力序列，使质心跟踪误差和接触力之和最小，同时满足接触力的摩擦锥约束（Friction Cone Constraint）。

MPC 层的输出为期望质心轨迹和期望接触力序列，该输出作为参考指令传递给下层的全身控制器。在实际工程实现中，MPC 通常以 10–100 Hz 的频率运行，而下层的 WBC 则以 1 kHz 的频率执行，两层之间通过环形缓冲区（Ring Buffer）传递参考轨迹。

### 全身控制（WBC）层与 IHMC 软件库

全身控制（Whole-Body Control, WBC）层负责将高层的运动指令转化为每个关节的力矩命令。在 DRC 期间，多支参赛队伍使用了由佛罗里达州人机认知研究所（Institute for Human and Machine Cognition, IHMC）开源的 IHMC Robotics Library，该库提供了针对人形机器人全身控制的成熟实现，包括：

- 基于层次化二次规划（Hierarchical Quadratic Programming, HQP）的优化求解器
- 接触状态估计器（Contact State Estimator）
- 质心动量控制器（Centroidal Momentum Controller）
- 用于运动学正逆解的工具链（Kinematics Toolchain）

IHMC 团队在 DRC 中使用 Atlas 平台取得了第二名的成绩，其控制代码基础随后演变为学术界广泛使用的开源框架，为全身控制领域的研究提供了重要的工程参考。

### 仿真到真实（Sim-to-Real）方法

液压版 Atlas 的运动轨迹开发大量依赖于物理仿真（Physics Simulation）。Boston Dynamics 内部使用定制的刚体动力学仿真器，在仿真环境中对运动轨迹进行优化，随后将结果迁移到真实机器人上执行。

在这一过程中，接触隐式轨迹优化（Contact-Implicit Trajectory Optimization, CITO）是一种关键技术。与传统轨迹优化方法需要预先固定接触序列（例如"左脚先落地，然后右脚"）不同，CITO 将接触时机和接触力同时作为优化变量，允许求解器在优化过程中自由探索接触模式（Contact Mode）空间。这一特性使得后空翻等含有飞行相的轨迹的自动生成成为可能。

仿真到真实的主要挑战在于克服仿真误差（Sim-to-Real Gap），即仿真器对以下因素的近似误差：

- 液压系统的非线性特性（阀门流量曲线、密封摩擦）
- 地面柔性与接触模型参数（刚度、阻尼、摩擦系数）
- 机体柔性（连杆弯曲、关节间隙）

Boston Dynamics 通过系统辨识（System Identification）实验测量真实机器人的动态参数，更新仿真器模型；并通过鲁棒控制（Robust Control）策略设计对参数不确定性具有鲁棒性的控制器，以缩小仿真误差对实机表现的影响。


## 全身控制（WBC）框架

全身控制框架是现代人形机器人控制的核心范式之一，Atlas 的控制系统对这一框架的发展与推广起到了重要作用。

### 基本原理

全身控制的核心思想是：将机器人视为一个统一的多体动力学系统，在满足所有物理约束的前提下，通过求解一个优化问题同时确定所有关节的力矩指令，而非逐个关节独立设计控制器。这一整体优化方法可以自然地处理关节间的耦合动力学（Coupled Dynamics），并在多个任务目标发生冲突时找到全局最优的折中方案。

全身控制的目标是在满足物理约束的前提下，使机器人的整体运动尽可能地跟踪上层规划器给出的多个任务目标。这些任务目标通常以层次化（Hierarchical）方式排列，例如：

1. **最高优先级**：运动学可行性与关节力矩限制（安全约束，不可违背）
2. **次优先级**：接触力约束（不滑动、不穿透地面）
3. **再次优先级**：质心轨迹跟踪（保证整体平衡）
4. **最低优先级**：上肢姿态调整与能耗最小化（次要任务）

### 数学建模

机器人全身动力学方程（Equation of Motion, EoM）可写为：

$$
M(q)\dot{v} + C(q, v) = S^\top \tau + J_c^\top f_c
$$

其中 \(q\) 为广义坐标（Generalized Coordinate），\(v\) 为广义速度，\(M(q)\) 为质量矩阵（Mass Matrix），\(C(q,v)\) 为包含科里奥利力（Coriolis Force）和重力（Gravity）的非线性项，\(S\) 为选择矩阵（Selection Matrix，区分驱动自由度与浮动基座的非驱动自由度），\(\tau\) 为关节力矩向量，\(J_c\) 为接触雅可比（Contact Jacobian），\(f_c\) 为接触力向量（Contact Force Vector）。

WBC 通过在每个控制周期（通常为 1 kHz）求解一个二次规划（Quadratic Programming, QP）问题来确定最优关节力矩：

$$
\min_{u} \|W(u - u_d)\|^2
$$

其中 \(u\) 为决策变量（包含关节力矩 \(\tau\) 和接触力 \(f_c\)），\(u_d\) 为期望值，\(W\) 为对角权重矩阵（Diagonal Weight Matrix），用于对不同任务分量的相对重要性进行加权。

约束条件包括：

- 全身动力学方程（等式约束）：\(M(q)\dot{v} + C(q,v) = S^\top \tau + J_c^\top f_c\)
- 接触摩擦锥（Friction Cone，不等式约束）：\(\|f_{c,t}\| \leq \mu f_{c,n}\)，其中 \(\mu\) 为摩擦系数，\(f_{c,n}\) 和 \(f_{c,t}\) 分别为接触力的法向分量和切向分量
- 关节力矩限制（不等式约束）：\(\tau_{\min} \leq \tau \leq \tau_{\max}\)
- 接触点无滑动条件（等式约束，在稳定接触阶段）：\(J_c \dot{v} + \dot{J}_c v = 0\)

该 QP 问题的规模通常为数十到数百个决策变量，可由高效的二次规划求解器（如 qpOASES 或 OSQP）在亚毫秒量级内求解，满足实时控制的时间要求。

### 任务优先级与层次 QP

在实践中，当多个任务目标发生冲突时，需要引入任务优先级机制。层次化二次规划（Hierarchical QP, HQP）方法通过严格层次（Strict Priority）或带权重的单层 QP（Weighted Single-Level QP）来处理优先级关系：

- **严格层次 HQP**：依次求解多个 QP 问题，每个低优先级问题在高优先级问题的残差构成的约束空间中求解，保证高优先级任务的残差不会被低优先级任务污染，但需要级联求解多个 QP，计算代价随任务层数增加而增大。
- **带权重的单层 QP**：将所有任务目标合并为一个加权目标函数，通过权重差异（通常相差数个数量级）近似实现优先级，计算代价仅为一个 QP，更适合实时实现，但权重调整需要经验。

Atlas 使用的控制框架在工程实现上采用了带任务层次的单层 QP，通过显著区分各任务的权重（例如安全相关约束权重为 \(10^6\)，次要任务权重为 \(1\)）来近似实现严格优先级，同时保持足够的求解速度以满足 1 kHz 实时控制要求。

### 接触状态切换

WBC 框架中另一个关键问题是接触状态的切换（Contact Mode Switching）。在步行过程中，每一步都伴随着足部从摆动状态（Swing State，无接触力约束）到支撑状态（Stance State，受接触力约束）的切换。切换时刻的判断依赖足底力传感器读数，当检测到接触力超过预设阈值时，控制器将对应足的接触雅可比加入约束集合，并在下一个控制周期开始以双支撑（Double Support）模式求解 QP。

接触状态的误判（例如将尚未触地的摆动足误判为已接触地面）会导致控制器施加错误的约束，进而导致关节力矩异常乃至机器人失稳，因此接触状态检测的可靠性对整个控制系统至关重要。


## 运动规划

### 落脚点规划（Footstep Planning）

Atlas 在非结构化地形上行走时，需要实时规划每一步的落脚点（Footstep）。落脚点规划是运动规划层的核心任务，其输出为一个有序的落脚点序列，供步态控制器（Gait Controller）逐步执行。

落脚点规划算法以点云（Point Cloud）或高程图（Elevation Map）为输入，结合机器人当前姿态与步态状态，在候选落脚区域中搜索满足以下条件的落脚序列：

- **运动学可达性**：落脚点必须在机器人腿部的运动学可达范围（Reachability）内
- **平坦性**：落脚区域的局部地形倾角不超过允许范围
- **稳定性**：落脚序列需保证质心运动的整体稳定性
- **连续性**：相邻落脚点之间的步幅（Step Length）和步宽（Step Width）需满足运动约束

常用方法包括：

- **A\* 图搜索**：在离散化的落脚点候选集合上进行启发式搜索，速度快，适合实时规划
- **混合整数规划（Mixed-Integer Programming, MIP）**：将地形约束建模为整数变量，实现全局最优搜索，但计算代价较高
- **强化学习（Reinforcement Learning, RL）策略**：近年来逐渐被引入落脚点规划，通过端到端学习直接从点云输出落脚点，无需显式的地形分析步骤

DRC 期间，多支团队采用了人机协同的半自主（Semi-Autonomous）策略：操作员在点云可视化界面中审核并确认系统给出的落脚点建议，以降低在复杂场景中的规划失败风险。这一"人在回路"（Human-in-the-Loop）模式在自主程度和安全性之间取得了当时条件下的最优折中。

### 摆动腿轨迹优化（Swing Trajectory Optimization）

在两个落脚点之间，摆动腿（Swing Leg）的轨迹需要在满足避障约束的同时，使关节力矩尽可能平滑以减少能耗并降低机械冲击。

常见做法是以多项式样条（Polynomial Spline，通常为五次或七次多项式）参数化摆动轨迹：给定起点（当前足部位置）、终点（目标落脚点）以及轨迹最高点（Apex），求解满足边界条件（起点和终点处速度、加速度为零）的多项式系数。在存在障碍物（如台阶边缘）的情况下，还需在中间添加路径点（Via Point）以约束轨迹形状，确保足部在抬起时不与障碍物发生碰撞。

对于更复杂的场景（如跨越宽幅障碍），可以采用数值轨迹优化方法，以避障约束和关节加速度最小化为目标，通过非线性规划（Nonlinear Programming, NLP）求解器得到全局最优摆动轨迹。

### ZMP 准则与质心动力学

早期双足机器人稳定性分析以零力矩点（Zero Moment Point, ZMP）准则为主——ZMP 定义为地面上使地面反力矩（Ground Reaction Moment）在水平方向分量为零的点，当 ZMP 落在支撑多边形（Support Polygon）内部时，机器人不会发生绕脚尖的准静态翻倒。

ZMP 准则的优点是计算简单，可直接通过足底力传感器测量 ZMP 位置，并以此作为稳定性实时监测指标。然而，ZMP 准则的根本局限在于它仅适用于质心高度恒定的准静态（Quasi-Static）运动假设，无法描述 Atlas 在跑步、跳跃或后空翻时的高动态运动状态——这些运动中质心高度剧烈变化，且存在完全失去地面接触的腾空相。

为此，Boston Dynamics 转向基于质心动力学（Centroidal Dynamics）的方法：

$$
\dot{h} = \sum_i f_{c,i} + m g
$$

$$
\dot{k} = \sum_i \left( r_i \times f_{c,i} + \tau_{c,i} \right)
$$

其中 \(h\) 为系统线动量，\(k\) 为系统角动量，\(r_i\) 为接触点位置向量，\(f_{c,i}\) 和 \(\tau_{c,i}\) 为各接触点处的接触力和力矩，\(m\) 为机器人总质量，\(g\) 为重力加速度向量。

质心动力学方法将全身的线动量和角动量作为统一的规划变量，允许在腾空相中对角动量的变化进行显式规划，从而支持包含飞行相的高动态运动轨迹的生成与控制。


## 参考资料

1. S. Kuindersma et al., "Optimization-Based Locomotion Planning, Estimation, and Control Design for the Atlas Humanoid Robot," *Autonomous Robots*, vol. 40, pp. 429-455, 2016.
2. S. Feng, E. Whitman, X. Xinjilefu, and C. G. Atkeson, "Optimization-Based Full Body Control for the DARPA Robotics Challenge," *Journal of Field Robotics*, vol. 32, no. 2, pp. 293-312, 2015.
3. Boston Dynamics, "Atlas Gets a Grip" / "Flipping the Script with Atlas" 技术博客. https://bostondynamics.com/blog/
4. [Atlas 总览](atlas.md)
5. [全身控制与力控](../manipulation/force-control.md)
6. [模型预测控制](../control/mpc/mpc.md)
