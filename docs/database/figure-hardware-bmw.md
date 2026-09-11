# Figure 硬件与 BMW 工厂部署

!!! note "引言"
    再强的策略模型也要通过硬件落地。Figure 02 的每只手拥有约 16 个自由度，配合指尖力觉与本体感知构成灵巧操作的物理基础；机身则集成多目视觉与高频惯性测量，为 Helix 提供观测输入。2024 年起，Figure 与 BMW 在其美国 Spartanburg 工厂开展的合作，成为人形机器人首批进入真实汽车产线的案例之一。本页面介绍其灵巧手与感知系统设计，以及 BMW 项目的部署细节与工程挑战。


## 灵巧手技术

### 设计理念

Figure 02 的灵巧手（Dexterous Hand）是整机最具技术含量的子系统之一。其设计理念源于对人类手部解剖学（Hand Anatomy）的深入研究：人类手部拥有约 27 块骨骼、29 个关节和超过 30 块肌肉，能够实现从粗力量抓取（Power Grasp）到精密捏取（Precision Pinch）的宽泛操作模式。Figure 02 的 16 自由度手部设计旨在尽可能近似这一能力范围，同时在工程可行性（材料强度、电机体积、重量限制）方面做出合理的工程折中。

### 关节与驱动方案

每只手的 16 个自由度分布于拇指（Thumb）、食指（Index Finger）、中指（Middle Finger）、无名指（Ring Finger）和小指（Little Finger）五根手指，以及手腕（Wrist）关节。各手指通常配置 2—3 个主动自由度（Active DOF），手腕配置 2 个自由度（屈伸与侧偏）。

驱动方案采用微型电机（Micro Motor）配合肌腱传动（Tendon Drive）的组合方式：

- **微型电机**：布置于手掌或前臂，避免将大质量部件置于手指末端，降低手指的转动惯量（Moment of Inertia），从而提升手指运动的动态响应速度
- **肌腱传动**：细钢缆（Cable）模拟肌腱，将电机力矩传递至手指关节，实现较高的传动效率（Transmission Efficiency）和较小的传动间隙（Backlash）

与液压驱动（Hydraulic Actuation）相比，全电动腱驱方案在力控精度（Force Control Precision）方面略有不足，但在能量效率（Energy Efficiency）、维护成本（Maintenance Cost）和环境适应性方面具有明显优势，更适合工厂环境长时间连续运行的要求。

### 触觉感知

Figure 02 的手部集成了触觉传感器（Tactile Sensor），能够在手指指尖和手掌关键接触区域实时采集接触力（Contact Force）和接触面积（Contact Area）信息。触觉感知数据与视觉数据融合后，为抓取策略提供关键反馈：

- **抓取力调节**：根据物体表面材质（硬/软）和重量，动态调节抓取力大小，避免压碎易碎物体（如玻璃容器）或因力度不足导致物体滑落
- **接触检测**：在视觉遮挡情况下（如手指被物体遮挡时），通过触觉信号判断是否已建立有效接触
- **滑移检测（Slip Detection）**：通过高频采样触觉信号，检测物体在手指间的微小滑移趋势，并及时触发补偿动作

### 操作模式

Figure 02 的手部支持两种主要操作模式，两种模式之间可根据任务需求动态切换：

**精密捏取模式（Precision Pinch Mode）**：主要使用拇指和食指（或拇指与中指）的指尖进行小物体的精细夹持，适用于拾取螺钉、插拔连接器、操作按钮等需要高精度定位的任务。在此模式下，控制系统优先保证位置精度（Position Accuracy），接触力维持在较小水平（通常 0.5—2 N）。

**力量抓取模式（Power Grasp Mode）**：五根手指同时包裹物体，利用手掌和手指的综合接触面积提供最大抓持力，适用于搬运较重零件、推拉机构等需要大抓持力的任务。在此模式下，控制系统优先保证抓持稳定性，接触力可达数十牛顿。

### 手部视觉引导

Figure 02 在手腕附近集成了 RGB 摄像头（Hand-Mounted RGB Camera），为近距离精细操作提供"手眼协调"（Hand-Eye Coordination）能力。头部摄像头负责全局场景感知和目标定位，手部摄像头则负责在操作执行阶段提供高分辨率的近距离视觉反馈，两者协同工作，显著提升了机器人在工件对准（Workpiece Alignment）、插孔（Peg-in-Hole）等高精度任务中的成功率。

手眼协调的标定（Calibration）是保证视觉引导精度的关键步骤。Figure 02 采用眼在手上（Eye-in-Hand）配置，需要精确标定手部摄像头相对于手腕坐标系（Wrist Frame）的外参矩阵（Extrinsic Matrix）\(\mathbf{T}_{cam}^{wrist}\)，以及摄像头内参（Intrinsic Parameters）。标定误差直接影响视觉引导操作的精度，通常要求总体定位误差控制在 1—2 mm 以内。

### 手部与整机的力控集成

灵巧手的力控能力并非孤立运作，而是与整机力控系统深度集成。当机器人执行需要力柔顺（Force Compliance）的任务（如将零件插入有公差的孔位）时，手部关节的力矩传感器数据与腕部六维力/力矩传感器（6-Axis Force/Torque Sensor）数据融合，共同驱动阻抗控制（Impedance Control）算法：

$$\mathbf{F} = \mathbf{K}(\mathbf{x}_d - \mathbf{x}) + \mathbf{D}(\dot{\mathbf{x}}_d - \dot{\mathbf{x}}) + \mathbf{M}(\ddot{\mathbf{x}}_d - \ddot{\mathbf{x}})$$

其中 \(\mathbf{K}\)、\(\mathbf{D}\)、\(\mathbf{M}\) 分别为刚度矩阵（Stiffness Matrix）、阻尼矩阵（Damping Matrix）和惯性矩阵（Inertia Matrix），\(\mathbf{x}_d\) 为期望位置，\(\mathbf{x}\) 为实际位置。通过调节刚度矩阵，可以在不同任务中灵活切换位置控制主导模式和力控柔顺主导模式，是实现精密装配操作的关键控制技术。


## 感知系统

### 多摄像头立体视觉

Figure 02 在头部配置了 6 个摄像头，构成多目立体视觉系统（Multi-Camera Stereo Vision System）。与单目摄像头（Monocular Camera）相比，多目配置具有以下优势：

- **深度估计（Depth Estimation）**：通过双目或多目几何关系直接计算场景中物体的三维深度，精度优于纯单目深度估计网络（Monocular Depth Network），在近距离操作场景下尤为重要
- **宽视场覆盖（Wide Field of View Coverage）**：6 个摄像头提供近乎 360° 的环境感知覆盖，消除单摄像头在机器人运动过程中可能出现的视觉盲区
- **冗余容错（Redundancy）**：当某个方向的摄像头受到遮挡或反光干扰时，其他摄像头的数据可以提供补充信息

视觉处理流程通常包括以下步骤：原始图像采集 → 图像去畸变（Undistortion） → 多目立体匹配（Stereo Matching） → 点云生成（Point Cloud Generation） → 目标检测与语义分割（Object Detection & Semantic Segmentation） → 三维目标位姿估计（6-DoF Object Pose Estimation）。

### 视觉惯性里程计

Figure 02 集成了视觉惯性里程计（Visual-Inertial Odometry, VIO）系统，通过融合摄像头图像与惯性测量单元（Inertial Measurement Unit, IMU）的加速度计（Accelerometer）和陀螺仪（Gyroscope）数据，实现机器人在无 GPS 环境（如室内工厂）下的高精度自身位姿估计（Self-Pose Estimation）：

$$\hat{\mathbf{T}}_{WB} = \text{VIO}(\{\mathbf{I}_t\}, \{\mathbf{a}_t, \boldsymbol{\omega}_t\})$$

其中 \(\hat{\mathbf{T}}_{WB}\) 为机器人本体坐标系（Body Frame）相对于世界坐标系（World Frame）的估计位姿，\(\mathbf{I}_t\) 为图像序列，\(\mathbf{a}_t\) 为加速度测量，\(\boldsymbol{\omega}_t\) 为角速度测量。

准确的自身位姿估计是机器人在工厂内自主导航（Autonomous Navigation）的基础，也是全局操作规划（Global Manipulation Planning）中准确定位工作台和目标物体的前提条件。

### 本体感知

除外部环境感知外，Figure 02 还具备完整的本体感知（Proprioception）系统，包括：

- **关节编码器（Joint Encoder）**：精确测量每个关节的角度和角速度，为运动控制提供反馈
- **关节力矩传感器（Joint Torque Sensor）**：直接测量每个关节输出的力矩，用于力控和碰撞检测
- **足底压力传感器（Foot Pressure Sensor）**：测量双足与地面的接触力分布，为步态控制（Gait Control）和平衡控制（Balance Control）提供关键反馈

这些本体感知数据以高频（通常 500 Hz 以上）实时采集，与低层运动控制器紧密耦合，是机器人在动态环境中保持稳定平衡的基础。


## BMW 工厂合作

### 合作背景

宝马集团（BMW Group）斯帕坦堡工厂（Spartanburg Plant）位于美国南卡罗来纳州（South Carolina），是宝马全球最大的单一生产基地，主要生产 X 系列运动型多用途车（SUV）。该工厂年产能超过 40 万辆，雇用约 11,000 名工人。

宝马长期以来是工业机器人的大规模用户，其生产线上部署了大量传统工业机械臂（Industrial Robot Arm）。然而，传统工业机器人通常被固定在特定工位，仅能在高度结构化的环境中重复执行预编程动作，缺乏在工厂内自主移动和灵活执行多种任务的能力。人形机器人的引入被视为填补"最后一英里灵活性"（Last-Mile Flexibility）缺口的可能方案。

### 部署任务与场景

Figure 02 在 BMW 斯帕坦堡工厂的首期部署集中在车身车间（Body Shop），主要执行以下任务：

**冲压件搬运（Stamped Parts Handling）**：将金属冲压件（Stamped Sheet Metal Parts，即已成型的车身钣金件）从存放区取出，搬运至指定放置点或传送带入口。此类任务要求机器人能够识别形状各异的钣金件、规划无碰撞的抓取姿态，并在搬运过程中保持零件姿态稳定，防止划伤或变形。

**零件放置与对位（Part Placement and Alignment）**：将零件精确放置到工装夹具（Fixture）或装配工位上，对对位精度有一定要求。这类任务对机器人的手眼协调能力和末端执行器（End-Effector）控制精度要求较高。

### 概念验证的意义

BMW 部署案例作为概念验证（Proof of Concept, PoC），其意义超越了具体任务本身：

1. **安全合规验证**：证明人形机器人能够在有人类工人共同作业的工厂环境中，满足工业安全标准（如 ISO 10218 协作机器人安全要求），不对工人造成伤害风险
2. **任务可靠性验证**：在真实工业环境（非实验室洁净条件）下，验证机器人对噪声、振动、灰尘、光照变化等干扰因素的鲁棒性（Robustness）
3. **商业模式验证**：探索人形机器人以"机器人即服务"（Robot as a Service, RaaS）模式向制造商提供的商业可行性，为后续大规模商业化积累数据和经验

BMW 官方表示，此次合作是"评估人形机器人在宝马生产运营中长期潜力"的重要一步，并未承诺大规模采购，但为 Figure AI 提供了极具价值的真实工业场景验证机会。

### 工厂部署的技术挑战

将人形机器人部署到真实汽车工厂面临的技术挑战远比实验室演示复杂：

**环境感知鲁棒性**：工厂车身车间的光照条件复杂，存在强反光的金属表面、局部遮挡、烟雾（如焊接产生的烟尘）等干扰因素，对机器人视觉系统的鲁棒性（Robustness）提出严苛要求。Figure 02 的头部 6 摄像头配置提供了更宽的视野覆盖，并通过多视图融合（Multi-View Fusion）降低单一视角遮挡带来的感知盲区。

**人机协作安全（Human-Robot Collaboration Safety）**：斯帕坦堡工厂的车身车间同时有人类工人作业，Figure 02 必须满足 ISO/TS 15066（协作机器人安全技术规范）的要求，确保在与人类发生意外接触时能够立即停止运动或限制接触力。机器人配备了多层安全机制：视觉感知的人体检测（Human Detection）、力矩传感器驱动的碰撞检测（Collision Detection）以及冗余的安全停机电路（Safety Stop Circuit）。

**节拍匹配（Cycle Time Matching）**：汽车制造业的生产节拍（Takt Time）极为严格，通常以秒为单位管理。机器人的任务执行时间必须与整体生产节拍相匹配，否则将成为产线瓶颈。在试点阶段，Figure 02 的执行效率通常低于熟练工人，因此被安排在节拍要求相对宽松的上下料（Material Handling）工位，而非直接替代核心装配工位。

**可靠性与维护**：工厂环境要求机器人具备极高的运行可靠性（MTBF，Mean Time Between Failures），以及便于快速维修和更换零部件的可维护性设计（Maintainability Design）。这对 Figure AI 而言是重要的工程挑战，因为当前阶段的人形机器人样机与大规模量产的工业机械臂相比，在可靠性和耐用性方面仍存在明显差距。


## 参考资料

1. Figure AI, "Figure 02" 产品发布资料, 2024. https://www.figure.ai/
2. BMW Group, "BMW Group Plant Spartanburg 人形机器人试点项目" 新闻稿, 2024. https://www.press.bmwgroup.com/
3. [Figure 总览](figure.md)
4. [机器人抓取](../manipulation/grasping.md)
5. [力控与柔顺控制](../manipulation/force-control.md)
