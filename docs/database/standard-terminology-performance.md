# 机器人术语与性能评估标准

!!! note "引言"
    术语与性能标准是机器人标准体系中最基础的一层：它们本身不规定安全底线，却为其余所有标准提供共同语言。ISO 8373 统一了「机器人」「操作机」「工作空间」等概念的定义，ISO 9787 规定了坐标系与运动轴的命名约定，而 ISO 9283 则给出了位姿准确度、重复性与轨迹特性的标准测试方法——正是它使不同厂商标称的「重复定位精度 ±0.02 mm」具备可比性。本页面整理这两类标准的主要文件与适用范围。


## 分类 I. 术语公约

此类标准旨在制定机器人系统的通行术语与系统结构，对机器人系统已约定俗成的基本概念进行统一定义。统一的术语体系是跨团队、跨企业、跨国界协作的基础，也是技术文档与学术论文写作的重要依据。

| 标准名称        | 描述                                                                                                                 |
|-----------------|----------------------------------------------------------------------------------------------------------------------|
| ISO 8373-2012   | Robots and robotic devices – Vocabulary                                                                              |
| ISO 9787-2013   | Robots and robotic devices – Coordinate systems and motion nomenciatures                                             |
| ISO 19649-2017  | Mobile robots – Vocabulary                                                                                           |
| ASTM F3200-2018 | Standard Terminology for Driverless Automatic Guided Industrial Vehicles                                             |
| VDI 2510-2005   | Automated Guided Vehicle Systems (AGVS)                                                                              |
| VDI 2510-3-2017 | Automated guided vehicle systems (AGVS) Interfaces to infrastructure and peripherals                                 |
| VDI 2710-3-2014 | Applications of simulation for automated guided vehicle systems (AGVS)                                               |
| VDI 4451-2-2000 | Compatibility of Automated Guide Vehicle Systems (AGVS) Power supply and charging technology                         |
| VDI 4451-3-1998 | Compatibility of automated guided vehicle systems (AGVS) Driving and steering motors                                 |
| VDI 4451-4-1998 | CompatIbility of automated guided vehicle systems (AGVS) Open control system for automated guided vehicles (AGV)     |
| VDI 4451-5-2005 | Compatibility of Automated Guided Vehicle Systems (AGVS) Interface between command initiator and AGVS control system |
| VDI 4451-6-2003 | Compatibility of automated guided vehicle systems (AGVS) Sensor systems for navigation and control                   |
| VDI 4451-7-2005 | Compatibility of Automated Guided Vehicle Systems (AGVS) AGVS guidance control system                                |
| REP 103         | Standard Units of Measure and Coordinate Conventions                                                                 |
| REP 105         | Coordinate Frames for Mobile Platforms                                                                               |
| REP 120         | Coordinate Frames for Humanoid Robots                                                                                |

### ISO 8373：机器人通用词汇

ISO 8373 是机器人领域的基础性词汇标准，定义了约 200 个核心术语，覆盖机器人系统的各个层面。该标准对以下概念给出了精确的技术定义：

- **机械臂**（Manipulator）：由一系列连杆（Link）通过关节（Joint）连接而成的机构，用于抓取或移动物体或工具。
- **末端执行器**（End-Effector）：安装在机械臂末端、直接与工作对象交互的装置，例如夹爪（Gripper）、焊枪（Welding Torch）或真空吸盘（Suction Cup）。
- **工作空间**（Workspace）：机器人末端执行器所能到达的空间范围，分为最大工作空间（Maximum Workspace）和限制工作空间（Restricted Workspace）。
- **自由度**（Degree of Freedom，DOF）：描述机器人运动能力的参数，通常指独立运动关节的数量。
- **重复定位精度**（Pose Repeatability）：机器人反复到达同一目标位姿时的离散程度，是衡量工业机器人性能的核心指标之一。
- **协作机器人**（Collaborative Robot，Cobot）：设计用于与人在共同工作空间内协同工作的工业机器人。

掌握 ISO 8373 定义的术语是阅读其他机器人标准的前提，也是与国际客户和合作伙伴进行技术交流的基础。

### ISO 9787：坐标系与运动命名

ISO 9787 定义了机器人系统中各类坐标系（Coordinate Frame）的命名规则和方向约定，是机器人运动学建模与控制程序开发的重要参考。标准中定义的主要坐标系包括：

- **基坐标系**（Base Frame）：固定在机器人安装底座上的参考坐标系，是描述机器人运动的基准。
- **关节坐标系**（Joint Frame）：与每个关节相关联的局部坐标系，用于描述关节的转动或平移。
- **工具坐标系**（Tool Frame / Tool Center Point，TCP）：固定在末端执行器上的坐标系，原点通常位于工具的作用点。
- **工件坐标系**（Work Object Frame）：固定在工件或工作台上的坐标系，便于描述机器人相对于工件的运动。
- **世界坐标系**（World Frame）：整个机器人系统的全局参考坐标系，通常与基坐标系重合或相差一个已知的固定变换。

ISO 9787 采用右手坐标系（Right-Hand Coordinate System）约定，X 轴指向前方，Y 轴指向左方，Z 轴指向上方，这与 ROS REP 103 的约定一致。

### ROS REP 系列：开源社区的事实标准

ROS 的 REP 文件是开源机器人开发社区的重要规范，其中三个最基础的 REP 分别规定了单位、坐标系和人形机器人约定：

**REP 103 - 标准测量单位与坐标约定**：规定 ROS 系统中使用国际单位制（SI，Système International d'unités），例如长度单位为米（m）、角度单位为弧度（rad）、线速度单位为米每秒（m/s）、角速度单位为弧度每秒（rad/s）。坐标轴方向遵循右手定则（Right-Hand Rule）：X 轴向前，Y 轴向左，Z 轴向上。旋转正方向为右手螺旋方向（从 Z 轴正方向俯视为逆时针）。

**REP 105 - 移动平台坐标系**：为移动机器人定义了一套标准的坐标系命名体系，包括 `base_link`（机器人本体坐标系）、`odom`（里程计坐标系，局部连续但存在漂移）、`map`（全局地图坐标系，不连续但全局一致）和 `earth`（地球坐标系，用于多机器人或 GPS 场景）。这套坐标系层级是所有 ROS 移动机器人导航包的基础。

**REP 120 - 人形机器人坐标系**：在 REP 105 的基础上，为双足人形机器人（Humanoid Robot）额外定义了 `base_footprint`（机器人在地面的投影点）、`l_sole` / `r_sole`（左右脚掌坐标系）、`l_wrist` / `r_wrist`（左右手腕坐标系）等特定坐标系，为人形机器人的步态控制和全身运动规划提供了统一的参考框架。


## 分类 II. 性能评估

此类标准旨在为各类机器人系统提供性能评估方法上的实用性建议，为研发者提供最基础的可行性实验方法。性能测试的标准化使得不同实验室、不同时间的测试结果具有可比性，是产品迭代和技术比较的客观依据。

| 标准名称         | 描述                                                                                                                          |
|------------------|-------------------------------------------------------------------------------------------------------------------------------|
| ISO 9283-1998    | Manipulating industrial robots – Performance criteria and related test methods                                                |
| ISO 18646-1-2016 | Robotics — Performance criteria and related test methods for service robots — Part 1 Locomotion for wheeled robots            |
| ISO 18646-2-2019 | Robotics — Performance criteria and related test methods for service robots — Part 2 Navigation                               |
| ASTM F3218-2017  | Standard Practice for Recording Environmental Effects for Utilization with A-UGV Test Methods                                 |
| ASTM F3244-2017  | Standard Test Method for Navigation Defined Area ASTM F3327-2018 Standard Practice for Recording the A-UGV Test Configuration |
| ASTM             | International Autonomous Industrial Vehicles From the Laboratory to the Factory Floor                                         |
| NISTIR 8168      | Guideline for Automatic Guided Vehicle Calibration                                                                            |
| VDI 2710-1-2007  | Interdisciplinary design of automated guided vehicle systems (AGVS) — Decision criteria for the choice of a conveyor system   |
| VDI 2710-2-2008  | AGVS check list Planning support for operators and manufacturers of automated guided vehicle-systems (AGVS)                   |
| VDI 2710-4-2011  | Evaluation of economic efficiency of Automated Guided Vehicles Systems (AGVS)                                                 |
| VDI 2710-5-2013  | Acceptance specification for automated guided vehicle systems (AGVS)                                                          |

### ISO 9283：工业机器人性能准则与测试方法

ISO 9283 是评估工业机械臂（Industrial Manipulator）性能的核心标准，定义了一系列可量化的性能指标及对应的测试方法。该标准被广泛应用于工业机器人的选型采购、出厂检验和竞品对比。主要测试项目包括：

**位姿准确度（Pose Accuracy）**：机器人实际到达位姿与指令位姿之间的偏差。测试方法是让机器人以规定速度重复到达同一目标位姿 30 次，记录每次实际位姿，计算平均偏差。

**位姿重复性（Pose Repeatability）**：机器人反复到达同一目标位姿时，实际位姿的离散程度（即精度的稳定性）。用统计方法计算 30 次测试结果的分散范围。重复性是判断机器人是否适合精密装配任务的关键指标，工业机器人的重复定位精度通常在 ±0.01 mm 到 ±0.1 mm 量级。

**路径准确度（Path Accuracy）**：机器人沿指令路径（通常为直线或圆弧）运动时，实际轨迹与指令轨迹之间的偏差。该指标对焊接、切割、喷涂等连续路径作业至关重要。

**ISO 循环时间（ISO Cycle Time）**：标准规定了一个特定的测试路径和负载条件，在此条件下完成一个完整运动循环所需的时间，用于标准化比较不同机器人的速度性能。

**速度准确度（Velocity Accuracy）**和**加速度稳定性**也在标准中有所涉及，用于评估机器人在动态运动中的控制精度。

### ASTM F3218 / F3244：自主工业车辆测试方法

针对在仓储物流环境中运行的自主移动机器人（Autonomous Mobile Robot，AMR）和自主工业车辆（Autonomous Industrial Vehicle，A-IV），ASTM F3218 和 F3244 提供了系统化的测试框架：

**ASTM F3218** 规定了在进行 A-UGV（Autonomous Unmanned Ground Vehicle，自主无人地面车辆）测试时，如何记录和报告环境条件，包括地面类型、光照条件、温湿度、障碍物密度等，确保测试结果的可重复性和可解释性。

**ASTM F3244** 定义了在特定导航测试区域内评估 A-UGV 导航性能的标准方法，包括路径跟踪精度、避障响应时间、定位漂移等指标的量化测试流程，适用于仓库、工厂等结构化室内环境。

这两个标准与 NISTIR 8168（AGV 标定指南）共同构成了移动机器人性能评估的北美参考体系，是自主物流机器人进入北美市场的重要技术依据。


## 参考资料

1. ISO. *ISO Online Browsing Platform*. https://www.iso.org/obp
2. ISO 8373:2021, *Robotics — Vocabulary*.
3. ISO 9283:1998, *Manipulating industrial robots — Performance criteria and related test methods*.
4. [机器人领域行业标准总览](standard.md)
