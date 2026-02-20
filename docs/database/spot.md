# Boston Dynamics Spot

!!! note "引言"
    Spot 是由美国 Boston Dynamics 公司研发的四足机器人，于 2019 年面向企业客户正式商业化，是全球首款大规模量产销售的电动四足机器人。凭借出色的地形适应能力、丰富的载荷接口和开放的软件开发工具包（Software Development Kit, SDK），Spot 已在工业巡检、建筑测量、危险环境探查等领域得到广泛应用，成为四足机器人商业化的标志性产品。


## 发展历程

Spot 的诞生并非一蹴而就，其背后有一条清晰的技术演进脉络。

- **2005 年**：Boston Dynamics 在美国国防高级研究计划局（Defense Advanced Research Projects Agency, DARPA）的资助下，推出四足机器人 BigDog。BigDog 采用液压驱动，体重约 109 kg，能在雪地、碎石等复杂地形行走，并可承载约 154 kg 的负荷。这是 Spot 系列的直接前身，奠定了公司在腿足机器人领域的核心技术积累。
- **2012 年**：推出 LS3（Legged Squad Support System），作为 BigDog 的军用衍生版，旨在为士兵分担装备负重，但因噪音过大未被美国军方采购。
- **2015 年**：Boston Dynamics 发布 Spot 初代原型（当时内部称为 Spot Classic），采用液压驱动，体重约 73 kg，展示了更紧凑的四足行走能力，被用于火山口探查等实验性任务。
- **2016 年**：推出全电动四足机器人 SpotMini，体重仅约 25–30 kg（带机械臂版本），面向家用和办公场景演示，展示了开门、上下楼梯、抓取物品等能力。SpotMini 是公司首款无液压驱动的四足机器人，标志着技术路线从液压向电动的战略转型。
- **2019 年 6 月**：正式对企业客户开放销售，商业版 Spot 发布，售价约 74,500 美元。这是公司首次将四足机器人推向商业市场，初期客户包括石油天然气公司、建筑企业及科研机构。
- **2020 年**：新冠疫情（COVID-19）期间，Spot 被部署至美国波士顿布里格姆妇女医院，协助医护人员对疑似感染患者进行远程问诊和生命体征筛查，引发广泛关注。同年新加坡将 Spot 投入公园巡逻，用于提醒市民保持社交距离。
- **2021 年**：发布 Spot Arm（机械臂模块），使 Spot 具备了操纵门把手、阀门等物体的能力，大幅扩展了应用场景。同年，Shell 等能源公司将 Spot 部署至海上油气平台，执行仪表读数、热成像巡检等任务。美国陆军也启动了针对 Spot 的集成视觉增强系统（Integrated Visual Augmentation System, IVAS）评估项目。
- **2022–2023 年**：推出 Spot Enterprise 企业版，集成更强的机载计算能力和更完善的云端管理功能，面向大规模工业部署场景。
- **2024 年**：发布第三代 Spot（Spot 3），在感知系统、关节效率、有效载荷和续航时间等方面全面升级，进一步巩固其在商业四足机器人市场的领先地位。


## 技术规格

### 整机参数

| 参数 | 规格 |
|------|------|
| 站立肩高 | 约 0.6 m |
| 整机长度 | 约 1.1 m |
| 整机宽度 | 约 0.5 m |
| 整机重量 | 约 32 kg |
| 有效载荷 | 最高 14 kg |
| 最高行走速度 | 1.6 m/s |
| 续航时间 | 约 90 分钟（标准负载） |
| 防护等级 | IP54（防尘防泼溅） |
| 工作温度 | -20°C ~ 45°C |

### 运动系统

| 参数 | 规格 |
|------|------|
| 腿部数量 | 4 条 |
| 每条腿自由度（Degrees of Freedom, DoF） | 3 DoF（髋关节俯仰、髋关节偏航、膝关节） |
| 总自由度 | 12 DoF |
| 驱动方式 | 全电动，自研高扭矩密度关节电机 |
| 爬坡能力 | 最大 30° 坡面 |
| 台阶适应 | 可攀爬高度约 30 cm 的台阶 |

### 感知系统

| 传感器类型 | 配置 |
|------|------|
| 立体视觉相机（Stereo Camera） | 前置 × 1 对、后置 × 1 对、左右侧 × 各 1 对，共 5 组 |
| 飞行时间深度传感器（Time-of-Flight, ToF） | 前置 1 个，用于近距离障碍物检测 |
| 惯性测量单元（Inertial Measurement Unit, IMU） | 机身内置，用于姿态估计 |
| 可选载荷传感器 | Spot CAM 全景摄像头、FLIR 热成像仪、激光雷达（LiDAR）等 |


## 软件平台

### Spot SDK

Boston Dynamics 提供开源的 Spot SDK（主要支持 Python，同时提供 C++ 绑定），覆盖以下核心功能：

- **图像获取**：从机身各立体相机和深度传感器获取图像流
- **运动控制**：发送速度指令、姿态控制、站立/坐下等基础指令
- **任务自动化**：通过 Mission API 编排复杂的多步骤自动化任务
- **状态监控**：读取关节扭矩、电池电量、本体位姿等实时状态

以下为使用 Spot SDK 连接机器人并创建运动控制客户端的最小示例：

```python
import bosdyn.client
from bosdyn.client.robot_command import RobotCommandClient

# 初始化 SDK 并连接到机器人
sdk = bosdyn.client.create_standard_sdk('SpotClient')
robot = sdk.create_robot('ROBOT_IP')
robot.authenticate('admin', 'password')

# 获取运动控制客户端
command_client = robot.ensure_client(RobotCommandClient.default_service_name)
```

### 自主导航

- **GraphNav**：Spot 的地图构建与自主导航框架（Graph Navigation）。操作人员首先手动引导 Spot 遍历目标区域，系统同步录制传感器数据并构建拓扑-度量混合地图（Topometric Map）；之后 Spot 可在地图范围内完全自主导航，无需人工干预。
- **Autowalk**：基于 GraphNav 的录制-回放巡检功能。操作人员遥控 Spot 走一遍巡检路线并记录各检查点动作（如拍照、读取仪表），此后 Spot 可按计划自动重复执行，适用于定期巡检场景。

### 机载计算与载荷接口

- **Spot Core**：可选的机载计算模块，搭载 Intel NUC 主机，运行 Ubuntu Linux，支持第三方应用直接在 Spot 机身上部署，无需外部计算资源。
- **Spot CAM**：360° 全景摄像头载荷，提供球形视野覆盖，并支持双向音频通信，常用于远程巡逻和安防场景。
- **有效载荷接口**：机身背部提供标准化机械接口和以太网/电源接口，支持激光雷达、热成像仪、机械臂等第三方载荷快速挂载。

### Scout 机队管理平台

Scout 是 Boston Dynamics 提供的云端机队管理平台，支持多台 Spot 同时监控、任务调度、历史数据查询和远程操控，面向大规模工业部署场景。


## 行业应用

### 工业巡检

工业巡检是 Spot 最成熟的商业应用场景。在石油天然气（Oil & Gas）行业，Spot 被部署至海上钻井平台和陆地炼化设施，定期执行仪表板读数、管道热成像扫描、设备异常声音检测等任务，替代人工进入潜在危险区域。美国壳牌公司（Shell）、英国石油公司（BP）均已将 Spot 引入日常运营。在电力行业，Spot 配备热成像仪对变电站设备进行巡检，识别过热节点，提前预防故障。

### 建筑测量与 BIM 建模

Spot 与瑞士天宝公司（Trimble）和徕卡公司（Leica Geosystems）合作，背载三维激光扫描仪（3D Laser Scanner）在建筑工地自主行走扫描，生成高精度点云数据，用于构建建筑信息模型（Building Information Model, BIM）。与人工测量相比，Spot 可在夜间或危险区域持续作业，大幅提升测量效率和覆盖密度。

### 危险环境探查

在核电站、化工厂检修及自然灾害现场，Spot 可代替人员进入放射性污染区或结构不稳定的建筑，搭载辐射剂量仪、气体传感器等专用仪器，实时回传现场数据。Spot 的 IP54 防护等级和宽温度工作范围使其能够在恶劣条件下稳定运行。

### 公共安全与安保

2020 年，新加坡国家公园局（National Parks Board）将 Spot 部署至公园，用于在疫情期间广播社交距离提示并估算人群密度。此后，多个机场和公共设施开始评估将 Spot 用于夜间安保巡逻。这一应用也引发了公众对机器人执法与隐私保护的讨论，美国部分城市警察局曾因争议暂停或终止相关合同。

### 科研与学术平台

麻省理工学院（Massachusetts Institute of Technology, MIT）、苏黎世联邦理工学院（ETH Zürich）、卡内基梅隆大学（Carnegie Mellon University, CMU）等顶尖高校使用 Spot 作为研究平台，开展四足步态研究、强化学习（Reinforcement Learning）运动策略训练、人机交互以及多机器人协作等方向的实验。开放的 SDK 和标准化的载荷接口降低了二次开发门槛，使 Spot 成为学术界最广泛使用的商业四足平台之一。


## 与 ROS/ROS 2 集成

### spot_ros（ROS 1 社区包）

`spot_ros` 是由社区维护的 ROS 1（Robot Operating System）驱动包，将 Spot SDK 封装为标准 ROS 话题和服务：

- **发布话题**：里程计（`/odometry`）、关节状态（`/joint_states`）、各相机图像（`/spot/camera/*/image_raw`）、点云（`/spot/depth/*/points`）
- **订阅话题**：速度指令（`/cmd_vel`）用于控制机器人移动
- **服务**：站立（`/stand`）、坐下（`/sit`）、自我归位等控制服务

### spot_ros2（ROS 2 版本）

`spot_ros2` 是对应的 ROS 2 驱动包，支持 ROS 2 Humble 及更高版本，充分利用 ROS 2 的 DDS（Data Distribution Service）通信机制，提供更低的延迟和更好的多机支持。

### 集成注意事项

- **时间同步**：Spot SDK 使用机器人本体时钟，接入 ROS 系统时需通过 `robot.time_sync` 接口校准时间偏差，否则传感器数据时间戳会出现漂移，影响 SLAM（Simultaneous Localization and Mapping）和传感器融合精度。
- **坐标系对齐**：Spot 内部坐标系（机体坐标系、视觉里程计坐标系、地图坐标系）需正确映射到 ROS 的 `base_link`、`odom`、`map` 坐标系，避免导航算法出现方向错误。
- **带宽限制**：同时流式传输多路高分辨率相机图像时，需注意 WiFi 带宽限制，建议使用压缩图像话题（`image_transport` 的 `compressed` 插件）降低传输负荷。


## 竞品对比

| 指标 | Boston Dynamics Spot | 宇树 B2（Unitree B2） | ANYbotics ANYmal C |
|------|------|------|------|
| 整机重量 | 约 32 kg | 约 60 kg | 约 50 kg |
| 有效载荷 | 14 kg | 10 kg | 10 kg |
| 续航时间 | 约 90 分钟 | 约 120 分钟 | 约 90 分钟 |
| 最高速度 | 1.6 m/s | 1.5 m/s（持续）/ 6.0 m/s（峰值） | 1.0 m/s |
| 开放 SDK | 是（Python/C++，开源） | 是（Python/C++，开源） | 是（C++，部分开源） |
| 防护等级 | IP54 | IP67 | IP67 |
| 价格定位 | 高（约 7.5 万美元起） | 中（面向商业市场，低于 Spot） | 高（工业专用定制） |
| 主要市场 | 北美、欧洲、全球 | 全球（中国品牌出海） | 欧洲工业 |


## 参考资料

1. [Spot](https://bostondynamics.com/products/spot/), Boston Dynamics 官网
2. [Spot SDK Documentation](https://dev.bostondynamics.com/), Boston Dynamics Developer Portal
3. [spot_ros](https://github.com/heuristicus/spot_ros), GitHub 社区仓库
4. [spot_ros2](https://github.com/bdaiinstitute/spot_ros2), GitHub（Boston Dynamics AI Institute）
5. [Spot (robot)](https://en.wikipedia.org/wiki/Spot_(robot)), Wikipedia
6. [BigDog](https://en.wikipedia.org/wiki/BigDog), Wikipedia
7. [How Boston Dynamics' Spot Robot Is Being Used in Real Industries](https://www.bostondynamics.com/resources/case-studies), Boston Dynamics Case Studies
8. [Singapore uses Spot to enforce social distancing](https://www.theguardian.com/world/2020/may/08/singapore-police-robot-dog-patrol-social-distancing), The Guardian, 2020
9. [Spot at Work: Oil and Gas](https://bostondynamics.com/resources/case-studies), Boston Dynamics
10. [GraphNav Overview](https://dev.bostondynamics.com/docs/concepts/autonomy/graphnav_overview), Boston Dynamics Developer Documentation

