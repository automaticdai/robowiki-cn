# 机器人领域行业标准

!!! note "引言"
    行业标准（Industry Standard）是机器人工程师日常工作中不可回避的重要参考依据。标准的存在解决了多个核心问题：首先是**互操作性**（Interoperability），不同厂商生产的机器人部件、传感器和控制器需要能够协同工作，统一的接口与术语定义使之成为可能；其次是**安全认证**（Safety Certification），工业机器人、服务机器人在进入市场前必须通过相应安全标准的认证，否则无法合法销售；第三是**采购要求**（Procurement Requirements），大型制造企业在招标时通常明确要求供应商的产品符合特定标准；第四是**监管合规**（Regulatory Compliance），欧盟机械指令（Machinery Directive）、美国职业安全与健康管理局（OSHA）法规等均直接引用机器人安全标准。

    对于机器人工程师而言，标准的影响贯穿整个产品开发周期。在设计阶段，坐标系定义（ISO 9787）和术语规范（ISO 8373）确保团队内部沟通无歧义；在测试阶段，性能评估标准（ISO 9283）提供了可重复的测试方法，使不同实验室的测试结果具有可比性；在产品上市前，安全标准（ISO 10218、ISO 13482）规定了必须满足的最低安全要求，协作机器人标准（ISO/TS 15066）则定义了人机共工的具体边界条件。忽视这些标准，不仅可能导致产品认证失败，还可能引发严重的安全事故和法律责任。

    本列表整理了国际上常用的机器人相关标准，涵盖 ISO、IEC、ASTM、VDI 等主要标准化机构发布的文件，以及 ROS REP 等开源平台的约定规范。列表按照对于标准的需求类型划分，而非机器人系统的种类或标准的制定机构。


## 标准化机构介绍

机器人领域的标准由多个国际和地区性机构制定，了解这些机构有助于工程师找到权威的参考文件。

### ISO/TC 299（国际标准化组织机器人技术委员会）

国际标准化组织（International Organization for Standardization，ISO）是全球最具影响力的标准制定机构。其下设的第 299 技术委员会（Technical Committee 299）专门负责机器人领域的标准制定工作，前身为 ISO/TC 184/SC 2。TC 299 的工作范围涵盖工业机器人、服务机器人、协作机器人、移动机器人等各类机器人系统。ISO 标准通常需要成员国投票通过后正式发布，代表了全球范围内的最广泛共识。

ISO 标准可通过 ISO 官网（iso.org）购买，部分标准也可通过各国国家标准机构获取。在中国，ISO 标准往往会被等同采用（等同采用标志为"IDT"）或修改采用（"MOD"）转化为 GB/T 国家标准，由国家市场监督管理总局负责发布。

### IEC（国际电工委员会）

国际电工委员会（International Electrotechnical Commission，IEC）负责电气、电子和相关技术领域的国际标准化工作。在机器人领域，IEC 的贡献主要集中在功能安全（Functional Safety）、工业通信协议和电气安全等方面，例如 IEC 61508（功能安全基础标准）、IEC 62541（OPC-UA 通信标准）和 IEC 61158（工业以太网通信）。IEC 与 ISO 在机器人领域存在密切合作，部分标准以 ISO/IEC 联合发布。

### ASTM International

ASTM International 前身为美国材料与测试协会（American Society for Testing and Materials），现为全球性标准组织。其 F45 委员会专注于无人驾驶系统（Driverless Automatic Guided Vehicles，DAGV）和自主工业车辆（Autonomous Industrial Vehicles，A-IVs）的标准制定。ASTM 标准在北美制造业中被广泛采用，尤其是仓储物流和 AGV（Automated Guided Vehicle，自动导引车）领域。

### VDI（德国工程师协会）

德国工程师协会（Verein Deutscher Ingenieure，VDI）是德国最具影响力的工程技术学会，发布的 VDI 指南（Richtlinie）在自动导引车系统（AGVS）领域具有重要地位，尤其在欧洲制造业中被广泛参考。VDI 标准侧重于工程实践指导，许多 AGV 系统的设计规范、接口定义和经济效益评估方法均源自 VDI 系列文件。

### ROS REP（ROS 增强提案）

ROS 增强提案（ROS Enhancement Proposal，REP）是 ROS（Robot Operating System，机器人操作系统）开源社区的技术规范文件，类似于 Python 的 PEP 或 IETF 的 RFC。REP 定义了 ROS 生态系统内的约定俗成，包括坐标系方向、单位规范、话题命名等。虽然 REP 不具有法律约束力，但在 ROS/ROS 2 开发社区中具有极高的权威性，遵循 REP 规范是保证代码与第三方包兼容的基本前提。


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


## 分类 III. 系统安全

此类标准旨在为各类机器人系统需要满足的最基本安全性功能提供强制性规范，其中也包含对于安全性功能性能测验的实用性方法。安全标准的遵从通常是产品合法上市的硬性要求，而非可选建议。

| 标准名称            | 描述                                                                                                              |
|---------------------|---------------------------------------------------------------------------------------------------------------------------|
| ISO 10218-1-2011    | Robots and robotic devices – Safety requirements for industrial robots — Part 1 Robots                                    |
| ISO 10218-2-2011    | Robots and robotic devices – Safety requirements for industrial robots — Part 2 Robot systems and integration             |
| ISO 13482-2014      | Robots and robotic devices – Safety requirements for personal care robots                                                 |
| ISO/TR 20218-1-2018 | Robotics — Safety design for industrial robot systems — Part 1 End-effectors                                              |
| ISO/TR 20218-2-2018 | Robotics — Safety design for industrial robot systems — Part 2 Manual load unload stations                                |
| ISO/TR 23482-1-2019 | Robotics — Application of ISO 13482 — Part 1 Safety-related test methods                                                 |
| ISO/TR 23482-2-2019 | Robotics — Application of ISO 13482 — Part 2 Application guidelines                                                      |
| ISO/TS 15066-2016   | Robots and robotic devices — Collaborative robots                                                                         |
| BS-EN 1525-1998     | Safety of industrial trucks Driverless trucks and their systems                                                           |
| ISO 3691-4-2019     | Industrial trucks – Safety requirements and verification — Part 4 Driverless industrial trucks and their systems          |
| ANSI B56.5-2012     | Safety Standard For Driverless Automatic Guided Industrial Vehicles And Automated Functions Of Manned Industrial Vehicles |
| VDI 2510-2-2013     | Automated guided vehicle systems (AGVS) — Safety of AGVS                                                                  |

### ISO 10218：工业机器人安全要求

ISO 10218 分为两个部分，共同构成工业机器人（Industrial Robot）最核心的安全法规框架，欧盟机械指令（2006/42/EC）直接将其列为协调标准（Harmonized Standard）：

**ISO 10218-1** 规定了机器人本体（Robot Unit）的安全设计要求，涵盖机械安全（如关节限位、防意外启动）、电气安全、控制系统安全和软件安全等方面。标准要求机器人必须配备安全额定的停止功能，并根据 ISO 13849 或 IEC 62061 达到相应的性能等级（Performance Level，PL）或安全完整性等级（Safety Integrity Level，SIL）。

**ISO 10218-2** 规定了机器人系统集成（Robot System Integration）和机器人单元（Robot Cell）的安全要求，即在机器人本体之外，围栏、安全光帘（Safety Light Curtain）、互锁装置（Interlocking Device）和操作员界面等外围设施的安全设计规范。这部分主要面向系统集成商和最终用户。

2021 年，ISO 10218 经历了重大修订，新版本（ISO 10218:2021）在协作机器人和移动工业机器人等方面做了大幅更新，并与 ISO/TS 15066 进行了更好的整合。

### ISO 13482：个人护理机器人安全要求

ISO 13482 是首个专门针对服务机器人（Service Robot）的国际安全标准，于 2014 年发布，将个人护理机器人（Personal Care Robot）分为三种类型：

- **A 型（移动助行机器人，Mobile Servant Robot）**：能够在人的周围自主移动并执行任务的机器人，例如家用清洁机器人、送餐机器人。标准要求此类机器人具备可靠的碰撞检测（Collision Detection）与紧急停止（Emergency Stop）功能。
- **B 型（身体辅助机器人，Physical Assistant Robot）**：与人体直接接触，辅助人的运动功能的机器人，例如外骨骼（Exoskeleton）、康复训练机器人。此类机器人的安全要求最为严格，需要对人机接触力（Contact Force）和运动范围加以严格限制。
- **C 型（乘人机器人，Person Carrier Robot）**：搭载人员移动的机器人，例如轮椅机器人、代步机器人。此类机器人需满足乘员防坠落、速度控制等额外安全要求。

ISO/TR 23482-1 和 ISO/TR 23482-2 作为 ISO 13482 的应用指南，分别提供了安全相关测试方法和应用场景指导，帮助工程师将抽象的安全要求转化为具体可操作的测试程序。

### ISO/TS 15066：协作机器人安全

ISO/TS 15066 是协作机器人（Collaborative Robot，Cobot）领域最重要的技术规范，定义了工业机器人与人在共同工作空间内协同工作时必须满足的安全要求。标准定义了四种协作工作模式（Collaborative Operation Mode）：

**安全额定监控停止（Safety-Rated Monitored Stop，SRMS）**：当人进入协作工作区时，机器人停止运动并保持停止状态，人离开后机器人自动恢复。适用于机器人运动速度快但人机交互频率低的场景。

**手动引导（Hand Guiding，HG）**：操作员通过手动引导装置（Hand Guiding Device）直接引导机器人末端执行器运动，机器人跟随操作员施加的力进行运动。适用于示教编程（Teach Programming）和精细装配场景。

**速度和间距监控（Speed and Separation Monitoring，SSM）**：通过传感器（如激光雷达、视觉相机）持续检测人与机器人之间的距离，并根据距离动态调整机器人运动速度，距离越近速度越慢，确保任何情况下机器人都能在人到达其最近点之前停止。

**功率和力限制（Power and Force Limiting，PFL）**：机器人本身具备力控制能力，将与人体接触时的力和压力限制在人体可耐受的安全阈值以内。ISO/TS 15066 附录 A 提供了不同身体部位（手、手臂、躯干、头部等）对应的最大允许接触力（Maximum Permissible Force）和压力数值表，这是 PFL 模式下设计协作机器人的定量依据。

PFL 模式是目前大多数商用协作机器人（如 Universal Robots UR 系列、FANUC CR 系列）主要采用的协作模式，其核心优势在于无需额外的安全围栏，可实现真正的人机共工（Human-Robot Collaboration，HRC）。


## 分类 IV. 功能安全标准

功能安全（Functional Safety）是指系统在设备或外部异常（包括硬件故障、软件错误、电磁干扰等）情况下仍能正确执行安全功能的能力。功能安全标准通过引入安全完整性等级（SIL）或性能等级（PL）的概念，为机器人控制系统的安全设计提供了量化框架。

| 标准名称      | 描述                                                                             |
|---------------|----------------------------------------------------------------------------------|
| IEC 61508     | 电气/电子/可编程电子安全相关系统的功能安全 (SIL 1-4)                             |
| ISO 26262     | 道路车辆功能安全 (ASIL A-D)，自动驾驶机器人及轮式移动机器人适用                  |
| IEC 62061     | 机械安全 - 机械控制系统功能安全                                                   |
| EN ISO 13849-1| 机械安全 - 控制系统安全相关部件，性能等级 (PL a-e)                                |

### IEC 61508：功能安全基础标准

IEC 61508 是电气、电子和可编程电子（Electrical/Electronic/Programmable Electronic，E/E/PE）安全相关系统的基础功能安全标准，也是 ISO 26262、IEC 62061 等行业专用功能安全标准的上层框架。标准将安全完整性等级（Safety Integrity Level，SIL）分为四级（SIL 1 至 SIL 4），数字越高代表安全要求越严格，对应系统在一定时间内发生危险失效的概率越低。

对于机器人系统而言，IEC 61508 主要适用于安全控制器（Safety Controller）、安全传感器（Safety Sensor，如安全激光雷达 Safety Laser Scanner）和安全执行器的设计与认证。大多数工业机器人的安全停止功能需要满足 SIL 2 或 PLd 的要求。

### ISO 26262：道路车辆功能安全

ISO 26262 是专门面向道路车辆（Road Vehicle）的功能安全标准，将汽车安全完整性等级（Automotive Safety Integrity Level，ASIL）分为 A、B、C、D 四级（ASIL D 要求最严格）以及 QM（Quality Management，质量管理，无安全要求）。

虽然 ISO 26262 的直接目标是汽车行业，但对于自动驾驶底盘（Autonomous Driving Chassis）、轮式移动机器人（Wheeled Mobile Robot）以及使用车规级（Automotive-Grade）硬件的机器人系统，ISO 26262 的方法论和工具链同样具有重要参考价值。随着人形机器人和自动驾驶技术的融合，预计未来将出现专门针对轮式/足式移动机器人的 ASIL 类功能安全标准。

### EN ISO 13849-1：机械安全性能等级

EN ISO 13849-1 专注于机械控制系统中安全相关部件（Safety-Related Parts of Control System，SRP/CS）的设计和评估，引入了性能等级（Performance Level，PL）的概念，分为 PLa（最低）到 PLe（最高）五级。与 SIL 不同，PL 的计算综合考虑了结构类别（Category，反映架构冗余度）、平均危险失效间隔时间（Mean Time to Dangerous Failure，MTTFd）和诊断覆盖率（Diagnostic Coverage，DC）三个维度。

在机器人领域，安全停止功能（Safety Stop Function）、防意外启动功能（Prevention of Unexpected Start-up）和使能装置（Enabling Device）通常需要达到 PLd 或 PLe，这直接影响安全控制器和传感器的选型。


## 分类 V. 通信与接口标准

机器人系统内部各模块之间，以及机器人与外部系统之间的通信需要遵循标准化协议，以保证实时性、可靠性和互操作性。以下是机器人领域常用的通信与接口标准。

| 标准名称             | 描述                                                                    |
|----------------------|-------------------------------------------------------------------------|
| EtherCAT (IEC 61158) | 实时工业以太网通信协议，广泛用于机器人关节伺服控制                       |
| CANopen (CiA 301)    | 基于 CAN 总线的设备通信协议，用于关节控制器和传感器接口                  |
| OPC-UA (IEC 62541)   | 工业物联网数据通信标准，用于机器人与 MES/SCADA 系统的数据交换及数字孪生  |
| ROS REP-2000         | ROS 2 目标平台和策略，定义各 ROS 2 发行版支持的操作系统和硬件平台        |
| MTConnect (ANSI/MTC1)| 制造设备数据采集标准，用于机器人与生产线监控系统的集成                    |
| MQTT (ISO/IEC 20922) | 轻量级消息传输协议，用于机器人云端连接和远程监控                          |

### EtherCAT：实时工业以太网

EtherCAT（Ethernet for Control Automation Technology）由德国倍福（Beckhoff Automation）开发，现已被纳入 IEC 61158 工业现场总线标准体系。EtherCAT 采用"飞行处理"（Processing on the Fly）技术，使得数据帧在经过每个从站节点时无需等待完整接收即可被处理并转发，实现了极低的通信延迟（通常在 1 ms 以内）和高度的时间确定性（Determinism）。

在机器人领域，EtherCAT 是目前主流工业机器人和协作机器人关节伺服驱动器（Servo Driver）通信的事实标准。使用 EtherCAT 的代表性机器人系统包括 KUKA、ABB 等品牌的工业机器人控制器，以及采用 EtherCAT 主站（Master）的开源机器人控制框架（如 ethercat_master、SOEM 等）。

### CANopen：嵌入式设备总线协议

CANopen 基于控制器局域网络（Controller Area Network，CAN）总线，由 CAN in Automation（CiA）组织制定，核心规范为 CiA 301。CANopen 定义了设备对象字典（Object Dictionary）、过程数据对象（Process Data Object，PDO）和服务数据对象（Service Data Object，SDO）等机制，为不同厂商的设备提供了统一的通信接口。

在机器人领域，CANopen 常用于连接关节控制器（Joint Controller）、力矩传感器（Torque Sensor）、编码器（Encoder）等低层设备，尤其在成本敏感的中小型机械臂和移动机器人中应用广泛。CiA 402 是 CANopen 的电机驱动设备子协议，定义了速度模式、位置模式、力矩模式等标准控制模式。

### OPC-UA：工业物联网通信标准

OPC 统一架构（OPC Unified Architecture，OPC-UA，IEC 62541）是一种平台无关的工业数据通信标准，提供安全、可靠的机器间（Machine-to-Machine，M2M）通信。与传统 OPC 不同，OPC-UA 不依赖 Windows COM/DCOM 技术，可运行于嵌入式 Linux、实时操作系统（RTOS）甚至微控制器上。

在智能制造（Smart Manufacturing）背景下，OPC-UA 被 RAMI 4.0（工业 4.0 参考架构模型）和 IIC（工业互联网联盟）列为工业物联网（Industrial Internet of Things，IIoT）的推荐通信标准。机器人通过 OPC-UA 服务器将状态数据（位置、速度、温度、错误代码等）暴露给上层制造执行系统（Manufacturing Execution System，MES）、数字孪生（Digital Twin）平台和云端分析服务。


## 分类 VI. 人形机器人相关标准

人形机器人（Humanoid Robot）作为近年来机器人领域发展最快的方向之一，其标准化工作仍处于起步阶段。现有标准的制定速度远落后于技术发展速度，这既是挑战，也为从业者参与标准制定提供了机会。

### 现有标准的适用性

目前，人形机器人在安全性方面主要参考以下已有标准，但均需要结合人形机器人的特殊性加以解释和补充：

- **ISO 13482（B 型：身体辅助机器人）**：人形机器人若用于辅助人体运动（如老年人助行）可适用此分类，但标准的许多具体要求是针对轮椅机器人等传统护理设备制定的，对于全身运动控制的人形机器人适用性有限。
- **ISO/TR 23482-1 和 ISO/TR 23482-2**：作为 ISO 13482 的应用指南，提供了服务机器人安全测试方法和应用场景指导，可为人形机器人安全测试方案的设计提供参考框架。
- **ISO/TS 15066（PFL 模式）**：若人形机器人手臂被用于与人协作的工业任务，PFL 模式的力限制要求和身体部位接触力阈值表同样适用。
- **IEC 61508 / EN ISO 13849-1**：人形机器人的关节驱动控制系统、安全停止功能等安全相关控制部件，需要按照功能安全标准进行设计和认证。

### 标准的空白与挑战

当前针对人形机器人的标准化工作面临以下主要挑战：

**动态稳定性（Dynamic Stability）**：传统机器人标准主要针对固定底座机器人或轮式移动机器人，缺乏针对双足步行（Bipedal Walking）机器人动态平衡失效（如跌倒）风险的评估方法。

**全身运动安全**：人形机器人具有数十个自由度，其运动空间大、速度快，与单臂工业机器人相比，人机碰撞的几何复杂度和风险评估难度大幅提升。

**意图感知与决策安全**：随着人形机器人开始搭载大型语言模型（Large Language Model，LLM）和具身智能（Embodied Intelligence）系统，机器人的行为决策不再完全可预测，传统基于确定性逻辑的安全标准面临根本性挑战。

**电池安全**：大容量锂电池组的热失控（Thermal Runaway）风险、跌倒时的结构安全等问题，需要新的测试方法和安全设计规范。

ISO/TC 299 已于 2023 年前后开始讨论专门针对人形机器人的新标准项目，预计未来几年内将陆续出现针对人形机器人测试方法、安全要求和性能评估的专项标准。


## 标准获取渠道

了解标准的获取途径对于工程师快速找到权威文本至关重要。以下是主要的标准获取渠道：

### 官方购买渠道

- **ISO 官网（iso.org）**：所有 ISO 标准的权威发布平台，可直接购买 PDF 版本。部分标准提供免费预览（通常为前几页）。ISO 标准价格通常在 100-300 瑞士法郎之间。
- **IEC 官网（iec.ch）**：IEC 系列标准的官方来源，提供与 ISO 联合发布标准的查询和购买服务。
- **ASTM 官网（astm.org）**：ASTM 系列标准的官方来源，支持按单份标准或打包订阅的方式购买。
- **VDI 官网（vdi.de）**：VDI 指南文件的官方来源，部分文件提供德语和英语双语版本。

### 国家标准机构

- **中国（SAC，国家标准化管理委员会）**：通过全国标准信息公共服务平台（std.samr.gov.cn）可以查询和购买 GB/T 国家标准。许多 ISO 标准被等同采用为 GB/T 标准，价格远低于直接购买 ISO 原版。例如 ISO 10218-1 对应的国标为 GB 11291.1，ISO 8373 对应 GB/T 12643。
- **美国（ANSI）**：通过 ANSI 网上商店（webstore.ansi.org）可购买美国采用的 ISO/IEC 标准（ANSI/ISO/IEC 联合发布版）。
- **欧洲（CEN/CENELEC）**：欧盟协调标准（EN 系列）通过各成员国国家标准机构（如德国 DIN、英国 BSI、法国 AFNOR）购买。

### 免费资源

- **ROS REP 文档（ros.org/reps）**：所有 REP 文件均免费在线访问，这是 ROS 开发者最常查阅的标准资源。
- **标准草案（Draft Standards）**：部分标准在正式发布前会公开征求意见，草案（Draft International Standard，DIS 或 Final Draft International Standard，FDIS）有时可免费获取。
- **ISO 免费标准计划**：ISO 设有面向发展中国家和学术机构的免费或优惠访问计划（如 RIDES 计划），相关高校和研究机构可申请。
- **学术图书馆**：许多高校图书馆订阅了 IHS Markit、Techstreet 或 BSI Knowledge 等标准数据库，在校师生可免费访问。


## 参考资料

[1] [机器人领域行业标准汇总 - 云飞机器人实验室](https://www.yfworld.com/?p=5753)

[2] [ISO/TC 299 Robotics - ISO](https://www.iso.org/committee/5915511.html)

[3] [REP 103 - Standard Units of Measure and Coordinate Conventions - ROS](https://www.ros.org/reps/rep-0103.html)

[4] [REP 105 - Coordinate Frames for Mobile Platforms - ROS](https://www.ros.org/reps/rep-0105.html)

[5] [REP 120 - Coordinate Frames for Humanoid Robots - ROS](https://www.ros.org/reps/rep-0120.html)

[6] [ISO/TS 15066:2016 - Robots and robotic devices — Collaborative robots](https://www.iso.org/standard/62996.html)

[7] [IEC 61508 - Functional Safety of E/E/PE Safety-related Systems](https://www.iec.ch/functionalsafety/)

[8] [EtherCAT Technology Group - ethercat.org](https://www.ethercat.org)

