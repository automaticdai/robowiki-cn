# 机器人安全与功能安全标准

!!! note "引言"
    安全标准是机器人产品进入市场的硬性门槛。系统安全类标准规定机器人在何种条件下可以与人共处：ISO 10218 面向传统工业机器人及其单元集成，ISO/TS 15066 给出协作作业中的力与压强限值，ISO 13482 覆盖个人护理机器人。功能安全类标准则关注控制系统本身失效时的行为，以 IEC 61508 为总纲，配合 ISO 13849 的性能等级（Performance Level, PL）与 IEC 62061 的安全完整性等级（Safety Integrity Level, SIL）构成完整的评估框架。本页面整理这两类标准及其相互关系。


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


## 参考资料

1. ISO 10218-1/-2:2011, *Robots and robotic devices — Safety requirements for industrial robots*.
2. ISO/TS 15066:2016, *Robots and robotic devices — Collaborative robots*.
3. IEC 61508, *Functional safety of electrical/electronic/programmable electronic safety-related systems*.
4. ISO 13849-1, *Safety of machinery — Safety-related parts of control systems*.
5. [机器人领域行业标准总览](standard.md)
6. [功能安全](../hardware/functional-safety.md)
7. [协作机器人安全](../hardware/safety-collaborative-robot.md)
