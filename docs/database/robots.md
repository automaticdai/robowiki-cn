# 机器人图鉴

!!! note "引言"
    本页面汇总了当前具有代表性的机器人产品，按形态和应用场景分类，涵盖人形机器人、四足机器人、轮式移动机器人、工业机械臂、协作机器人、医疗机器人、无人飞行机器人、水下机器人、太空机器人及仓储物流机器人等类别。各类机器人在驱动方式（Actuation）、感知（Perception）、自主性（Autonomy）和人机交互（Human-Robot Interaction，HRI）等维度上存在显著差异，共同构成了现代机器人技术谱系的全貌。本页内容持续更新，欢迎贡献补充。点击机器人名称可跳转至详细介绍页面（如有链接）。


## 人形机器人（Humanoid Robots）

人形机器人（Humanoid Robot）模仿人类外形，通常具备双足行走（Bipedal Locomotion）和双臂操作（Bimanual Manipulation）能力。其核心挑战在于动态平衡控制（Dynamic Balance Control）、全身运动规划（Whole-Body Motion Planning）与鲁棒感知，是当前机器人产业最受关注的方向之一。

驱动方式上，早期人形机器人多采用液压驱动（Hydraulic Actuation），具有功率密度高的优点，但系统复杂、噪音大、维护困难；现代人形机器人主流转向电机驱动（Electric Actuation），配合高减速比谐波减速器（Harmonic Drive）或行星减速器（Planetary Gearbox）实现力矩放大；部分机器人探索线驱动（Tendon-Driven）架构以降低腿部惯量（Leg Inertia）。

从技术路线看，Boston Dynamics 的 Atlas 长期代表液压路线的顶峰，而 2024 年发布的全电动 Atlas 则象征行业向电动方向的全面转型。中国团队在 2023–2024 年间集中爆发，宇树、傅利叶、智元、优必选等企业密集发布产品，推动了人形机器人商业化进程。

| 名称 | 公司/机构 | 国家 | 首发年份 | 身高 | 体重 | 自由度 | 驱动方式 | 主要应用 |
|------|---------|------|----------|------|------|--------|----------|----------|
| [Atlas](atlas.md) | Boston Dynamics | 美国 | 2013（液压）/ 2024（电动） | 1.5 m | ~89 kg | 28+ | 液压→电动 | 研究与演示 |
| [Optimus](optimus.md) | Tesla | 美国 | 2022 | 1.73 m | ~73 kg | 28+ | 电动 | 通用任务 |
| [Figure 02](figure.md) | Figure AI | 美国 | 2024 | 1.67 m | ~60 kg | 16+ | 电动 | 仓储物流 |
| [ASIMO](asimo.md) | Honda | 日本 | 2000 | 1.3 m | 54 kg | 57 | 电动 | 研究与展示 |
| [Digit](digit.md) | Agility Robotics | 美国 | 2019 | 1.75 m | ~65 kg | 16+ | 电动 | 物流搬运 |
| [H1](unitree-h1.md) | 宇树科技（Unitree） | 中国 | 2023 | 1.8 m | ~47 kg | 19 | 电动 | 研究与通用任务 |
| G1 | 宇树科技（Unitree） | 中国 | 2024 | 1.27 m | ~35 kg | 23 | 电动 | 研究与教育 |
| [NAO](nao.md) | SoftBank Robotics | 法国/日本 | 2008 | 0.574 m | 5.48 kg | 25 | 电动 | 教育与研究 |
| Pepper | SoftBank Robotics | 法国/日本 | 2014 | 1.2 m | 28 kg | 20 | 电动 | 商业接待 |
| Sophia | Hanson Robotics | 美国/香港 | 2016 | — | — | 头部 62+ | 电动 | 社交互动演示 |
| Phoenix | Sanctuary AI | 加拿大 | 2023 | 1.7 m | ~70 kg | 20+ | 电动 | 通用任务 |
| Apollo | Apptronik | 美国 | 2023 | 1.73 m | ~73 kg | 24+ | 电动 | 物流与制造 |
| GR-1 | 傅利叶智能（Fourier） | 中国 | 2023 | 1.65 m | ~55 kg | 40 | 电动 | 康复与研究 |
| GR-2 | 傅利叶智能（Fourier） | 中国 | 2024 | 1.75 m | ~63 kg | 53 | 电动 | 通用人形 |
| Agibot（远征 A2） | 智元机器人 | 中国 | 2024 | 1.75 m | ~65 kg | 40+ | 电动 | 通用任务 |
| Walker S | 优必选（UBTECH） | 中国 | 2023 | 1.7 m | ~77 kg | 41 | 电动 | 工业与服务 |
| CyberOne | 小米（Xiaomi） | 中国 | 2022 | 1.77 m | ~52 kg | 21 | 电动 | 展示与研究 |
| HRP-4 | 川田工业（Kawada） | 日本 | 2010 | 1.51 m | 39 kg | 34 | 电动 | 研究与演示 |
| iCub | 意大利技术研究院（IIT） | 意大利 | 2008 | 1.04 m | ~33 kg | 53 | 电动 | 认知与具身智能研究 |
| Valkyrie（R5） | NASA / JSC | 美国 | 2015 | 1.8 m | ~125 kg | 44 | 电动 | 太空探索研究 |
| TALOS | PAL Robotics | 西班牙 | 2017 | 1.75 m | ~95 kg | 32 | 电动（力控） | 学术研究平台 |
| Surena IV | 德黑兰大学 | 伊朗 | 2019 | 1.7 m | ~74 kg | 43 | 电动 | 学术研究 |


## 四足机器人（Quadruped Robots）

四足机器人（Quadruped Robot）以四条腿为支撑，具备出色的地形适应能力（Terrain Adaptability），可在不平整、泥泞或危险环境中执行巡检（Inspection）、测绘（Mapping）和搜救（Search and Rescue）等任务。

**控制方法演进**：早期四足机器人依赖预先设计的步态库（Gait Library）和零力矩点（Zero Moment Point，ZMP）准则；现代系统广泛采用模型预测控制（Model Predictive Control，MPC）和凸优化（Convex Optimization），结合接触力规划（Contact Force Planning）实现动步态（Dynamic Gait）。近年来，基于深度强化学习（Deep Reinforcement Learning，DRL）的端到端步态控制取得突破，MIT Mini Cheetah 和宇树 Go2 均展示了在仿真中训练、在真实世界部署（Sim-to-Real Transfer）的能力。

**商业化进展**：Boston Dynamics 的 Spot 是目前商业化程度最高的四足机器人，已在石油化工、电力、矿山等行业部署超过数千台，执行例行巡检任务。宇树科技凭借极具竞争力的价格策略，将四足机器人推向科研和消费市场。

| 名称 | 公司/机构 | 国家 | 首发年份 | 体重 | 最大速度 | 主要应用 |
|------|---------|------|----------|------|----------|----------|
| Spot | Boston Dynamics | 美国 | 2019 | ~32 kg | 1.6 m/s | 工业巡检与测绘 |
| LS3（骡子机器人） | Boston Dynamics / DARPA | 美国 | 2012 | ~590 kg | 3.2 m/s | 军用负载运输 |
| BigDog | Boston Dynamics / DARPA | 美国 | 2005 | ~109 kg | 1.6 m/s | 军用早期研究平台 |
| ANYmal C | ANYbotics | 瑞士 | 2020 | ~50 kg | 1.0 m/s | 工业巡检 |
| ANYmal D | ANYbotics | 瑞士 | 2023 | ~50 kg | 1.0 m/s | 工业巡检（升级版） |
| HyQ | 意大利技术研究院（IIT） | 意大利 | 2010 | ~80 kg | 2.0 m/s | 学术研究平台 |
| MIT Mini Cheetah | 麻省理工学院（MIT） | 美国 | 2019 | ~9 kg | 3.7 m/s | 学术步态与 RL 研究 |
| Go1 | 宇树科技（Unitree） | 中国 | 2021 | ~12 kg | 3.5 m/s | 消费与教育 |
| Go2 | 宇树科技（Unitree） | 中国 | 2023 | ~15 kg | 3.5 m/s | 科研与消费 |
| B1 | 宇树科技（Unitree） | 中国 | 2021 | ~50 kg | 1.6 m/s | 工业巡检 |
| B2 | 宇树科技（Unitree） | 中国 | 2023 | ~60 kg | 1.5 m/s | 工业与科研 |
| Laikago | 宇树科技（Unitree） | 中国 | 2018 | ~22 kg | 3.0 m/s | 早期研究平台 |
| A1 | 宇树科技（Unitree） | 中国 | 2020 | ~12 kg | 3.3 m/s | 学术步态研究 |
| CyberDog 1 | 小米（Xiaomi） | 中国 | 2021 | ~14 kg | 3.2 m/s | 消费与开发 |
| CyberDog 2 | 小米（Xiaomi） | 中国 | 2023 | ~8.9 kg | 3.2 m/s | 消费与开发（升级版） |
| Jueying X20 | 云深处科技 | 中国 | 2022 | ~60 kg | 1.5 m/s | 工业巡检 |
| Spot Mini（原型） | Boston Dynamics | 美国 | 2016 | ~25 kg | 1.4 m/s | Spot 的前身平台 |


## 轮式移动机器人（Mobile Wheeled Robots）

轮式移动机器人（Wheeled Mobile Robot）凭借结构简单、能效高、控制成熟等优势，广泛应用于室内科研（Indoor Research）、仓储物流（Warehouse Logistics）和工业巡检（Industrial Inspection）等场景。

常见底盘类型及其特点：

- **差速驱动（Differential Drive）**：两轮独立控制，结构最简，适合平坦室内环境，是 ROS 教学平台的首选。
- **阿克曼转向（Ackermann Steering）**：类似汽车转向机构，适合室外高速行驶，最小转弯半径较大。
- **全向轮（Omnidirectional Wheel）**：包括麦克纳姆轮（Mecanum Wheel）和球形轮，可实现任意方向平移，适合空间受限的室内作业场景。
- **履带式（Tracked）**：越野能力强，适合松软地面，常用于搜救和军事领域。

| 名称 | 公司/机构 | 国家 | 类型 | 主要应用 |
|------|---------|------|------|----------|
| TurtleBot 4 | Clearpath Robotics | 加拿大 | 差速驱动 | ROS 教育研究 |
| TurtleBot 3 Waffle | ROBOTIS | 韩国 | 差速驱动 | ROS 入门教学 |
| Husky A200 | Clearpath Robotics | 加拿大 | 差速驱动 | 室外科研平台 |
| Jackal | Clearpath Robotics | 加拿大 | 差速驱动 | 室外导航研究 |
| Dingo | Clearpath Robotics | 加拿大 | 全向轮（麦克纳姆） | 室内轻载科研 |
| Pioneer 3-DX | Adept MobileRobots（现 Omron） | 美国 | 差速驱动 | 经典科研平台 |
| Fetch Robot | Fetch Robotics（现 Zebra） | 美国 | 全向轮 | 仓储物流 |
| iRobot Create 3 | iRobot | 美国 | 差速驱动 | 教育与开发 |
| SUMMIT-XL | Robotnik | 西班牙 | 全向轮（麦克纳姆） | 工业巡检 |
| Ridgeback | Clearpath Robotics | 加拿大 | 全向轮（麦克纳姆） | 室内重载搬运 |
| ROSbot 2R | Husarion | 波兰 | 差速驱动 | ROS 开发平台 |
| AgileX Scout Mini | AgileX Robotics | 中国 | 差速驱动 | 室外科研与教育 |
| AgileX LIMO | AgileX Robotics | 中国 | 多模式（差速/阿克曼/全向） | 多模式科研教育平台 |
| MiR100 | Mobile Industrial Robots（MiR） | 丹麦 | 差速驱动 | 工厂自主物料运输 |
| Pepper（轮式底盘） | SoftBank Robotics | 法国/日本 | 全向轮 | 商业接待与服务 |


## 工业机械臂（Industrial Robot Arms）

工业机械臂（Industrial Robot Arm）是目前市场规模最大的机器人类别，广泛应用于焊接（Welding）、装配（Assembly）、搬运（Material Handling）、喷涂（Painting）和机床上下料（Machine Tending）等制造场景。

**关键技术指标**：

- **额定负载（Rated Payload）**：末端执行器（End-Effector）和工件的最大合计重量。
- **最大臂展（Maximum Reach）**：末端可达的最远距离，决定作业空间大小。
- **重复定位精度（Repeatability）**：多次返回同一位置时的位置误差，高精度机械臂可达 ±0.02 mm。
- **循环时间（Cycle Time）**：完成标准测试轨迹所需时间，反映机器人作业效率。
- **IP 防护等级（IP Rating）**：用于喷涂和食品等场合时需考虑防尘防水性能。

按负载分类，工业机械臂可分为轻型（负载 ≤20 kg）、中型（20–100 kg）和重型（>100 kg）三类。重型机械臂主要用于汽车制造的车身焊接和搬运。

| 名称 | 公司 | 国家 | 负载 | 自由度 | 最大臂展 | 主要应用 |
|------|------|------|------|--------|----------|----------|
| M-20iD/25 | FANUC | 日本 | 25 kg | 6 | 1,853 mm | 焊接与搬运 |
| M-410iC/185 | FANUC | 日本 | 185 kg | 4 | 3,143 mm | 重载码垛 |
| R-2000iC/210F | FANUC | 日本 | 210 kg | 6 | 2,655 mm | 汽车车身搬运 |
| IRB 6700 | ABB | 瑞士/瑞典 | 150–300 kg | 6 | 2,850 mm | 重载搬运与焊接 |
| IRB 1200 | ABB | 瑞士/瑞典 | 5–7 kg | 6 | 901 mm | 精密装配 |
| IRB 120 | ABB | 瑞士/瑞典 | 3 kg | 6 | 580 mm | 电子装配 |
| KR AGILUS KR6 R900 | KUKA | 德国 | 6 kg | 6 | 900 mm | 高速轻载装配 |
| KR 1000 Titan | KUKA | 德国 | 1,000 kg | 6 | 3,202 mm | 超重载搬运 |
| KR QUANTEC | KUKA | 德国 | 120–300 kg | 6 | 2,900 mm | 汽车制造焊接 |
| Motoman GP7 | 安川电机（Yaskawa） | 日本 | 7 kg | 6 | 927 mm | 精密装配 |
| Motoman GP225 | 安川电机（Yaskawa） | 日本 | 225 kg | 6 | 2,702 mm | 重载搬运 |
| BX200L | 川崎机器人（Kawasaki） | 日本 | 200 kg | 6 | 2,600 mm | 重载点焊 |
| Doosan M0617 | 斗山机器人（Doosan） | 韩国 | 6 kg | 6 | 1,700 mm | 长臂展装配 |
| TM5-700 | 达明机器人（Techman） | 台湾 | 6 kg | 6 | 700 mm | 内置视觉检测 |
| SIASUN SR210 | 新松机器人（SIASUN） | 中国 | 210 kg | 6 | 2,688 mm | 重载工业搬运 |
| ESTUN ER50 | 埃斯顿（ESTUN） | 中国 | 50 kg | 6 | 2,033 mm | 焊接与搬运 |
| 汇川 IR616 | 汇川技术（Inovance） | 中国 | 16 kg | 6 | 1,629 mm | 中载装配搬运 |
| Stäubli TX2-90 | Stäubli | 瑞士 | 15 kg | 6 | 1,000 mm | 洁净室装配 |


## 协作机器人（Collaborative Robots / Cobots）

协作机器人（Collaborative Robot，Cobot）设计用于与人类在同一工作空间安全共存，依据 ISO/TS 15066 标准实现安全协作。其核心安全特性包括力矩传感（Torque Sensing）、碰撞检测（Collision Detection）、速度与间距监控（Speed and Separation Monitoring，SSM）及功率/力限制（Power and Force Limiting，PFL）。

协作机器人与传统工业机械臂的关键区别在于：无需安全围栏（Fenceless Operation）、支持拖动示教（Lead-Through Programming）、可快速换线（Flexible Deployment）。其负载通常在 3–20 kg 范围内，重复定位精度一般为 ±0.03–0.1 mm，略逊于高精度工业机械臂。

**市场格局**：Universal Robots 的 e-Series 系列长期占据协作机器人市场份额第一，约占全球市场的 50%（2022 年数据）。中国本土品牌（遨博、珞石、节卡等）凭借价格优势快速增长。

| 名称 | 公司 | 国家 | 负载 | 主要特点 |
|------|------|------|------|---------|
| UR3e | Universal Robots | 丹麦 | 3 kg | 桌面级，适合精细装配 |
| UR5e | Universal Robots | 丹麦 | 5 kg | 市场标杆，生态最成熟 |
| UR10e | Universal Robots | 丹麦 | 12.5 kg | 中载，柔性产线首选 |
| UR16e | Universal Robots | 丹麦 | 16 kg | 较大负载的协作应用 |
| LBR iiwa 7 R800 | KUKA | 德国 | 7 kg | 关节力矩传感，阻抗控制，精密装配 |
| LBR iiwa 14 R820 | KUKA | 德国 | 14 kg | 重型协作，汽车零部件装配 |
| Panda | Franka Emika | 德国 | 3 kg | 科研首选，开源 libfranka SDK |
| FR3 | Franka Robotics | 德国 | 3 kg | Panda 后继，更高动态性能 |
| TM5-700 / TM12 / TM14 | 达明机器人（Techman） | 台湾 | 6–14 kg | 内置视觉，无需外部相机 |
| AUBO-i5 | 遨博智能（AUBO） | 中国 | 5 kg | 国产协作机器人代表，价格亲民 |
| AUBO-i10 | 遨博智能（AUBO） | 中国 | 10 kg | 中载国产协作 |
| Rokae xMate ER3 / ER7 | 珞石机器人（Rokae） | 中国 | 3 / 7 kg | 高精度，低成本，科研友好 |
| JAKA Zu 3 / Zu 7 / Zu 12 | 节卡机器人（JAKA） | 中国 | 3–12 kg | 易用性强，无线示教，快速部署 |
| CRX-10iA | FANUC | 日本 | 10 kg | 手推示教，绿色外观，易于集成 |
| HC10DT | 安川电机（Yaskawa） | 日本 | 10 kg | 皮肤传感，整机安全性高 |
| GoFa CRB 15000 | ABB | 瑞士/瑞典 | 5 kg | 快速轻量协作，IRC5 控制器 |
| SARA（SR6） | 遨博 × 新松 | 中国 | 6 kg | 国产联合研发协作机器人 |


## 医疗机器人（Medical Robots）

医疗机器人（Medical Robot）以高精度、稳定性和可重复性深刻改变现代医学实践。主要分类包括手术机器人（Surgical Robot）、康复机器人（Rehabilitation Robot）、辅助机器人（Assistive Robot）和诊断机器人（Diagnostic Robot）。监管认证（如美国 FDA 510(k) 或欧盟 MDR CE 标志）是医疗机器人商业化的核心门槛，认证周期通常长达数年。

手术机器人的核心价值在于：通过主从操作（Master-Slave Control）滤除术者手部抖动（Tremor Cancellation）、提供三维高清放大视野、减小切口和缩短患者恢复时间。


### 手术机器人（Surgical Robots）

| 名称 | 公司 | 国家 | 首发年份 | 主要应用 |
|------|------|------|----------|----------|
| da Vinci Xi | Intuitive Surgical | 美国 | 2014 | 腔镜微创手术，全球市场主导 |
| da Vinci 5 | Intuitive Surgical | 美国 | 2024 | 新一代 da Vinci，力反馈 |
| Versius | CMR Surgical | 英国 | 2019 | 模块化腔镜手术，床旁独立臂 |
| Hugo RAS | Medtronic（美敦力） | 美国 | 2021 | 腔镜微创手术 |
| Mako SmartRobotics | Stryker（史赛克） | 美国 | 2006 | 骨科关节置换（髋/膝） |
| 天玑（TiRobot） | 天智航（Tinavi） | 中国 | 2016 | 骨科与脊柱手术，国内首款 |
| 图迈（Toumai） | 微创机器人 | 中国 | 2022 | 腔镜微创手术，国产 da Vinci |
| 康多（Kangduo） | 术锐机器人 | 中国 | 2023 | 单孔腔镜手术 |
| ROSA One | Zimmer Biomet | 美国 | 2019 | 脑外科与骨科 |
| Mazor X Stealth | Medtronic | 美国 | 2018 | 脊柱手术导航机器人 |


### 康复机器人（Rehabilitation Robots）

康复机器人（Rehabilitation Robot）辅助神经损伤（如脑卒中 Stroke、脊髓损伤 Spinal Cord Injury）和骨科术后患者恢复运动功能，通过重复性运动训练促进神经可塑性（Neuroplasticity）。外骨骼（Exoskeleton）是其典型形态，分为下肢外骨骼（Lower-Limb Exoskeleton）和上肢外骨骼（Upper-Limb Exoskeleton）两类。

| 名称 | 公司 | 国家 | 类型 | 主要应用 |
|------|------|------|------|----------|
| Lokomat Pro | Hocoma | 瑞士 | 悬吊式下肢外骨骼 | 步态康复训练 |
| EksoGT | Ekso Bionics | 美国 | 下肢外骨骼 | 脑卒中/脊髓损伤康复 |
| ReWalk Personal 6.0 | ReWalk Robotics | 以色列/美国 | 下肢外骨骼 | 脊髓损伤患者日常辅助行走 |
| Myopro Motion G | Myomo | 美国 | 上肢外骨骼 | 偏瘫上肢功能辅助 |
| Hybrid Assistive Limb（HAL） | Cyberdyne | 日本 | 全身外骨骼 | 运动功能障碍康复 |
| Indego | Parker Hannifin | 美国 | 下肢外骨骼 | 脊髓损伤步态训练 |
| MATE-XT | Comau | 意大利 | 上肢被动外骨骼 | 工业辅助，减轻肩部负担 |
| 傅利叶 X2 | 傅利叶智能（Fourier） | 中国 | 下肢外骨骼 | 康复训练 |


## 无人飞行机器人（Unmanned Aerial Vehicles / Drones）

无人飞行机器人（Unmanned Aerial Vehicle，UAV）按旋翼数量和构型分为固定翼（Fixed-Wing）、旋翼（Rotary-Wing，包括单旋翼直升机和多旋翼 Multi-Rotor）以及固定翼多旋翼混合（Hybrid VTOL）等。多旋翼无人机结构简单、垂直起降（Vertical Take-Off and Landing，VTOL）性能好，在消费娱乐、农业植保（Agricultural Spraying）、工业巡检和应急救援等领域获得广泛应用。

飞行控制器（Flight Controller）是无人机的计算核心，负责姿态估计（Attitude Estimation）、控制律计算和传感器融合（Sensor Fusion）。开源飞控平台 PX4 和 ArduPilot 极大地推动了无人机科研与产品开发，已成为学术研究的事实标准。

大疆创新（DJI）占据全球消费级无人机市场约 70% 的份额（2023 年数据），在农业植保领域也是全球领先者。

| 名称 | 公司 | 国家 | 类型 | 主要应用 |
|------|------|------|------|----------|
| DJI Mini 4 Pro | 大疆创新（DJI） | 中国 | 消费级折叠多旋翼 | 入门航拍 |
| DJI Mavic 3 Pro | 大疆创新（DJI） | 中国 | 消费级多旋翼 | 专业航拍摄影 |
| DJI Agras T50 | 大疆创新（DJI） | 中国 | 农业植保多旋翼 | 农业精准喷洒 |
| DJI Matrice 350 RTK | 大疆创新（DJI） | 中国 | 行业级多旋翼 | 测绘（Mapping）与工业巡检 |
| DJI Dock 2 | 大疆创新（DJI） | 中国 | 无人机机巢系统 | 无人值守自动巡检 |
| Skydio 2+ | Skydio | 美国 | 自主避障多旋翼 | 自主跟踪与基础设施巡检 |
| Parrot ANAFI USA | Parrot | 法国 | 消费/行业级多旋翼 | 安防与应急响应 |
| Autel EVO II Pro | Autel Robotics | 美国 | 消费级多旋翼 | 专业航拍 |
| PX4 / ArduPilot | 开源社区 | 国际 | 开源飞控平台 | 科研开发与定制产品 |
| Wingcopter 198 | Wingcopter | 德国 | 固定翼多旋翼混合（VTOL） | 医疗物资配送 |
| Zipline Platform 2 | Zipline | 美国 | 固定翼 VTOL | 医疗物资与商品配送 |
| 极飞 P100 Pro | 极飞科技（XAG） | 中国 | 农业植保多旋翼 | 农业精准作业 |


## 水下机器人（Underwater Robots）

水下机器人分为自主水下航行器（Autonomous Underwater Vehicle，AUV）和遥控水下航行器（Remotely Operated Vehicle，ROV）两大类。AUV 预先编程任务后自主执行，适合大范围海洋调查（Oceanographic Survey）和海底地形测绘（Bathymetric Survey）；ROV 由水面人员通过脐带缆（Umbilical Cable）实时操控，适合精细作业，如海底油气管道检修（Subsea Pipeline Inspection）和水下考古（Underwater Archaeology）。

水下环境对通信提出严苛挑战：无线电波（Radio Wave）在水中衰减极快，水声通信（Acoustic Communication）带宽低、延迟高，光学通信（Optical Communication）作用距离短。因此，AUV 自主性要求极高，ROV 则依赖有缆实时控制。

**国内发展**：中国在深海技术领域持续投入，"蛟龙号"（载人潜水器）和"海斗一号"（全海深 AUV）代表了国内最高水平。

### 自主水下航行器（AUV）

| 名称 | 公司/机构 | 国家 | 最大深度 | 主要应用 |
|------|---------|------|----------|----------|
| REMUS 100 | Kongsberg Maritime（原 Hydroid） | 挪威/美国 | 100 m | 近海测绘与浅水海洋调查 |
| REMUS 600 | Kongsberg Maritime | 挪威/美国 | 600 m | 中深海调查与军用侦察 |
| REMUS 6000 | Kongsberg Maritime | 挪威/美国 | 6,000 m | 深海测绘，曾用于搜寻 AF447 |
| Bluefin-21 | General Dynamics Mission Systems | 美国 | 4,500 m | 深海测绘与军用 |
| Seaglider | Kongsberg Maritime（原 iRobot） | 美国 | 1,000 m | 长航程海洋环境监测 |
| Aquanaut | Houston Mechatronics | 美国 | 3,000 m | 水下变形机器人，设施检修 |
| Ocean One | 斯坦福大学（Stanford） | 美国 | — | 深海科考，仿人形水下机器人 |
| 海斗一号 | 中国科学院沈阳自动化所 | 中国 | 10,900 m | 全海深 AUV，马里亚纳海沟探测 |

### 遥控水下航行器（ROV）

| 名称 | 公司/机构 | 国家 | 最大深度 | 主要应用 |
|------|---------|------|----------|----------|
| BlueROV2 | Blue Robotics | 美国 | 100 m | 低成本开源科研与教育 |
| VideoRay Defender | VideoRay | 美国 | 305 m | 安防检查与搜救 |
| Saab Seaeye Falcon DR | Saab Seaeye | 英国 | 300 m | 近海设施检修 |
| Oceaneering Millennium Plus | Oceaneering | 美国 | 3,000 m+ | 深海油气工程作业 |
| SuBastian | Schmidt Ocean Institute | 美国 | 4,500 m | 科学考察 ROV |
| 海马号 | 中国地质调查局 | 中国 | 4,500 m | 深海地质与冷泉调查 |


## 太空机器人（Space Robots）

太空机器人（Space Robot）在人类直接操控受限的极端环境下执行任务，包括空间站维护（Space Station Maintenance）、在轨卫星服务（On-Orbit Servicing）和行星表面探测（Planetary Surface Exploration）。

太空环境的三大挑战：**高辐射**（High Radiation，需特殊辐射加固电子器件）、**极端温差**（从 -150 °C 到 +150 °C）、**通信延迟**（Communication Delay，地火距离导致单向延迟最长约 22 分钟），要求太空机器人具备高可靠性和较强的自主决策能力。

| 名称 | 机构 | 国家/组织 | 类型 | 任务/应用 |
|------|------|---------|------|-----------|
| Canadarm2（SSRMS） | 加拿大航天局（CSA） | 加拿大 | 空间站机械臂，17 m | 国际空间站（ISS）组件装配与维护 |
| Dextre（SPDM） | 加拿大航天局（CSA） | 加拿大 | 双臂精细操作机器人 | ISS 轨道更换单元（ORU）维护 |
| Robonaut 2（R2） | NASA / 通用汽车 | 美国 | 人形上半身机器人 | ISS 内部任务辅助与力交互研究 |
| Curiosity 火星车（MSL） | NASA / JPL | 美国 | 核动力火星探测车 | 火星地质与宜居性科学探测 |
| Perseverance 火星车（Mars 2020） | NASA / JPL | 美国 | 核动力火星探测车 | 样本采集（MOXIE 制氧实验），搜寻生命迹象 |
| Ingenuity 火星直升机 | NASA / JPL | 美国 | 火星旋翼无人机 | 首次实现地外天体动力飞行验证 |
| 祝融号（Zhurong） | 中国国家航天局（CNSA） | 中国 | 太阳能火星探测车 | 乌托邦平原地质与气候探测 |
| 玉兔二号（Yutu-2） | 中国国家航天局（CNSA） | 中国 | 月面巡视探测车 | 月球背面地形与矿物探测 |
| ERA（欧洲机械臂） | ESA / Roscosmos | 欧洲/俄罗斯 | 空间站机械臂，11 m | 俄罗斯舱段 MLM 外部维护 |
| Justin | DLR（德国航空航天中心） | 德国 | 轮式双臂机器人 | 遥操作与在轨服务研究平台 |


## 仓储与物流机器人（Warehouse & Logistics Robots）

仓储与物流机器人（Warehouse and Logistics Robot）是近年来增长最快的机器人细分市场之一。核心技术包括自主移动机器人（Autonomous Mobile Robot，AMR）、机器人拣选（Robotic Picking）和货到人（Goods-to-Person，GTP）系统。

区分自主移动机器人（AMR）与自动导引车（Automated Guided Vehicle，AGV）：AGV 依赖磁条、二维码或激光反射板等固定导航基础设施，路径固定；AMR 则基于激光雷达（LiDAR）和即时定位与地图构建（Simultaneous Localization and Mapping，SLAM），可动态规划路径、绕过障碍物，部署更灵活。

**市场规模**：据 IFR 统计，2023 年全球仓储物流机器人市场规模超过 90 亿美元，年均增长率约 25%。

| 名称 | 公司 | 国家 | 类型 | 主要应用 |
|------|------|------|------|----------|
| Stretch | Boston Dynamics | 美国 | 移动拆码垛机器人 | 集装箱卸载，箱体拆码垛 |
| Handle | Boston Dynamics | 美国 | 轮腿式码垛机器人 | 物流中心托盘码垛 |
| Kiva（Amazon Robotics） | Amazon | 美国 | 货架搬运 AMR | 亚马逊仓储核心系统 |
| M-series AMR | 快仓（Quicktron） | 中国 | 货架搬运 AMR | 电商仓储货到人拣选 |
| R-series AMR | 极智嘉（Geek+） | 中国 | 货架搬运 AMR | 快递分拣与电商仓储 |
| MiR250 | Mobile Industrial Robots（MiR） | 丹麦 | 自主移动机器人 | 工厂内部物料自动运输 |
| MiR600 | Mobile Industrial Robots（MiR） | 丹麦 | 重载 AMR | 工厂重载物料运输 |
| Locus Origin | Locus Robotics | 美国 | 协同拣选 AMR | 电商仓储订单拣选 |
| 6 River Systems Chuck | 6 River Systems（Shopify） | 美国 | 协同拣选机器人 | 引导人工拣选 |
| Autostore System | AutoStore | 挪威 | 三维立体仓储机器人网格系统 | 高密度仓储与自动拣选 |
| 哈工智造 SP100 | 哈工智造 | 中国 | 托盘搬运 AGV | 重载托盘仓内搬运 |


## 服务机器人（Service Robots）

服务机器人（Service Robot）面向专业服务（Professional Service）和个人/家用（Personal/Domestic）两大场景。专业服务机器人包括用于餐厅配送（Delivery）、酒店礼宾和机场引导的商业服务机器人；个人服务机器人则以家用扫地机器人（Robotic Vacuum Cleaner）最为普及。

| 名称 | 公司 | 国家 | 类型 | 主要应用 |
|------|------|------|------|----------|
| Roomba j9+ | iRobot | 美国 | 家用扫地机器人 | 家庭自动清洁 |
| 石头 G20 | 石头科技（Roborock） | 中国 | 家用扫地拖地机器人 | 家庭清洁，自清洁基站 |
| 科沃斯 X2 Pro | 科沃斯（Ecovacs） | 中国 | 家用扫地拖地机器人 | 家庭清洁，激光导航 |
| Whiz | SoftBank Robotics | 日本 | 商用清洁机器人 | 大型场馆地面清洁 |
| Bear Robotics Servi | Bear Robotics | 美国 | 餐厅配送机器人 | 餐厅送餐与收盘 |
| 擎朗 Keenon T8 | 擎朗智能（Keenon） | 中国 | 室内配送机器人 | 酒店/餐厅配送 |
| Spot（导览版） | Boston Dynamics | 美国 | 导览与交互机器人 | 博物馆、展馆导览 |
| Pepper | SoftBank Robotics | 法国/日本 | 社交机器人 | 商业接待与客户服务 |
| Aethon TUG | Aethon（现 ST Engineering） | 美国 | 室内自主配送机器人 | 医院药品与物资配送 |
| Savioke Relay | Savioke | 美国 | 室内配送机器人 | 酒店客房物品配送 |
| HEXA | VINCROSS | 中国 | 六足桌面机器人 | 开发与教育 |
| Misty II | Misty Robotics | 美国 | 个人社交机器人 | 开发平台与教育 |


## 特种与搜救机器人（Special-Purpose & Search-and-Rescue Robots）

特种机器人（Special-Purpose Robot）用于人类难以或无法进入的危险环境，包括核电站事故现场（如福岛第一核电站）、城市搜救（Urban Search and Rescue，USAR）、排爆（Explosive Ordnance Disposal，EOD）和极地探测等场景。此类机器人对环境鲁棒性（Environmental Robustness）要求极高，通常具备遥控操作（Teleoperation）和有限自主（Semi-Autonomous）能力。

DARPA 机器人挑战赛（DARPA Robotics Challenge，DRC，2013–2015）是推动灾难响应机器人发展的重要里程碑，参赛机器人需完成驾车、开门、使用工具等拟人任务，极大促进了人形机器人运动控制与自主性的进步。

| 名称 | 公司/机构 | 国家 | 类型 | 主要应用 |
|------|---------|------|------|----------|
| PackBot | iRobot（现 Endeavor Robotics） | 美国 | 履带式遥控机器人 | 排爆与战场侦察 |
| TALON | QinetiQ | 美国 | 履带式遥控机器人 | 排爆，EOD，军用 |
| Thermite RS3 | Howe & Howe（现 Textron） | 美国 | 履带式消防机器人 | 灭火与消防救援 |
| SPOT（防爆版） | Boston Dynamics | 美国 | 四足巡检机器人 | 危险环境巡检与测绘 |
| Quince | 千叶大学 / 东北大学 | 日本 | 履带式核辐射机器人 | 核事故现场勘察（福岛） |
| KOHGA2 | 日立（Hitachi） | 日本 | 核电站维护机器人 | 核电站设施检修 |
| Coyote | Cobalt Robotics | 美国 | 轮式安保巡逻机器人 | 室内安保与异常检测 |
| 哈工大 SJT | 哈尔滨工业大学 | 中国 | 六足搜救机器人 | 复杂地形搜救研究 |
| ASALA | NIST（美国国家标准与技术研究院） | 美国 | 参考测试平台 | USAR 机器人性能标准制定 |
| Husky（改装排爆版） | Clearpath Robotics | 加拿大 | 轮式遥控机器人底盘 | 排爆任务改装研究平台 |
| ANYmal（检测版） | ANYbotics | 瑞士 | 四足核电巡检机器人 | 核电站辐射区巡检 |


## 机器人关键技术参数说明

了解机器人性能规格时，以下关键术语和参数有助于横向比较不同产品：

### 机械参数

- **自由度（Degrees of Freedom，DoF）**：机器人可独立运动的关节数量。6 自由度是工业机械臂的最低配置，可实现末端执行器在三维空间的任意位置和姿态；人形机器人通常需要 20 个以上自由度才能完成灵巧操作。
- **额定负载（Rated Payload）**：在标准速度和臂展条件下，机器人末端可承受的最大有效载荷，通常不包括末端执行器自身重量。
- **最大臂展（Maximum Reach）**：末端执行器可到达的最远距离，决定了机器人的作业空间（Workspace）大小。
- **重复定位精度（Repeatability，RP）**：机器人多次（通常 ≥30 次）返回同一示教点时，实际位置的最大偏差范围，是衡量机器人精度的核心指标。高端工业机械臂可达 ±0.02 mm，而协作机器人一般在 ±0.03–0.1 mm 范围内。

### 驱动与传感

- **谐波减速器（Harmonic Drive）**：利用柔性齿轮的弹性形变实现大减速比（通常 50:1–320:1），具有零背隙（Zero Backlash）、高扭矩密度等优点，广泛用于工业机械臂和协作机器人关节。
- **力矩传感器（Torque Sensor）**：安装于关节或腕部，用于测量关节输出力矩，是阻抗控制（Impedance Control）和力控（Force Control）的基础。协作机器人的碰撞检测依赖关节力矩传感器实现。
- **编码器（Encoder）**：测量关节旋转角度的传感器，分为增量式（Incremental）和绝对式（Absolute）两类。绝对式编码器在断电后仍能保持位置信息，是关节位置控制的核心器件。

### 移动平台参数

- **最大速度（Maximum Speed）**：机器人在平地直线行进时的最大速度，受电机功率、控制策略和安全限制约束。
- **有效载荷（Payload Capacity）**：移动机器人可携带的最大有效载荷，影响其搭载传感器和执行器的能力。
- **续航时间（Battery Life）**：满载工作条件下，单次充电可持续工作的时间。四足机器人通常 1–2 小时，部分工业 AGV 可实现换电或无线充电。
- **防护等级（IP Rating）**：依据 IEC 60529 标准，反映设备防尘（第一位数字，0–6）和防水（第二位数字，0–9）能力。户外机器人通常需 IP54 以上，水下机器人需 IP68 乃至特殊压力防护。

### 自主等级（Levels of Autonomy）

机器人的自主程度通常分为以下几个层级，参考美国国防部（DoD）和 SAE 自动驾驶分级框架改编：

1. **遥控（Teleoperation）**：人类实时控制机器人每一个动作，机器人不具备自主决策能力（如早期 EOD 机器人）。
2. **辅助控制（Assisted Control）**：机器人可执行简单的底层稳定和避障，人类负责高层路径和任务规划（如大多数 ROV）。
3. **有监督自主（Supervised Autonomy）**：机器人能自主执行预设任务，人类监督并可随时接管（如 AMR 自主导航）。
4. **高度自主（High Autonomy）**：机器人可独立完成复杂多步骤任务，仅在遇到超出能力边界时请求人类协助（如火星探测车）。
5. **全自主（Full Autonomy）**：机器人在无人干预的情况下完整执行任务，目前仅在高度受控环境中实现。


## 参考资料

1. [IEEE Spectrum: Robot Database](https://robots.ieee.org/)，IEEE
2. [International Federation of Robotics](https://ifr.org/)，IFR，《World Robotics Report》年度报告
3. [Boston Dynamics 官方网站](https://www.bostondynamics.com/)，Boston Dynamics
4. [Unitree Robotics 官方网站](https://www.unitree.com/)，宇树科技
5. [ANYbotics 官方网站](https://www.anybotics.com/)，ANYbotics
6. [Intuitive Surgical 官方网站](https://www.intuitivesurgical.com/)，Intuitive Surgical
7. [NASA Robotics](https://robotics.nasa.gov/)，NASA
8. Bruno Siciliano 等著，《Robotics: Modelling, Planning and Control》，Springer，2009
9. [DJI 官方网站](https://www.dji.com/)，大疆创新
10. [Universal Robots 官方网站](https://www.universal-robots.com/)，Universal Robots
11. [KUKA 官方网站](https://www.kuka.com/)，KUKA
12. [ABB Robotics 官方网站](https://new.abb.com/products/robotics)，ABB
13. [Clearpath Robotics 官方网站](https://clearpathrobotics.com/)，Clearpath Robotics
14. [Blue Robotics 官方网站](https://bluerobotics.com/)，Blue Robotics
15. [Franka Robotics 官方网站](https://franka.de/)，Franka Robotics
16. [CMR Surgical 官方网站](https://cmrsurgical.com/)，CMR Surgical
17. 中国机器人产业联盟（CRIA），《中国机器人产业发展报告》，2024
18. Niku, S. B.，《Introduction to Robotics: Analysis, Control, Applications》，Wiley，2020
19. [MathWorks Robotics Toolbox 文档](https://www.mathworks.com/products/robotics.html)，MathWorks
20. [ROS.org 官方文档](https://www.ros.org/)，Open Robotics
21. [Clearpath Robotics 机器人研究指南](https://clearpathrobotics.com/robots/)，Clearpath Robotics
22. Spong, M. W. 等著，《Robot Modeling and Control》，Wiley，2005
23. [DARPA Robotics Challenge 官方总结报告](https://www.darpa.mil/program/darpa-robotics-challenge)，DARPA，2015

