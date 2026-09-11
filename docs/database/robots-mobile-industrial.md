# 移动与工业机器人图鉴

!!! note "引言"
    移动与工业机器人是目前出货量最大、商业模式最成熟的机器人品类。轮式移动机器人（含自主移动机器人 AMR）与仓储物流机器人构成了现代仓库自动化的主体；工业机械臂在汽车与 3C 制造中已有数十年积累；协作机器人（Collaborative Robot, Cobot）则通过力矩感知与安全限速，把机械臂从安全围栏中解放出来，使其可与人共享工作空间。本页面汇总这四类平台的代表产品。


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


## 参考资料

1. [IEEE Spectrum: Robot Database](https://robots.ieee.org/)，IEEE
2. [机器人图鉴总览](robots.md)
3. [协作机器人安全](../hardware/safety-collaborative-robot.md)
4. [机器人操作](../manipulation/index.md)
5. [导航](../planning/navigation.md)
