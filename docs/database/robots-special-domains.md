# 专用领域机器人图鉴

!!! note "引言"
    当作业环境对人类而言过于危险、狭小或遥远时，专用机器人往往是唯一可行的方案。医疗机器人追求亚毫米级精度与绝对的操作安全，无人飞行器受制于严苛的功重比，水下机器人需应对高压与通信受限，太空机器人则必须在极端温差与不可维修的条件下工作数年。本页面汇总医疗、无人机、水下、太空、服务与特种搜救六类机器人的代表产品。


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


## 参考资料

1. [IEEE Spectrum: Robot Database](https://robots.ieee.org/)，IEEE
2. [机器人图鉴总览](robots.md)
3. [功能安全](../hardware/functional-safety.md)
4. [传感器](../sensing/sensors.md)
