# 机器人竞赛

!!! note "引言"
    机器人竞赛（Robotics Competitions）是推动机器人技术创新与人才培养的重要引擎。从仿真足球场上的 RoboCup 到 DARPA 荒漠越野挑战赛，从校园普及的 ROBOCON 到面向工业前沿的 Amazon Robotics Challenge，各类竞赛共同构成了机器人技术生态的重要组成部分。本页面系统收录国际与国内主要机器人竞赛，涵盖赛制规则、历史沿革与参赛建议，供研究人员、工程师及在校学生参考。

---

## RoboCup 机器人世界杯

### 概述

RoboCup（Robot Soccer World Cup，机器人世界杯）由人工智能与机器人学领域的研究者于 1997 年在日本名古屋创办，是目前全球规模最大、历史最悠久的综合性机器人竞赛之一。RoboCup 的终极目标（Grand Challenge）是：**到 2050 年，组建一支完全自主的人形机器人足球队，能够在遵守 FIFA 规则的前提下，击败当届人类世界杯冠军队**。

RoboCup 每年举办一届世界锦标赛，并在全球各大洲举办区域赛（Regional Open）。除足球联赛外，RoboCup 还设有救援（Rescue）、家庭服务（@Home）、工业（Industrial）和青少年（Junior）等多个联赛，形成覆盖多技术方向的完整竞赛体系。

### RoboCup Soccer — 标准平台联赛（Standard Platform League，SPL）

标准平台联赛（SPL）是 RoboCup 足球联赛中最具影响力的子项之一，所有参赛队伍使用**完全相同的机器人硬件平台**，以统一硬件来凸显软件算法的差异。

- **指定平台**：2008 年起统一使用软银机器人公司（SoftBank Robotics）的 NAO 人形机器人（Humanoid Robot）。NAO 身高约 58 cm，配备立体摄像头、超声波传感器、惯性测量单元（IMU）及关节力矩传感器，采用 Aldebaran 操作系统（现为 NAOqi OS）。
- **比赛规则**：5 对 5 全自主对抗，机器人需自主感知场地、定位自身、识别球与队友、规划路径并执行射门、传球等动作，禁止任何人工干预。
- **技术难点**：视觉感知（球与球门检测）、机器人定位（自主蒙特卡洛定位，Monte Carlo Localization）、步态控制（动态行走与快速起身）、多机协作（队形与战术协调）。
- **中国队伍**：北京大学 NaoBit 队、浙江大学 ZJUNlict 队（曾获 SSL 世界冠军，也活跃于 SPL）、同济大学等均参与过 SPL 竞赛，并在亚太区域赛中多次获奖。

### RoboCup Soccer — 小型机器人联赛（Small Size League，SSL）

小型机器人联赛（SSL）是 RoboCup 中节奏最快、对抗最激烈的联赛，以极高的运动速度与精密的多机协作著称。

- **平台规格**：每队 6 台（含守门员）圆柱形轮式机器人，直径不超过 18 cm，高度不超过 15 cm；球场尺寸为 12 m × 9 m（大场地，Large Field）。
- **视觉系统**：场地上方架设统一的顶视摄像头（Overhead Camera），通过 SSL-Vision 软件将场地状态（机器人位置、球位置）广播至各队的主控计算机；机器人本体较为简单，主控逻辑在场外计算机（Off-board Computer）中运行。
- **技术亮点**：多机器人协同运动规划（Multi-Robot Motion Planning）、高速路径规划（速度可达 3 m/s 以上）、发球机构（踢球和挑球装置）设计，以及实时博弈策略。
- **中国队伍**：浙江大学 ZJUNlict 队是 SSL 历史上最成功的中国队伍，多次问鼎世界冠军（2015、2016、2017、2019、2022 年）；此外，华南理工大学、上海交通大学、北京信息科技大学等也有参赛队伍。

### RoboCup Soccer — 中型机器人联赛（Middle Size League，MSL）

中型机器人联赛（MSL）使用真实大小的足球，机器人需完全自主感知、定位与决策，对机器人本体设计要求极高。

- **平台规格**：每队最多 5 台机器人，单台机器人最大外形尺寸约 52 cm × 52 cm × 80 cm；球场为标准足球场的缩小版（18 m × 12 m）；使用标准橙色足球。
- **感知系统**：机器人自带全向视觉系统（Omnidirectional Vision）或多目摄像头，以及激光雷达（LiDAR）用于自身定位，无外部摄像头辅助。
- **技术难点**：带球运动（Dribbling）机构设计、全向移动底盘（Omnidirectional Chassis）与快速动态稳定控制、分布式多机协作通信。
- **代表队伍**：荷兰埃因霍芬理工大学（TU/e）Tech United 队长期处于顶尖水平；中国参赛队伍相对较少，但中科大等高校曾有参与经历。

### RoboCup Soccer — 人形机器人联赛（Humanoid League）

人形机器人联赛（Humanoid League）是技术挑战最大的联赛，使用完全双足的人形机器人在真实三维动态环境中踢球。

- **子类别**：按机器人身高分为成人组（AdultSize，身高 ≥ 140 cm）、青少年组（TeenSize，身高 100–140 cm）和青少年组（KidSize，身高 40–90 cm）。
- **挑战内容**：双足步态稳定性、动态平衡与摔倒恢复、球的视觉识别与跟踪、人形机器人射门动作规划。
- **代表队伍**：德国汉堡大学 Hamburg Bit-Bots、波恩大学 Nimbro 队长期处于领先地位；中国的北京航空航天大学（北航）、哈尔滨工业大学（哈工大）等曾参与 KidSize 联赛。

### RoboCup @Home — 家庭服务机器人联赛

RoboCup @Home 联赛旨在推动家庭服务机器人（Domestic Service Robot）的研究，参赛机器人需在模拟家庭环境中完成日常生活辅助任务。

- **典型任务**：物体识别与抓取（Object Recognition & Grasping）、人员跟随（Person Following）、语音指令理解与执行、场景中的自主导航与避障、"找到并拿来饮料"等综合生活任务。
- **联赛子项**：@Home DSPL（Domestic Standard Platform League，使用 Toyota HSR 或 Softbank Pepper 等固定平台）、@Home OPL（Open Platform League，参赛队自带机器人）、@Home SSPL（Social Standard Platform League）。
- **中国队伍**：中国科学技术大学（中科大）RoboWaiter 队、上海交通大学 RoboHome 队、北京大学等均参与过 @Home 联赛并获得优异成绩；中科大曾多次获得世界前三名。

### RoboCup Rescue — 救援机器人联赛

RoboCup Rescue 联赛起源于 1999 年土耳其大地震后，旨在推动机器人在灾难搜救（Disaster Response）场景中的应用研究。

- **Robot League（机器人实体赛）**：参赛机器人在模拟废墟（Disaster Arena）中执行搜索与定位幸存者任务，评分指标包括地图构建（Mapping）精度、幸存者识别率和遥操作效率。
- **Simulation League（仿真赛）**：在 RoboCupRescue Simulation 环境中，多智能体系统协调灭火、救援。
- **技术重点**：非结构化地形导航（Unstructured Terrain Navigation）、三维地图构建（3D Mapping）、幸存者热成像检测（Thermal Detection）、遥操作（Teleoperation）与半自主控制。

### RoboCup Industrial

RoboCup Industrial 聚焦工业物流（Industrial Logistics）场景，设有以下子赛事：

- **RoboCup@Work**：机器人在工厂模拟环境中执行物料搬运、工件识别与精密装配任务，使用 KUKA youBot 等移动操作平台。
- **RoboCup Logistics League（RCLL）**：多台机器人协作完成工厂物流模拟，包括从原材料仓库取件、加工并运送至交付点的完整流程，强调多机协调与任务规划。

### RoboCup Junior

RoboCup Junior 面向 19 岁以下青少年，设有足球（Soccer）、救援（Rescue）和表演（OnStage）三类子赛，是 RoboCup 体系中入门门槛最低、参与人数最多的联赛，在全球 50 余个国家设有地区预选赛。

---

## DARPA 挑战赛系列

美国国防高等研究计划署（Defense Advanced Research Projects Agency，DARPA）通过设立高奖金挑战赛（Grand Challenge）的方式推动机器人与自动驾驶技术的跨越式发展，是近 30 年来最具影响力的政府机构机器人推动力量。

### DARPA Grand Challenge — 自动驾驶沙漠挑战赛（2004、2005）

- **背景**：2001 年，美国国会要求国防部在 2015 年前使三分之一的地面作战车辆实现无人化。DARPA 以设立挑战赛的方式加速这一进程。
- **2004 年首届**：比赛地点为加利福尼亚州莫哈韦沙漠（Mojave Desert），全程约 240 km，奖金 100 万美元。结果：**无一参赛车辆完赛**，行驶最远的 Carnegie Mellon University（CMU）Sandstorm 也仅完成约 11.78 km（约 5%）。
- **2005 年第二届**：DARPA 将奖金提升至 200 万美元，吸引了 195 支报名队伍，最终 23 支完成资格赛进入决赛。斯坦福大学（Stanford University）Sebastian Thrun 团队的 **Stanley** 以 6 小时 54 分完成全程（212 km），夺得冠军；CMU 的 Sandstorm 和 Highlander 分列第二、三名。这次成功标志着自动驾驶技术从不可能走向可能，是自动驾驶历史上的里程碑事件。
- **核心技术**：斯坦福 Stanley 采用激光雷达（SICK LiDAR）、摄像头与 GPS 融合感知，基于机器学习（Machine Learning）进行路面分类，开创了数据驱动感知在自动驾驶中应用的先例。

### DARPA Urban Challenge — 城市自动驾驶挑战赛（2007）

- **赛制升级**：比赛场景从沙漠迁移至模拟城市街道（Victorville，加州），全程 96 km，要求参赛车辆**遵守交通法规（California Driver Code）**、与真实驾驶车辆混行、在路口礼让、完成停车等复杂城市驾驶任务，奖金提升至 200 万美元。
- **冠军**：CMU 与通用汽车（General Motors）联队的 **Boss**（一辆改装雪佛兰 Tahoe），以 4 小时 10 分完赛；斯坦福 Junior 获得第二名。
- **意义**：Urban Challenge 直接促成了谷歌（Google）自动驾驶项目（后来的 Waymo）的成立——Sebastian Thrun 和多名参赛选手加入谷歌，成为 Waymo 早期核心团队。

### DARPA Robotics Challenge（DRC）— 灾难救援机器人挑战赛（2012—2015）

- **背景**：2011 年日本福岛核电站（Fukushima Daiichi）事故后，DARPA 认识到灾难现场缺乏能替代人类进入危险环境的机器人，于 2012 年启动 DRC，奖金为 200 万美元（冠军）。
- **赛制**：参赛人形机器人（或半人形机器人）需在模拟工业灾难现场中完成 8 项任务，包括：驾驶车辆（Drive a Vehicle）、步行穿越废墟（Walk Over Rubble）、清除障碍物（Remove Debris）、开门（Open Door）、爬梯（Climb Ladder）、关闭阀门（Close Valve）、接插头（Connect Hose/Plug）和使用电动工具。通信受限（Degraded Communication）以模拟真实灾难条件。
- **2013 年 Trials（预赛）**：佛罗里达州举行，Atlas（Boston Dynamics 制造，软件由各队自主开发）是最主要平台，共 16 支队伍参赛，IHMC Robotics 和 Shaft（日本）表现最佳。
- **2015 年 Finals（决赛）**：加利福尼亚州 Pomona 举行，共 23 支队伍参赛。**韩国科学技术院（KAIST）DRC-HUBO** 以 44 分 28 秒完成全部 8 项任务夺冠（冠军奖金 200 万美元）；IHMC Robotics Running Man 获第二名；CMU Tartan Rescue 获第三名。
- **技术遗产**：DRC 催生了众多后续研究，Boston Dynamics Atlas 在 DRC 期间大幅迭代；许多关键技术（全身运动控制、实时运动规划）成为后来人形机器人研究的基础；KAIST DRC-HUBO 的改进版本 HUBO 后来成为重要的研究平台。

### DARPA Subterranean Challenge（SubT）— 地下探索挑战赛（2018—2021）

- **背景**：GPS 信号在隧道、矿洞、城市地下空间等地下环境中失效，DARPA 于 2018 年启动 SubT，旨在推动机器人在地下非结构化环境中自主探索的能力。
- **赛制**：参赛团队使用地面机器人与空中机器人（无人机）的组合，在三类环境（隧道系统 Tunnel Circuit、城市地下空间 Urban Circuit、洞穴系统 Cave Circuit）中自主搜索并定位人工制品（物品、幸存者假人、气体泄漏点等），奖金 200 万美元（最终冠军）。
- **2021 年总决赛**：路易斯安那州 Louisville Mega Cavern 举行。卡内基梅隆大学（CMU）与 Oregon State University 联合的 **CERBERUS 队**（由 ETH Zurich 领队的多高校联队）获得决赛冠军；CMU Explorer 队获亚军。
- **技术亮点**：多模态 SLAM（激光雷达 + 视觉 + IMU）、空地协同探索（Aerial-Ground Collaboration）、无通信辅助的分布式多机器人协调，以及对黑暗、烟雾等极端感知条件的鲁棒性设计。
- **中国参与**：浙江大学等高校在 SubT 相关技术方向有研究跟进，但中国队伍未直接参加 DARPA SubT 决赛（因参赛资格限制主要面向美国机构）。

---

## 中国主要机器人竞赛

### RoboMaster 机甲大师赛

RoboMaster（机甲大师赛）由大疆创新（DJI）于 2015 年发起，是中国规模最大、影响力最强的机器人对抗类竞赛，面向全球高校学生。

- **赛制概述**：每支参赛队伍需设计、制造并操控多种类型的机器人（英雄机器人、工程机器人、步兵机器人、哨兵机器人、无人机等）进行团队对抗。比赛分为线上初赛和现场决赛，决赛在深圳举办，规模宏大、观赏性强。
- **机器人类型**（以 2024 赛季为例）：
  - **步兵机器人（Infantry）**：数量最多（最多 3 台），是进攻主力，需快速移动并精确射击对方能量核心；
  - **英雄机器人（Hero）**：发射大弹丸，火力强大，每队 1 台；
  - **工程机器人（Engineer）**：负责复活己方阵亡机器人、夺取战场资源，每队 1 台；
  - **哨兵机器人（Sentry）**：固定在滑轨上自主巡逻防守，全自主运行；
  - **飞镖（Dart）**：发射导弹攻击对方基地，每队 1 台；
  - **无人机（Aerial）**：侦察与骚扰，2024 赛季已具备自主功能；
  - **基地（Base）**：需要保护的核心目标。
- **RoboMaster AI Challenge（RMAC）**：RoboMaster 框架下专门面向人工智能的竞赛子项，所有参赛机器人需**完全自主**运行（无人操控），考察感知、决策与运动控制算法。
- **技术培养**：RoboMaster 被公认为中国高校机器人实践能力培养最全面的平台之一，涵盖机械设计、电子硬件、嵌入式软件、计算机视觉与人工智能等全栈技术栈。许多 RoboMaster 参赛者后来成为大疆、华为、字节跳动等企业的核心技术人才。
- **代表队伍**：南方科技大学（南科大）、哈尔滨工业大学（哈工大）、上海交通大学（上交大）、北京航空航天大学（北航）、华南理工大学等院校的战队多次获得全国总冠军。

### ROBOCON 亚太大学机器人大赛

ROBOCON（Asia-Pacific Robot Contest）是由亚洲太平洋广播联盟（ABU）主办、各成员国广播电视台协办的年度大学生机器人竞赛，每年在不同亚太国家举办。

- **历史沿革**：ROBOCON 创办于 2002 年，首届在日本举办。中国赛区（全国大学生机器人大赛）由中国中央电视台（CCTV）组织承办，选拔国家代表队参加亚太决赛。
- **赛制特点**：每届比赛主题不同，设有固定场地、特定任务目标和规定时间，参赛队伍需设计**手动控制机器人（Manual Robot）**和**自动控制机器人（Automatic Robot）**各至少一台，协同完成任务。
- **典型历届主题**：投掷、过桥、堆叠积木、传球、模拟农耕等，主题通常与举办国文化相关。
- **中国成绩**：中国代表队在 ROBOCON 历史上成绩优异，多次获得亚太决赛冠军（Champion Award）；哈尔滨工业大学、浙江大学、北京航空航天大学等名校战队是传统强队。
- **意义**：ROBOCON 是中国高校机器人竞赛的"黄埔军校"，培养了大批机器人与自动化领域的工程技术人才。

### 中国机器人大赛（China Robot Competition，CRC）

中国机器人大赛（CRC）由中国自动化学会（Chinese Association of Automation，CAA）主办，是中国历史最悠久的机器人综合性竞赛之一。

- **赛事规模**：每年举办，参赛团队涵盖全国数百所高校和中学，设有足球、武术擂台、救援、水中机器人、服务机器人、仿生机器人等数十个竞赛项目。
- **主要子项**：
  - **机器人武术擂台**：机器人在擂台上对抗，将对手推出圈外或击倒；
  - **机器人足球**：分为仿真组、小型组、中型组和人形机器人组；
  - **水中机器人**：水下自主机器人完成识别、打捞等任务；
  - **服务机器人**：在模拟家庭或办公环境中完成抓取、导航、交互任务；
  - **创新创意赛**：鼓励原创性机器人系统设计。
- **中学赛道**：CRC 设有专门面向中学生的竞赛项目，是衔接青少年机器人教育与高校机器人竞赛的重要桥梁。

### 全国大学生机器人竞赛（其他）

除上述赛事外，教育部与各省市教育主管部门还组织了多项全国性竞赛：

- **全国大学生智能汽车竞赛**：基于飞思卡尔（现 NXP）微控制器的智能小车循迹与避障竞赛，参赛规模超过千所高校。
- **全国大学生电子设计竞赛（NUEDC）**：每两年一届，机器人控制是重要题目类别之一。
- **世界机器人大赛（WRC）**：由工业和信息化部（工信部）指导、中国电子学会（CIE）主办，是世界机器人大会（World Robot Conference）的重要组成部分，设有共融机器人、BCI 脑控机器人、青少年机器人创意赛等多项赛事。

---

## 无人机竞赛

### IROS 自主无人机竞速赛（IROS Autonomous Drone Racing）

依托 IEEE/RSJ 智能机器人与系统国际会议（IROS）举办的自主无人机竞速挑战赛，是学术界最具影响力的无人机竞赛之一。

- **赛制**：参赛无人机需在室内或室外赛道中，**完全自主**（禁止人工操控）地穿越一系列由 LED 框架标识的门（Gate），以最短时间完成全程为胜。
- **技术挑战**：高速状态下的实时视觉定位（Visual Odometry）、动态避障与门检测（Gate Detection）、高响应控制器设计（侵略性飞行机动，Aggressive Maneuvers）。
- **代表性研究**：苏黎世大学（University of Zurich）Davide Scaramuzza 团队的深度强化学习（Deep Reinforcement Learning）无人机竞速方案 CPC（Champion Policy Controller）曾在比赛中击败人类飞手，引发广泛关注（2023 年发表于 *Nature*）。

### AlphaPilot — 无人机自主飞行挑战赛

AlphaPilot 由 Lockheed Martin（洛克希德·马丁）和 Drone Racing League（DRL）于 2019 年联合发起，是奖金规模最大的无人机人工智能挑战赛。

- **规则**：参赛团队开发完全自主的无人机飞行算法，在真实赛道中与人类顶级 FPV 飞手同台竞速，无人机直径约 80 cm，飞行速度可达 80 mph（约 130 km/h）以上。
- **总奖金**：100 万美元，吸引了全球超过 400 支团队报名，最终 9 支入围决赛。
- **冠军**：SWIFT 团队（由苏黎世大学主导）获得最高成绩，其算法在速度上接近顶级人类飞手。

### IMAV — 国际微型飞行器大赛

国际微型飞行器大赛（International Micro Air Vehicle Conference and Competition，IMAV）是结合学术会议与工程竞赛的国际活动，每年在欧洲举办。

- **赛制**：分室内赛（Indoor）和室外赛（Outdoor）两大类，参赛微型飞行器（Micro Air Vehicle，MAV）需完成导航、目标识别、有效载荷投递等多种自主任务；室内赛严格禁止 GPS 等全球定位系统，仅依赖机载传感器完成自主飞行。
- **参与门槛**：面向全球大学和研究机构，中国高校如北京航空航天大学、浙江大学等曾参赛并取得较好成绩。

### 中国无人系统大赛

国内也涌现出多项专注无人机的竞赛，包括：

- **全国大学生无人飞行器智能感知技术竞赛**：教育部产学合作协同育人项目支持，重点考察无人机感知、建图与导航能力。
- **中国无人机创意大赛**：面向创新应用场景的无人机系统设计竞赛。
- **大疆 Sky City 大学生挑战赛**：依托 DJI 无人机平台，考察控制算法与自主任务能力。

---

## 物流与工业操作类竞赛

### Amazon Robotics Challenge（ARC）/ Amazon Picking Challenge（APC）

- **历史**：亚马逊物流机器人挑战赛（Amazon Picking Challenge，APC）于 2015 年首次举办，2017 年改名为 Amazon Robotics Challenge（ARC），此后暂停举办。
- **赛制**：参赛机器人需从标准货架（Shelf）或料箱（Bin）中自主抓取指定物品，并将其放置到指定位置，评分指标为单位时间内正确拣选物品的数量与准确率。物品种类繁多（数十至数百种），包括软包装、反光金属罐、书本等形状各异、材质不同的商品。
- **技术挑战**：多类别物体识别（Object Recognition）、姿态估计（Pose Estimation）、鲁棒抓取规划（Grasp Planning）与执行、软体物品处理（Deformable Object Manipulation）。
- **影响**：APC/ARC 极大地推动了机器人抓取（Robot Grasping）领域的研究，催生了 GraspNet 等开源数据集与抓取算法基准，促进了末端执行器（End-Effector）的多样化设计。

### Mohamed Bin Zayed International Robotics Challenge（MBZIRC）

- **概述**：MBZIRC 是由阿联酋阿布扎比高级技术研究委员会资助、哈利法科学技术大学（Khalifa University）承办的国际机器人挑战赛，奖金总额 500 万美元，每两年举办一届，面向全球顶尖研究机构。
- **赛制**（以 2020 年第二届为例）：包含三项挑战：
  - **挑战一**：无人机在城市建筑物墙面上自主定位并灭火（Firefighting）；
  - **挑战二**：空地协同机器人在港口环境中完成货物搬运任务；
  - **挑战三**：无人机与地面机器人协作，在指定区域内识别并拆除模拟炸弹。
- **参赛团队**：顶级机构包括苏黎世联邦理工学院（ETH Zurich）、卡内基梅隆大学（CMU）、麻省理工学院（MIT）等；中国的浙江大学、哈尔滨工业大学也曾参赛。
- **意义**：MBZIRC 是目前奖金最高、技术难度最大的国际机器人挑战赛之一，代表了空地协同自主机器人的前沿水平。

---

## 学术基准挑战赛

### NIST 应急响应机器人性能标准（NIST ARIAC）

NIST（美国国家标准与技术研究院）Agile Robotics for Industrial Automation Competition（ARIAC）是一项面向工业自动化场景的在线仿真竞赛。

- **赛制**：参赛队伍在 ROS（机器人操作系统）+ Gazebo 仿真环境中，开发机械臂抓取与装配算法，在模拟装配线上高效完成零件拣选与装配任务，同时应对传送带卡顿、零件丢失等随机干扰事件（Agility Challenges）。
- **特点**：纯软件竞赛，无需硬件投入，适合以算法研究为主的科研团队参与。

### ICRA / IROS 机器人竞赛子项

IEEE 机器人与自动化国际会议（ICRA）和 IEEE/RSJ 智能机器人与系统国际会议（IROS）每届均设置若干竞赛子项，题目覆盖：

- **灵巧操作（Dexterous Manipulation）**：如物体重新抓取（Regrasping）、工具使用；
- **移动操作（Mobile Manipulation）**：如 ICRA Robothon 挑战赛；
- **仿真到真实迁移（Sim-to-Real Transfer）**；
- **自主导航（Autonomous Navigation）**：如 BARN Challenge（密集障碍物导航基准）。

这些竞赛规模较小但学术影响力高，获奖成果通常直接发表于顶级机器人学术期刊。

---

## FIRST 系列竞赛

FIRST（For Inspiration and Recognition of Science and Technology）由美国工程师 Dean Kamen 于 1989 年创立，旨在通过机器人竞赛培养青少年对科学、技术、工程和数学（STEM）的兴趣。

### FIRST Robotics Competition（FRC）

- **面向群体**：高中生（14-18 岁），全球规模最大的高中生机器人竞赛之一，每年参赛队伍超过 3500 支。
- **赛制**：每年 1 月发布新赛题，各队有**6 周时间（Build Season）** 设计、制造并编程一台约 55 kg 的机器人。机器人需在规定场地内完成当年的主题任务（如投篮、攀爬架构等）。
- **精神内核**："Coopertition"（合作与竞争并重），强调团队协作、工程思维与工匠精神（Gracious Professionalism）。
- **资源保障**：每支队伍获得统一的硬件包（Kit of Parts），含电控系统、传感器和基础结构材料；软件支持 LabVIEW、Java 和 C++ 三种编程语言。

### FIRST Tech Challenge（FTC）

- **面向群体**：初中与高中生（12-18 岁），规模仅次于 FRC，每年参赛队伍超过 6000 支。
- **赛制**：使用 TETRIX 或 REV Robotics 等套件，机器人尺寸更小（18 英寸立方体内），编程语言支持 Java（Android Studio）和 Blocks 图形化编程。

### FIRST LEGO League（FLL）

- **面向群体**：分 FLL Explore（6-10 岁）和 FLL Challenge（9-14 岁）两档。
- **赛制**：使用 LEGO SPIKE Prime 或 LEGO Mindstorms 套件，结合机器人任务赛（Robot Game）、项目展示（Innovation Project）和核心价值评审（Core Values）三个维度综合评分。
- **中国推广**：FLL 在中国已有多个赛区，每年从各省赛区选拔团队参加全国赛，再遴选代表队参加世界锦标赛（FLL World Festival）。

---

## 参赛建议与备赛指南

### 选择适合的竞赛

| 竞赛方向 | 推荐竞赛 | 适合阶段 |
|----------|----------|----------|
| 自主移动与导航 | RoboCup SSL/MSL、DARPA SubT 相关、BARN Challenge | 本科高年级至研究生 |
| 人形机器人与步态 | RoboCup Humanoid League、CRC 人形赛 | 研究生及以上 |
| 家庭服务机器人 | RoboCup @Home | 研究生及以上 |
| 工业操作与抓取 | APC/ARC、ARIAC、ICRA 竞赛子项 | 研究生及以上 |
| 无人机自主飞行 | IROS Drone Racing、IMAV、DJI Sky City | 本科高年级至研究生 |
| 多机器人对抗 | RoboMaster 机甲大师赛 | 本科全阶段 |
| 青少年入门 | FRC、FTC、FLL、ROBOCON、WRO | 中学至本科低年级 |

### 技术准备建议

1. **软件栈**：熟练掌握 ROS（Robot Operating System）或 ROS 2，了解 Gazebo/Isaac Sim 等仿真工具，能够独立完成传感器驱动集成与调试。
2. **感知算法**：掌握摄像头标定、目标检测（YOLO 等）、激光雷达点云处理（PCL）、视觉里程计（Visual Odometry）等基础感知模块。
3. **规划与控制**：了解路径规划（A*、RRT、DWA）、运动控制（PID、MPC）及多机器人协调的基本原理。
4. **硬件集成**：具备一定的电路焊接、PCB 调试和机械加工能力，能独立排查硬件故障。
5. **团队协作**：使用 Git 进行版本管理，建立清晰的代码规范和文档习惯，定期进行内部技术分享。

### 参赛资源推荐

- **开源项目**：RoboCup 历届冠军队的代码通常开源（如 UT Austin Villa、ZJUNlict），是学习的宝贵资源。
- **官方文档**：各竞赛官网均提供规则手册（Rulebook）、技术描述文件（Team Description Paper，TDP）要求，认真研读是备赛的第一步。
- **论文研读**：在参赛前系统阅读该竞赛领域的 3-5 篇代表性论文，了解当前技术前沿与主流方案。
- **仿真先行**：在购置昂贵硬件前，优先在仿真环境中验证算法方案，降低试错成本。

---

## 参考资料

1. RoboCup Federation. *RoboCup Official Website*. https://www.robocup.org/
2. RoboCup Technical Committee. *RoboCup Standard Platform League (NAO) Rule Book 2024*. https://spl.robocup.org/
3. ZJUNlict Team. *ZJUNlict Extended Team Description Paper for RoboCup 2019*. https://github.com/ZJUNlict
4. DARPA. *DARPA Grand Challenge: Ten Years Later*. https://www.darpa.mil/news-events/2014-02-11
5. Thrun, S. et al. "Stanley: The Robot That Won the DARPA Grand Challenge." *Journal of Field Robotics*, 23(9), 2006. https://doi.org/10.1002/rob.20147
6. DARPA. *DARPA Robotics Challenge (DRC) Finals Results*. https://www.darpa.mil/program/darpa-robotics-challenge
7. DARPA. *DARPA Subterranean Challenge Final Results*. https://www.subtchallenge.com/
8. RoboMaster. *RoboMaster 机甲大师赛官方网站*. https://www.robomaster.com/
9. ABU ROBOCON. *ABU Asia-Pacific Robot Contest Official Website*. https://www.aburobоcon.tv/
10. 中国自动化学会. *中国机器人大赛（CRC）官方网站*. http://www.caa.net.cn/
11. Loquercio, A. et al. "Champion-level drone racing using deep reinforcement learning." *Nature*, 620, 2023. https://doi.org/10.1038/s41586-023-06419-4
12. Correll, N. et al. "Analysis and Observations from the First Amazon Picking Challenge." *IEEE Transactions on Automation Science and Engineering*, 15(1), 2018.
13. MBZIRC. *Mohamed Bin Zayed International Robotics Challenge Official Website*. https://www.mbzirc.com/
14. FIRST. *FIRST Robotics Competition Official Website*. https://www.firstinspires.org/robotics/frc
15. NIST. *Agile Robotics for Industrial Automation Competition (ARIAC)*. https://www.nist.gov/el/intelligent-systems-division-73500/agile-robotics-industrial-automation-competition

