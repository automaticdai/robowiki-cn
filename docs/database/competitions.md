# 机器人竞赛

!!! note "引言"
    机器人竞赛（Robotics Competitions）是推动机器人技术创新与人才培养的重要引擎。从仿真足球场上的 RoboCup 到 DARPA 荒漠越野挑战赛，从校园普及的 ROBOCON 到面向工业前沿的 Amazon Robotics Challenge，各类竞赛共同构成了机器人技术生态的重要组成部分。本页面系统收录国际与国内主要机器人竞赛，涵盖赛制规则、历史沿革与参赛建议，供研究人员、工程师及在校学生参考。

---

## RoboCup 机器人世界杯

### 概述

RoboCup（Robot Soccer World Cup，机器人世界杯）由人工智能与机器人学领域的研究者于 1997 年在日本名古屋创办，是目前全球规模最大、历史最悠久的综合性机器人竞赛之一。创始人包括日本大阪大学的浅田稔（Minoru Asada）和九州大学的安西祐一郎（Yuichiro Anzai），以及多伦多大学的 Alan Mackworth 等。

RoboCup 的终极目标（Grand Challenge）是：**到 2050 年，组建一支完全自主的人形机器人足球队，能够在遵守 FIFA 规则的前提下，击败当届人类世界杯冠军队**。这一目标不仅是技术愿景，更是推动感知、规划、控制、多机协作等核心机器人技术协同发展的重要牵引力。

RoboCup 每年举办一届世界锦标赛，赛址轮流在世界各地举行（历届举办城市包括名古屋、斯德哥尔摩、巴黎、波士顿、大阪、里斯本、上海等）；并在全球各大洲举办区域赛（Regional Open）。除足球联赛外，RoboCup 还设有救援（Rescue）、家庭服务（@Home）、工业（Industrial）和青少年（Junior）等多个联赛，形成覆盖多技术方向的完整竞赛体系，每届参赛队伍来自全球 40 余个国家和地区，参赛人数超过 3000 人。

### RoboCup Soccer — 标准平台联赛（Standard Platform League，SPL）

标准平台联赛（Standard Platform League，SPL）是 RoboCup 足球联赛中最具影响力的子项之一，所有参赛队伍使用**完全相同的机器人硬件平台**，以统一硬件来凸显软件算法的差异。

- **指定平台**：2008 年起统一使用软银机器人公司（SoftBank Robotics）的 NAO 人形机器人（Humanoid Robot）。NAO 身高约 58 cm，重约 5.4 kg，配备立体摄像头、超声波传感器、惯性测量单元（Inertial Measurement Unit，IMU）及关节力矩传感器，采用 Aldebaran 操作系统（现为 NAOqi OS），具有 25 个自由度（Degrees of Freedom，DoF）。
- **比赛规则**：5 对 5 全自主对抗，机器人需自主感知场地、定位自身、识别球与队友、规划路径并执行射门、传球等动作，禁止任何人工干预。比赛分上下半场，每半场 10 分钟。
- **技术难点**：视觉感知（球与球门检测）、机器人自定位（基于场地线特征的蒙特卡洛定位，Monte Carlo Localization）、步态控制（动态行走与快速起身）、多机协作（队形与战术协调）。
- **发展历程**：SPL 的前身为 Four-Legged League（四足犬联赛），使用索尼爱宝（AIBO）机器狗。2008 年转换平台至 NAO，成为目前的 SPL。
- **中国队伍**：北京大学 NaoBit 队、浙江大学 ZJUNlict 队（曾获 SSL 世界冠军，也活跃于 SPL）、同济大学等均参与过 SPL 竞赛，并在亚太区域赛中多次获奖。

### RoboCup Soccer — 小型机器人联赛（Small Size League，SSL）

小型机器人联赛（Small Size League，SSL）是 RoboCup 中节奏最快、对抗最激烈的联赛，以极高的运动速度与精密的多机协作著称，被誉为机器人领域的"方程式赛车"。

- **平台规格**：每队 6 台（含守门员）圆柱形轮式机器人，直径不超过 18 cm，高度不超过 15 cm；球场尺寸为 12 m × 9 m（大场地，Large Field）；机器人底盘通常采用全向移动轮（Omnidirectional Wheel）设计，以实现任意方向的快速移动。
- **视觉系统**：场地上方架设统一的顶视摄像头（Overhead Camera），通过 SSL-Vision 软件将场地状态（机器人位置、球位置）以约 60 Hz 的频率广播至各队的主控计算机；机器人本体较为简单，主控逻辑在场外计算机（Off-board Computer）中运行，通过无线通信下发指令。
- **技术亮点**：多机器人协同运动规划（Multi-Robot Motion Planning）、高速路径规划（速度可达 3–4 m/s）、发球机构（踢球和挑球装置）设计，以及基于博弈论（Game Theory）的实时战术决策。
- **历届冠军**：CMU 的 CMDragons 曾长期统治该联赛；浙江大学 ZJUNlict 自 2015 年起多次夺冠（2015、2016、2017、2019、2022 年），成为 SSL 历史上最成功的中国队伍。
- **中国其他队伍**：华南理工大学（HUST Wolves）、上海交通大学、北京信息科技大学（BISTU Robotics）等也参与 SSL 竞赛。

### RoboCup Soccer — 中型机器人联赛（Middle Size League，MSL）

中型机器人联赛（Middle Size League，MSL）使用真实大小的足球，机器人需完全自主感知、定位与决策，对机器人本体设计要求极高，是最接近真实足球场景的 RoboCup 联赛。

- **平台规格**：每队最多 5 台机器人，单台机器人最大外形尺寸约 52 cm × 52 cm × 80 cm，最大重量约 40 kg；球场为标准足球场的缩小版（18 m × 12 m）；使用标准橙色足球。
- **感知系统**：机器人自带全向视觉系统（Omnidirectional Vision，通过鱼眼镜头或抛物面镜实现 360° 视野）或多目摄像头，以及激光雷达（LiDAR）用于自身定位，无任何外部摄像头辅助，机器人需完全依赖自身传感器完成感知与定位。
- **技术难点**：带球运动（Dribbling）机构设计（需主动夹持球）、全向移动底盘（Omnidirectional Chassis）与快速动态稳定控制、分布式多机协作通信（基于 WiFi 的去中心化协调）。
- **代表队伍**：荷兰埃因霍芬理工大学（TU/e）Tech United 队长期处于顶尖水平；葡萄牙 CAMBADA 队、巴西 THUNDERBOTS 队也是传统强队；中国参赛队伍相对较少，但中科大等高校曾有参与经历。

### RoboCup Soccer — 人形机器人联赛（Humanoid League）

人形机器人联赛（Humanoid League）是技术挑战最大的联赛，使用完全双足的人形机器人在真实三维动态环境中踢球，其难度远超其他足球联赛。

- **子类别**：按机器人身高分为成人组（AdultSize，身高 ≥ 140 cm）、青少年组（TeenSize，身高 100–140 cm）和儿童组（KidSize，身高 40–90 cm）三档，各档均有独立的比赛规则。
- **挑战内容**：双足步态稳定性（保持动态平衡）、摔倒恢复（Fall Recovery）、球的视觉识别与跟踪（球为标准黑白足球）、人形机器人射门动作规划（兼顾速度与稳定性）。
- **技术进展**：2019 年 RoboCup 世界杯期间，KidSize 组首次实现了真正意义上的全自主 2 对 2 对抗；成人组的机器人运动速度和稳定性也在逐届竞赛中显著提升。
- **代表队伍**：德国汉堡大学 Hamburg Bit-Bots 队、波恩大学 Nimbro 队（AdultSize 组强队）长期处于领先地位；荷兰代尔夫特理工大学（TU Delft）ToroD 队擅长动态步态；中国的北京航空航天大学（北航）、哈尔滨工业大学（哈工大）等曾参与 KidSize 联赛。

### RoboCup @Home — 家庭服务机器人联赛

RoboCup @Home 联赛旨在推动家庭服务机器人（Domestic Service Robot）的研究，参赛机器人需在模拟家庭环境中完成日常生活辅助任务，是 RoboCup 中与产业应用结合最紧密的联赛。

- **典型任务**：物体识别与抓取（Object Recognition & Grasping）、人员跟随（Person Following）、语音指令理解与执行（自然语言处理，Natural Language Processing）、场景中的自主导航与避障、"找到并拿来饮料"等综合生活任务、人物识别（Face Recognition）等。
- **联赛子项**：
  - @Home DSPL（Domestic Standard Platform League）：使用 Toyota HSR（Human Support Robot）或 SoftBank Pepper 等固定平台；
  - @Home OPL（Open Platform League）：参赛队自行设计与制造机器人，灵活度最高；
  - @Home SSPL（Social Standard Platform League）：强调社交互动与人机协作场景。
- **评分方式**：采用任务积分制，机器人每成功完成一个子任务步骤获得对应积分，总积分最高者获胜；评委包括技术评审和普通观众评审，兼顾技术水平与用户体验。
- **中国队伍**：中国科学技术大学（中科大）RoboWaiter 队（后更名为 KeJia 队）是中国在 @Home 联赛中成绩最好的队伍，多次获得世界前三名，并于 2014 年首次获得 OPL 组世界冠军；上海交通大学、北京大学等也有参与。

### RoboCup Rescue — 救援机器人联赛

RoboCup Rescue 联赛起源于 1999 年土耳其大地震和 2001 年美国 9·11 事件后的反思——真实灾难中缺乏能够代替人类进入危险环境执行搜救任务的机器人。该联赛旨在推动机器人在灾难搜救（Disaster Response）场景中的应用研究。

- **Robot League（机器人实体赛）**：参赛机器人在模拟废墟（Disaster Arena）中执行搜索与定位幸存者任务，场地模拟地震废墟，包含不同难度的地形（平坦、斜面、楼梯、碎石堆）。评分指标包括地图构建（Mapping）精度、幸存者识别率（基于摄像头、热成像等传感器）和遥操作效率。
- **Simulation League（仿真赛）**：在 RoboCupRescue Simulation（RCRS）平台中，多智能体系统（Multi-Agent System）协调灭火、救援、疏散，强调高层任务规划与分布式协调。
- **技术重点**：非结构化地形导航（Unstructured Terrain Navigation）、三维地图构建（3D Mapping，通常基于点云 SLAM）、幸存者热成像检测（Thermal Detection）、遥操作（Teleoperation）与半自主控制（Shared Autonomy）、气体检测等多传感器融合。

### RoboCup Industrial

RoboCup Industrial 聚焦工业物流（Industrial Logistics）场景，旨在推动机器人在制造业和物流业中的自动化应用研究，设有以下子赛事：

- **RoboCup@Work**：机器人在工厂模拟环境中执行物料搬运、工件识别与精密装配任务。参赛机器人需在工厂地图中自主导航，从指定位置抓取特定工件并运送至目标位置，同时应对静态和动态障碍物。早期指定使用 KUKA youBot 平台，后逐步开放平台。
- **RoboCup Logistics League（RCLL）**：多台机器人协作完成工厂物流模拟，包括从原材料仓库取件、在加工站加工并运送至交付点的完整生产流程，强调多机协调（Multi-Robot Coordination）与任务规划（Task Planning）。场地模拟 Festo 模块化生产系统（Modular Production System，MPS）。

### RoboCup Junior

RoboCup Junior 面向 19 岁以下青少年，设有足球（Soccer）、救援（Rescue）和表演（OnStage）三类子赛，是 RoboCup 体系中入门门槛最低、参与人数最多的联赛，在全球 50 余个国家和地区设有地区预选赛，是青少年机器人教育的重要平台。

- **Soccer 子项**：2 对 2 的小型足球对抗，机器人需具备自主红外寻球和射门能力；分 LightWeight 和 Open 两个级别。
- **Rescue 子项**：机器人在模拟灾难场地（迷宫形式）中自主导航并识别幸存者（颜色标识或热成像目标）；分 Line 和 Maze 两个难度级别。
- **OnStage 子项**：机器人配合音乐进行表演，评分维度包括技术复杂度、创意性和舞台效果。

---

## DARPA 挑战赛系列

美国国防高等研究计划署（Defense Advanced Research Projects Agency，DARPA）通过设立高奖金挑战赛（Grand Challenge）的方式推动机器人与自动驾驶技术的跨越式发展，是近三十年来最具影响力的政府机构机器人推动力量。DARPA 的挑战赛模式后来被世界各国广泛借鉴，成为政府推动前沿技术发展的重要政策工具。

### DARPA Grand Challenge — 自动驾驶沙漠挑战赛（2004、2005）

- **政策背景**：2001 年，美国国会通过《2001 年国防授权法案》（National Defense Authorization Act），要求国防部在 2015 年前使三分之一的地面作战车辆实现无人化。DARPA 以设立挑战赛的方式加速这一进程，并以奖金激励机制吸引全美最优秀的工程团队参与。
- **2004 年首届**：比赛地点为加利福尼亚州莫哈韦沙漠（Mojave Desert），全程约 240 km，奖金 100 万美元。结果：**无一参赛车辆完赛**，行驶最远的卡内基梅隆大学（Carnegie Mellon University，CMU）Sandstorm 也仅完成约 11.78 km（约 5%）即发生故障停车，反映出当时自动驾驶技术的巨大局限。
- **2005 年第二届**：DARPA 将奖金提升至 200 万美元，吸引了 195 支报名队伍，经过资格赛最终 23 支进入决赛。斯坦福大学（Stanford University）Sebastian Thrun 团队的 **Stanley**（改装大众途锐）以 6 小时 54 分完成全程（212 km），夺得冠军；CMU 的 Sandstorm 和 Highlander 分列第二、三名，共 5 支队伍完成全程。这次成功标志着自动驾驶技术从"理论上可行"走向"工程上实现"，是自动驾驶历史上划时代的里程碑事件。
- **核心技术**：斯坦福 Stanley 采用 5 台激光雷达（SICK LiDAR）、摄像头与 GPS/INS 融合感知，通过机器学习（Machine Learning）对地形进行可通行性分类（Terrain Classification），并采用基于概率的路径规划。CMU 采用了更为保守的纯几何方法，但机器人可靠性略低。

### DARPA Urban Challenge — 城市自动驾驶挑战赛（2007）

- **赛制升级**：比赛场景从沙漠迁移至加利福尼亚州维克托维尔（Victorville）封闭式模拟城市街道，全程 96 km，要求参赛车辆**遵守交通法规（California Driver Code）**、与真实驾驶的无人车混行、在路口礼让行人与车辆、完成停车入库、双向道行驶等复杂城市驾驶任务，奖金提升至 200 万美元（冠军）、100 万美元（亚军）、50 万美元（季军）。
- **冠军**：CMU 与通用汽车（General Motors）联合团队的 **Boss**（一辆改装雪佛兰 Tahoe 皮卡），以 4 小时 10 分完赛，平均时速约 22 km/h；斯坦福 Junior（改装大众帕萨特）获得第二名；弗吉尼亚理工大学 Victor Tango 获第三名。
- **技术升级**：与沙漠赛相比，城市赛要求车辆具备对其他运动目标（Moving Objects）的感知与预测能力，以及高精度 HD 地图（High-Definition Map）构建与使用能力，难度大幅提升。
- **深远意义**：Urban Challenge 直接催生了谷歌（Google）自动驾驶项目——Sebastian Thrun、Mike Montemerlo 等斯坦福核心成员，以及 CMU 团队的多位成员加入谷歌，成为后来 Waymo 的早期核心团队，开启了商业化自动驾驶时代。

### DARPA Robotics Challenge（DRC）— 灾难救援机器人挑战赛（2012—2015）

- **政策背景**：2011 年 3 月，日本福岛第一核电站（Fukushima Daiichi Nuclear Power Plant）因地震引发的海啸严重损毁，由于辐射环境极为危险，无法派遣人工进行应急处置。DARPA 认识到机器人技术在灾难响应领域的巨大缺口，于 2012 年 10 月启动 DRC，总奖金 350 万美元（冠军 200 万、亚军 100 万、季军 50 万美元）。
- **赛制设计**：参赛人形机器人（或半人形机器人）需在模拟工业灾难现场（Simulated Disaster Site）中完成 8 项递进式任务：
  1. 驾驶车辆（Drive a Vehicle）并下车
  2. 步行穿越碎石废墟（Walk Over Rubble）
  3. 清除障碍物（Remove Debris）——开门前的障碍物清理
  4. 开门（Open Door）并进入建筑
  5. 爬梯（Climb Ladder）
  6. 关闭工业阀门（Close Valve）
  7. 插接软管或电连接器（Connect Hose/Plug）
  8. 使用电动工具（Cut through Wall with Power Tool）破墙
  - 比赛时通信受限（Degraded Communication），模拟真实灾难条件下无线电通信受干扰的情况，每队通信带宽限制为约 300 bps，并引入随机延迟（Latency）。
- **2013 年 Trials（预赛）**：在佛罗里达州 Homestead 举行，Boston Dynamics 为多个参赛队提供 Atlas 机器人硬件，各队自主开发软件控制系统；此外还有团队自研机器人参赛。共 16 支队伍参赛，结果 IHMC Robotics（使用 Atlas）和日本 Shaft（自研机器人）表现最佳。
- **2015 年 Finals（决赛）**：在加利福尼亚州波莫纳（Pomona）举行，共 23 支队伍参赛。**韩国科学技术院（KAIST）的 DRC-HUBO** 以 44 分 28 秒完成全部 8 项任务夺冠（总冠军，奖金 200 万美元）；IHMC Robotics Running Man 获第二名（50 分 26 秒）；CMU Tartan Rescue 获第三名。大量机器人因稳定性问题在测试中摔倒，引发广泛关注，反映出人形机器人在复杂任务中的脆弱性。
- **技术遗产**：DRC 直接推动了人形机器人技术的快速发展。Boston Dynamics 在 DRC 期间多次迭代 Atlas，最终发展出当前的全电动版本；KAIST DRC-HUBO 的独特设计（膝盖可向前或向后弯曲，支持跪姿移动）成为工程创新的经典案例；全身运动控制（Whole-Body Control，WBC）、运动规划（Motion Planning）等核心算法在此期间得到大幅提升。

### DARPA Subterranean Challenge（SubT）— 地下探索挑战赛（2018—2021）

- **技术背景**：GPS（全球定位系统）信号在隧道、矿洞、地下城市空间、自然洞穴等地下环境中完全失效，传统机器人导航方法难以适用。DARPA 于 2018 年启动 SubT，旨在推动机器人在地下非结构化环境中自主探索、感知与映射的能力，奖金 200 万美元（最终冠军）。
- **三阶段赛制**：
  - **Tunnel Circuit（2019 年）**：隧道系统，模拟矿井或市政隧道；
  - **Urban Circuit（2020 年）**：城市地下空间，模拟地铁站或停车场；
  - **Cave Circuit（2020 年）**：自然洞穴系统，地形最为不规则。
  - 每阶段均在真实地下设施中举行，机器人需在 60 分钟内自主搜索并精确定位尽可能多的"物品"（包括幸存者假人、手机、气体泄漏标识等）。
- **2021 年总决赛**：在路易斯安那州路易斯维尔大洞（Louisville Mega Cavern）举行，综合三类地下环境。由瑞士联邦理工学院（ETH Zurich）领队、联合多所高校的 **CERBERUS 联队**获得冠军（10 个物品，1000 万美元研究基金）；CMU + Oregon State 联队的 **Explorer** 队获得亚军（9 个物品，虚拟决赛冠军）。
- **技术亮点**：多模态同步定位与建图（Multi-Modal SLAM：激光雷达 + 视觉 + IMU + 气压计）、空地协同自主探索（Aerial-Ground Collaborative Exploration）、无中央通信的分布式多机器人协调（Communication-Denied Multi-Robot Coordination），以及针对黑暗、粉尘、烟雾等极端感知条件的鲁棒性设计。

---

## 中国主要机器人竞赛

### RoboMaster 机甲大师赛

RoboMaster（机甲大师赛）由大疆创新（DJI）于 2015 年发起创办，面向全球高校学生，是中国规模最大、影响力最强、观赏性最佳的机器人对抗类竞赛，也是中国高校机器人工程实践教育的标杆平台。

- **赛制概述**：每支参赛队伍需设计、制造并操控多种类型的机器人进行团队对抗。比赛分为线上初赛（算法挑战赛）、线下区域赛和深圳总决赛三个阶段，总决赛在深圳举办，现场观众席超过万人，并同步进行网络直播。
- **机器人类型**（以 2024 赛季规则为例）：

  | 机器人类型 | 数量上限 | 主要功能 | 操控方式 |
  |------------|----------|----------|----------|
  | 步兵机器人（Infantry） | 3 台 | 进攻主力，高速移动精确射击 | 人工遥控 |
  | 英雄机器人（Hero） | 1 台 | 发射大弹丸，重火力输出 | 人工遥控 |
  | 工程机器人（Engineer） | 1 台 | 复活队友机器人、夺取资源 | 人工遥控 |
  | 哨兵机器人（Sentry） | 1 台 | 滑轨巡逻防守 | 全自主运行 |
  | 飞镖（Dart） | 1 台 | 远程导弹攻击对方基地 | 人工遥控 |
  | 无人机（Aerial） | 1 台 | 侦察、骚扰、空中射击 | 2024 赛季引入自主模式 |

- **RoboMaster AI Challenge（RMAC）**：RoboMaster 框架下专门面向人工智能研究的子赛项，所有参赛机器人需**完全自主**运行（禁止任何人工干预），重点考察目标检测（Object Detection）、自主导航（Autonomous Navigation）、决策规划（Decision Planning）算法的综合能力。RMAC 参赛队伍通常来自各高校机器人实验室或 AI 研究团队。
- **技术培养价值**：RobOMaster 被公认为中国高校机器人实践能力培养最全面的竞赛平台，覆盖机械设计（CAD/CAM）、电子硬件（PCB 设计、电机驱动）、嵌入式软件（STM32、RTOS）、计算机视觉（OpenCV、深度学习）与人工智能等全栈技术。许多 RoboMaster 参赛者后来成为大疆、华为、字节跳动、百度、商汤等企业的核心技术人才。
- **代表队伍**：南方科技大学（南科大）、哈尔滨工业大学（哈工大）、上海交通大学（上交大）、北京航空航天大学（北航）、华南理工大学、电子科技大学等院校的战队多次获得全国总冠军；深圳大学、西北工业大学、湖南大学等校也是近年崛起的强队。

### ROBOCON 亚太大学机器人大赛

ROBOCON（Asia-Pacific Robot Contest，亚太大学机器人大赛）是由亚洲太平洋广播联盟（Asia-Pacific Broadcasting Union，ABU）主办、各成员国广播电视台协办的年度大学生机器人竞赛，每年在不同亚太国家和地区举办，是亚太地区影响力最大的大学生机器人竞赛之一。

- **历史沿革**：ROBOCON 创办于 2002 年，首届在日本举办。中国赛区（全国大学生机器人大赛，CCTV 机器人大赛）由中国中央电视台（CCTV）组织承办，每年选拔优秀队伍代表中国参加亚太决赛。
- **赛制特点**：每届比赛主题不同（由当年举办国提出），设有固定场地、特定任务目标和规定完成时间，参赛队伍需自行设计并制造**手动控制机器人（Manual Robot）**和**自动控制机器人（Automatic Robot）**各至少一台，两台机器人协同完成任务。
- **典型历届主题**：
  - 2002 年（日本）：机器人相扑（类似相扑的推拉对抗）
  - 2010 年（中国）：模拟古代文明投石车
  - 2017 年（越南）：传统越南竹竿舞（机器人从竹竿间穿越）
  - 2023 年（印度）：传统陀螺竞技
- **中国成绩**：中国代表队在 ROBOCON 历史上成绩优异，多次获得亚太决赛冠军（Champion Award）和最佳设计奖；哈尔滨工业大学、浙江大学、东北大学、北京航空航天大学等院校是传统强队。
- **意义**：ROBOCON 是中国高校机器人竞赛的重要培养平台，每年吸引百余所高校参与中国区选拔赛，培养了大批机器人与自动化领域的优秀工程师。

### 中国机器人大赛（China Robot Competition，CRC）

中国机器人大赛（China Robot Competition，CRC）由中国自动化学会（Chinese Association of Automation，CAA）主办，创办于 1999 年，是中国历史最悠久的机器人综合性竞赛之一，至今已举办二十余届。

- **赛事规模**：每年举办一届，参赛团队涵盖全国数百所高校和中学，设有足球、武术擂台、救援、水中机器人、服务机器人、仿生机器人、创新创意等数十个竞赛项目，是中国参赛院校数量最多的综合性机器人赛事。
- **主要子项**：
  - **机器人武术擂台**：机器人在规定擂台上进行自主对抗，将对手推出圈外或击倒，分为有线遥控和无线自主两个组别；
  - **机器人足球**：分为仿真组、小型组（3 对 3）、中型组和人形机器人组，与 RoboCup 规则体系对接；
  - **水中机器人**：水下自主机器人完成水下目标识别、打捞、定点运动等任务；
  - **服务机器人**：在模拟家庭或办公环境中完成物体识别抓取、自主导航、自然语言交互任务；
  - **创新创意赛**：鼓励原创性机器人系统设计，不设固定任务，由专家评委综合评分。
- **中学赛道**：CRC 设有专门面向中学生的竞赛项目（机器人综合信息技术应用、仿真机器人等），是衔接青少年机器人科普教育与高校机器人竞赛的重要桥梁。

### 全国大学生机器人相关竞赛（其他）

除上述主体赛事外，教育部与各省市教育主管部门还组织了多项与机器人技术密切相关的全国性竞赛，为不同技术方向的学生提供展示平台：

- **全国大学生智能汽车竞赛**：由教育部高等学校自动化专业教学指导委员会主办，基于 NXP（原飞思卡尔）微控制器的智能小车循迹与避障竞赛，参赛规模超过千所高校，分摄像头组、激光组、电磁组、信标组等多个赛道。
- **全国大学生电子设计竞赛（NUEDC）**：每两年一届，机器人控制（如送餐机器人、桥形起重机控制等）是历届重要题目类别之一。
- **世界机器人大赛（World Robot Contest，WRC）**：由工业和信息化部（工信部）指导、中国电子学会（CIE）主办，是世界机器人大会（World Robot Conference）的核心竞赛组成部分，设有共融机器人挑战赛、BCI 脑控机器人大赛、青少年机器人创意赛等多项赛事，奖励丰厚且与产业结合紧密。
- **中国工程机器人大赛暨国际公开赛**：由教育部工程训练教学指导委员会主办，涵盖擂台对抗、竞速、创意设计等多个子项，面向工科高校学生。

---

## 无人机竞赛

### IROS 自主无人机竞速赛（IROS Autonomous Drone Racing）

依托 IEEE/RSJ 智能机器人与系统国际会议（International Conference on Intelligent Robots and Systems，IROS）每年举办的自主无人机竞速挑战赛，是学术界最具影响力的无人机自主飞行竞赛之一，每届均附设于 IROS 大会。

- **赛制**：参赛无人机需在室内或室外预设赛道中，**完全自主**（禁止任何形式的人工遥控）地依次穿越一系列由 LED 灯环或彩色方框标识的门（Gate），以最短时间完成全程为胜。机器人仅允许使用机载传感器（Onboard Sensors），禁止使用外部定位系统（如动作捕捉 Motion Capture 或 GPS）。
- **技术挑战**：高速飞行状态（速度可达 10 m/s 以上）下的实时视觉里程计（Visual Odometry）与状态估计（State Estimation）、动态门检测（Gate Detection，通常基于深度学习目标检测）、高带宽低延迟的姿态控制器设计（侵略性飞行机动，Aggressive Maneuvers 如急转弯和翻滚），以及对传感器噪声的鲁棒性。
- **代表性研究成果**：苏黎世大学（University of Zurich）机器人与感知团队（Robotics and Perception Group，RPG）Davide Scaramuzza 教授领导的研究团队，于 2023 年在 *Nature* 杂志发表论文，报告其深度强化学习（Deep Reinforcement Learning）训练的无人机算法 Swift 在 IROS Drone Racing 赛道上首次以高成功率击败世界顶级人类 FPV 飞手，成为机器人竞速领域的里程碑事件。

### AlphaPilot — 无人机自主飞行挑战赛

AlphaPilot 由洛克希德·马丁（Lockheed Martin）和无人机竞速联盟（Drone Racing League，DRL）于 2019 年联合发起，是迄今为止奖金规模最大的无人机人工智能（Drone AI）挑战赛。

- **规则设计**：参赛团队开发完全自主的无人机飞行算法（软件），由 DRL 提供统一的 DRL RacerAI 竞速无人机硬件（直径约 80 cm 的竞速穿越机）；算法需在真实 DRL 赛道中与人类顶级 FPV 飞手同台竞速，无人机飞行速度可达 80 mph（约 130 km/h）以上。
- **规模与奖金**：总奖金 100 万美元，吸引了全球超过 400 支团队报名参赛，经过网络仿真预赛筛选，最终 9 支入围 2019—2020 赛季现场决赛。
- **成绩**：由宾夕法尼亚大学（UPenn）、苏黎世大学等成员组成的 SWIFT 联合团队获得最高技术成绩，其自主飞行速度与人类顶级飞手的差距已大幅缩短，验证了自主无人机竞速在工程上的可行性。

### IMAV — 国际微型飞行器大赛

国际微型飞行器大赛（International Micro Air Vehicle Conference and Competition，IMAV）是一项结合学术研讨会议与工程竞赛的国际活动，通常每年在欧洲国家举办，由相关高校或研究机构轮流承办。

- **赛制**：分室内赛（Indoor Flight）和室外赛（Outdoor Flight）两大类，参赛微型飞行器（Micro Air Vehicle，MAV）需完成自主导航（Autonomous Navigation）、目标识别（Target Recognition）、有效载荷精确投递（Payload Delivery）等多种任务；室内赛严格禁止使用 GPS 等全球卫星定位系统，仅允许依赖机载传感器（摄像头、深度传感器、IMU 等）完成自主飞行，技术难度极高。
- **参赛门槛与推荐**：面向全球大学和研究机构，适合具备一定无人机控制与感知算法基础的研究团队；中国高校如北京航空航天大学（北航）、浙江大学无人系统研究组等曾参赛并取得较好成绩。

### 中国国内无人机竞赛

国内也涌现出多项专注无人机技术的竞赛，推动高校无人机技术的发展：

- **全国大学生无人飞行器智能感知技术竞赛**：教育部产学合作协同育人项目支持，重点考察无人机感知算法（目标检测、深度估计）、自主建图（Autonomous Mapping）与自主导航能力；赛题覆盖固定翼、多旋翼等多种机型。
- **中国无人机创意大赛**：面向创新应用场景的无人机系统设计与应用竞赛，鼓励将无人机用于农业、物流、搜救等实际场景。
- **大疆 Sky City 大学生飞行挑战赛**：依托 DJI 提供的无人机硬件平台，考察参赛团队的控制算法优化与自主任务执行能力，提供大疆 SDK 开放接口，降低了硬件门槛。
- **RoboMaster 无人机联赛**：作为 RoboMaster 机甲大师赛的组成部分，专注无人机在对抗环境中的自主飞行与协同任务。

---

## 物流与工业操作类竞赛

### Amazon Robotics Challenge（ARC）/ Amazon Picking Challenge（APC）

- **历史沿革**：亚马逊物流机器人挑战赛（Amazon Picking Challenge，APC）于 2015 年在美国西雅图首次举办，其直接动机是解决亚马逊仓库中机器人"货架取物"自动化的技术瓶颈；2017 年改名为 Amazon Robotics Challenge（ARC），并扩大了任务范围（增加"物品放置"任务），此后因技术成熟度提升而暂停举办。
- **赛制设计**：参赛机器人系统（通常为工业机械臂 + 定制末端执行器 + 视觉系统）需从标准亚马逊货架（Shelf）或随机散装料箱（Tote/Bin）中自主识别并抓取指定物品，将其放置到目标位置。物品种类繁多（ARC 2017 设 112 种物品），包括软包装食品、反光金属罐、书本、毛绒玩具等形状各异、材质不同的商品，考察机器人抓取算法的泛化能力（Generalization）。
- **技术挑战**：多类别密集堆叠物体的三维识别（RGB-D Object Recognition）、6-DoF 姿态估计（Pose Estimation）、对未知物体的鲁棒抓取规划（Grasp Planning for Novel Objects）、软体与易形变物品处理（Deformable Object Manipulation），以及高速高可靠的整体系统集成。
- **历届优胜团队**：澳大利亚机器人视觉中心（Australian Centre for Robotic Vision，ACRV）、德国卡尔斯鲁厄理工学院（KIT）、MIT 等团队表现突出；中国队伍（如清华大学、上海交通大学）也曾参赛并获得良好名次。
- **深远影响**：APC/ARC 极大地推动了机器人抓取（Robot Grasping）领域的整体研究水平，催生了 GraspNet-1Billion、YCB Object Dataset 等开源数据集与抓取算法基准，并直接推动了吸盘（Suction Cup）+ 指状夹爪（Finger Gripper）混合末端执行器的工程化应用。

### Mohamed Bin Zayed International Robotics Challenge（MBZIRC）

- **概述**：MBZIRC 是由阿联酋阿布扎比高级技术研究委员会（Advanced Technology Research Council，ATRC）资助、哈利法科学技术大学（Khalifa University of Science and Technology）承办的国际机器人挑战赛，以每届高达 500 万美元的总奖金著称，每两年举办一届，面向全球顶尖研究机构，是目前奖金规模最大的国际机器人竞赛之一。
- **2020 年第二届赛制**（三项独立挑战赛）：
  - **挑战一**：多架无人机自主搜索一栋建筑物外墙上的火焰（模拟火灾），并使用机载灭火系统精确扑灭，考察无人机的视觉感知、自主定位与精准作动能力；
  - **挑战二**：空地协同机器人系统在港口仿真环境中自主完成集装箱识别、货物搬运与精确放置任务；
  - **挑战三**：无人机与地面机器人协作，在指定区域内自主搜索、定位并"拆除"模拟爆炸物（标有颜色编码的目标），考察多机器人协调与快速任务规划能力。
- **参赛团队**：历届顶级参赛机构包括苏黎世联邦理工学院（ETH Zurich）、卡内基梅隆大学（CMU）、麻省理工学院（MIT）、宾夕法尼亚大学（UPenn）、Imperial College London 等；中国的浙江大学自主机器人实验室（ARL）、哈尔滨工业大学也曾参赛。

---

## 学术基准挑战赛

### NIST 工业自动化机器人敏捷竞赛（ARIAC）

NIST（美国国家标准与技术研究院，National Institute of Standards and Technology）发起的 Agile Robotics for Industrial Automation Competition（ARIAC）是一项完全基于仿真的在线工业机器人竞赛。

- **赛制**：参赛队伍在 ROS（Robot Operating System，机器人操作系统）配合 Gazebo 仿真环境中，开发机械臂（通常为 UR10 仿真模型）的抓取、搬运与装配算法，在模拟装配线上高效完成零件拣选（Kitting）和装配（Assembly）任务，同时需实时应对传送带卡顿、零件丢失、订单插队、传感器故障等随机干扰事件（Agility Challenges），考察算法的鲁棒性与自适应能力。
- **特点与优势**：纯软件竞赛，无需实体机器人硬件投入，参赛门槛低，特别适合以算法研究为主的科研团队或正在学习 ROS/工业机器人的学生团队。ARIAC 自 2017 年起每年举办，并持续更新任务难度和仿真场景的逼真度。

### ICRA / IROS 机器人竞赛子项

IEEE 机器人与自动化国际会议（International Conference on Robotics and Automation，ICRA）和 IEEE/RSJ 智能机器人与系统国际会议（IROS）每届均设置若干学术竞赛子项（Competitions），这些竞赛规模相对较小，但由于附属于顶级学术会议，学术曝光度高，获奖成果通常直接发表于顶级机器人学术期刊或会议，具有较强的学术影响力。

常见竞赛方向：

- **灵巧操作（Dexterous Manipulation）**：如开放式物体操作挑战、物体重新抓取（Regrasping）、工具使用等；
- **移动操作（Mobile Manipulation）**：如 ICRA Robothon 家庭物品操作挑战赛（参赛机器人需自主折叠毛巾、拔插头、使用工具等）；
- **仿真到真实迁移（Sim-to-Real Transfer）**：强调算法在仿真中训练后在真实机器人上部署的能力；
- **自主导航（Autonomous Navigation）**：如 BARN Challenge（在随机生成的密集静态障碍物环境中的导航基准，Benchmark for Autonomous Robot Navigation）；
- **人机协作（Human-Robot Collaboration）**：如手势识别、意图预测与协同搬运等。

---

## FIRST 系列竞赛

FIRST（For Inspiration and Recognition of Science and Technology）由美国发明家 Dean Kamen 于 1989 年在美国新罕布什尔州创立，是全球最具影响力的青少年 STEM 教育竞赛组织之一。FIRST 的核心理念是通过真实的机器人工程挑战，让青少年像专业工程师一样思考与工作，培养科学思维、工程实践能力与团队协作精神。FIRST 旗下设有四个层级的竞赛，覆盖 4 岁至 18 岁全年龄段。

### FIRST Robotics Competition（FRC）

- **面向群体**：高中生（14–18 岁），是 FIRST 旗下规模最大、技术水平最高的竞赛，全球参赛队伍超过 3500 支，遍布 100 余个国家，被称为"青少年工程师的超级碗"。
- **赛制流程**：每年 1 月上旬举行"启动日"（Kickoff），公布当年赛题；各队伍有**6 周时间（Build Season）**设计、制造并编程一台约 55 kg 的机器人，比赛在 16 m × 8 m 的标准场地上举行，通常为 3 对 3 的联盟（Alliance）对抗赛制，兼有自主期（Autonomous Period，15 秒）和操控期（Teleoperated Period，2 分 15 秒）两个阶段。
- **精神内核**：FIRST 独创的"Coopertition"精神（合作与竞争并重，Cooperation + Competition），强调团队协作、工程思维与工匠精神（Gracious Professionalism）；参赛队伍被鼓励在竞赛期间互相帮助，甚至与对手共享零件。
- **资源保障**：每支队伍获得统一的"硬件包"（Kit of Parts），包含 RoboRIO 控制器（National Instruments 提供）、电机驱动器、传感器和基础结构材料；软件支持 Java、C++ 和 LabVIEW 三种编程语言，并提供完整的 WPILib 机器人库。
- **赞助与奖学金生态**：FRC 拥有完善的企业赞助和大学奖学金体系，许多美国顶尖工科大学为 FRC 参赛者提供专项奖学金，Google、Boeing、FIRST 基金会等企业每年资助数百支初创队伍。

### FIRST Tech Challenge（FTC）

- **面向群体**：初中与高中生（12–18 岁），规模仅次于 FRC，每年参赛队伍超过 6000 支，是 FRC 的"入门台阶"版本。
- **赛制**：使用 TETRIX Metal、REV Robotics 或 goBILDA 等标准化搭建套件，机器人尺寸限制在 18 英寸（约 45 cm）立方体内，2 对 2 联盟对抗；编程语言支持 Java（Android Studio + FTC SDK）和 Blocks 图形化编程（类似 Scratch），对编程基础要求相对 FRC 更低。

### FIRST LEGO League（FLL）

- **面向群体**：分 FLL Explore（6–10 岁）和 FLL Challenge（9–14 岁）两个年龄档，是 FIRST 系列中参与人数最多的竞赛，全球每年参赛队伍超过 6 万支。
- **赛制**：使用 LEGO Education SPIKE Prime（FLL Challenge）套件，结合机器人任务赛（Robot Game，在场地上完成自主任务积分）、探究项目展示（Innovation Project，针对每年主题进行科学探究并设计方案）和核心价值评审（Core Values，评估团队文化与精神）三个维度综合评分。
- **中国推广**：FLL 在中国已有北京、上海、广州、深圳、成都等多个赛区，每年从各省赛区选拔团队参加全国赛，再遴选代表队参加世界锦标赛（FLL World Festival）；已有中国队伍在世界锦标赛上获得冠军和多项单项奖。

---

## 其他值得关注的国际竞赛

### VEX Robotics Competition（VRC）

VEX Robotics Competition 是全球规模最大的学生机器人竞赛平台之一，由美国 REC Foundation 组织，使用 VEX EDR（面向中学）和 VEX IQ（面向小学）标准套件。全球每年参赛队伍超过 2 万支，世界锦标赛（VEX Robotics World Championship）在美国达拉斯举行，是 FIRST 系列之外参赛规模最大的青少年机器人竞赛。中国参赛队伍众多，多次在世界锦标赛上获得冠军。

### World Robot Olympiad（WRO，世界机器人奥林匹克）

WRO 是面向 8–25 岁青少年的国际机器人竞赛，每年在不同国家举办世界决赛，设有常规赛（Regular Category，基于 LEGO 或同类积木构建）、高级创意赛（Open Category）和 WRO 足球赛（Football）三大类别，以及面向大学生的 RoboSports 子项。WRO 在 50 余个国家和地区设有国家委员会，中国由中国青少年机器人竞赛（CASC）对应衔接。

### Eurobot — 欧洲业余机器人大赛

Eurobot 创办于 1998 年，面向业余机器人爱好者和学生团队，每年设定不同的比赛主题，参赛机器人需在规定场地内自主完成指定任务，以欧洲参赛队伍为主，是欧洲规模最大的非商业性机器人竞赛。

---

## 参赛建议与备赛指南

### 按技术方向选择竞赛

| 竞赛方向 | 推荐竞赛 | 适合阶段 |
|----------|----------|----------|
| 自主移动与导航 | RoboCup SSL/MSL、DARPA SubT 相关技术、BARN Challenge | 本科高年级至研究生 |
| 人形机器人与步态控制 | RoboCup Humanoid League、CRC 人形赛 | 研究生及以上 |
| 家庭服务机器人 | RoboCup @Home OPL/DSPL | 研究生及以上 |
| 工业操作与机器人抓取 | APC/ARC、NIST ARIAC、ICRA 竞赛子项 | 研究生及以上 |
| 无人机自主飞行 | IROS Drone Racing、IMAV、大疆 Sky City 挑战赛 | 本科高年级至研究生 |
| 多机器人对抗系统 | RoboMaster 机甲大师赛 | 本科全阶段 |
| 空地协同与多机协作 | MBZIRC、DARPA SubT 相关 | 研究生及以上 |
| 青少年机器人入门 | FRC、FTC、FLL、ROBOCON、WRO、VEX | 中学至本科低年级 |

### 技术准备建议

1. **软件栈基础**：熟练掌握 ROS（Robot Operating System）或 ROS 2 的基本使用，了解 Gazebo、Isaac Sim、Webots 等主流仿真工具，能够独立完成传感器驱动集成、话题订阅与服务调用。
2. **感知算法**：掌握摄像头内外参标定（Camera Calibration）、基于深度学习的目标检测（YOLOv8 等）、激光雷达点云处理（PCL 库）、深度估计与视觉里程计（Visual Odometry）等基础感知模块的原理与工程实现。
3. **规划与控制**：了解全局路径规划（A*、Dijkstra）与局部路径规划（动态窗口法 DWA、TEB）、运动控制（PID、模型预测控制 MPC）及基本的多机器人协调与任务分配（Task Allocation）原理。
4. **硬件集成能力**：具备一定的电路调试（万用表、示波器使用）、PCB 焊接和机械加工基础能力，能独立排查从电源故障到传感器驱动问题的常见硬件故障。
5. **团队协作与工程规范**：使用 Git 进行版本管理（建议采用 Git Flow 工作流），建立清晰的代码注释规范和文档写作习惯，定期开展内部技术分享（Code Review 与算法讲解）。

### 参赛资源推荐

- **开源代码**：RoboCup 历届冠军队的代码通常在赛后开源，如 ZJUNlict（SSL）、UT Austin Villa（SPL）等，是入门该联赛技术栈的宝贵资源；GitHub 搜索对应赛事名称可找到大量参考实现。
- **规则手册**：各竞赛官网均提供完整的规则手册（Rulebook）PDF，以及历届技术描述文件（Team Description Paper，TDP）；认真研读规则并分析历届 TDP 的技术选型，是高效备赛的第一步。
- **论文研读**：在正式备赛前，系统阅读该竞赛联赛近 3 年发表在 ICRA/IROS/RoboCup Symposium 的 3–5 篇代表性论文，了解当前技术前沿、主流方案与尚未解决的关键问题。
- **仿真优先原则**：在购置昂贵硬件（尤其是人形机器人或无人机）前，优先在仿真环境中完整验证算法流程，通过仿真-实物迁移（Sim-to-Real Transfer）的方式降低试错成本；建议预留至少 1–2 个月的真机调试时间。
- **社区与交流**：积极参加对应竞赛的官方论坛、邮件列表或 Discord 社群，与其他参赛队伍和组委会直接交流疑问；出席竞赛现场并观摩强队操作，往往比单纯读论文更有收获。

---

## 主要竞赛速查表

### 综合对比

| 竞赛名称 | 主办方 | 创办年份 | 主要对象 | 技术方向 | 中国参与度 |
|----------|--------|----------|----------|----------|------------|
| RoboCup SSL | RoboCup Federation | 1997 | 本科生/研究生 | 多机协作、运动规划 | 高（ZJUNlict 多次夺冠） |
| RoboCup SPL | RoboCup Federation | 1997 | 本科生/研究生 | 视觉感知、步态、AI | 中 |
| RoboCup @Home | RoboCup Federation | 1997 | 研究生及以上 | 服务机器人、NLP | 高（中科大曾获冠军） |
| RoboCup Humanoid | RoboCup Federation | 2002 | 研究生及以上 | 双足步态、动态平衡 | 低至中 |
| DARPA Grand Challenge | DARPA（美国政府） | 2004 | 顶尖研究机构 | 自动驾驶、感知融合 | 不对外开放 |
| DARPA DRC | DARPA（美国政府） | 2012 | 顶尖研究机构 | 人形机器人、灾难救援 | 不对外开放 |
| DARPA SubT | DARPA（美国政府） | 2018 | 顶尖研究机构 | 地下 SLAM、空地协同 | 不对外开放 |
| RoboMaster 机甲大师赛 | 大疆（DJI） | 2015 | 本科生 | 全栈机器人工程 | 极高（中国主场） |
| ROBOCON | ABU（亚洲广播联盟） | 2002 | 本科生 | 机械设计、控制 | 高（中国常年强队） |
| 中国机器人大赛（CRC） | 中国自动化学会 | 1999 | 高校及中学生 | 综合机器人技术 | 极高（中国国内赛） |
| Amazon Robotics Challenge | Amazon | 2015 | 研究团队 | 机器人抓取、视觉 | 中 |
| MBZIRC | Khalifa University | 2017 | 顶尖研究机构 | 空地协同、UAV | 中 |
| IROS Drone Racing | IROS（IEEE） | 2016 | 研究生及以上 | 自主无人机、RL | 中 |
| AlphaPilot | Lockheed/DRL | 2019 | 研究团队 | 自主无人机竞速 | 低 |
| FRC | FIRST | 1992 | 高中生 | 机器人工程入门 | 中（国际参赛） |
| FLL | FIRST/LEGO | 1998 | 小学/初中生 | STEM 启蒙、编程 | 高（全国多赛区） |
| NIST ARIAC | NIST（美国） | 2017 | 研究生及以上 | 工业仿真、ROS | 中 |

### 机器人竞赛历史里程碑

| 年份 | 事件 |
|------|------|
| 1992 | FIRST Robotics Competition（FRC）创办，开启青少年机器人竞赛时代 |
| 1997 | RoboCup 在日本名古屋首届举办，提出"2050 年击败人类世界杯冠军"终极目标 |
| 1999 | 中国机器人大赛（CRC）首届举办 |
| 2002 | ABU ROBOCON 亚太大学机器人大赛首届举办；RoboCup Junior 正式设立 |
| 2004 | DARPA Grand Challenge 首届举办，无一车辆完赛，揭示当时自动驾驶技术局限 |
| 2005 | 斯坦福 Stanley 完成 DARPA Grand Challenge 全程，自动驾驶技术里程碑 |
| 2007 | DARPA Urban Challenge 举办，CMU Boss 夺冠；Sebastian Thrun 随后创立 Google 自动驾驶项目 |
| 2008 | RoboCup SPL 从 AIBO 平台切换至 NAO 人形机器人平台 |
| 2012 | DARPA Robotics Challenge（DRC）启动，以福岛核电站事故为背景 |
| 2014 | 中科大 KeJia 队获 RoboCup @Home OPL 组世界冠军，中国机器人首次问鼎 |
| 2015 | KAIST DRC-HUBO 夺得 DARPA DRC 总冠军；RoboMaster 机甲大师赛首届举办；Amazon Picking Challenge 首届举办 |
| 2017 | MBZIRC 首届举办，总奖金 500 万美元创国际机器人竞赛奖金纪录 |
| 2018 | DARPA SubT 启动，开启地下机器人探索竞赛新赛道 |
| 2019 | ZJUNlict 再获 RoboCup SSL 世界冠军；AlphaPilot 无人机挑战赛首届举办 |
| 2021 | DARPA SubT 总决赛，CERBERUS 队夺冠 |
| 2023 | UZH/RPG 团队自主无人机首次在竞速中击败人类飞手，成果发表于 *Nature* |

---

## 参考资料

1. RoboCup Federation. *RoboCup Official Website*. https://www.robocup.org/
2. RoboCup Technical Committee. *RoboCup Standard Platform League (NAO) Rule Book 2024*. https://spl.robocup.org/
3. ZJUNlict Team. *ZJUNlict Extended Team Description Paper for RoboCup 2019 SSL*. https://github.com/ZJUNlict
4. DARPA. *DARPA Grand Challenge: Ten Years Later*. https://www.darpa.mil/news-events/2014-02-11
5. Thrun, S. et al. "Stanley: The Robot That Won the DARPA Grand Challenge." *Journal of Field Robotics*, 23(9), pp. 661–692, 2006. https://doi.org/10.1002/rob.20147
6. DARPA. *DARPA Robotics Challenge (DRC) Finals Official Results and Overview*. https://www.darpa.mil/program/darpa-robotics-challenge
7. DARPA. *DARPA Subterranean Challenge Final Results 2021*. https://www.subtchallenge.com/
8. Oh, J. et al. "Team KAIST at the DARPA Robotics Challenge Finals 2015." *Journal of Field Robotics*, 34(2), 2017.
9. RoboMaster. *RoboMaster 机甲大师赛官方网站*. https://www.robomaster.com/
10. ABU ROBOCON. *ABU Asia-Pacific Robot Contest — Official Website*. https://www.aburobocup.tv/
11. 中国自动化学会. *中国机器人大赛（CRC）官方网站*. http://www.caa.net.cn/
12. Loquercio, A. et al. "Champion-level drone racing using deep reinforcement learning." *Nature*, 620, pp. 982–987, 2023. https://doi.org/10.1038/s41586-023-06419-4
13. Correll, N. et al. "Analysis and Observations from the First Amazon Picking Challenge." *IEEE Transactions on Automation Science and Engineering*, 15(1), pp. 172–188, 2018. https://doi.org/10.1109/TASE.2016.2600527
14. MBZIRC Organizing Committee. *Mohamed Bin Zayed International Robotics Challenge Official Website*. https://www.mbzirc.com/
15. FIRST. *FIRST Robotics Competition — Official Website and Resources*. https://www.firstinspires.org/robotics/frc
16. NIST. *Agile Robotics for Industrial Automation Competition (ARIAC) 2023*. https://www.nist.gov/el/intelligent-systems-division-73500/agile-robotics-industrial-automation-competition
17. Behnke, S. et al. *RoboCup 2023: Robot World Cup XXVI*. Springer, Lecture Notes in Artificial Intelligence, 2024.
18. Kitano, H. et al. "RoboCup: A Challenge Problem for AI." *AI Magazine*, 18(1), pp. 73–85, 1997.

