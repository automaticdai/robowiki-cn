# RoboCup 与 DARPA 挑战赛

!!! note "引言"
    RoboCup 机器人世界杯与 DARPA 挑战赛系列是机器人领域影响力最大的两类旗舰赛事，但定位截然不同。RoboCup 以「2050 年击败人类足球世界冠军」为长期目标，通过持续数十年的标准化联赛体系推动多机器人协作、实时视觉与双足运动的研究；DARPA 挑战赛则以一次性、高奖金、面向具体国防与救灾需求的形式，在无人驾驶、灾难救援与地下探测等方向上多次引发技术拐点。本页面梳理两者的赛制、发展历程与技术影响。

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

## 参考资料

1. RoboCup Federation. *RoboCup Official Website*. https://www.robocup.org/
2. H. Kitano, M. Asada, Y. Kuniyoshi, I. Noda, and E. Osawa, "RoboCup: The Robot World Cup Initiative," *Proceedings of the First International Conference on Autonomous Agents*, 1997.
3. DARPA. *DARPA Grand Challenge / Urban Challenge / Robotics Challenge / Subterranean Challenge* 官方存档. https://www.darpa.mil/
4. M. Buehler, K. Iagnemma, and S. Singh (eds.), *The DARPA Urban Challenge: Autonomous Vehicles in City Traffic*, Springer, 2009.
5. [机器人竞赛总览](competitions.md)
