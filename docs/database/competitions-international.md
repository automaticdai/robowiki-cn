# 国际专项与教育类竞赛

!!! note "引言"
    除 RoboCup 与 DARPA 之外，国际上还存在大量面向特定技术方向或特定人群的机器人竞赛。无人机竞速考验高速状态估计与敏捷控制，物流与工业操作类竞赛直接对接仓储自动化的真实需求，学术基准挑战赛把可复现的评测标准嵌入顶级会议，而 FIRST、WRO、VEX 等教育类赛事则承担着面向青少年的普及职能。本页面汇总这几类竞赛的赛制与参与方式。

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

## 参考资料

1. FIRST. *FIRST Robotics Competition / FTC / FLL 官方网站*. https://www.firstinspires.org/
2. VEX Robotics Competition. *官方网站*. https://www.vexrobotics.com/competition
3. World Robot Olympiad Association. *WRO 官方网站*. https://wro-association.org/
4. Mohamed Bin Zayed International Robotics Challenge. *MBZIRC 官方网站*. https://www.mbzirc.com/
5. NIST. *Agile Robotics for Industrial Automation Competition (ARIAC)*. https://www.nist.gov/el/intelligent-systems-division-73500/agile-robotics-industrial-automation-competition
6. [机器人竞赛总览](competitions.md)
