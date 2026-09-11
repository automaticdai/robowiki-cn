# Atlas

!!! note "引言"
    Atlas 是由美国 Boston Dynamics 公司研发的人形机器人（Humanoid Robot），是全球最具标志性的双足机器人之一。Atlas 最初于 2013 年作为液压驱动的研究平台亮相，在动态运动控制领域不断突破人类对机器人运动能力的认知。经过十余年迭代，Atlas 的每一次公开演示都引发全球机器人领域的广泛讨论，并深刻影响了整个行业的技术路线。2024 年，Boston Dynamics 推出全新的全电动 Atlas，标志着该平台从研究原型向商业化工业应用的重大转型。


## 发展历程

Atlas 的发展历程横跨十余年，每个阶段都伴随着技术上的重大突破和广泛的社会影响。

### 起源与 DARPA 机器人挑战赛（2013–2015）

Atlas 的诞生与美国国防高级研究计划局（Defense Advanced Research Projects Agency, DARPA）密切相关。2011 年日本福岛核电站事故后，DARPA 认识到现有机器人无法在核泄漏等极端灾害环境中执行应急操作，因此于 2012 年启动了 DARPA 机器人挑战赛（DARPA Robotics Challenge, DRC），旨在推动能够替代人类进入危险环境作业的机器人的研发。

DRC 设计了一套高度模拟真实灾害场景的任务序列，要求参赛机器人在无人协助的情况下完成以下八项任务：

1. 驾驶车辆穿越障碍场地
2. 下车并步行至目标建筑
3. 打开并通过一扇门
4. 操作工业阀门（旋转关闭）
5. 使用电动工具在墙壁上切割出开口
6. 爬越瓦砾堆等非结构化地形
7. 攀爬工业楼梯
8. 连接消防水管接头

任务设计的核心挑战在于：机器人需要在通信受限（模拟真实灾害中通信中断的场景）的条件下独立决策，操作员与机器人之间的通信带宽被人为限制，且存在额外的网络延迟。这要求各参赛团队不仅要解决运动控制难题，还要设计具备一定自主感知与决策能力的软件架构。

2013 年，DARPA 选定 Boston Dynamics 作为硬件供应商，为各参赛研究团队提供统一的 Atlas 平台。第一代 Atlas 身高约 1.88 m，体重约 150 kg，采用液压驱动，具备 28 个液压自由度（Degree of Freedom, DOF），并配备立体摄像头与激光雷达（LiDAR）用于环境感知。参赛队伍包括麻省理工学院（MIT）、卡内基梅隆大学（Carnegie Mellon University, CMU）、佛罗里达人机认知研究所（IHMC）等顶级研究机构。

2015 年 DRC 决赛中，韩国科学技术院（Korea Advanced Institute of Science and Technology, KAIST）的 DRC-HUBO 机器人夺冠，多支使用 Atlas 平台的团队亦进入前列。其中 IHMC 战队的 Atlas 获得第二名，完成了六项任务。比赛期间机器人频繁跌倒的画面经网络广泛传播，一方面展示了任务的高难度，另一方面也反映出当时双足机器人在实用性上与人类的巨大差距——绝大多数机器人完成全部八项任务耗时超过 40 分钟，而训练有素的人类不到 10 分钟即可完成。

### 2016 年：新一代 Atlas 与平衡恢复

2016 年 2 月，Boston Dynamics 发布新一代 Atlas（Next Generation Atlas），体型大幅缩小至身高 1.75 m、体重约 80 kg，并首次实现电池自主供电，不再依赖脐带式外接电源线。这一版本在结构上进行了大幅优化：通过采用更紧凑的液压回路布局和轻量化材料，在缩小体型的同时保持了与第一代相当的力矩输出能力。

这一版本展示了多项令人印象深刻的能力：在雪地中稳定行走、被人用曲棍球棒反复推打后通过实时重心调整恢复平衡、被踢翻放置在地上的纸箱后重新拾取物体并完成搬运。这些演示的技术意义在于，Atlas 能够将意外扰动（Disturbance）视为需要在线响应的事件，而非导致任务失败的故障——这依赖于对冲击力的实时感知与快速重规划能力。

发布视频在 YouTube 上数日内突破数百万次播放，并引发了关于机器人伦理的广泛讨论——部分观众对"虐待机器人"的测试画面感到不安，这一现象本身折射出社会对机器人拟人化的投射心理，也引发了学界对机器人道德地位（Moral Status）问题的讨论。

### 2017 年：后空翻突破

2017 年 11 月，Boston Dynamics 发布 Atlas 完成后空翻（Backflip）的视频。这是人形机器人首次在公开演示中完成完整的空中翻转动作，意味着机器人已能够在短暂脱离地面支撑的情况下完成高度非线性的动态运动。

从技术角度看，后空翻的挑战性在于整个腾空阶段（大约 0.5 秒）机器人完全失去与地面的接触，无法获得任何地面反力（Ground Reaction Force）来修正姿态。因此，起跳前的状态必须被精确控制——起跳角速度（Angular Velocity）的微小偏差都会导致落地时姿态严重偏离目标，进而引发跌倒。Boston Dynamics 通过接触隐式轨迹优化（Contact-Implicit Trajectory Optimization）生成了完整的后空翻参考轨迹，并借助鲁棒控制设计使 Atlas 能够在真实机器人上稳定复现仿真中优化出的轨迹。

该视频在 48 小时内获得超过 500 万次观看，被《自然》（Nature）、《科学》（Science）等学术媒体引用报道，并登上全球多家主流媒体头版，成为机器人领域的标志性事件。多位顶尖机器人学家在社交媒体上表示震惊，认为这一演示将人形机器人的动态运动能力的认知上限大幅提前了数年。

### 2018 年：跑酷与连续障碍跨越

2018 年 10 月，Atlas 展示了在户外场地独立完成跑步、跳上箱体、跨越原木等跑酷（Parkour）动作的能力，并以近乎流畅的方式连续完成多个动作。这一演示的核心意义在于：所有动作均在非事先精确测量的真实场地中完成，机器人需要依赖实时感知数据动态规划落脚点，而非依赖预设轨迹。

与此同时，演示中 Atlas 表现出了一定的错误恢复能力——在起跳时机略有偏差的情况下，Atlas 能够通过调整着陆姿态来避免跌倒，而非完全依赖预设动作的精确复现。这一特性被研究者解读为从"开环动作回放"向"闭环在线规划"过渡的重要迹象。

### 2021 年：体操例程与协同舞蹈

2021 年 8 月，Boston Dynamics 发布了 Atlas 完成完整体操例程（Gymnastics Routine）的视频，包括侧手翻、前后空翻的连续组合，以及单手支撑跳跃、旋转等高难度动作，最后以与 Spot 机器人狗、Handle 机器人共同舞蹈的片段收尾。

这一演示将先前碎片化的高难度动作整合为一个连贯的编排序列，证明了 Atlas 不仅能完成单个高难动作，还能可靠地将其串联执行。从工程角度看，连续动作之间的状态转换（Transition）是一大挑战——每个动作结束时的状态（位置、速度、角速度）必须落在下一个动作允许的初始状态范围内，否则无法成功衔接。Boston Dynamics 通过在轨迹优化阶段引入动作间的兼容性约束来解决这一问题。

视频获得超过 3000 万次播放，成为 Boston Dynamics 有史以来传播最广的内容之一。视频发布后，MIT、CMU、ETH Zürich 等高校的研究团队纷纷在社交媒体上讨论其中涉及的技术细节，进一步推动了相关算法的公开讨论与研究跟进。

### 2023 年：物体操作与工具使用

2023 年，Boston Dynamics 公开了 Atlas 在仓储环境中执行物体搬运与工具使用任务的演示。场景设定为：一名工人站在高台上需要工具包，工具包位于地面某处。Atlas 被要求自主识别工具包位置、拾取工具包并将其传递至高台上的工人，随后攀爬楼梯完成整个任务链。

此次演示涉及多个此前鲜见于 Atlas 演示中的能力：一是对非结构化场景（工具包散落于地面）的物体检测与位姿估计（Pose Estimation）；二是在运动状态下完成抓取与持握（Grasp-while-locomoting）；三是面向高台工人的物体交接（Handover），要求 Atlas 协调手臂运动与步态，在与人类的近距离交互中完成任务。

这一演示标志着 Atlas 从纯粹展示运动能力转向探索实际操作任务，也是液压版 Atlas 向工业应用方向迈出的重要一步，同时为同年宣布的电动版方向提供了铺垫。

### 2024 年：液压版退役与全电动版发布

2024 年 4 月，Boston Dynamics 正式宣布液压版 Atlas 退役，并发布全新的全电动 Atlas。Boston Dynamics 发布了一段名为"Farewell to HD Atlas"的告别视频，以略带幽默的方式展现了液压版 Atlas 十余年来的"高光时刻"与"糗事集锦"，包括多次跌倒的画面——这种自嘲式的告别在社交媒体引发大量情感共鸣，被广泛认为是一次成功的品牌传播。

同月发布的电动版 Atlas 宣传片以一段令人不安的画面开场：机器人从蜷缩的姿态通过关节超自然旋转缓慢起身，动作方式与人类完全不同，但极为流畅。这一设计被 Boston Dynamics 解释为电动版 Atlas 的关键特性之一：不受人体关节旋转范围的限制，可以选择最优的运动路径而非最"像人"的运动路径。


## 技术规格

### 电动版 Atlas（2024）

| 参数 | 规格 |
|------|------|
| 身高 | 约 1.5 m |
| 体重 | 约 89 kg |
| 驱动方式 | 全电动（Electric Actuators） |
| 自由度（DOF） | 28+ |
| 关节特性 | 多轴旋转关节，超 360° 旋转范围 |
| 感知系统 | 多传感器融合（摄像头、LiDAR、力/力矩传感器） |
| 手部 | 多指灵巧手，可抓取不规则物体 |

### 液压版 Atlas（2016–2024）

| 参数 | 规格 |
|------|------|
| 身高 | 1.75 m |
| 体重 | 约 80 kg |
| 驱动方式 | 液压驱动（Hydraulic Actuators） |
| 自由度（DOF） | 28 |
| 行走速度 | 约 1.5 m/s |
| 感知系统 | LiDAR、立体视觉（Stereo Vision）、IMU |
| 电源 | 外接电源或机载电池组 |
| 液压系统压力 | 约 2000 psi |

### 第一代 Atlas（DRC 版，2013–2015）

| 参数 | 规格 |
|------|------|
| 身高 | 约 1.88 m |
| 体重 | 约 150 kg |
| 驱动方式 | 液压驱动 |
| 自由度（DOF） | 28 |
| 电源 | 外接电源线（脐带供电） |
| 感知系统 | 激光雷达、立体摄像头 |
| 主要用途 | DRC 参赛平台 |


## 研究影响

Atlas 对机器人研究领域的影响是多维度的，其贡献远超一台具体的硬件产品。

### 推动全身控制理论发展

Atlas 的高动态运动演示为全身控制（WBC）理论的快速发展提供了最具说服力的验证平台。自 2016 年以来，以 WBC 为核心的论文数量在 ICRA（IEEE International Conference on Robotics and Automation）、IROS（IEEE/RSJ International Conference on Intelligent Robots and Systems）和 RAL（IEEE Robotics and Automation Letters）等顶级会议和期刊中显著增长。许多研究团队基于 Atlas 的开放规格（在 DRC 期间由 DARPA 提供）开发了新的控制算法，并在 Gazebo 或 MuJoCo 仿真器中与 Atlas 模型进行验证。

Drake（MIT 开发的机器人数学软件库）专门为 Atlas 模型提供了详细的仿真支持，成为控制算法开发的重要工具。Kuindersma 等人在 2016 年发表于 *Autonomous Robots* 的论文系统总结了 Atlas 的运动控制框架，成为人形机器人控制领域引用量最高的论文之一。

### 激励竞品机器人的研发

Atlas 的持续突破极大地激励了其他机构加速人形机器人研发。特斯拉（Tesla）在 2021 年宣布研发 Optimus（擎天柱）时，马斯克（Elon Musk）明确将 Atlas 作为参照对象，并承诺 Optimus 将比 Atlas 更便宜、更实用。国内方面，宇树科技（Unitree Robotics）、智元机器人、宇航科技、傅利叶智能等公司，以及国际上的 Agility Robotics（Digit）、Figure AI（Figure 01/02）等，均在 Atlas 奠定的技术基础上进行了针对商业化场景的差异化研发。

Atlas 的竞品效应不仅体现在技术层面，也体现在资本市场：Atlas 的每一次重大演示往往伴随着整个人形机器人赛道的融资活跃度上升，投资者将 Atlas 的演示视为行业技术可行性的重要信号。

### 推动动态操作研究

2023 年 Atlas 的工具操作演示将机器人研究的关注点从纯粹的运动控制拓展到动态操作（Dynamic Manipulation）——即在机器人运动过程中同步完成物体抓取、搬运和放置任务。这一方向要求运动控制（Locomotion Control）与操作规划（Manipulation Planning）的深度耦合（Loco-Manipulation），是当前人形机器人研究最活跃的前沿之一。

以 Atlas 为背景的具身智能（Embodied Intelligence）研究也日益增多，探索如何将大型语言模型（Large Language Model, LLM）或视觉语言模型（Vision-Language Model, VLM）与低层运动控制结合，使 Atlas 能够理解自然语言指令并将其转化为具体的运动序列。

### 改变公众对机器人的认知

Atlas 的系列视频在非专业群体中广泛传播，使"人形机器人"从科幻概念变为具体可感的技术现实。这一认知转变对政策制定者、投资者和公众产生了深远影响，在一定程度上加速了全球范围内对人形机器人产业的资本投入和政策关注。

2021 年体操视频之后，多国政府将人形机器人纳入科技战略规划，中国工业和信息化部（Ministry of Industry and Information Technology）于 2023 年发布的《人形机器人创新发展指导意见》明确将人形机器人列为战略新兴产业，这一政策背景与 Atlas 等机器人营造的全球认知高度有着直接关联。


## 与竞品对比

下表对比了目前主要人形机器人平台的关键参数（以电动版 Atlas 为参照，数据截至 2024 年底）：

| 参数 | Atlas（电动版，2024） | Optimus Gen 2（Tesla） | Figure 02（Figure AI） | H1（Unitree） |
|------|----------------------|------------------------|------------------------|----------------|
| 制造商 | Boston Dynamics | Tesla | Figure AI | Unitree Robotics |
| 发布年份 | 2024 | 2023 | 2024 | 2023 |
| 身高 | 约 1.5 m | 约 1.73 m | 约 1.68 m | 约 1.8 m |
| 体重 | 约 89 kg | 约 57 kg | 约 70 kg | 约 47 kg |
| 驱动方式 | 全电动 | 全电动 | 全电动 | 全电动 |
| 自由度（DOF） | 28+ | 28 | 35 | 19 |
| 行走速度 | 未公布 | 约 0.5 m/s | 约 1.2 m/s | 约 1.8 m/s |
| 灵巧手 | 有（多指） | 有（多指） | 有（多指） | 有（三指） |
| 主要定位 | 工业制造（Hyundai） | 特斯拉工厂 | 工业通用 | 研究与工业 |
| 开放 SDK | 否 | 否 | 否 | 是 |
| 运动控制方法 | WBC + MPC | 强化学习为主 | 强化学习为主 | 强化学习为主 |
| 代表性能力 | 高动态运动、超 360° 关节旋转 | 精细手部操作 | 整车零件操作 | 高速行走、开发者友好 |

注：上表中的参数来自各公司公开发布的数据，部分参数（尤其是仍在迭代中的产品）可能在后续更新中发生变化。各机器人的实际性能会随软件版本迭代而持续演进，上表仅反映公开信息的快照。

从对比中可以观察到以下几个行业趋势：

- **轻量化**：除 Atlas 电动版外，其他竞品均以减重为重要设计目标，以降低安全风险并提高能效。Atlas 电动版体重偏高，部分原因在于其优先考虑输出力矩和动态运动能力，而非便携性。
- **强化学习渗透**：大多数新一代人形机器人的运动控制以端到端强化学习（End-to-End Reinforcement Learning）为主要技术路线，通过在大规模仿真环境中训练策略网络（Policy Network），再迁移至真实机器人；这与 Atlas 长期坚持的基于模型的优化控制方法形成对比。两种路线各有优劣：基于模型的方法可解释性强、对约束的处理更精确；强化学习方法在面对未知扰动和新环境时泛化能力更强，但训练代价高且策略行为难以解释。
- **开放生态差异**：宇树等公司通过开放 SDK（Software Development Kit）和提供 ROS（Robot Operating System）接口，积极吸引高校和独立开发者构建第三方应用生态；而 Boston Dynamics 选择聚焦专有工业应用场景，以闭环系统提供更高的可靠性保证和定制化服务。
- **操作能力的差距**：目前大多数竞品的灵巧操作（Dexterous Manipulation）能力仍显著弱于运动能力，如何将高动态运动与精细操作有机结合，是整个行业面临的共同挑战，也是下一阶段竞争的核心维度。


## Boston Dynamics 与 Atlas

Boston Dynamics 成立于 1992 年，由马克·雷伯特（Marc Raibert）创立，脱胎于麻省理工学院（Massachusetts Institute of Technology, MIT）的腿足实验室（Leg Laboratory）。公司的核心研究基因来自 Raibert 在 MIT 期间对弹跳腿（Raibert Hoppers）的基础研究——这些仅有一条弹跳腿的早期机器人通过独特的能量泵入与姿态控制机制实现了动态稳定奔跑，奠定了动态稳定性控制（Dynamic Stability Control）的理论基础，并直接影响了后来 Atlas 的控制哲学。

公司先后经历了以下所有权变更：

- **2013 年**：Google（现 Alphabet）以约 3.75 亿美元收购 Boston Dynamics，时值 DRC 项目开展，此次收购引发了关于科技巨头进入军用机器人领域的广泛讨论
- **2017 年**：软银（SoftBank）以约 1.65 亿美元从 Alphabet 收购，期间 Boston Dynamics 在软银旗下专注于技术研发，产品化进程相对缓慢
- **2020 年**：现代汽车集团以约 11 亿美元收购约 80% 股份，标志着 Boston Dynamics 进入以商业化为核心战略的新阶段

在所有权多次更迭中，Atlas 项目始终是公司核心技术展示窗口，其持续迭代所积累的控制技术、工程经验和品牌效应，是 Boston Dynamics 在竞争激烈的人形机器人市场中维持领先地位的重要资产。

Boston Dynamics 目前同时运营多个机器人产品线：Spot（四足机器人，已商业化）、Stretch（仓储码垛机器人）和 Atlas（人形机器人，仍处于从研究到商业化的过渡阶段）。Spot 的商业化成功为公司提供了稳定的现金流，也积累了大量在真实工业场景部署和维护机器人的工程经验，这些经验对 Atlas 的工业化落地具有重要的参考价值。


## 技术前景与挑战

尽管 Atlas 在动态运动领域已处于世界领先水平，其实现大规模商业部署仍面临多项需要突破的技术挑战。

### 具身智能与任务泛化

当前 Atlas 的能力主要依赖于针对特定任务事先设计和优化的控制策略，每类新任务（如新的抓取形态或新的地形类型）通常需要专门的工程开发投入。实现真正意义上的任务泛化（Task Generalization）——即让 Atlas 能够根据自然语言描述或少量示范自动学习新任务——是当前具身人工智能（Embodied AI）研究的核心目标。

Boston Dynamics 已公开表示正在探索将基础模型（Foundation Model）与底层运动控制结合的技术路径，以实现更灵活的任务规划与执行能力。这一方向的进展将决定 Atlas 能否从"为特定任务定制的专用系统"进化为"可通用部署的通用机器人劳动力"。

### 续航与能源管理

电动版 Atlas 的电池续航时间目前尚未公开披露，但这是工业部署中的关键约束之一。人形机器人在执行搬运等高负载任务时的功耗显著高于四足机器人，如何在可接受的机体重量和体积约束下实现足够长的工作续航（目标通常为 4 小时以上的连续作业），是电动驱动方案需要持续优化的重要方向。

### 成本与规模化制造

Atlas 当前的制造成本预计远高于传统工业机器人臂，这是其大规模工业部署的主要经济障碍。Boston Dynamics 需要在保持技术性能的前提下，通过供应链优化、模块化设计和规模化生产显著降低单台成本，使 Atlas 的总拥有成本（Total Cost of Ownership, TCO）与其在工业应用中创造的价值相匹配。


## 本章内容导览

| 页面 | 主要内容 |
|------|---------|
| [Atlas](atlas.md) | 发展历程、技术规格、研究影响与竞品对比 |
| [Atlas 控制架构](atlas-control-architecture.md) | 液压伺服底层、全身控制（WBC）框架、基于 MPC 的运动规划 |
| [全电动 Atlas 与工业应用](atlas-electric-applications.md) | 2024 全电动版技术亮点、Hyundai Metaplant 工厂部署 |
| [Boston Dynamics](boston-dynamics.md) | 公司历史、技术路线与全系产品 |
| [人形机器人图鉴](robots.md) | 各厂商人形机器人横向对比 |


## 参考资料

1. [Atlas](https://bostondynamics.com/atlas/), Boston Dynamics 官网
2. [Atlas (robot)](https://en.wikipedia.org/wiki/Atlas_(robot)), Wikipedia
3. [Farewell to HD Atlas](https://bostondynamics.com/blog/electric-new-satisfying-atlas/), Boston Dynamics Blog, 2024
4. [DARPA Robotics Challenge Finals](https://www.darpa.mil/program/darpa-robotics-challenge), DARPA 官网
5. Koolen T. et al., "Design of a Momentum-Based Control Framework and Application to the Humanoid Robot Atlas", *International Journal of Humanoid Robotics*, 2016
6. Feng S. et al., "Optimization-based Full Body Control for the DARPA Robotics Challenge", *Journal of Field Robotics*, 2015
7. Kuindersma S. et al., "Optimization-based locomotion planning, estimation, and control design for the Atlas humanoid robot", *Autonomous Robots*, 2016
8. Winkler A. W. et al., "Gait and Trajectory Optimization for Legged Systems Through Phase-Based End-Effector Parameterization", *IEEE Robotics and Automation Letters*, 2018
9. [Atlas Does Gymnastics](https://www.youtube.com/watch?v=_sBBaNYex3E), Boston Dynamics YouTube 频道, 2021
10. [New Atlas](https://bostondynamics.com/blog/the-new-atlas/), Boston Dynamics Blog, 2024
11. Posa M. et al., "A Direct Method for Trajectory Optimization of Rigid Bodies Through Contact", *International Journal of Robotics Research*, 2014
12. [人形机器人创新发展指导意见](https://www.miit.gov.cn/), 中国工业和信息化部, 2023
