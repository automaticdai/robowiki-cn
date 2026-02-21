# Figure 02

!!! note "引言"
    Figure 02 是由美国 Figure AI 公司研发的第二代通用人形机器人（General-Purpose Humanoid Robot）。Figure AI 成立于 2022 年，是人形机器人领域融资规模最大的初创公司之一，2024 年 2 月完成约 6.75 亿美元 B 轮融资，投资方涵盖 Microsoft、OpenAI、NVIDIA、Jeff Bezos 等顶级科技机构和个人。Figure 02 深度集成了视觉语言基础模型（Vision-Language Foundation Model, VLM），代表了将大规模预训练模型能力引入物理世界操作的重要探索方向。Figure AI 的核心理念是"AI 优先"（AI-First），即将前沿人工智能作为机器人能力的根本驱动力，而非将 AI 视为传统机器人系统的附加模块。


## 发展历程

Figure AI 从成立到进入工厂商业部署，仅用约两年时间，其发展速度在硬件机器人领域极为罕见。

### 公司创立阶段（2022 年）

Figure AI 由布雷特·阿德科克（Brett Adcock）于 2022 年创立，总部位于美国加利福尼亚州桑尼维尔（Sunnyvale, California）。阿德科克曾创立劳动力市场平台 Vettery（后被 Adecco 收购）和电动垂直起降飞行器（eVTOL）公司 Archer Aviation，并将其推上纳斯达克。

公司从一开始便以"商业化的通用人形机器人"为核心使命，区别于传统的高校研究型机器人项目。Figure AI 的愿景是：人形机器人应当能够在无需对环境做任何改造的前提下，进入工厂、仓库等真实工业场所并胜任人类工人的工作，从而缓解全球劳动力短缺问题。

### Figure 01 研发与演示阶段（2023—2024 年初）

**2023 年 10 月**：Figure 01（第一代原型机）首次公开展示自主行走能力，视频在社交媒体上迅速传播。Figure 01 展示了双足稳定行走、跨越障碍物以及在非平整地面上保持平衡等基础运动能力，证明 Figure AI 的硬件团队具备从零构建高性能人形机器人底盘的工程能力。

**2024 年 1 月**：Figure 01 展示了在咖啡机前自主完成冲泡咖啡全流程操作的演示视频。这一演示意义重大，它表明 Figure 01 不仅能够行走，还能够感知桌面上的物体、规划操作序列，并用机械手执行拿取咖啡豆、放入咖啡机、按下按钮等一系列精细操作。该演示验证了机器人在非结构化工作台环境中完成多步骤任务的可行性。

**2024 年 2 月**：Figure AI 宣布完成约 6.75 亿美元 B 轮融资（Series B），公司估值达约 26 亿美元。本轮融资创下当时人形机器人行业单轮融资金额记录。参与本轮投资的机构和个人包括：

| 投资方 | 类型 |
|--------|------|
| Microsoft | 战略投资 |
| OpenAI Startup Fund | AI 生态战略投资 |
| NVIDIA | 芯片与 AI 生态战略投资 |
| Jeff Bezos（个人） | 个人战略投资 |
| Parkway Venture Capital | 风险投资机构 |
| Intel Capital | 半导体与计算生态战略投资 |
| LG Innotek | 电子元器件产业战略投资 |

这批投资方的构成极具战略意义：既有算力基础设施层（NVIDIA、Intel）、AI 模型层（OpenAI）、云与操作系统层（Microsoft），也有消费电子供应链层（LG Innotek），形成了覆盖从芯片到应用的完整生态布局。

**2024 年 3 月**：Figure AI 发布与 OpenAI 联合开发的语音交互演示视频，在全球范围内引发广泛关注。演示中，Figure 01 通过自然语言对话理解用户指令（如"我可以吃些什么？"），借助视觉感知识别桌面上的苹果，将苹果递给人类，同时用自然语言解释自己的推理过程和行动意图。这一演示首次向公众展示了将 OpenAI 大语言模型与物理机器人行动系统相融合的可能性，被媒体称为"机器人与 AI 结合的里程碑"。

### Figure 02 发布与商业化阶段（2024 年下半年）

**2024 年 8 月**：Figure 02 正式发布。相较于 Figure 01，Figure 02 在硬件结构、AI 算力、能效和灵巧操作能力方面进行了全面升级。Figure AI 同步宣布与宝马（BMW）斯帕坦堡（Spartanburg）工厂签署商业部署协议，Figure 02 成为全球首批在汽车制造工厂实现商业化运营的人形机器人之一。

**2024 年底**：BMW 斯帕坦堡工厂的试点部署进入实际生产验证阶段，Figure AI 在官方渠道发布了机器人在车身车间（Body Shop）执行冲压件搬运任务的实录视频。


## 技术规格对比

Figure 01 与 Figure 02 在关键技术指标上存在显著差异，以下对比表格反映了两代产品的主要改进方向：

| 参数 | Figure 01 | Figure 02 |
|------|-----------|-----------|
| 身高 | 1.70 m | 1.67 m |
| 体重 | 约 65 kg | 约 60 kg |
| 负载能力 | 约 20 kg | 约 25 kg |
| 行走速度 | 约 1.2 m/s | 约 1.2 m/s |
| 续航时间 | 约 4 小时 | 约 5 小时 |
| 整机自由度 | 约 27 | 约 44 |
| 每只手自由度 | 约 6 | 16 |
| 摄像头数量 | 头部 4 个 | 头部 6 个 + 手部摄像头 |
| 驱动方式 | 全电动 | 全电动（效率优化） |
| 内置算力 | 有限本地推理 | 增强本地 AI 推理芯片 |
| 语音交互 | 基础 | 支持自然语言对话 |
| AI 基础模型 | 无原生集成 | Helix VLM 集成 |
| 商业化状态 | 技术演示阶段 | BMW 工厂商业部署 |

Figure 02 的整机自由度从 Figure 01 的约 27 个大幅提升至约 44 个，其中手部自由度的提升（从每只手约 6 个提升至 16 个）是最核心的改进之一，直接决定了机器人灵巧操作能力的上限。


## Helix 基础模型

### 概述

Helix 是 Figure AI 为 Figure 02 开发的机器人专用视觉语言基础模型（Vision-Language Foundation Model），也是 Figure AI "AI 优先"战略的核心技术载体。Helix 的设计目标是打通从自然语言指令和视觉感知到机器人物理动作之间的端到端链路（End-to-End Pipeline），使机器人能够在无需针对每个任务单独编程的前提下，理解并执行多样化的操作任务。

### 模型架构

Helix 采用分层架构（Hierarchical Architecture），将机器人控制问题分解为两个紧密耦合的层次：

**高层语义层（High-Level Semantic Layer）**

高层语义层负责处理来自机器人头部摄像头的 RGB 图像输入和来自操作员或用户的自然语言指令输入。该层基于大规模预训练的视觉语言模型，能够：

- 理解任务语义（如"把红色零件放到传送带上"）
- 在视觉场景中定位目标物体
- 规划任务步骤序列（Task Planning）
- 评估当前任务执行状态并在必要时调整计划

**低层运动控制层（Low-Level Motor Control Layer）**

低层运动控制层接受来自高层语义层的目标状态（Target State）或关键姿态（Key Pose）作为输入，生成具体的关节力矩（Joint Torque）或关节轨迹（Joint Trajectory）指令，驱动机器人的四肢和手部执行实际动作。该层通常基于模型预测控制（Model Predictive Control, MPC）或学习型运动策略（Learned Motor Policy）实现。

两层之间通过统一的接口协议进行通信，高层语义层以较低的频率（约 5—10 Hz）输出语义级别的控制信号，低层运动控制层以更高的频率（约 200—1000 Hz）执行精细的关节控制。

### 零样本泛化能力

Helix 的核心优势之一是零样本泛化（Zero-Shot Generalization）能力，即在面对训练数据中从未出现过的新物体、新场景或新任务描述时，仍能生成合理的行动策略。

这一能力来源于大规模视觉语言预训练所积累的语义知识。具体而言，Helix 在预训练阶段学习了大量关于物体类别、空间关系、动作语义的通用知识，这些知识在微调（Fine-Tuning）和机器人具体任务训练阶段被迁移和专门化，使得模型在面对新任务时能够借助预训练知识进行合理推断。

以数学形式表达，Helix 的目标可以描述为学习一个策略函数 \(\pi\)，使得：

$$a_t = \pi(o_t, \ell)$$

其中 \(o_t\) 为 \(t\) 时刻的视觉观测（Visual Observation），\(\ell\) 为自然语言任务指令（Language Instruction），\(a_t\) 为机器人在 \(t\) 时刻执行的动作（Action）。Helix 的零样本泛化能力意味着该策略函数在 \(\ell\) 描述的任务超出训练分布时仍能给出合理的 \(a_t\)。

### 训练数据与学习流程

Helix 的训练流程分为多个阶段：

**第一阶段：视觉语言预训练（VLP Pre-Training）**

利用互联网规模的图像-文本配对数据对视觉编码器（Visual Encoder）和语言模型（Language Model）进行联合预训练，使模型学习通用的视觉-语义对齐（Visual-Semantic Alignment）能力。此阶段数据量通常在数十亿图文对规模，模型参数量从数十亿到数百亿不等。

**第二阶段：机器人操作数据微调（Robot Manipulation Fine-Tuning）**

在通用预训练模型基础上，使用机器人操作专属数据进行微调，包括：

- **遥操作数据（Teleoperation Data）**：操作员通过遥操作设备（Teleoperation Device）控制机器人完成任务，同步记录机器人的视觉观测、关节状态和动作序列
- **仿真数据（Simulation Data）**：在 MuJoCo、Isaac Sim 等物理仿真环境（Physics Simulation Environment）中自动生成大量操作轨迹，弥补真实数据稀缺的问题
- **人类视频数据（Human Video Data）**：利用人类执行相似操作的视频数据，通过跨形态模仿（Cross-Embodiment Imitation）方法迁移人类操作技能

**第三阶段：在线强化学习（Online Reinforcement Learning）**

将微调后的模型部署到真实机器人上，通过自主试错（Trial-and-Error）收集交互数据，以任务完成率（Task Success Rate）为奖励信号，进一步优化策略。此阶段的数据质量远高于仿真数据，但收集成本也显著更高。

三阶段训练构成了一个从通用到专用、从仿真到真实的渐进精化（Progressive Refinement）流程，是当前机器人基础模型训练的主流范式之一。

### 与 OpenAI 合作的关系

2024 年 3 月的演示展示了 Figure AI 与 OpenAI 的早期合作成果：OpenAI 提供语言理解和推理能力，Figure AI 提供机器人硬件和低层控制能力，两者通过 API 接口（Application Programming Interface）耦合。Helix 的发展是在这一合作基础上的深化，Figure AI 逐步将语言理解模型与机器人控制深度融合，形成专为机器人操作任务优化的一体化基础模型，而非简单调用通用 LLM 接口。

值得注意的是，Helix 并非单纯将 GPT-4o 等通用模型直接用于机器人控制，而是针对机器人场景的独特需求做出了以下关键适配：

- **时序建模（Temporal Modeling）**：机器人操作是时序过程，需要模型对历史动作和观测序列进行有效建模，而通用 VLM 主要处理静态图文对。Helix 引入了对动作历史的显式建模机制
- **3D 空间感知（3D Spatial Perception）**：操作任务需要对物体三维位置和朝向的精确估计，Helix 在视觉编码阶段引入了深度信息（Depth Information）和立体视觉（Stereo Vision）处理能力
- **实时性约束（Real-Time Constraint）**：机器人控制对推理延迟极为敏感，Helix 通过模型量化（Model Quantization）和硬件加速优化，将推理延迟控制在可接受范围内


## 灵巧手技术

### 设计理念

Figure 02 的灵巧手（Dexterous Hand）是整机最具技术含量的子系统之一。其设计理念源于对人类手部解剖学（Hand Anatomy）的深入研究：人类手部拥有约 27 块骨骼、29 个关节和超过 30 块肌肉，能够实现从粗力量抓取（Power Grasp）到精密捏取（Precision Pinch）的宽泛操作模式。Figure 02 的 16 自由度手部设计旨在尽可能近似这一能力范围，同时在工程可行性（材料强度、电机体积、重量限制）方面做出合理的工程折中。

### 关节与驱动方案

每只手的 16 个自由度分布于拇指（Thumb）、食指（Index Finger）、中指（Middle Finger）、无名指（Ring Finger）和小指（Little Finger）五根手指，以及手腕（Wrist）关节。各手指通常配置 2—3 个主动自由度（Active DOF），手腕配置 2 个自由度（屈伸与侧偏）。

驱动方案采用微型电机（Micro Motor）配合肌腱传动（Tendon Drive）的组合方式：

- **微型电机**：布置于手掌或前臂，避免将大质量部件置于手指末端，降低手指的转动惯量（Moment of Inertia），从而提升手指运动的动态响应速度
- **肌腱传动**：细钢缆（Cable）模拟肌腱，将电机力矩传递至手指关节，实现较高的传动效率（Transmission Efficiency）和较小的传动间隙（Backlash）

与液压驱动（Hydraulic Actuation）相比，全电动腱驱方案在力控精度（Force Control Precision）方面略有不足，但在能量效率（Energy Efficiency）、维护成本（Maintenance Cost）和环境适应性方面具有明显优势，更适合工厂环境长时间连续运行的要求。

### 触觉感知

Figure 02 的手部集成了触觉传感器（Tactile Sensor），能够在手指指尖和手掌关键接触区域实时采集接触力（Contact Force）和接触面积（Contact Area）信息。触觉感知数据与视觉数据融合后，为抓取策略提供关键反馈：

- **抓取力调节**：根据物体表面材质（硬/软）和重量，动态调节抓取力大小，避免压碎易碎物体（如玻璃容器）或因力度不足导致物体滑落
- **接触检测**：在视觉遮挡情况下（如手指被物体遮挡时），通过触觉信号判断是否已建立有效接触
- **滑移检测（Slip Detection）**：通过高频采样触觉信号，检测物体在手指间的微小滑移趋势，并及时触发补偿动作

### 操作模式

Figure 02 的手部支持两种主要操作模式，两种模式之间可根据任务需求动态切换：

**精密捏取模式（Precision Pinch Mode）**：主要使用拇指和食指（或拇指与中指）的指尖进行小物体的精细夹持，适用于拾取螺钉、插拔连接器、操作按钮等需要高精度定位的任务。在此模式下，控制系统优先保证位置精度（Position Accuracy），接触力维持在较小水平（通常 0.5—2 N）。

**力量抓取模式（Power Grasp Mode）**：五根手指同时包裹物体，利用手掌和手指的综合接触面积提供最大抓持力，适用于搬运较重零件、推拉机构等需要大抓持力的任务。在此模式下，控制系统优先保证抓持稳定性，接触力可达数十牛顿。

### 手部视觉引导

Figure 02 在手腕附近集成了 RGB 摄像头（Hand-Mounted RGB Camera），为近距离精细操作提供"手眼协调"（Hand-Eye Coordination）能力。头部摄像头负责全局场景感知和目标定位，手部摄像头则负责在操作执行阶段提供高分辨率的近距离视觉反馈，两者协同工作，显著提升了机器人在工件对准（Workpiece Alignment）、插孔（Peg-in-Hole）等高精度任务中的成功率。

手眼协调的标定（Calibration）是保证视觉引导精度的关键步骤。Figure 02 采用眼在手上（Eye-in-Hand）配置，需要精确标定手部摄像头相对于手腕坐标系（Wrist Frame）的外参矩阵（Extrinsic Matrix）\(\mathbf{T}_{cam}^{wrist}\)，以及摄像头内参（Intrinsic Parameters）。标定误差直接影响视觉引导操作的精度，通常要求总体定位误差控制在 1—2 mm 以内。

### 手部与整机的力控集成

灵巧手的力控能力并非孤立运作，而是与整机力控系统深度集成。当机器人执行需要力柔顺（Force Compliance）的任务（如将零件插入有公差的孔位）时，手部关节的力矩传感器数据与腕部六维力/力矩传感器（6-Axis Force/Torque Sensor）数据融合，共同驱动阻抗控制（Impedance Control）算法：

$$\mathbf{F} = \mathbf{K}(\mathbf{x}_d - \mathbf{x}) + \mathbf{D}(\dot{\mathbf{x}}_d - \dot{\mathbf{x}}) + \mathbf{M}(\ddot{\mathbf{x}}_d - \ddot{\mathbf{x}})$$

其中 \(\mathbf{K}\)、\(\mathbf{D}\)、\(\mathbf{M}\) 分别为刚度矩阵（Stiffness Matrix）、阻尼矩阵（Damping Matrix）和惯性矩阵（Inertia Matrix），\(\mathbf{x}_d\) 为期望位置，\(\mathbf{x}\) 为实际位置。通过调节刚度矩阵，可以在不同任务中灵活切换位置控制主导模式和力控柔顺主导模式，是实现精密装配操作的关键控制技术。


## 感知系统

### 多摄像头立体视觉

Figure 02 在头部配置了 6 个摄像头，构成多目立体视觉系统（Multi-Camera Stereo Vision System）。与单目摄像头（Monocular Camera）相比，多目配置具有以下优势：

- **深度估计（Depth Estimation）**：通过双目或多目几何关系直接计算场景中物体的三维深度，精度优于纯单目深度估计网络（Monocular Depth Network），在近距离操作场景下尤为重要
- **宽视场覆盖（Wide Field of View Coverage）**：6 个摄像头提供近乎 360° 的环境感知覆盖，消除单摄像头在机器人运动过程中可能出现的视觉盲区
- **冗余容错（Redundancy）**：当某个方向的摄像头受到遮挡或反光干扰时，其他摄像头的数据可以提供补充信息

视觉处理流程通常包括以下步骤：原始图像采集 → 图像去畸变（Undistortion） → 多目立体匹配（Stereo Matching） → 点云生成（Point Cloud Generation） → 目标检测与语义分割（Object Detection & Semantic Segmentation） → 三维目标位姿估计（6-DoF Object Pose Estimation）。

### 视觉惯性里程计

Figure 02 集成了视觉惯性里程计（Visual-Inertial Odometry, VIO）系统，通过融合摄像头图像与惯性测量单元（Inertial Measurement Unit, IMU）的加速度计（Accelerometer）和陀螺仪（Gyroscope）数据，实现机器人在无 GPS 环境（如室内工厂）下的高精度自身位姿估计（Self-Pose Estimation）：

$$\hat{\mathbf{T}}_{WB} = \text{VIO}(\{\mathbf{I}_t\}, \{\mathbf{a}_t, \boldsymbol{\omega}_t\})$$

其中 \(\hat{\mathbf{T}}_{WB}\) 为机器人本体坐标系（Body Frame）相对于世界坐标系（World Frame）的估计位姿，\(\mathbf{I}_t\) 为图像序列，\(\mathbf{a}_t\) 为加速度测量，\(\boldsymbol{\omega}_t\) 为角速度测量。

准确的自身位姿估计是机器人在工厂内自主导航（Autonomous Navigation）的基础，也是全局操作规划（Global Manipulation Planning）中准确定位工作台和目标物体的前提条件。

### 本体感知

除外部环境感知外，Figure 02 还具备完整的本体感知（Proprioception）系统，包括：

- **关节编码器（Joint Encoder）**：精确测量每个关节的角度和角速度，为运动控制提供反馈
- **关节力矩传感器（Joint Torque Sensor）**：直接测量每个关节输出的力矩，用于力控和碰撞检测
- **足底压力传感器（Foot Pressure Sensor）**：测量双足与地面的接触力分布，为步态控制（Gait Control）和平衡控制（Balance Control）提供关键反馈

这些本体感知数据以高频（通常 500 Hz 以上）实时采集，与低层运动控制器紧密耦合，是机器人在动态环境中保持稳定平衡的基础。


## BMW 工厂合作

### 合作背景

宝马集团（BMW Group）斯帕坦堡工厂（Spartanburg Plant）位于美国南卡罗来纳州（South Carolina），是宝马全球最大的单一生产基地，主要生产 X 系列运动型多用途车（SUV）。该工厂年产能超过 40 万辆，雇用约 11,000 名工人。

宝马长期以来是工业机器人的大规模用户，其生产线上部署了大量传统工业机械臂（Industrial Robot Arm）。然而，传统工业机器人通常被固定在特定工位，仅能在高度结构化的环境中重复执行预编程动作，缺乏在工厂内自主移动和灵活执行多种任务的能力。人形机器人的引入被视为填补"最后一英里灵活性"（Last-Mile Flexibility）缺口的可能方案。

### 部署任务与场景

Figure 02 在 BMW 斯帕坦堡工厂的首期部署集中在车身车间（Body Shop），主要执行以下任务：

**冲压件搬运（Stamped Parts Handling）**：将金属冲压件（Stamped Sheet Metal Parts，即已成型的车身钣金件）从存放区取出，搬运至指定放置点或传送带入口。此类任务要求机器人能够识别形状各异的钣金件、规划无碰撞的抓取姿态，并在搬运过程中保持零件姿态稳定，防止划伤或变形。

**零件放置与对位（Part Placement and Alignment）**：将零件精确放置到工装夹具（Fixture）或装配工位上，对对位精度有一定要求。这类任务对机器人的手眼协调能力和末端执行器（End-Effector）控制精度要求较高。

### 概念验证的意义

BMW 部署案例作为概念验证（Proof of Concept, PoC），其意义超越了具体任务本身：

1. **安全合规验证**：证明人形机器人能够在有人类工人共同作业的工厂环境中，满足工业安全标准（如 ISO 10218 协作机器人安全要求），不对工人造成伤害风险
2. **任务可靠性验证**：在真实工业环境（非实验室洁净条件）下，验证机器人对噪声、振动、灰尘、光照变化等干扰因素的鲁棒性（Robustness）
3. **商业模式验证**：探索人形机器人以"机器人即服务"（Robot as a Service, RaaS）模式向制造商提供的商业可行性，为后续大规模商业化积累数据和经验

BMW 官方表示，此次合作是"评估人形机器人在宝马生产运营中长期潜力"的重要一步，并未承诺大规模采购，但为 Figure AI 提供了极具价值的真实工业场景验证机会。

### 工厂部署的技术挑战

将人形机器人部署到真实汽车工厂面临的技术挑战远比实验室演示复杂：

**环境感知鲁棒性**：工厂车身车间的光照条件复杂，存在强反光的金属表面、局部遮挡、烟雾（如焊接产生的烟尘）等干扰因素，对机器人视觉系统的鲁棒性（Robustness）提出严苛要求。Figure 02 的头部 6 摄像头配置提供了更宽的视野覆盖，并通过多视图融合（Multi-View Fusion）降低单一视角遮挡带来的感知盲区。

**人机协作安全（Human-Robot Collaboration Safety）**：斯帕坦堡工厂的车身车间同时有人类工人作业，Figure 02 必须满足 ISO/TS 15066（协作机器人安全技术规范）的要求，确保在与人类发生意外接触时能够立即停止运动或限制接触力。机器人配备了多层安全机制：视觉感知的人体检测（Human Detection）、力矩传感器驱动的碰撞检测（Collision Detection）以及冗余的安全停机电路（Safety Stop Circuit）。

**节拍匹配（Cycle Time Matching）**：汽车制造业的生产节拍（Takt Time）极为严格，通常以秒为单位管理。机器人的任务执行时间必须与整体生产节拍相匹配，否则将成为产线瓶颈。在试点阶段，Figure 02 的执行效率通常低于熟练工人，因此被安排在节拍要求相对宽松的上下料（Material Handling）工位，而非直接替代核心装配工位。

**可靠性与维护**：工厂环境要求机器人具备极高的运行可靠性（MTBF，Mean Time Between Failures），以及便于快速维修和更换零部件的可维护性设计（Maintainability Design）。这对 Figure AI 而言是重要的工程挑战，因为当前阶段的人形机器人样机与大规模量产的工业机械臂相比，在可靠性和耐用性方面仍存在明显差距。


## AI 优先策略

### 传统机器人编程的局限

传统工业机器人的编程范式（Programming Paradigm）以"示教—再现"（Teach and Playback）为核心：工程师通过手动引导机器人运动轨迹或编写运动程序（Motion Program），将特定任务的操作步骤硬编码（Hard-Coded）到机器人控制器中。机器人随后精确重复这些预定义动作，几乎不具备对环境变化的自适应能力。

这种范式在高度结构化、高度重复的大批量生产（Mass Production）中效率极高，但存在以下根本性局限：

- **部署成本高**：每引入一种新任务，需要专业工程师重新示教和编程，部署周期以天到周计
- **泛化能力差**：当物体位置、姿态或环境布局发生细微变化时，预编程的机器人可能直接失败
- **灵活性低**：无法在任务之间动态切换，更无法处理非预期情况（Unexpected Situations）

### Figure AI 的 AI 优先哲学

Figure AI 的 AI 优先（AI-First）策略是对上述局限的正面回应。其核心主张是：

> 机器人的"智能"不应来自人类工程师事先编写的规则和程序，而应来自从大规模数据中学习得到的通用能力模型。

具体体现在以下几个层面：

**基础模型驱动（Foundation Model-Driven）**：使用在互联网规模数据上预训练的视觉语言基础模型作为机器人的"认知大脑"，使机器人天然具备对日常物体、空间关系和人类指令的语义理解能力，而无需为每个物体类别单独编写识别程序。

**端到端学习（End-to-End Learning）**：尽量减少人工设计的中间模块（如独立的物体检测模块、轨迹规划模块），让模型直接从原始感知输入（图像、语音）学习到动作输出，以减少模块间接口引入的误差积累和信息损失。

**数据飞轮（Data Flywheel）**：通过在真实工厂环境中积累大量机器人操作数据，持续改进基础模型的能力，形成"部署越多、数据越多、模型越强、部署越成功"的正向循环。

### 与 OpenAI 合作的独特性

Figure AI 与 OpenAI 的合作不同于一般企业对 AI 能力的简单调用，其独特性体现在：

**联合开发而非单纯 API 调用**：双方团队深度合作，针对机器人操作场景的特殊需求（如对三维空间的精确感知、对物理接触的理解、对实时性的严格要求）对基础模型进行专项优化，而不仅仅是通过 HTTP 接口调用通用 GPT 模型。

**具身智能（Embodied Intelligence）的共同探索**：OpenAI 视此次合作为探索"具身智能"的重要实验场——通过真实物理交互积累经验，可能反过来促进更通用 AI 能力的发展。这种双向价值使合作超越了单纯的商业服务关系。

**数据产权与安全**：在工厂环境部署过程中收集的操作数据，其产权归属和使用权限是双方合作协议中的关键条款，也是整个机器人 AI 产业在数据战略层面面临的共同挑战。

### 与传统机器人编程范式的根本差异

为直观说明 AI 优先策略与传统范式的差异，以"将工件从料箱（Bin）中取出并放到传送带上"这一典型工业任务为例：

**传统编程范式下的实现**：工程师使用机器人视觉软件（如 HALCON 或 OpenCV）开发专用的工件识别程序，定义特征模板（Feature Template）；使用运动规划软件（如 MoveIt）规划从识别位置到放置位置的轨迹；将整套程序部署到机器人控制器，调试参数直至达标。全程需要 1—4 周的工程时间，且当工件型号更换时需重新开发。

**Helix AI 优先范式下的实现**：操作员通过自然语言描述任务（"从左侧料箱中取出蓝色零件，放到右侧传送带的入口处"），Helix 利用预训练知识理解"蓝色零件"的视觉特征、"料箱"和"传送带"的空间关系，直接生成操作策略。若任务描述或场景发生变化，只需更新自然语言指令，而无需修改任何代码。

这一对比揭示了 AI 优先策略的根本价值主张：将任务部署的成本从"工程师-周"压缩到"操作员-分钟"量级，从而在经济上实现人形机器人在中小批量、多品种生产场景中的商业可行性。


## Figure 03 展望

截至本文撰写时（2026 年初），Figure AI 尚未正式发布 Figure 03 的详细技术规格，但根据公司公开声明和行业分析人士的推断，下一代产品可能在以下方向进行重点迭代：

**更高的手部灵巧度**：进一步增加手部自由度，或引入更高密度的触觉传感器阵列，以支持更复杂的装配任务（如线束插接、精密螺纹紧固）。

**更强的端侧算力（Edge Computing）**：随着专用 AI 推理芯片（如 NVIDIA Jetson 系列的后续产品）的发展，预计 Figure 03 将集成更强大的本地推理能力，降低对云端网络连接的依赖，提升在网络受限工厂环境中的可靠性。

**提升能量密度**：电池技术的进步（如固态电池 Solid-State Battery 的逐步商用化）可能使 Figure 03 在相同体重约束下获得更长的续航时间，或在相同续航要求下减轻电池重量，改善整机重量分布。

**双臂协调能力**：Figure AI 已在演示中展示了双臂协调操作的初步能力，预计 Figure 03 将在双臂任务规划（Bimanual Task Planning）和力控协调（Force-Controlled Bimanual Coordination）方面取得更大突破，支持需要双手配合的复杂装配操作。

**Helix 模型的持续迭代**：随着工厂部署数据积累，Helix 基础模型的零样本泛化能力和任务成功率预计将持续提升。Figure 03 可能搭载更大参数规模（Parameter Scale）或更优架构的 Helix 后续版本。

**扩展商业化场景**：BMW 合作的成功验证将为 Figure AI 打开汽车行业其他 OEM（原始设备制造商）以及电子制造、食品饮料等行业的商业谈判大门。Figure 03 的推出很可能伴随面向更多垂直行业（Vertical Industry）的专项功能包（Feature Package）。

**感知模态的扩展**：当前 Figure 02 主要依赖视觉和触觉两种感知模态，Figure 03 可能引入力觉（Force Sensing）、听觉（Auditory Sensing）乃至嗅觉（Olfactory Sensing）等额外感知通道，以应对更广泛的任务场景需求。例如，在食品行业的应用中，听觉感知可用于检测包装密封状态，嗅觉传感器可用于质量检测（Quality Inspection）。

**软件平台开放化**：部分行业分析人士预测，Figure AI 可能效仿 NVIDIA 的商业模式，在销售机器人硬件的同时开放 Helix 模型的 API，向第三方开发者和系统集成商（System Integrator）提供软件许可（Software License），构建类似 App Store 的机器人应用生态。

**扩大生产规模**：随着需求增长，Figure AI 需要从小批量手工装配（Low-Volume Manual Assembly）过渡到规模化量产（High-Volume Manufacturing）。这意味着公司需要建立或与代工伙伴合作，将精密机电系统的生产成本大幅压缩，以实现具有商业竞争力的出厂价格（Bill of Materials, BOM 成本控制）。


## 行业定位与竞争

### 主要竞争产品对比

人形机器人市场正在快速发展，多家公司推出了各具特色的产品。以下表格对 Figure 02 与同期主要竞品在关键维度上进行横向对比（数据来源于各公司官方公布信息，部分指标为估算值）：

| 参数 | Figure 02 | Tesla Optimus Gen 2 | Boston Dynamics Atlas | Unitree G1 | UBTECH Walker S |
|------|-----------|---------------------|-----------------------|------------|-----------------|
| 研发公司 | Figure AI（美国） | Tesla（美国） | Boston Dynamics（美国） | 宇树科技（中国） | 优必选（中国） |
| 发布时间 | 2024 年 8 月 | 2024 年 初 | 2024 年（电动版） | 2024 年 | 2023 年 |
| 身高 | 1.67 m | 1.73 m | 1.50 m | 1.27 m | 1.70 m |
| 体重 | 约 60 kg | 约 57 kg | 约 89 kg | 约 35 kg | 约 70 kg |
| 负载能力 | 约 25 kg | 约 20 kg | 约 11 kg | 约 3 kg | 约 10 kg |
| 行走速度 | 约 1.2 m/s | 约 0.5 m/s | 约 1.5 m/s | 约 2.0 m/s | 约 0.6 m/s |
| 驱动方式 | 全电动 | 全电动 | 全电动 | 全电动 | 全电动 |
| 手部自由度 | 16（每只手） | 11（每只手） | 未公开 | 7（每只手） | 12（每只手） |
| AI 集成深度 | 高（Helix VLM） | 高（Tesla FSD 技术迁移） | 中（感知与规划） | 中（学习型控制） | 中（语音交互）|
| 主要商业场景 | 汽车制造（BMW） | Tesla 工厂内部 | 研究与特种作业 | 教育、科研 | 工业物流 |
| 公开售价 | 未公开 | 未公开（预计数万美元） | 不对外销售 | 约 16,000 美元 | 未公开 |

### 差异化竞争优势

**与 Tesla Optimus 的差异**：Tesla 的 Optimus 项目具有独特的"工厂自用"逻辑——Optimus 首先服务于 Tesla 自身的超级工厂（Gigafactory），具有天然的封闭场景优势，无需向外部客户证明产品价值。Figure AI 则走完全市场化路线，其商业压力更大，但潜在市场也更广泛。

**与 Boston Dynamics Atlas 的差异**：Atlas 代表了传统动力学驱动（Dynamics-Driven）机器人的最高水准，在运动能力（跑步、跳跃、翻跟头）上远超当前所有竞争对手，但其商业化进展明显慢于 Figure AI，且在 AI 基础模型集成方面起步较晚。Boston Dynamics 于 2024 年退役了液压版 Atlas，推出全电动新版本，并明确向工业应用场景转型，是 Figure AI 未来最直接的竞争对手之一。

**与 Unitree G1 的差异**：宇树科技（Unitree Robotics）的 G1 以极具竞争力的价格（约 16,000 美元）面向科研和教育市场，主打高性价比和开放软件生态。G1 在运动敏捷性方面表现出色，但在负载能力和手部灵巧度上与 Figure 02 存在较大差距，两者目前面向的主要市场也不同。

**与 UBTECH Walker S 的差异**：优必选（UBTECH）是中国最早商业化人形机器人的企业之一，Walker S 在工业物流场景中已有一定规模的部署案例，在国内市场具有先发优势。Figure AI 在 AI 基础模型集成深度和单机灵巧操作能力上具有优势，但 Walker S 在工业物流场景的软件集成（与 WMS、ERP 系统的对接）方面积累更为成熟。

### 行业整体趋势

当前人形机器人行业正处于从"技术演示"向"早期商业化"的关键过渡阶段，主要趋势包括：

1. **AI 基础模型成为标配**：随着 Figure Helix、Tesla FSD-for-robots 等方案的出现，将大规模预训练模型用于机器人控制逐渐成为行业共识，而非 Figure AI 独有的差异化优势
2. **工厂场景成为主战场**：由于工业制造和仓储物流对灵活劳动力的需求明确且支付意愿较强，大多数人形机器人企业都将工厂作为首要商业化场景
3. **国际竞争格局形成**：美国（Figure、Boston Dynamics、Tesla）和中国（宇树、优必选、智元、宇联、达闼等）正形成两大竞争阵营，双方在技术路线、商业模式和政策支持方面各有侧重
4. **数据积累成为核心竞争壁垒**：在基础模型架构趋于同质化的背景下，在真实工厂环境中积累高质量机器人操作数据的能力，将成为决定各家公司长期竞争力的关键因素

### 商业模式分析

Figure AI 目前的商业模式尚在探索阶段，但已初步呈现出以下特征：

**硬件销售与租赁并行**：与传统工业机器人的一次性销售模式不同，Figure AI 可能采取部分出租（Leasing）或机器人即服务（RaaS）的订阅制（Subscription）模式，以降低制造商的初始采购门槛，同时为自身创造持续的软件和数据服务收入来源。

**软件与数据增值服务**：Helix 模型的持续训练和更新是 Figure AI 的核心价值主张之一。通过为客户提供持续的模型迭代升级（Over-the-Air Update，OTA），Figure AI 可以建立与客户的长期服务关系，类似于特斯拉汽车的软件订阅模式。

**垂直行业解决方案**：Figure AI 可能逐步向汽车、物流、半导体等特定垂直行业推出针对性的解决方案包，包括预训练的行业特定技能库（Skill Library）和与客户现有制造执行系统（Manufacturing Execution System, MES）的集成接口。

### 主要技术风险

尽管 Figure AI 取得了快速进展，但其技术路线仍面临若干重要的不确定性：

- **推理延迟与实时控制的矛盾**：大型 VLM 的推理延迟通常在数百毫秒量级，而精密操作任务要求控制频率达到数百赫兹。如何在不牺牲模型能力的前提下大幅压缩推理延迟，是当前最核心的工程瓶颈
- **长尾场景的可靠性**：在高频出现的典型任务上，AI 模型可以达到较高的成功率；但在工厂中不可避免出现的罕见异常情况（如零件变形、设备故障、工人干预）下，模型的表现往往不稳定，这是影响实际商业部署规模的主要障碍
- **监管与认证**：在欧美市场，将 AI 驱动的自主机器人引入有人作业的工厂场景，需要通过 CE 认证（欧盟）或 OSHA 合规审查（美国），监管框架的不确定性可能延缓大规模商业化进程


## 参考资料

1. [Figure AI 官网](https://www.figure.ai/), Figure AI
2. [Figure 02 发布公告](https://www.figure.ai/figure-02), Figure AI, 2024
3. [Figure raises $675M from Microsoft, OpenAI and others](https://techcrunch.com/2024/02/29/figure-ai-raises-675m/), TechCrunch, 2024
4. [Figure 01 x OpenAI — Figure's First Conversation](https://www.youtube.com/watch?v=Sq1QZB5baNw), Figure AI, YouTube, 2024
5. [BMW and Figure AI to bring robots to auto factory](https://techcrunch.com/2024/01/18/bmw-and-figure-ai/), TechCrunch, 2024
6. [Figure AI announces Helix, an AI model for humanoid robots](https://techcrunch.com/2024/11/07/figure-ai-announces-helix/), TechCrunch, 2024
7. [Boston Dynamics unveils new electric Atlas robot](https://techcrunch.com/2024/04/17/boston-dynamics-unveils-all-electric-atlas-robot/), TechCrunch, 2024
8. [Unitree G1 humanoid robot](https://www.unitree.com/g1/), 宇树科技官网, 2024
9. [UBTECH Walker S industrial humanoid](https://www.ubtrobot.com/), 优必选官网, 2024
10. [Robot as a Service: The emerging business model for humanoid robots](https://spectrum.ieee.org/), IEEE Spectrum, 2024
11. [Visual-Inertial Odometry: A Survey](https://arxiv.org/), arXiv, 2023
12. [Dexterous Manipulation: From Biological Inspiration to Robotic Implementation](https://ieeexplore.ieee.org/), IEEE Transactions on Robotics, 2023
