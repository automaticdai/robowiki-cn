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


## 本章内容导览

| 页面 | 主要内容 |
|------|---------|
| [Figure](figure.md) | 发展历程、技术规格对比、Figure 03 展望与行业定位 |
| [Helix 模型与 AI 优先策略](figure-helix-ai.md) | Helix VLA 架构、双系统设计、训练方式与部署成本结构 |
| [Figure 硬件与 BMW 工厂部署](figure-hardware-bmw.md) | 灵巧手、感知系统、BMW Spartanburg 产线试点 |
| [人形机器人图鉴](robots.md) | 各厂商人形机器人横向对比 |
| [Atlas](atlas.md) | Boston Dynamics 人形机器人平台 |
| [Optimus](optimus.md) | 特斯拉人形机器人 |


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
