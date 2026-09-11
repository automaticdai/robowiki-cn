# 波士顿动力 | Boston Dynamics

!!! note "引言"
    波士顿动力（Boston Dynamics）是全球最具影响力的机器人公司之一，以其在动态运动控制领域的突破性成就著称。从马克·雷伯特（Marc Raibert）在麻省理工学院腿部实验室的早期研究出发，经过三十余年的技术积累，波士顿动力已将动态平衡理论转化为 BigDog、Atlas、Spot、Handle 和 Stretch 等一系列里程碑式的机器人产品，深刻改变了人类对机器人运动能力极限的认知。


## 公司历史

### 创始背景与 MIT 腿部实验室

波士顿动力的技术根脉可以追溯到 20 世纪 80 年代麻省理工学院（MIT）的腿部实验室（Leg Laboratory）。马克·雷伯特（Marc Raibert）于 1980 年加入 MIT，专注于研究动态平衡与奔跑机器的控制问题。

1983 年，雷伯特团队研制出单腿跳跃机器人"Raibert Hopper"，这是世界上首台能够动态自稳定跳跃的机器人。该机器人证明了一个关键思想：通过在每个落脚点精确调节脚的落点位置，即可维持奔跑过程中的动态平衡，而无需依赖静态支撑多边形。这一理论后来被形式化为弹跳腿倒立摆（Spring-Loaded Inverted Pendulum，SLIP）模板模型，成为整个足式机器人领域的理论基础。

雷伯特团队随后将单腿跳跃机器人的原理推广到双腿和四腿系统，研制了一系列奔跑机器人，最高速度达到当时的记录水平。这些早期研究确立了一套以"运动中控制、落脚点调节、姿态恢复"为核心的动态运动哲学，这一哲学贯穿了波士顿动力此后所有产品的设计。

### 公司成立与早期 DARPA 合作

1992 年，马克·雷伯特从 MIT 腿部实验室独立出来，正式创立波士顿动力公司，初始核心团队由其在 MIT 的研究伙伴组成。公司创立之初即与美国国防高级研究计划局（Defense Advanced Research Projects Agency，DARPA）建立了深度合作关系。

DARPA 在此后二十年间为波士顿动力提供了大量研发资金，支持了 BigDog、LS3、Atlas 等重量级项目。这种军方科研资助模式使得波士顿动力得以在商业化压力有限的环境下专注于技术突破，形成了其独特的"技术驱动、演示先行"的公司文化。

公司早期还承接了 DI-Guy 人物仿真软件业务，与美国系统公司合作，为美国海军航空作战中心训练处（NAWCTSD）开发基于三维人物互动仿真的训练系统，以取代传统的舰载机弹射任务训练影片。

### 所有权变迁史

| 时间 | 事件 | 交易金额 |
|------|------|----------|
| 1992 年 | Marc Raibert 从 MIT 独立创立，私有公司 | — |
| 2013 年 12 月 | Google 母公司 Alphabet 收购 | 未公开（估计数亿美元） |
| 2017 年 6 月 | 软银集团（SoftBank Group）收购 | 约 1 亿美元 |
| 2021 年 6 月 | 现代汽车集团（Hyundai Motor Group）完成收购 80% 股权 | 约 11 亿美元 |

**Google/Alphabet 时期（2013—2017）**：Google 在 2013 年底密集收购了多家机器人公司，波士顿动力是其中最重要的标的。在 Google 旗下期间，波士顿动力相对独立运作，持续推进 Atlas、Spot（早期版本 SpotMini）等项目的研发，但据报道与 Google 在商业化路径上存在分歧，最终导致出售。

**软银时期（2017—2021）**：软银以约 1 亿美元的相对较低价格接手波士顿动力。这一时期，波士顿动力首次将 Spot 推向商业市场（2019 年），标志着公司从纯研究驱动转向产品商业化的重要转型。

**现代时期（2021 年至今）**：韩国现代汽车集团于 2020 年 12 月宣布、2021 年 6 月完成以约 11 亿美元收购波士顿动力约 80% 股权的交易，软银通过附属公司继续持有约 20% 股份。现代集团的战略意图在于将机器人技术与智能制造、未来出行（Future Mobility）生态深度融合。


## 关键技术方法

### 模板模型：SLIP 与弹簧质量系统

波士顿动力的运动控制哲学建立在"模板模型（Template Model）"的理论框架之上。模板模型是一种高度简化的动力学模型，用于捕捉复杂生物体或机器人运动的本质规律，再将其映射到真实系统的控制上。

最核心的模板模型是**弹簧腿倒立摆**（Spring-Loaded Inverted Pendulum，SLIP）。SLIP 模型将整个机器人简化为一个质点加一根无质量弹性腿，通过控制弹簧刚度和落脚角度来维持奔跑中的动态平衡。其状态方程可以写为：

$$
m\ddot{\mathbf{r}} = \mathbf{F}_{\text{spring}} + m\mathbf{g}
$$

其中 \(\mathbf{r}\) 为质心位置，\(\mathbf{F}_{\text{spring}} = k(l_0 - |\mathbf{r} - \mathbf{r}_{\text{foot}}|)\hat{l}\) 为弹性腿产生的恢复力，\(k\) 为等效腿刚度，\(l_0\) 为自然腿长。

在 SLIP 框架下，落脚点的选择至关重要。雷伯特提出的落脚点调节法则为：

$$
x_{\text{foot}} = x_{\text{hip}} + \frac{\dot{x} T_{\text{stance}}}{2} + \frac{1}{2}\sqrt{\frac{l_0}{g}}(\dot{x} - \dot{x}_d)
$$

其中 \(T_{\text{stance}}\) 为支撑相时长，\(\dot{x}_d\) 为期望速度，最后一项为速度误差补偿项。这一简洁规则使机器人即便受到推搡也能快速恢复平衡。

### 轨迹优化与接触隐式规划

对于更复杂的动作序列（如翻跟斗、跑酷），波士顿动力采用**序列轨迹优化**（Sequential Trajectory Optimization，STO）和**接触隐式优化**（Contact-Implicit Optimization）方法。

接触隐式优化将接触力和接触时序作为优化变量，与运动轨迹联合求解，无需预先指定接触序列。其优化问题一般形式为：

$$
\min_{\mathbf{q}, \dot{\mathbf{q}}, \mathbf{u}, \boldsymbol{\lambda}} \int_0^T \ell(\mathbf{q}, \dot{\mathbf{q}}, \mathbf{u}) \, dt
$$

$$
\text{s.t.} \quad M(\mathbf{q})\ddot{\mathbf{q}} + C(\mathbf{q}, \dot{\mathbf{q}}) = \mathbf{S}^T\mathbf{u} + \mathbf{J}_c^T\boldsymbol{\lambda}
$$

$$
\boldsymbol{\lambda} \geq 0, \quad \phi(\mathbf{q}) \geq 0, \quad \boldsymbol{\lambda} \cdot \phi(\mathbf{q}) = 0
$$

其中 \(\mathbf{q}\) 为广义坐标，\(\boldsymbol{\lambda}\) 为接触力，\(\phi(\mathbf{q})\) 为距离约束，最后一个方程为互补条件（Complementarity Condition），表达"要么有接触力、要么有间隙"的物理约束。

### 全身控制与分层二次规划

**全身控制**（Whole-Body Control，WBC）是波士顿动力在 Atlas 等人形机器人上实现高动态性能的核心技术。WBC 将机器人的多个控制目标（质心轨迹、末端执行器位置、关节力矩限制、摩擦锥约束等）表达为一个**分层二次规划**（Hierarchical QP）问题，按优先级依次求解：

1. **第一优先级**：运动方程约束、接触约束（等式约束）
2. **第二优先级**：质心动量控制目标（最小化加速度误差）
3. **第三优先级**：末端执行器任务（操作臂跟踪目标）
4. **第四优先级**：关节阻尼、姿势参考（正则化）

每一层在不违反更高优先级约束的前提下，最小化本层的加权代价函数：

$$
\min_{\ddot{\mathbf{q}}, \boldsymbol{\lambda}, \boldsymbol{\tau}} \| \mathbf{J}_i \ddot{\mathbf{q}} + \dot{\mathbf{J}}_i \dot{\mathbf{q}} - \mathbf{a}_i^{\text{ref}} \|_{\mathbf{W}_i}^2
$$

### 线性二次型调节器与平衡控制

对于直立平衡控制，波士顿动力大量使用**线性二次型调节器**（Linear Quadratic Regulator，LQR）。将机器人线性化为倒立摆模型后，状态反馈控制律为：

$$
u = -K(x - x_d)
$$

其中 \(x\) 为系统状态（质心位置、速度、姿态角及其导数），\(x_d\) 为期望状态，增益矩阵 \(K\) 通过求解代数黎卡提方程（Algebraic Riccati Equation）得到：

$$
A^T P + PA - PBR^{-1}B^T P + Q = 0, \quad K = R^{-1}B^T P
$$

其中 \(Q\) 为状态代价权重矩阵，\(R\) 为控制代价权重矩阵，两者的选取决定了平衡控制的响应速度与能量消耗之间的权衡。

### 感知系统：多传感器融合

波士顿动力机器人的感知系统采用多传感器融合架构，典型配置包括：

- **立体视觉相机（Stereo Camera）**：用于近距离障碍物检测和地形估计，提供稠密深度图
- **激光雷达（LiDAR）**：提供中远距离的精确三维点云，用于建图和定位
- **惯性测量单元（Inertial Measurement Unit，IMU）**：高频（通常 1 kHz）测量机体加速度和角速度，是动态控制的核心传感器
- **足端力传感器（Foot Force Sensor）**：测量各腿接触力，用于判断接触状态和地形硬度

地形估计算法将 LiDAR 点云和立体视觉深度图融合为高程图（Elevation Map），机器人基于此地图进行步态规划和落脚点选择。Spot 使用的视觉里程计（Visual-Inertial Odometry，VIO）将相机和 IMU 数据紧耦合融合，在无 GPS 环境下实现厘米级定位精度。


## 当前研究方向

### 波士顿动力人工智能研究所

2022 年，马克·雷伯特宣布成立**波士顿动力人工智能研究所**（Boston Dynamics AI Institute），与波士顿动力公司相关但独立运营，专注于机器人基础研究，目标是解决当前商业机器人在感知、认知和操作能力上的根本性局限。研究所获得了现代汽车集团的大量资金支持，早期规模约数百名研究人员。

研究所的核心研究议题包括：

1. **操作智能（Manipulation Intelligence）**：使机器人能够在非结构化环境中灵活操作各类物体，解决接触丰富（Contact-Rich）的操作任务
2. **全身操作（Loco-Manipulation）**：将移动能力与操作能力深度融合，使机器人能够在移动过程中执行操作任务
3. **任务与运动规划（Task-and-Motion Planning，TAMP）**：使机器人能够理解高层次任务语义并自动规划详细的运动序列，是连接大语言模型与机器人执行层的关键
4. **自适应学习（Adaptive Learning）**：利用强化学习（Reinforcement Learning）和模仿学习（Imitation Learning）使机器人能够从少量示范中快速习得新技能

### 强化学习与仿真训练

近年来，波士顿动力在传统基于模型的控制方法之外，开始深度整合**深度强化学习**（Deep Reinforcement Learning，DRL）技术。通过大规模仿真环境（Sim-to-Real Transfer）训练神经网络策略，然后将策略迁移到真实机器人，可以大幅扩展机器人的技能库，并提升对未见地形的泛化能力。

Atlas 新版本的部分动作（如抓取任务、全身协调运动）已引入基于学习的策略，与传统优化控制器并行运行或层叠使用，代表了混合控制架构（Hybrid Control）的发展趋势。


## 商业成绩与市场地位

### 营收与规模

波士顿动力作为私有企业（现代集团子公司）不公开详细财务数据，但根据公开披露和行业分析，大致情况如下：

- **Spot 销售量**：截至 2023 年底累计销售数千台，单价约 74,500 美元，仅 Spot 产品线的总销售额已超过数亿美元
- **Stretch 商业合同**：已与 DHL 等大型物流企业签署规模化部署协议，合同总额估计达数千万至数亿美元量级
- **研发投入**：现代集团持续向波士顿动力和 AI 研究所注入大量资金，仅 AI 研究所在初期即获得承诺超过 4 亿美元的资助

### 公司定位演变

波士顿动力已从一家以 DARPA 资助为主的纯研发机构，成功转型为兼具研发能力和商业化产品的机器人企业。Spot 是这一转型最重要的标志——它不仅是技术展示平台，更是能够在真实工业环境中持续创造价值的商业产品。

然而，波士顿动力的商业规模相较于其技术声誉和研发投入仍有一定差距，如何将 Atlas 的高动态能力转化为可复制的商业价值，仍是公司面临的核心战略挑战。


## 与竞品对比

以下对比了波士顿动力与主要竞争对手在关键维度上的差异：

| 维度 | 波士顿动力（Spot + Atlas + Stretch） | ANYbotics（ANYmal） | 宇树科技 Unitree（Go2 + H1） | Agility Robotics（Digit） |
|------|--------------------------------------|---------------------|-------------------------------|---------------------------|
| **产品成熟度** | 高（Spot 已量产数千台） | 中高（ANYmal C/D 已商业化） | 中（快速迭代中） | 中（Digit 开始量产） |
| **主要产品** | 四足（Spot）、人形（Atlas）、仓储（Stretch） | 四足（ANYmal） | 四足（Go1/Go2）、人形（H1/G1） | 双足人形（Digit） |
| **运动性能** | 顶级（行业标杆） | 高 | 中高（性价比突出） | 中高（针对物流优化） |
| **定价** | 高（Spot 约 7.4 万美元） | 高（ANYmal 约 15 万美元以上） | 低至中（Go2 约 1.6 万美元） | 中（Digit 具体定价未公开） |
| **市场定位** | 工业巡检、物流、研究、国防 | 工业巡检（油气、采矿）、研究 | 研究、消费级、工业低成本 | 仓储物流（亚马逊合作） |
| **技术路线** | 优化控制 + 少量强化学习 | 优化控制 + 强化学习 | 强化学习为主 | 强化学习为主 |
| **商业客户** | Chevron、DHL、Exelon 等 | BP、Petronas 等 | 大量学术和研究机构 | Amazon（战略合作） |
| **开放生态** | Spot SDK（较完善） | ANYmal SDK | 提供 SDK | 提供 SDK |
| **母公司背景** | 现代汽车集团 | 独立（苏黎世联邦理工背景） | 独立（中国初创公司） | 独立（获亚马逊等战略投资） |

**综合评价**：波士顿动力在运动性能和品牌声誉上处于绝对领先地位，但定价较高，主要服务于大型企业客户。ANYbotics 在工业巡检领域深耕，客户结构稳定。Unitree 以极具竞争力的价格快速扩大市场覆盖，尤其在学术研究领域影响力显著提升。Agility Robotics 凭借亚马逊的战略背书，在仓储物流场景具有独特竞争优势。


## 本章内容导览

| 页面 | 主要内容 |
|------|---------|
| [Boston Dynamics](boston-dynamics.md) | 公司历史、关键技术方法、商业成绩与竞品对比 |
| [机器人平台谱系](boston-dynamics-platforms.md) | BigDog、Atlas、SpotMini、Handle、Stretch 与 Spot 各代硬件演进 |
| [Spot 商业应用与驱动技术路线](boston-dynamics-spot-applications.md) | 工业巡检与建筑测绘落地、液压与电驱动路线对比 |
| [Atlas](atlas.md) | Atlas 平台详解 |
| [Spot](spot.md) | Spot 四足机器人 |
| [机器人企业](companies.md) | 全球机器人公司概览 |


## 参考资料

1. [波士顿动力](https://zh.wikipedia.org/wiki/%E6%B3%A2%E5%A3%AB%E9%A0%93%E5%8B%95%E5%8A%9B)词条，维基百科
2. [Boston Dynamics 官方网站](https://bostondynamics.com/)
3. [Spot 产品页面](https://bostondynamics.com/products/spot/)
4. [Atlas 产品页面](https://bostondynamics.com/atlas/)
5. [Stretch 产品页面](https://bostondynamics.com/products/stretch/)
6. [Handle 产品页面](https://bostondynamics.com/products/handle/)
7. Raibert, M. H. (1986). *Legged Robots That Balance*. MIT Press.
8. Raibert, M., et al. (2008). BigDog, the rough-terrain quadruped robot. *IFAC Proceedings Volumes*, 41(2), 10822–10825.
9. DARPA Robotics Challenge (DRC) 官方网站，archive.darpa.mil/roboticschallenge/
10. Kuindersma, S., et al. (2016). Optimization-based locomotion planning, estimation, and control design for the Atlas humanoid robot. *Autonomous Robots*, 40(3), 429–455.
11. Hutter, M., et al. (2016). ANYmal – a highly mobile and dynamic quadrupedal robot. *IEEE/RSJ IROS 2016*.
12. [Boston Dynamics AI Institute 官方网站](https://theaiinstitute.com/)
13. [现代汽车集团收购波士顿动力新闻稿](https://www.hyundai.com/worldwide/en/media-center/pressrelease/boston-dynamics-acquisition)
