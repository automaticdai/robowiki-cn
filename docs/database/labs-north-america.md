# 北美机器人实验室

!!! note "引言"
    北美是现代机器人研究的发源地与当前的重心所在。卡内基梅隆大学机器人研究所（CMU RI）是全球第一个授予机器人学博士学位的机构，MIT CSAIL 与 Stanford 在操作、学习与人机交互方向长期领先，加州大学伯克利分校的 BAIR 则在机器人强化学习与模仿学习上产出了大量奠基性工作。这一区域的另一特点是学术与产业的高度流动——Boston Dynamics、Skydio、Covariant 等公司均直接源于实验室成果转化。本页收录北美主要机器人实验室的研究方向、代表成果与衍生企业。


## 北美 (North America)

### 美国 (United States)

#### MIT CSAIL（麻省理工学院计算机科学与人工智能实验室）

| 属性 | 详情 |
|------|------|
| 所在机构 | 麻省理工学院（Massachusetts Institute of Technology） |
| 地点 | 美国马萨诸塞州剑桥市 |
| 成立时间 | 2003 年（前身 AI Lab 成立于 1959 年） |
| 实验室网站 | [csail.mit.edu](https://csail.mit.edu) |

**主要研究方向**

- 欠驱动机器人（Underactuated Robotics）与运动规划（Motion Planning）
- 可微仿真（Differentiable Simulation）：Drake 平台
- 软体机器人（Soft Robotics）与可穿戴机器人
- 操作（Manipulation）与抓取学习
- 大语言模型（LLM）与具身智能（Embodied AI）

**主要研究人员（PI）**

| 姓名 | 研究方向 |
|------|---------|
| Russ Tedrake | 运动规划、欠驱动控制、可微仿真 |
| Daniela Rus | 软体机器人、自重构机器人、移动机器人 |
| Leslie Kaelbling | 操作学习、任务与运动规划（TAMP） |
| Pulkit Agrawal | 强化学习、灵巧操作 |

**代表性成果**

- **Drake**：用于机器人运动规划与控制的可微仿真与优化框架，已被谷歌、丰田研究院等广泛采用
- **MIT Cheetah**（猎豹四足机器人）：在跑步速度与能效方面长期处于世界领先水平
- 基于强化学习（Reinforcement Learning, RL）的灵巧操作研究，推动了接触丰富任务的求解

**知名校友与孵化公司**

| 公司/人物 | 背景 |
|-----------|------|
| Boston Dynamics | Marc Raibert 于 MIT Leg Lab 创立，后独立发展为全球最知名动态运动机器人公司 |
| Rethink Robotics | 联合创始人 Rodney Brooks 来自 MIT AI Lab |
| 丰田研究院（TRI）机器人团队 | 多名核心研究员来自 CSAIL |

---

#### Stanford AI Lab / Robotics（斯坦福大学人工智能与机器人实验室）

| 属性 | 详情 |
|------|------|
| 所在机构 | 斯坦福大学（Stanford University） |
| 地点 | 美国加利福尼亚州斯坦福市 |
| 实验室网站 | [cs.stanford.edu/groups/manips](https://cs.stanford.edu/groups/manips/) |

**主要研究方向**

- 人机交互（Human-Robot Interaction, HRI）与协作机器人
- 水下机器人（Underwater Robotics）
- 操作学习（Learning for Manipulation）
- 运动规划与力控制

**主要研究人员（PI）**

| 姓名 | 研究方向 |
|------|---------|
| Dorsa Sadigh | 人机交互、安全强化学习、协作规划 |
| Oussama Khatib | 机械臂力控制、水下机器人、仿人机器人 |
| Jeannette Bohg | 感知驱动操作（Perception-driven Manipulation） |
| Chelsea Finn | 元学习（Meta-learning）、少样本操作 |

**代表性成果**

- **ROBOSUITE**：操作学习基准（Benchmark）平台，被学术界广泛用于算法评估
- **Stanford DEXTROUS HAND**：早期多指灵巧手研究，奠定操作方向基础
- **OROCOS**：开源机器人控制系统（Open RObot COntrol Software），在欧洲工业界广泛应用

**知名孵化公司**

| 公司 | 背景 |
|------|------|
| Diligent Robotics | 医院服务机器人，创始人来自斯坦福 HRI 方向 |
| Nuro | 自动驾驶配送机器人，Google 前员工与斯坦福背景联合创立 |

---

#### CMU Robotics Institute（卡内基梅隆大学机器人研究所）

| 属性 | 详情 |
|------|------|
| 所在机构 | 卡内基梅隆大学（Carnegie Mellon University） |
| 地点 | 美国宾夕法尼亚州匹兹堡市 |
| 成立时间 | 1979 年，全球最早的独立机器人学位授予机构 |
| 实验室网站 | [ri.cmu.edu](https://www.ri.cmu.edu/) |

**主要研究方向**

- 自主导航（Autonomous Navigation）与自动驾驶
- 操作学习（Learning for Manipulation）
- 人形机器人（Humanoid Robotics）
- 医疗机器人（Medical Robotics）
- 野外机器人（Field Robotics）

**主要研究人员（PI）**

| 姓名 | 研究方向 |
|------|---------|
| Chris Atkeson | 学习控制、仿人运动、物理仿真 |
| Deepak Pathak | 主动学习（Active Learning）、世界模型、视觉机器人 |
| Abhinav Gupta | 视觉操作、自监督学习 |
| Henny Admoni | 人机协作、辅助机器人 |
| Sebastian Scherer | 自主飞行、野外导航 |

**代表性成果**

- **Navlab**（自主导航实验室）：1980–2000 年代自动驾驶先驱，完成横穿美国自动驾驶壮举
- **LocoBot**：低成本移动操作平台，已成为学术界广泛使用的研究基准
- 对 ROS（Robot Operating System，机器人操作系统）有大量贡献

**知名孵化公司**

| 公司 | 背景 |
|------|------|
| Uber ATG | 自动驾驶部门，大量招募 CMU RI 人才 |
| Argo AI | 自动驾驶，创始人来自 CMU RI |
| Aurora Innovation | 联合创始人 Sterling Anderson 来自 CMU |

---

#### UC Berkeley BAIR（伯克利人工智能研究实验室）

| 属性 | 详情 |
|------|------|
| 所在机构 | 加州大学伯克利分校（University of California, Berkeley） |
| 地点 | 美国加利福尼亚州伯克利市 |
| 实验室网站 | [bair.berkeley.edu](https://bair.berkeley.edu) |

**主要研究方向**

- 深度强化学习（Deep Reinforcement Learning, DRL）
- 模仿学习（Imitation Learning）与行为克隆（Behavior Cloning）
- 机器人操作与灵巧抓取
- 大模型驱动的具身智能（Foundation Model for Robotics）

**主要研究人员（PI）**

| 姓名 | 研究方向 |
|------|---------|
| Pieter Abbeel | 模仿学习、强化学习、机器人操作 |
| Sergey Levine | 离线强化学习、视觉运动控制 |
| Ken Goldberg | 机器人抓取、工业自动化 |
| Jitendra Malik | 计算机视觉、3D 感知 |
| Alexei Efros | 视觉学习、图像合成 |

**代表性成果**

- **SAC（Soft Actor-Critic）**：在连续控制任务中应用广泛的 RL 算法
- **BridgeData**：大规模机器人操作数据集，推动数据驱动机器人学习
- **RoboAgent**：多任务操作学习框架
- **RLHF（Reinforcement Learning from Human Feedback，人类反馈强化学习）**：Pieter Abbeel 团队早期工作，后被 OpenAI 广泛应用于 LLM 对齐

**知名孵化公司**

| 公司 | 背景 |
|------|------|
| Covariant | 机器人抓取 AI，Pieter Abbeel 联合创立 |
| Embodied Intelligence | 机器人学习，Pieter Abbeel 参与创立 |
| Physical Intelligence (π) | Sergey Levine 参与创立，专注通用机器人策略 |

---

#### UPenn GRASP Lab（通用机器人、自动化、感知与传感实验室）

| 属性 | 详情 |
|------|------|
| 所在机构 | 宾夕法尼亚大学（University of Pennsylvania） |
| 地点 | 美国宾夕法尼亚州费城 |
| 实验室网站 | [grasp.upenn.edu](https://www.grasp.upenn.edu/) |

**主要研究方向**

- 微型飞行器（Micro Aerial Vehicle, MAV）与空中机器人
- 蜂群机器人（Swarm Robotics）与多机器人系统
- 感知驱动控制（Perception-driven Control）

**主要研究人员（PI）**

| 姓名 | 研究方向 |
|------|---------|
| Vijay Kumar | 多机器人协作、空中机器人动力学 |
| Kostas Daniilidis | 计算机视觉、3D 感知 |
| Pratik Chaudhari | 优化理论、深度学习 |

**代表性成果**

- 基于四旋翼的精确轨迹规划研究，Vijay Kumar 的 TED 演讲累计观看逾数百万次
- 多机器人编队与协同搬运（Cooperative Manipulation）
- 与 DARPA 合作的室内自主导航项目

---

#### 其他北美知名实验室

| 实验室 | 所在机构 | 研究方向 | 网站 |
|--------|---------|---------|------|
| LCSR（实验室计算感觉-运动研究） | 约翰·霍普金斯大学 | 手术机器人、微创手术 | [lcsr.jhu.edu](https://lcsr.jhu.edu/) |
| University of Michigan Robotics | 密歇根大学 | 足式机器人、假肢、双足行走 | [robotics.umich.edu](https://robotics.umich.edu/) |
| Georgia Tech IRIM | 佐治亚理工学院 | 医疗机器人、HRI、制造机器人 | [research.gatech.edu/robotics](https://research.gatech.edu/robotics) |
| Harvard SSR | 哈佛大学自组织系统小组 | 集群机器人、软体机器人 | [eecs.harvard.edu/ssr](https://eecs.harvard.edu/ssr) |
| MIT Personal Robots Group | MIT 媒体实验室 | 社交机器人、人机交互 | [robotic.media.mit.edu](https://robotic.media.mit.edu) |
| OpenAI Robotics | OpenAI（非传统学术实验室） | 灵巧操作、强化学习、大模型机器人 | [openai.com](https://openai.com/) |

## 参考资料

1. [CSRankings: Robotics](https://csrankings.org/#/index?robotics)
2. [机器人实验室总览](labs.md)
3. [欧洲机器人实验室](labs-europe.md)
4. [亚太机器人实验室](labs-asia.md)
