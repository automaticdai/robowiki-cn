# 欧洲机器人实验室

!!! note "引言"
    欧洲机器人研究的显著特征是工程化与标准化传统深厚，且长期由欧盟框架计划（Horizon Europe 及其前身）提供跨国协作资助。苏黎世联邦理工学院（ETH Zürich）的自主系统实验室与机器人系统实验室在足式机器人和无人机领域处于世界前列，并孵化出 ANYbotics 与 Flyability；德国宇航中心（DLR）机器人与机电一体化研究所在轻型机械臂与空间机器人上积累深厚，其技术直接衍生出 KUKA LBR iiwa 系列。本页收录欧洲主要机器人实验室的研究方向与代表成果。


## 欧洲 (Europe)

### ETH Zürich ASL（自主系统实验室）

| 属性 | 详情 |
|------|------|
| 所在机构 | 苏黎世联邦理工学院（ETH Zürich） |
| 地点 | 瑞士苏黎世 |
| 实验室网站 | [asl.ethz.ch](https://asl.ethz.ch) |

**主要研究方向**

- 移动机器人（Mobile Robotics）与野外自主系统
- 视觉-惯性导航（Visual-Inertial Odometry, VIO）
- 腿式机器人（Legged Robotics）与复杂地形行走
- 自主无人机导航

**主要研究人员（PI）**

| 姓名 | 研究方向 |
|------|---------|
| Roland Siegwart | 移动机器人、感知与定位 |
| Marco Hutter | 腿式机器人、接触运动规划 |

**代表性成果**

- **ANYmal**：四足机器人，在工业巡检与搜救领域已商业化部署
- **libpointmatcher**：开源点云（Point Cloud）配准库，被工业和学术界广泛使用
- **VILENS**（Visual-Inertial-Legged Navigation System）：腿式机器人专用 VIO 系统

**知名孵化公司**

| 公司 | 背景 |
|------|------|
| ANYbotics | 四足机器人商业化，Marco Hutter 参与创立 |
| Sevensense Robotics | 工业感知导航，来自 ASL |

---

### ETH Zürich RPG（机器人与感知小组）

| 属性 | 详情 |
|------|------|
| 所在机构 | 苏黎世联邦理工学院 & 苏黎世大学 |
| 地点 | 瑞士苏黎世 |
| 实验室网站 | [rpg.ifi.uzh.ch](http://rpg.ifi.uzh.ch/) |

**主要研究方向**

- 基于视觉的自主无人机（UAV）导航
- 事件相机（Event Camera）感知
- 视觉惯性里程计（Visual-Inertial Odometry）

**代表性成果**

- 利用神经网络直接从事件相机数据进行高速无人机飞行
- 在 UAV 竞速赛中击败人类飞手（2023 年）
- **RPGQ** 无人机仿真平台

---

### Imperial College London Dyson Robotics Lab（帝国理工 Dyson 机器人实验室）

| 属性 | 详情 |
|------|------|
| 所在机构 | 帝国理工学院（Imperial College London） |
| 地点 | 英国伦敦 |
| 实验室网站 | [imperial.ac.uk/dyson-robotics-lab](https://www.imperial.ac.uk/dyson-robotics-lab/) |

**主要研究方向**

- 实时三维重建（Real-time 3D Reconstruction）
- 语义 SLAM（Semantic Simultaneous Localisation and Mapping）
- 神经辐射场（Neural Radiance Field, NeRF）在机器人中的应用

**主要研究人员（PI）**

| 姓名 | 研究方向 |
|------|---------|
| Andrew Davison | 单目 SLAM、实时稠密重建，SLAM 领域奠基人之一 |
| Stefan Leutenegger | 视觉-惯性 SLAM、神经隐式表示 |

**代表性成果**

- **MonoSLAM**（2003）：首个实时单目 SLAM（同步定位与建图）系统
- **ElasticFusion**：基于面元（Surfel）的实时稠密 RGB-D SLAM
- **CodeSLAM**：将变分自编码器（VAE）引入 SLAM 场景表示

---

### Oxford Robotics Institute（牛津机器人研究所）

| 属性 | 详情 |
|------|------|
| 所在机构 | 牛津大学（University of Oxford） |
| 地点 | 英国牛津 |
| 实验室网站 | [ori.ox.ac.uk](https://ori.ox.ac.uk/) |

**主要研究方向**

- 长期自主性（Long-term Autonomy）
- 野外移动机器人
- 自动驾驶感知与规划

**主要研究人员（PI）**

| 姓名 | 研究方向 |
|------|---------|
| Ingmar Posner | 概率感知、深度学习在驾驶中的应用 |
| Nick Hawes | 任务规划、长期自主机器人 |
| Maurice Fallon | 状态估计、腿式机器人感知 |

**代表性成果**

- **Oxford RobotCar Dataset**：包含超过 1000km 自动驾驶数据的长期数据集，覆盖全天候、全季节变化
- **Navtech 雷达 SLAM**：基于旋转毫米波雷达的全天候定位与建图

---

### DLR Institute of Robotics（德国航空航天中心机器人研究所）

| 属性 | 详情 |
|------|------|
| 所在机构 | 德国航空航天中心（Deutsches Zentrum für Luft- und Raumfahrt, DLR） |
| 地点 | 德国慕尼黑韦斯林 |
| 实验室网站 | [dlr.de/rm](https://www.dlr.de/rm/) |

**主要研究方向**

- 空间机器人（Space Robotics）：国际空间站（ISS）任务支持
- 手术机器人（Surgical Robotics）
- 轻量化机械臂（Lightweight Robot Arm）

**代表性成果**

- **DLR LWR（轻量化机械臂）**：采用关节力矩传感器实现柔顺控制，后授权给 KUKA 商业化为 LBR iiwa，成为协作机器人（Cobot）的重要里程碑
- **Robonaut 协作**：与 NASA 合作研发空间机器人末端执行器
- **DLR Hand II**：高度集成的仿人五指机器人手

---

### 其他欧洲知名实验室

| 实验室 | 所在机构 | 研究方向 | 网站 |
|--------|---------|---------|------|
| IIT iCub Lab | 意大利理工学院 | iCub 仿人机器人、认知机器人 | [iit.it](https://www.iit.it/) |
| INRIA Lagadic | 法国国家信息与自动化研究所 | 视觉伺服（Visual Servoing）、机器人控制 | [inria.fr](https://www.inria.fr/en) |
| TU Munich AIS | 慕尼黑工业大学 | 机器人感知、操作、人机协作 | [ce.cit.tum.de](https://www.ce.cit.tum.de/en/ais/home/) |
| KTH RPL | 瑞典皇家理工学院 | 感知与学习、长期自主性 | [kth.se](https://www.kth.se/is/rpl) |
| Bristol Robotics Lab | 布里斯托大学 & 西英格兰大学 | 多学科机器人、仿生机器人 | [brl.ac.uk](https://brl.ac.uk) |
| DeepMind | Alphabet（英国伦敦） | 强化学习、机器人控制、基础模型 | [deepmind.com](https://deepmind.com) |
| Edinburgh Centre of Robotics | 爱丁堡大学 & 赫瑞-瓦特大学 | 服务机器人、自主系统 | [edinburgh-robotics.org](https://edinburgh-robotics.org) |

## 参考资料

1. euRobotics AISBL. https://eu-robotics.net/
2. [机器人实验室总览](labs.md)
3. [北美机器人实验室](labs-north-america.md)
4. [亚太机器人实验室](labs-asia.md)
