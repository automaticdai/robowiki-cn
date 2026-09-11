# 机器人竞赛

!!! note "引言"
    机器人竞赛（Robotics Competitions）是推动机器人技术创新与人才培养的重要引擎。从仿真足球场上的 RoboCup 到 DARPA 荒漠越野挑战赛，从校园普及的 ROBOCON 到面向工业前沿的 Amazon Robotics Challenge，各类竞赛共同构成了机器人技术生态的重要组成部分。本页面系统收录国际与国内主要机器人竞赛，涵盖赛制规则、历史沿革与参赛建议，供研究人员、工程师及在校学生参考。

## 竞赛体系概览

机器人竞赛大致可分为四类，各自的目标、周期与参与门槛差别很大：

| 类别 | 代表赛事 | 组织形式 | 主要价值 | 参与门槛 |
|------|----------|----------|----------|----------|
| 旗舰学术联赛 | RoboCup 各联赛 | 年度周期、分联赛、规则渐进演化 | 长期推动多机器人协作、实时感知与双足运动 | 高，需成建制团队与多年积累 |
| 国家级挑战赛 | DARPA Grand/Urban/Robotics/SubT Challenge | 一次性、高奖金、面向具体任务 | 在自动驾驶、灾难救援、地下探测上引发技术拐点 | 极高，多为顶尖高校与企业联队 |
| 专项技术竞赛 | IROS Drone Racing、ARC、ARIAC、MBZIRC | 依托顶会或企业，聚焦单一技术方向 | 提供可复现的评测基准，直接对接产业需求 | 中到高，适合研究生课题组 |
| 教育普及赛事 | FIRST、WRO、VEX、ROBOCON | 分年龄段、标准化套件 | 培养工程素养与团队协作，输送后备人才 | 低，中学至本科低年级均可参与 |

选择竞赛时，建议先按技术方向（导航、操作、人形、无人机、多机对抗）定位，再按团队规模与经费匹配赛事级别。详细的方向-赛事对照见下方「参赛建议与备赛指南」。

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

## 本章内容导览

| 页面 | 主要内容 |
|------|---------|
| [机器人竞赛](competitions.md) | 竞赛体系分类、参赛建议与备赛指南、主要竞赛速查表 |
| [RoboCup 与 DARPA 挑战赛](competitions-robocup-darpa.md) | RoboCup 各联赛赛制与历史、DARPA 四大挑战赛及其技术影响 |
| [中国主要机器人竞赛](competitions-china.md) | RoboMaster、ROBOCON、中国机器人大赛、智能汽车竞赛等 |
| [国际专项与教育类竞赛](competitions-international.md) | 无人机竞速、物流与工业操作、学术基准挑战赛、FIRST 系列 |
| [机器人实验室](labs.md) | 全球主要机器人研究机构 |
| [学术会议](conferences.md) | ICRA、IROS、RSS 等会议与投稿周期 |

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

