# 机器人图鉴

!!! note "引言"
    本页面汇总了当前具有代表性的机器人产品，按形态和应用场景分类，涵盖人形机器人、四足机器人、轮式移动机器人、工业机械臂、协作机器人、医疗机器人、无人飞行机器人、水下机器人、太空机器人及仓储物流机器人等类别。各类机器人在驱动方式（Actuation）、感知（Perception）、自主性（Autonomy）和人机交互（Human-Robot Interaction，HRI）等维度上存在显著差异，共同构成了现代机器人技术谱系的全貌。本页内容持续更新，欢迎贡献补充。点击机器人名称可跳转至详细介绍页面（如有链接）。


## 分类总览

下表给出各类机器人在几个关键维度上的定位差异，详细的产品清单见各分册页面。

| 类别 | 典型形态 | 主要驱动方式 | 自主性水平 | 主要应用场景 | 商业成熟度 |
|------|----------|--------------|------------|--------------|------------|
| 人形机器人 | 双足、双臂 | 电动准直驱、谐波减速 | 半自主为主 | 通用操作、产线试点、科研 | 早期，试点阶段 |
| 四足机器人 | 四足 | 电动准直驱 | 半自主到全自主 | 工业巡检、测绘、科研 | 已商业化 |
| 轮式移动机器人 | 差速/麦克纳姆轮 | 电动轮毂或减速电机 | 全自主（结构化环境） | 仓储搬运、配送、导览 | 成熟 |
| 工业机械臂 | 串联 6 轴为主 | 伺服电机加减速器 | 示教再现，低自主 | 焊接、喷涂、搬运、装配 | 高度成熟 |
| 协作机器人 | 串联 6/7 轴 | 力矩传感关节 | 示教与力控 | 柔性装配、检测、上下料 | 成熟 |
| 医疗机器人 | 主从操作、导航定位 | 精密丝传动、直线电机 | 人在回路 | 微创外科、骨科、康复 | 成熟（监管严格） |
| 无人飞行机器人 | 多旋翼、固定翼 | 无刷电机螺旋桨 | 半自主到全自主 | 航拍、巡检、测绘、物流 | 成熟 |
| 水下机器人 | ROV / AUV | 推进器、浮力调节 | 遥控或预编程 | 海洋调查、管线检测、打捞 | 成熟（专业市场） |
| 太空机器人 | 巡视器、空间机械臂 | 高可靠减速电机 | 高延迟半自主 | 行星探测、在轨服务 | 定制化 |
| 仓储物流机器人 | AGV / AMR / 拣选臂 | 电动轮毂、伺服臂 | 全自主（集群调度） | 货架搬运、分拣、卸货 | 成熟 |
| 服务机器人 | 轮式底盘加交互模块 | 电动减速电机 | 半自主 | 清洁、餐饮、零售导购 | 成熟 |
| 特种与搜救机器人 | 履带、蛇形、混合 | 电动或液压 | 遥操作为主 | 消防、排爆、废墟搜救 | 专业市场 |


## 机器人关键技术参数说明

了解机器人性能规格时，以下关键术语和参数有助于横向比较不同产品：

### 机械参数

- **自由度（Degrees of Freedom，DoF）**：机器人可独立运动的关节数量。6 自由度是工业机械臂的最低配置，可实现末端执行器在三维空间的任意位置和姿态；人形机器人通常需要 20 个以上自由度才能完成灵巧操作。
- **额定负载（Rated Payload）**：在标准速度和臂展条件下，机器人末端可承受的最大有效载荷，通常不包括末端执行器自身重量。
- **最大臂展（Maximum Reach）**：末端执行器可到达的最远距离，决定了机器人的作业空间（Workspace）大小。
- **重复定位精度（Repeatability，RP）**：机器人多次（通常 ≥30 次）返回同一示教点时，实际位置的最大偏差范围，是衡量机器人精度的核心指标。高端工业机械臂可达 ±0.02 mm，而协作机器人一般在 ±0.03–0.1 mm 范围内。

### 驱动与传感

- **谐波减速器（Harmonic Drive）**：利用柔性齿轮的弹性形变实现大减速比（通常 50:1–320:1），具有零背隙（Zero Backlash）、高扭矩密度等优点，广泛用于工业机械臂和协作机器人关节。
- **力矩传感器（Torque Sensor）**：安装于关节或腕部，用于测量关节输出力矩，是阻抗控制（Impedance Control）和力控（Force Control）的基础。协作机器人的碰撞检测依赖关节力矩传感器实现。
- **编码器（Encoder）**：测量关节旋转角度的传感器，分为增量式（Incremental）和绝对式（Absolute）两类。绝对式编码器在断电后仍能保持位置信息，是关节位置控制的核心器件。

### 移动平台参数

- **最大速度（Maximum Speed）**：机器人在平地直线行进时的最大速度，受电机功率、控制策略和安全限制约束。
- **有效载荷（Payload Capacity）**：移动机器人可携带的最大有效载荷，影响其搭载传感器和执行器的能力。
- **续航时间（Battery Life）**：满载工作条件下，单次充电可持续工作的时间。四足机器人通常 1–2 小时，部分工业 AGV 可实现换电或无线充电。
- **防护等级（IP Rating）**：依据 IEC 60529 标准，反映设备防尘（第一位数字，0–6）和防水（第二位数字，0–9）能力。户外机器人通常需 IP54 以上，水下机器人需 IP68 乃至特殊压力防护。

### 自主等级（Levels of Autonomy）

机器人的自主程度通常分为以下几个层级，参考美国国防部（DoD）和 SAE 自动驾驶分级框架改编：

1. **遥控（Teleoperation）**：人类实时控制机器人每一个动作，机器人不具备自主决策能力（如早期 EOD 机器人）。
2. **辅助控制（Assisted Control）**：机器人可执行简单的底层稳定和避障，人类负责高层路径和任务规划（如大多数 ROV）。
3. **有监督自主（Supervised Autonomy）**：机器人能自主执行预设任务，人类监督并可随时接管（如 AMR 自主导航）。
4. **高度自主（High Autonomy）**：机器人可独立完成复杂多步骤任务，仅在遇到超出能力边界时请求人类协助（如火星探测车）。
5. **全自主（Full Autonomy）**：机器人在无人干预的情况下完整执行任务，目前仅在高度受控环境中实现。


## 本章内容导览

| 页面 | 主要内容 |
|------|---------|
| [机器人图鉴](robots.md) | 分类总览、关键技术参数说明 |
| [人形与足式机器人](robots-humanoid-legged.md) | 人形机器人、四足机器人产品清单 |
| [移动与工业机器人](robots-mobile-industrial.md) | 轮式移动、工业机械臂、协作机器人、仓储物流 |
| [专用领域机器人](robots-special-domains.md) | 医疗、无人机、水下、太空、服务、特种与搜救 |
| [机器人企业](companies.md) | 全球机器人公司概览 |
| [机器人实验室](labs.md) | 主要研究机构 |


## 参考资料

1. [IEEE Spectrum: Robot Database](https://robots.ieee.org/)，IEEE
2. [International Federation of Robotics](https://ifr.org/)，IFR，《World Robotics Report》年度报告
3. [Boston Dynamics 官方网站](https://www.bostondynamics.com/)，Boston Dynamics
4. [Unitree Robotics 官方网站](https://www.unitree.com/)，宇树科技
5. [ANYbotics 官方网站](https://www.anybotics.com/)，ANYbotics
6. [Intuitive Surgical 官方网站](https://www.intuitivesurgical.com/)，Intuitive Surgical
7. [NASA Robotics](https://robotics.nasa.gov/)，NASA
8. Bruno Siciliano 等著，《Robotics: Modelling, Planning and Control》，Springer，2009
9. [DJI 官方网站](https://www.dji.com/)，大疆创新
10. [Universal Robots 官方网站](https://www.universal-robots.com/)，Universal Robots
11. [KUKA 官方网站](https://www.kuka.com/)，KUKA
12. [ABB Robotics 官方网站](https://new.abb.com/products/robotics)，ABB
13. [Clearpath Robotics 官方网站](https://clearpathrobotics.com/)，Clearpath Robotics
14. [Blue Robotics 官方网站](https://bluerobotics.com/)，Blue Robotics
15. [Franka Robotics 官方网站](https://franka.de/)，Franka Robotics
16. [CMR Surgical 官方网站](https://cmrsurgical.com/)，CMR Surgical
17. 中国机器人产业联盟（CRIA），《中国机器人产业发展报告》，2024
18. Niku, S. B.，《Introduction to Robotics: Analysis, Control, Applications》，Wiley，2020
19. [MathWorks Robotics Toolbox 文档](https://www.mathworks.com/products/robotics.html)，MathWorks
20. [ROS.org 官方文档](https://www.ros.org/)，Open Robotics
21. [Clearpath Robotics 机器人研究指南](https://clearpathrobotics.com/robots/)，Clearpath Robotics
22. Spong, M. W. 等著，《Robot Modeling and Control》，Wiley，2005
23. [DARPA Robotics Challenge 官方总结报告](https://www.darpa.mil/program/darpa-robotics-challenge)，DARPA，2015

