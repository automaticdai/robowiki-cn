# 传感器融合

!!! note "引言"
    传感器融合（Sensor Fusion）是将来自多个传感器的数据加以综合处理，以获得比任意单一传感器更准确、更可靠的状态估计的技术。在无人机自主飞行、自动驾驶汽车、人形机器人等系统中，传感器融合是实现安全、鲁棒感知的基础。单一传感器受限于自身物理原理，存在固有缺陷，而融合互补的传感器可大幅提升系统整体性能。


## 为什么需要传感器融合

### 单一传感器的局限性

现实机器人系统依赖多种传感器，但每种传感器都有固有的缺陷：

**单目/双目相机**

- 依赖环境光照，夜间或强逆光下性能急剧下降
- 遮挡问题：前景物体遮挡背景目标，造成感知盲区
- 无法直接测量速度，仅能从图像序列中估算运动
- 运动模糊（Motion Blur）在高速场景下严重影响特征提取

**全球定位系统（Global Positioning System, GPS）**

- 城市峡谷（Urban Canyon）中信号被建筑物遮挡，产生多径效应（Multipath Effect）
- 室内环境无信号，无法使用
- 民用 GPS 精度约 2–5 m，更新频率仅 1–10 Hz，不满足高动态控制需求
- 信号延迟可达数百毫秒，无法用于实时反馈控制

**惯性测量单元（Inertial Measurement Unit, IMU）**

- 加速度计和陀螺仪存在零偏（Bias）和随机游走（Random Walk）噪声
- 通过积分估计速度和位置时，误差随时间平方增长（二次积分漂移）
- 温度变化导致零偏漂移（Thermal Drift），需要温度补偿
- 高采样率（100–1000 Hz）带来大量数据，但长期精度无保证

**激光雷达（Light Detection and Ranging, LiDAR）**

- 雨、雾、雪等恶劣天气下激光束被散射，有效测距距离大幅下降
- 无法感知颜色和纹理信息
- 稀疏点云对细小物体（如行人腿部）的检测率低
- 高端多线激光雷达成本高昂（数千至数万美元）

### 融合后的互补优势

传感器融合的核心思想是利用不同传感器在时间域、频率域、精度域上的互补性：

| 传感器组合 | 互补关系 | 典型应用 |
|-----------|---------|---------|
| IMU + GPS | IMU 高频（>100 Hz）填补 GPS 低频（1–10 Hz）间隙；GPS 校正 IMU 长期漂移 | 无人机惯性导航 |
| 相机 + IMU | 相机提供绝对尺度和纹理；IMU 在快速运动中辅助位姿预测 | 视觉惯性里程计（VIO） |
| LiDAR + 相机 | LiDAR 提供精确深度；相机提供颜色和语义 | 自动驾驶 3D 目标检测 |
| IMU + 编码器 | IMU 感知姿态变化；编码器提供轮式里程计 | 地面移动机器人 |
| LiDAR + IMU | LiDAR 提供高精度地图匹配；IMU 提供初始位姿预测 | LiDAR SLAM |

**典型应用场景**

- **无人机（Unmanned Aerial Vehicle, UAV）**：IMU + 气压计 + GPS + 光流传感器融合，实现室内外无缝切换的自主悬停
- **自动驾驶汽车**：LiDAR + 相机 + 毫米波雷达 + GPS/RTK 融合，满足 L4 级别自动驾驶的感知需求
- **人形机器人（Humanoid Robot）**：关节编码器 + IMU + 足底力传感器融合，实现稳定的动态平衡控制


## 融合层级架构

传感器融合按处理层次分为三个级别，各有适用场景和权衡取舍。三者的区别在于融合发生在流水线的哪个位置：

![数据级、特征级、决策级三种融合架构对比](assets/fusion-levels.svg)

### 数据级融合（Low-level Fusion）

数据级融合（也称原始数据级融合）在传感器原始数据层面直接进行融合，不经过特征提取或决策步骤。

**工作流程**：原始传感器数据 → 同步与配准 → 融合处理 → 后续处理

**典型示例：双目视差计算**

双目相机左右图像在像素级进行立体匹配（Stereo Matching），计算视差图（Disparity Map）后恢复深度：

$$
Z = \frac{f \cdot B}{d}
$$

其中 \(f\) 为焦距（像素单位），\(B\) 为基线距离（m），\(d\) 为视差（像素）。这是典型的数据级融合：两路图像数据在像素级别完成融合。

**另一典型示例：LiDAR 与相机的点云着色**

将相机采集的 RGB 图像投影到 LiDAR 点云上，为每个三维点附加颜色属性，属于数据级融合。

**特点**

- 保留原始数据的最大信息量，融合精度高
- 对传感器时间同步和空间标定要求极高
- 计算量大，通常需要专用硬件加速

### 特征级融合（Feature-level Fusion）

特征级融合先从各传感器数据中独立提取特征（如边缘、角点、语义标签），再在特征空间中进行融合。

**工作流程**：原始传感器数据 → 各自特征提取 → 特征对齐与融合 → 联合推理

**典型示例：LiDAR + 相机 3D 目标检测**

- 相机提取图像特征（ResNet 骨干网络输出的特征图）
- LiDAR 提取点云特征（PointNet++ 或体素化后的稀疏卷积特征）
- 两路特征在鸟瞰图（Bird's Eye View, BEV）空间对齐后融合，输入 3D 检测头

代表算法：BEVFusion（MIT）、PointPainting、MVP（Multi-view Pseudo-labeling）

**特点**

- 比数据级融合计算量小，因特征维度远低于原始数据
- 具备一定的传感器缺失鲁棒性（缺失一路传感器时可部分降级运行）
- 特征对齐需要精确的外参标定

### 决策级融合（Decision-level Fusion）

决策级融合让各传感器独立完成推理（如检测、分类），再对各自的输出结果进行投票或加权融合。

**工作流程**：原始传感器数据 → 各自独立推理 → 决策融合（投票/加权/D-S 证据理论）

**典型示例：多雷达目标检测投票**

三个方向的毫米波雷达各自输出目标置信度，采用多数投票（Majority Voting）或加权平均得到最终检测结果。

**特点**

- 各模块高度解耦，易于独立开发和替换
- 对单一传感器故障具有最强鲁棒性
- 信息损失最大：原始数据经过推理压缩后，大量细节信息已丢失

### 三级融合架构对比

| 指标 | 数据级融合 | 特征级融合 | 决策级融合 |
|------|-----------|-----------|-----------|
| 信息保留量 | 高 | 中 | 低 |
| 融合精度 | 最高 | 较高 | 较低 |
| 计算开销 | 大 | 中 | 小 |
| 同步要求 | 严格（μs 级） | 中等（ms 级） | 宽松（帧级） |
| 传感器异构性支持 | 弱（需相同数据格式） | 中 | 强 |
| 典型场景 | 双目深度、点云着色 | BEVFusion、VIO | 冗余系统表决 |


## 本章内容导览

传感器融合章节按「为什么融合 → 数学工具 → 系统架构 → 工程落地 → 速查」的顺序组织：

| 页面 | 主要内容 |
|------|---------|
| [传感器融合](sensor-fusion.md) | 融合动机、互补性分析、数据级/特征级/决策级融合层级 |
| [贝叶斯滤波器族](sensor-fusion-filters.md) | 概率估计框架、KF、EKF、UKF、粒子滤波与选型 |
| [融合架构设计](sensor-fusion-architecture.md) | 松耦合与紧耦合、传感器标定、时间同步、按机器人类型的架构 |
| [融合工程实践](sensor-fusion-engineering.md) | EKF 实现、互补滤波、robot_localization 配置、异常值剔除与调试 |
| [参考资料汇总](sensor-fusion-full-reference.md) | 教材、论文、开源库与数据集 |
| [SLAM](slam.md) | 同步定位与建图 |
| [IMU 标定](imu-calibration.md) | 惯性器件误差建模与标定 |


## 参考资料

1. Thrun, S., Burgard, W., & Fox, D. (2005). *Probabilistic Robotics*. MIT Press.

2. Kalman, R. E. (1960). A New Approach to Linear Filtering and Prediction Problems. *Journal of Basic Engineering*, 82(1), 35–45.

3. Julier, S. J., & Uhlmann, J. K. (1997). New Extension of the Kalman Filter to Nonlinear Systems. *Proceedings of SPIE — Signal Processing, Sensor Fusion, and Target Recognition VI*, 3068, 182–193.

4. Doucet, A., de Freitas, N., & Gordon, N. (Eds.). (2001). *Sequential Monte Carlo Methods in Practice*. Springer.

5. Mourikis, A. I., & Roumeliotis, S. I. (2007). A Multi-State Constraint Kalman Filter for Vision-aided Inertial Navigation. *Proceedings of IEEE ICRA 2007*, 3565–3572.

6. Qin, T., Li, P., & Shen, S. (2018). VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator. *IEEE Transactions on Robotics*, 34(4), 1004–1020.

7. Liu, Z., Tang, H., Amini, A., et al. (2022). BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation. *Proceedings of IEEE ICRA 2023*.

8. Furgale, P., Rehder, J., & Siegwart, R. (2013). Unified Temporal and Spatial Calibration for Multi-Sensor Systems. *Proceedings of IEEE/RSJ IROS 2013*, 1280–1286.

9. Moore, T., & Stouch, D. (2014). A Generalized Extended Kalman Filter Implementation for the Robot Operating System. *Proceedings of the 13th International Conference on Intelligent Autonomous Systems (IAS-13)*. Springer.

10. Siegwart, R., Nourbakhsh, I., & Scaramuzza, D. (2011). *Introduction to Autonomous Mobile Robots* (2nd ed.). MIT Press.

