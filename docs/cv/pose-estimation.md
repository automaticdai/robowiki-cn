# 位姿估计

!!! note "引言"
    位姿估计（Pose Estimation）是计算机视觉与机器人学的交汇领域，旨在从传感器数据中恢复物体或人体在三维空间中的位置与姿态。在机器人应用中，准确的位姿估计是实现自主抓取、人机协作和环境理解的基础。本文分别介绍物体 6DoF 位姿估计和人体姿态估计两大任务的方法、评测体系与工程实践。


## 位姿估计概述

位姿估计涵盖两类核心任务：

1. **物体 6DoF 位姿估计**（6 Degrees-of-Freedom Object Pose Estimation）：估计目标物体相对于相机坐标系的完整刚体变换，包含 3 个平移自由度和 3 个旋转自由度，广泛用于机器人抓取、工业质检和增强现实。

2. **人体姿态估计**（Human Pose Estimation）：检测并定位人体各关节点（如肩、肘、腕）在 2D 图像或 3D 空间中的坐标，广泛用于人机交互、动作识别和运动分析。

### 位姿的数学表示

一个刚体在三维空间中的位姿可用齐次变换矩阵（Homogeneous Transformation Matrix）表示。设旋转矩阵为 \(\mathbf{R}\)，平移向量为 \(\mathbf{t}\)，则变换矩阵 \(\mathbf{T}\) 为：

$$
\mathbf{T} = \begin{bmatrix} \mathbf{R} & \mathbf{t} \\ \mathbf{0}^\top & 1 \end{bmatrix} \in SE(3)
$$

其中 \(\mathbf{R} \in SO(3)\)，\(\mathbf{t} \in \mathbb{R}^3\)。特殊欧氏群 \(SE(3)\) 描述所有刚体变换，特殊正交群 \(SO(3)\) 描述所有旋转变换。

旋转的常见参数化方式如下表：

| 表示方式 | 参数数量 | 优点 | 缺点 |
|---|---|---|---|
| 旋转矩阵（Rotation Matrix） | 9（含 6 个约束） | 直观、无奇异性 | 冗余参数，不宜直接回归 |
| 四元数（Quaternion） | 4（单位约束） | 插值平滑、无万向锁 | 双覆盖问题（\(\mathbf{q}\) 与 \(-\mathbf{q}\) 等价） |
| 轴角（Axis-Angle） | 3 | 紧凑、物理意义清晰 | 在零旋转附近奇异 |
| 李代数 \(\mathfrak{so}(3)\) | 3 | 适合梯度优化 | 需指数映射还原矩阵 |

四元数表示为 \(\mathbf{q} = (q_w, q_x, q_y, q_z)\)，满足 \(\|\mathbf{q}\| = 1\)。轴角表示为 \(\mathbf{r} = \theta \hat{\mathbf{n}}\)，其中 \(\theta\) 为旋转角，\(\hat{\mathbf{n}}\) 为单位旋转轴。

针孔相机模型（Pinhole Camera Model）将三维点 \(\mathbf{X}_c\) 投影到图像平面：

$$
\lambda \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \mathbf{K} \begin{bmatrix} \mathbf{R} & \mathbf{t} \end{bmatrix} \begin{bmatrix} \mathbf{X}_w \\ 1 \end{bmatrix}
$$

其中 \(\mathbf{K}\) 为相机内参矩阵，\(\lambda\) 为深度比例因子，\(\mathbf{X}_w\) 为世界坐标系下的点坐标。


## 物体 6DoF 位姿估计

### 基于 RGB 的方法

仅使用单目 RGB 图像估计物体位姿，无需深度传感器，部署成本低，但对光照变化和遮挡较敏感。

#### PoseCNN

PoseCNN（Pose Convolutional Neural Network）由 Xiang 等人于 2018 年提出，是深度学习物体位姿估计的早期里程碑。其网络分为三个分支：

- **语义分割分支**：输出各物体的像素级掩码。
- **位移场分支**：预测每个像素到物体中心的投影偏移，通过霍夫变换（Hough Voting）定位物体中心。
- **回归分支**：直接回归物体深度和四元数 \(\hat{\mathbf{q}}\)。

最终利用迭代最近点算法（Iterative Closest Point，ICP）对初始位姿进行细化，提高估计精度。PoseCNN 在 YCB-Video 数据集上建立了重要基准。

#### DPOD

密集位姿目标检测器（Dense Pose Object Detector，DPOD）采用密集对应策略，网络为每个物体像素预测其在 CAD 模型表面的 UV 纹理坐标。建立密集 2D-3D 对应关系后，利用透视 n 点（Perspective-n-Point，PnP）算法结合 RANSAC（随机样本一致性）鲁棒求解位姿：

$$
\hat{\mathbf{R}}, \hat{\mathbf{t}} = \arg\min \sum_i \left\| \mathbf{u}_i - \pi\left(\mathbf{K}, \mathbf{R}, \mathbf{t}, \mathbf{X}_i\right) \right\|^2
$$

其中 \(\pi(\cdot)\) 为投影函数，\(\mathbf{u}_i\) 为第 \(i\) 个 2D 特征点，\(\mathbf{X}_i\) 为对应 3D 模型点。

#### GDR-Net

几何引导直接回归网络（Geometry-guided Direct Regression Network，GDR-Net）实现端到端位姿估计，无需 PnP 后处理。网络首先预测密集对应图（2D-3D 对应关系），再通过几何引导的解耦回归头直接输出旋转（使用李代数参数化 \(\mathfrak{so}(3)\)）和平移。

李代数指数映射将向量 \(\boldsymbol{\omega} \in \mathbb{R}^3\) 还原为旋转矩阵：

$$
\mathbf{R} = \exp(\hat{\boldsymbol{\omega}}) = \mathbf{I} + \frac{\sin\theta}{\theta}\hat{\boldsymbol{\omega}} + \frac{1-\cos\theta}{\theta^2}\hat{\boldsymbol{\omega}}^2
$$

其中 \(\theta = \|\boldsymbol{\omega}\|\)，\(\hat{\boldsymbol{\omega}}\) 为 \(\boldsymbol{\omega}\) 对应的反对称矩阵（skew-symmetric matrix）。GDR-Net 在 LineMOD 数据集上显著优于同期方法。


### 基于 RGB-D 的方法

结合彩色图和深度图，可直接获取物体点云，从而利用几何信息提升估计精度。

#### DenseFusion

DenseFusion 是首批将点云与 RGB 特征逐点融合的端到端网络之一。其处理流程如下：

1. 对 RGB 图像使用 PSPNet 提取像素级外观特征 \(\mathbf{f}_\text{rgb} \in \mathbb{R}^{N \times C}\)。
2. 将深度图反投影为有序点云，使用 PointNet 提取几何特征 \(\mathbf{f}_\text{geo} \in \mathbb{R}^{N \times C}\)。
3. 将两类特征在每个点处拼接，送入位姿估计网络，直接输出旋转四元数和平移向量。
4. 引入迭代细化模块（Iterative Refinement），将前一轮估计的位姿变换点云后重复上述过程，逐步收敛。

#### FFB6D

全流动双向融合网络（Full Flow Bidirectional Fusion Network，FFB6D）引入双向融合机制，允许 RGB 特征和点云特征在网络各层互相增强：

- 点云分支使用 PointNet++（Pointnet Plus Plus）处理局部几何特征，支持多尺度邻域聚合。
- RGB 分支使用标准卷积神经网络提取外观特征。
- 双向融合层（Bidirectional Fusion Layer）在每个分辨率层次交换两个分支的信息。

最终输出每个点的关键点偏移向量，通过最小二乘法（Least Squares）拟合关键点位置，再用 PnP 求解最终位姿。


### 无 CAD 模型方法

传统方法依赖精确的 CAD 模型。无 CAD 模型方法（Model-Free Pose Estimation）只需少量参考图像，适用于工业快速换线和未见新物体（Novel Object）场景。

#### FoundPose

FoundPose 利用 DINOv2（一种大规模自监督视觉基础模型）提取丰富的语义特征，构建物体的密集特征场（Feature Field）。推理时通过特征匹配在参考视图库中检索最近邻，建立 2D-3D 对应关系后用 PnP 求解位姿。其优势在于无需任何 3D 模型即可泛化至未见物体。

#### Gen6D

可泛化 6DoF 位姿估计（Generalizable 6DoF Pose Estimation，Gen6D）从少量参考图像（Reference Images）生成物体的隐式 3D 表示。给定目标图像，网络：

1. 在参考图库中检索视角相似的参考帧。
2. 通过特征几何变换预测粗略位姿。
3. 利用多视角一致性细化最终位姿。

Gen6D 在 GenMOP 等泛化位姿估计基准上取得了领先性能。


### 位姿细化

位姿细化（Pose Refinement）将粗略的初始位姿作为输入，通过迭代优化提高精度。

#### 迭代最近点算法

经典 ICP 算法通过交替执行点对匹配与变换求解来最小化点云间距离：

$$
\hat{\mathbf{T}} = \arg\min_{\mathbf{T}} \sum_i \left\| \mathbf{p}_i - \mathbf{T} \mathbf{q}_i \right\|^2
$$

其中 \(\mathbf{p}_i\) 为目标点云点，\(\mathbf{q}_i\) 为模型点云中的最近邻点。ICP 对初值敏感，需要较好的初始位姿。点对平面（Point-to-Plane）ICP 变体收敛更快。

#### DeepIM

深度迭代匹配（Deep Iterative Matching，DeepIM）将位姿细化建模为端到端学习问题。给定当前位姿渲染图和实际观测图，网络预测相对变换 \(\Delta \mathbf{T}\) 对位姿进行更新：

$$
\mathbf{T}_{k+1} = \Delta \mathbf{T}_k \cdot \mathbf{T}_k
$$

与 ICP 相比，DeepIM 更鲁棒于遮挡和纹理差异。


### 机器人抓取应用

在机器人抓取（Robot Grasping）中，位姿估计是核心感知模块：

- **6DoF 抓取点估计**（6DoF Grasp Pose Estimation）：不依赖 CAD 模型，直接从点云预测抓取位姿，代表性方法包括 GraspNet-1Billion 和 AnyGrasp。
- **料箱拣选**（Bin-Picking）：在随机堆叠的零件中定位目标物体并规划抓取路径。物体间遮挡严重，对位姿估计鲁棒性要求高。
- **装配与插件**：需要毫米级甚至亚毫米级的位姿精度，通常结合力传感器进行接触阶段的力控调整。


## 人体姿态估计

### 2D 人体姿态估计

2D 人体姿态估计（2D Human Pose Estimation）在图像平面内检测 N 个关节点（Keypoints）的像素坐标，构成人体骨架（Skeleton）。

#### 热图回归方法

热图（Heatmap）回归是主流范式，为每个关节点生成一张概率热图 \(\mathbf{H}_k \in \mathbb{R}^{H \times W}\)，像素值代表该点为关节 \(k\) 的概率。关节坐标通过取热图最大值位置（argmax）获得：

$$
\hat{\mathbf{p}}_k = \arg\max_{(u,v)} \mathbf{H}_k(u, v)
$$

**HRNet**（High-Resolution Representation Network，高分辨率表示网络）是热图回归方法的代表。其核心思想是全程维护高分辨率特征表示，同时引入多尺度特征并行分支和跨分辨率融合（Multi-Resolution Fusion）。与先降分辨率再上采样的 U-Net 风格方法相比，HRNet 的高分辨率分支始终保留精细空间信息，关节定位更精确。

**ViTPose**（Vision Transformer Pose Estimation）以视觉变换器（Vision Transformer，ViT）为骨干网络，在大规模预训练特征的基础上接热图解码头。ViTPose 展示了大模型在姿态估计任务上的强大迁移能力，并在 COCO Keypoints 数据集上刷新了多项纪录。

#### 回归方法

直接回归方法（Direct Regression）以全连接层直接输出归一化坐标 \((u/W, v/H)\)，结构简单，推理速度快，但定位精度通常略低于热图回归。

#### 自底向上方法

自底向上（Bottom-Up）方法先检测图像中全部关节点，再将其组合为各人的骨架。

**OpenPose** 是最具影响力的自底向上方法，引入了部位亲和力场（Part Affinity Field，PAF）来编码骨骼连接信息。PAF 为图像中每条肢体（Limb）生成一张向量场 \(\mathbf{L}_c \in \mathbb{R}^{H \times W \times 2}\)，指向从一端关节到另一端关节的方向。通过匈牙利算法（Hungarian Algorithm）对关节点进行最优分配，可同时处理图像中任意数量的人体，适合多人实时场景。


### 3D 人体姿态估计

3D 人体姿态估计（3D Human Pose Estimation）以毫米为单位预测各关节在三维空间中的坐标（相机坐标系或世界坐标系）。

#### VideoPose3D

VideoPose3D 采用提升网络（Lifting Network）策略：先用成熟的 2D 姿态估计器获取每帧关节点的 2D 坐标，再将时序序列输入时序 Transformer（Temporal Transformer）将其提升为 3D 坐标。时序建模使网络能利用运动上下文解决深度模糊性（Depth Ambiguity）——仅从单帧 2D 投影无法唯一确定 3D 深度，但结合前后帧的运动规律可显著改善估计。

#### MotionBERT

MotionBERT 将 BERT 的预训练-微调范式引入人体运动理解，构建统一的人体运动表示框架。模型以 2D 关节点时序作为输入，通过双流时空 Transformer（Dual-stream Spatial-Temporal Transformer）同时建模空间（关节间）和时序（帧间）依赖关系。预训练后的模型可迁移至：

- 3D 姿态估计
- 动作识别（Action Recognition）
- 网格恢复（Mesh Recovery）

MotionBERT 在 Human3.6M 和 MPI-INF-3DHP 数据集上均取得了当时的最优性能（State-of-the-Art）。


### 全身与手势估计

**MediaPipe Holistic** 是谷歌开发的实时全身姿态估计框架，在单一管线中同时输出：

- 全身骨架（33 个关键点）
- 左右手各 21 个关键点
- 面部 468 个关键点

其高效性来源于轻量级检测-跟踪联合优化：首帧运行完整检测器，后续帧仅在已知感兴趣区域（Region of Interest，ROI）内运行跟踪器，显著降低计算量，可在移动端实时运行。

手势估计（Hand Gesture Estimation）在手势交互、虚拟现实和手语识别中具有重要应用价值。


### 机器人应用

人体姿态估计为机器人系统提供了丰富的人体状态信息：

- **人机协作安全区域检测**（Human-Robot Collaboration Safety）：实时监测工人关节位置，当人体进入机器人工作空间时触发减速或急停，保障协作安全。
- **动作模仿学习**（Imitation Learning from Demonstration）：通过记录人类示教者的关节轨迹，使用行为克隆（Behavior Cloning）或逆强化学习（Inverse Reinforcement Learning）训练机器人复现复杂操作动作。
- **手势交互指令**（Gesture-based Interaction）：识别手势或肢体语言，将其映射为机器人控制指令，实现无接触式人机交互界面。
- **步态分析与康复辅助**：结合外骨骼机器人，通过人体姿态估计量化步态参数，辅助运动康复训练。


## 评测指标

### 物体位姿估计指标

#### ADD（平均 3D 距离）

平均点对点距离（Average Distance of Model Points，ADD）衡量预测位姿与真实位姿在模型点云上的平均偏差：

$$
\text{ADD} = \frac{1}{m} \sum_{\mathbf{x} \in \mathcal{M}} \left\| \left(\mathbf{R}\mathbf{x} + \mathbf{t}\right) - \left(\hat{\mathbf{R}}\mathbf{x} + \hat{\mathbf{t}}\right) \right\|
$$

其中 \(\mathcal{M}\) 为模型点云（共 \(m\) 个点），\((\mathbf{R}, \mathbf{t})\) 为真实位姿，\((\hat{\mathbf{R}}, \hat{\mathbf{t}})\) 为预测位姿。当 ADD 小于模型直径的 10% 时，通常认为位姿估计正确，报告正确率（ADD < 0.1d）。

#### ADD-S（对称物体版本）

对于旋转对称物体（如圆柱、球体），不同旋转角度对应相同的视觉外观，ADD 无法正确衡量此类物体的估计质量。ADD-S 将每个模型点与预测变换后最近邻点匹配：

$$
\text{ADD-S} = \frac{1}{m} \sum_{\mathbf{x}_1 \in \mathcal{M}} \min_{\mathbf{x}_2 \in \mathcal{M}} \left\| \left(\mathbf{R}\mathbf{x}_1 + \mathbf{t}\right) - \left(\hat{\mathbf{R}}\mathbf{x}_2 + \hat{\mathbf{t}}\right) \right\|
$$

BOP 挑战赛（Benchmark for 6DoF Object Pose Estimation）采用 ADD-S 的曲线下面积（AUC）作为主要评测指标。

#### 投影误差

2D 投影误差（2D Projection Error）将模型点分别用真实位姿和预测位姿投影到图像平面，计算对应点的平均像素距离：

$$
e_\text{proj} = \frac{1}{m} \sum_{\mathbf{x} \in \mathcal{M}} \left\| \pi(\mathbf{K}, \mathbf{R}, \mathbf{t}, \mathbf{x}) - \pi(\mathbf{K}, \hat{\mathbf{R}}, \hat{\mathbf{t}}, \mathbf{x}) \right\|
$$

当误差小于 5 像素时认为位姿正确。该指标对深度误差不敏感，适合评估 2D 对齐质量。

### 人体姿态估计指标

#### PCK（正确关键点百分比）

正确关键点百分比（Percentage of Correct Keypoints，PCK）将预测关键点与真实关键点的距离与参考长度（如头部长度或躯干长度）进行比较：

$$
\text{PCK}_\alpha = \frac{1}{N} \sum_{k=1}^{N} \mathbf{1}\left[ \frac{\|\hat{\mathbf{p}}_k - \mathbf{p}_k^*\|}{d_\text{ref}} < \alpha \right]
$$

其中 \(\alpha\) 为阈值比例（通常取 0.2），\(d_\text{ref}\) 为参考骨骼长度，\(\mathbf{1}[\cdot]\) 为指示函数。MPII 数据集常用 PCKh@0.5（以头部长度为参考，阈值 0.5）。

#### MPJPE（均方根关节位置误差）

均方根关节位置误差（Mean Per-Joint Position Error，MPJPE）衡量 3D 姿态估计中各关节的平均欧氏距离误差，单位为毫米：

$$
\text{MPJPE} = \frac{1}{N} \sum_{k=1}^{N} \left\| \hat{\mathbf{p}}_k^{3D} - \mathbf{p}_k^{3D*} \right\|_2
$$

PA-MPJPE（Procrustes Aligned MPJPE）在计算误差前先用 Procrustes 对齐消除全局旋转和缩放的影响，更关注骨架内部形状的准确性。Human3.6M 数据集上通常同时报告 MPJPE 和 PA-MPJPE。

#### COCO AP 指标

COCO Keypoints 数据集采用基于目标关键点相似度（Object Keypoint Similarity，OKS）的平均精度（Average Precision，AP）：

$$
\text{OKS} = \frac{\sum_k \exp\left(-\frac{d_k^2}{2s^2\sigma_k^2}\right) \delta(v_k > 0)}{\sum_k \delta(v_k > 0)}
$$

其中 \(d_k\) 为第 \(k\) 个关键点的预测误差，\(s\) 为物体尺度，\(\sigma_k\) 为每类关键点的归一化因子，\(v_k\) 为可见性标注。


## 常用数据集

### 物体位姿估计数据集

| 数据集 | 物体数 | 传感器 | 特点 |
|---|---|---|---|
| LineMOD | 15 | RGB-D | 单物体、弱纹理；早期标准基准 |
| YCB-Video | 21 | RGB-D | 多物体混合场景；来自 YCB 物体集 |
| T-LESS | 30 | RGB-D | 工业零件，无纹理，高度对称 |
| Occlusion LineMOD | 8 | RGB-D | LineMOD 子集，强遮挡场景 |
| BOP 挑战赛系列 | - | RGB/RGB-D | 统一评测协议，涵盖多个子数据集 |

### 人体姿态估计数据集

| 数据集 | 标注类型 | 规模 | 特点 |
|---|---|---|---|
| COCO Keypoints | 2D，17 关键点 | 20 万张图 | 多人、复杂场景；工业标准 |
| MPII Human Pose | 2D，16 关键点 | 2.5 万人 | 多样活动类别，PCKh 指标 |
| Human3.6M | 3D，17 关键点 | 360 万帧 | 室内采集，动捕真值，MPJPE 指标 |
| MPI-INF-3DHP | 3D，17 关键点 | 130 万帧 | 室内外混合，多摄像机系统 |
| AIST++ | 3D 舞蹈动作 | 1408 段 | 音乐驱动舞蹈，细粒度动作 |


## 常用工具与框架

### FoundationPose

英伟达（NVIDIA）开源的 FoundationPose 是目前性能最强的零样本（Zero-Shot）6DoF 位姿估计与跟踪框架之一。支持两种模式：

- **基于 CAD 模型**：提供精确 3D 网格模型时，直接渲染参考视图建立对应关系。
- **基于参考图像**：提供少量 RGBD 参考图时，通过神经隐式场（Neural Implicit Field）重建近似几何。

FoundationPose 在 BOP 挑战赛多个子集上取得了当时最优结果，并支持实时 6DoF 跟踪。

### FoundPose

FoundPose 专注于仅使用参考图像（无深度）的零样本位姿估计，以 DINOv2 特征为核心，提供从粗检索到精细 PnP 求解的完整管线，适合无法获得深度传感器的场景。

### OpenCV solvePnP

OpenCV 提供的 `cv2.solvePnP()` 是工程实践中最常用的 PnP 求解器，支持多种算法：

- `SOLVEPNP_ITERATIVE`：基于 Levenberg-Marquardt 迭代优化
- `SOLVEPNP_EPNP`：高效 PnP（EPnP），时间复杂度 \(O(n)\)
- `SOLVEPNP_IPPE`：适用于平面物体的解析解
- `SOLVEPNP_SQPNP`：基于半定规划（Semi-Definite Programming）的全局最优解

搭配 `cv2.solvePnPRansac()` 可在存在异常点时鲁棒求解。

### Open3D ICP

Open3D 提供高效的点云处理和 ICP 实现，支持：

- 点对点 ICP（Point-to-Point ICP）
- 点对平面 ICP（Point-to-Plane ICP）
- 带颜色权重的彩色 ICP（Colored ICP）
- 多尺度 ICP（Multi-Scale ICP）

Open3D 的 `o3d.pipelines.registration.registration_icp()` 接口简洁，适合快速原型开发和工程集成。

### MMPose

OpenMMLab 开源的 MMPose 提供统一的人体姿态估计训练与推理框架，内置 HRNet、ViTPose、RTMPose 等主流模型，支持 2D/3D 人体姿态估计、手部和全身估计，并提供完整的数据流水线和评测脚本。

### MediaPipe

谷歌开源的 MediaPipe 提供跨平台（移动端、Web、桌面端）的实时姿态估计解决方案，包括：

- `mediapipe.solutions.pose`：全身 33 关键点
- `mediapipe.solutions.hands`：手部 21 关键点
- `mediapipe.solutions.holistic`：全身 + 手部 + 面部联合估计


## 参考资料

- Xiang, Y. et al. "PoseCNN: A Convolutional Neural Network for 6D Object Pose Estimation in Cluttered Scenes." *RSS 2018*.
- Wang, C. et al. "DenseFusion: 6D Object Pose Estimation by Iterative Dense Fusion." *CVPR 2019*.
- He, Y. et al. "FFB6D: A Full Flow Bidirectional Fusion Network for 6D Pose Estimation." *CVPR 2021*.
- Wang, G. et al. "GDR-Net: Geometry-Guided Direct Regression Network for Monocular 6D Object Pose Estimation." *CVPR 2021*.
- Sun, K. et al. "Deep High-Resolution Representation Learning for Visual Recognition." *TPAMI 2021*. (HRNet)
- Xu, M. et al. "ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation." *NeurIPS 2022*.
- Zhu, W. et al. "MotionBERT: A Unified Perspective on Learning Human Motion Representations." *ICCV 2023*.
- Wen, B. et al. "FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects." *CVPR 2024*.
- Li, Z. et al. "VideoPose3D: Exploiting Temporal Context for 3D Human Pose Estimation in Video." *CVPR 2019*.
- OpenCV 官方文档：solvePnP. https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
- Open3D 官方文档：ICP Registration. http://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html
- BOP 挑战赛官网. https://bop.felk.cvut.cz/
- MMPose 文档. https://mmpose.readthedocs.io/
- MediaPipe 文档. https://mediapipe.dev/
