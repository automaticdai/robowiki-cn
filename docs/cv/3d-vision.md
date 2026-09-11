# 三维视觉

!!! note "引言"
    三维视觉（3D Vision）是计算机视觉的重要分支，研究如何从二维图像或传感器数据中恢复三维结构与空间信息。与传统二维视觉不同，三维视觉能够感知物体的深度、形状与空间位置，是机器人抓取、自主导航、场景重建和增强现实等应用的核心技术基础。随着深度传感器普及和神经网络方法的快速发展，三维视觉正经历从经典几何方法向深度学习方法演进的重大变革。


## 三维视觉的任务谱系

三维视觉要回答的核心问题是：如何从二维观测中恢复场景的三维结构。按输入形式与输出目标，可分为以下几类任务：

| 任务 | 输入 | 输出 | 代表方法 | 典型机器人用途 |
|------|------|------|----------|----------------|
| 深度估计 | 单目或双目图像 | 稠密深度图 | SGBM、MiDaS、立体匹配网络 | 避障、抓取前的场景感知 |
| 主动深度感知 | 结构光或飞行时间信号 | 稠密深度图 | 结构光、ToF、iToF/dToF | 近距离抓取、人机交互 |
| 位姿估计与配准 | 两帧点云或图像 | 刚体变换 | ICP、NDT、FPFH + RANSAC | 里程计、多帧拼接、物体定位 |
| 稀疏重建 | 多视角无序图像 | 相机位姿与稀疏点云 | SfM（COLMAP） | 场景建模、离线标定 |
| 稠密重建 | 多视角图像与位姿 | 稠密点云或网格 | MVS、泊松重建、TSDF | 数字孪生、碰撞模型生成 |
| 辐射场重建 | 多视角图像与位姿 | 可微渲染的隐式/显式场 | NeRF、3D Gaussian Splatting | 高保真仿真资产、新视角合成 |
| 点云理解 | 点云 | 语义标签或物体框 | PointNet++、VoxelNet | 三维目标检测、场景分割 |


## 深度获取方式对比

不同深度传感原理在量程、精度、环境适应性上差异显著，选型时通常先由工作距离与光照条件排除大部分方案：

| 方式 | 典型量程 | 精度趋势 | 室外表现 | 主要限制 | 成本 |
|------|----------|----------|----------|----------|------|
| 双目立体视觉 | 0.3–20 m | 误差随距离平方增长 | 好 | 依赖纹理，弱纹理区失效 | 低 |
| 结构光 | 0.2–3 m | 近距离亚毫米级 | 差（阳光干扰） | 量程短，受强光影响大 | 中 |
| 飞行时间（ToF） | 0.5–10 m | 全程较均匀 | 中等 | 多径反射、边缘飞点 | 中 |
| 激光雷达 | 1–200 m | 全程较均匀 | 好 | 点云稀疏，成本高 | 高 |
| 单目深度估计 | 不限 | 尺度不确定 | 好 | 绝对尺度需外部标定 | 极低 |

**选型经验**：需要绝对尺度且工作在室外中远距离，优先激光雷达或双目；桌面级精细抓取选结构光；室内移动机器人避障多用 ToF 或双目深度相机；仅需相对深度线索（如可行驶区域分割）时，单目估计已足够。


## 点云处理 (Point Cloud Processing)

点云（Point Cloud）是三维空间中离散点的集合，每个点包含 \((x, y, z)\) 坐标，可选附带颜色 \((r, g, b)\) 或法向量等属性。点云是激光雷达（LiDAR）、深度相机和 SfM/MVS 的主要输出格式。

### 点云数据格式

| 格式 | 全称 | 特点 |
|------|------|------|
| PCD | Point Cloud Data | PCL 原生格式，支持 ASCII 和二进制，含头部元数据 |
| PLY | Polygon File Format | 支持顶点、面、颜色等属性，通用性强 |
| LAS | LASer File Format | 地理空间测绘标准格式，支持 GPS 时间戳和分类信息 |
| E57 | ASTM E57 | 三维成像系统数据交换标准，支持大规模扫描仪数据 |

### 基础处理

#### 滤波

**直通滤波（PassThrough Filter）**：按照指定坐标轴范围裁剪点云，去除感兴趣区域之外的点：

```python
import open3d as o3d

pcd = o3d.io.read_point_cloud("scene.pcd")
# 保留 z 坐标在 [0.5, 3.0] 范围内的点
bbox = o3d.geometry.AxisAlignedBoundingBox(
    min_bound=(-10, -10, 0.5),
    max_bound=(10, 10, 3.0)
)
pcd_cropped = pcd.crop(bbox)
```

**体素下采样（Voxel Downsampling）**：将点云按体素网格划分，每个体素内取重心点，均匀降低点密度，减少后续处理计算量：

```python
pcd_down = pcd.voxel_down_sample(voxel_size=0.05)  # 5 cm 体素
```

**统计离群点去除（Statistical Outlier Removal，SOR）**：对每个点计算其邻域内的平均距离，将统计上显著偏离均值的点标记为噪声并移除：

```python
pcd_clean, ind = pcd_down.remove_statistical_outlier(
    nb_neighbors=20, std_ratio=2.0
)
```

#### 法线估计 (Normal Estimation)

法线（Normal Vector）描述点云局部表面的朝向，是许多后续算法（配准、分割、特征提取）的基础。法线估计通过对每个点的 \(k\) 个近邻进行主成分分析（PCA，Principal Component Analysis），将最小特征值对应的特征向量作为法线方向：

$$
\mathbf{n}_i = \arg\min_{\|\mathbf{v}\|=1} \mathbf{v}^T C_i \mathbf{v}
$$

其中协方差矩阵 \(C_i = \frac{1}{k}\sum_{j \in \mathcal{N}(i)} (\mathbf{p}_j - \bar{\mathbf{p}})(\mathbf{p}_j - \bar{\mathbf{p}})^T\)。

```python
pcd_down.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
)
# 确保法线朝向相机（法线方向一致性）
pcd_down.orient_normals_towards_camera_location(camera_location=[0, 0, 0])
```

#### 关键点检测

- **ISS（Intrinsic Shape Signatures）**：基于点邻域散度分析，检测几何显著点
- **SIFT-3D**：将 SIFT 思想扩展至三维，在尺度空间中检测极值点

### 深度学习点云处理

传统点云处理方法需要手工设计特征，深度学习方法可以端到端地学习三维表示。

#### PointNet

PointNet 由 Qi 等人于 2017 年提出，直接以原始点云（无序点集）作为输入，通过以下方式处理点集的置换不变性（Permutation Invariance）：

- 对每个点独立应用共享的多层感知机（MLP），提取点级特征
- 通过全局最大池化（Global Max Pooling）聚合所有点的特征，获得全局描述子
- 输入变换网络（Input Transform Network，T-Net）学习对齐变换，提升旋转鲁棒性

全局特征维度通常为 1024，可用于分类或与点级特征拼接用于分割。

#### PointNet++

PointNet++ 是 PointNet 的层次化扩展，引入分层抽象结构：

1. **最远点采样（Farthest Point Sampling，FPS）**：从点云中均匀选取关键点
2. **Ball Query**：以关键点为中心、固定半径内的邻域点
3. **PointNet 局部特征学习**：对每个邻域应用 PointNet 提取局部特征
4. **层次化聚合**：多个抽象层次逐步扩大感受野

#### VoxelNet

VoxelNet 将点云体素化，在每个非空体素内用小型 PointNet（体素特征编码器，VFE）提取特征，然后将体素特征排列成三维张量，输入三维卷积神经网络（3D CNN）进行目标检测。


## 本章内容导览

三维视觉章节按「深度获取 → 配准 → 重建 → 速查」的顺序组织：

| 页面 | 主要内容 |
|------|---------|
| [三维视觉](3d-vision.md) | 任务谱系、深度获取方式对比、点云数据结构与基础处理 |
| [立体视觉与深度](3d-vision-stereo-depth.md) | 双目标定与校正、立体匹配算法、单目深度估计、结构光与 ToF |
| [点云配准](3d-vision-pointcloud-registration.md) | ICP 及其变种、FPFH 特征配准、全局配准、位姿图优化 |
| [三维重建与辐射场](3d-vision-reconstruction-radiance.md) | SfM、MVS、NeRF、3D Gaussian Splatting、网格重建 |
| [三维视觉参考资料](3d-vision-full-reference.md) | 经典论文、教材、开源工具、数据集与公式速查 |
| [深度相机](../sensing/depth-camera.md) | 深度相机硬件选型与标定 |
| [SLAM](../sensing/slam.md) | 同步定位与建图 |


## 参考资料

1. Hirschmüller, H. (2008). Stereo processing by semiglobal matching and mutual information. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 30(2), 328–341.
2. Lipson, L., Teed, Z., & Deng, J. (2021). RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching. *International Conference on 3D Vision (3DV)*.
3. Mildenhall, B., Srinivasan, P. P., Tancik, M., Barron, J. T., Ramamoorthi, R., & Ng, R. (2020). NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. *ECCV*.
4. Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering. *ACM SIGGRAPH*, 42(4).
5. Qi, C. R., Su, H., Mo, K., & Guibas, L. J. (2017). PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation. *CVPR*.
6. Qi, C. R., Yi, L., Su, H., & Guibas, L. J. (2017). PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space. *NeurIPS*.
7. Schönberger, J. L., & Frahm, J.-M. (2016). Structure-from-Motion Revisited. *CVPR*.
8. Zhou, Q.-Y., Park, J., & Koltun, V. (2018). Open3D: A Modern Library for 3D Data Processing. *arXiv:1801.09847*.
9. Rusu, R. B., & Cousins, S. (2011). 3D is here: Point Cloud Library (PCL). *ICRA*.
10. Biber, P., & Straßer, W. (2003). The Normal Distributions Transform: A New Approach to Laser Scan Matching. *IROS*.

