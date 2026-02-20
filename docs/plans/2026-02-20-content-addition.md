# Wiki Content Addition Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 新增 9 篇高质量中文技术文章，覆盖 CV 分割/跟踪/位姿/三维视觉、传感器融合、深度学习、数据集、Spot 档案、Shell 脚本，并同步更新导航。

**Architecture:** 每篇文章独立写入对应 Markdown 文件，遵循现有 wiki 规范（admonition 引言、数学公式、参考资料末节）。最后一步统一更新 `mkdocs.yml` 导航树。

**Tech Stack:** MkDocs + Material Theme, Markdown, MathJax (LaTeX), Python Markdown Math extension

**写作规范速查：**
- 引言用 `!!! note "引言"` admonition
- 行内数学：`\(...\)`，块数学：`$$ ... $$`
- 标题最多 4 级（`#` 到 `####`）
- 同级段落间两个空行
- 末节为 `## 参考资料`
- 文件末尾留一空行

---

### Task 1: 图像分割 `cv/segmentation.md`

**Files:**
- Create: `docs/cv/segmentation.md`

**Step 1: 写入文件**

写入以下完整内容：

```markdown
# 图像分割

!!! note "引言"
    图像分割 (Image Segmentation) 是计算机视觉中将图像划分为若干有意义区域的任务。与目标检测只输出边界框不同，图像分割为每个像素赋予类别标签或实例 ID，提供更精细的场景理解。在机器人系统中，图像分割广泛用于障碍物识别、可行驶区域检测、抓取目标精确定位和人机安全协作。


## 分割任务类型

图像分割按照输出粒度可分为三类：

### 语义分割 (Semantic Segmentation)

语义分割为图像中的每个像素分配一个类别标签，但不区分同类别的不同个体。例如，将所有行人像素标记为"行人"，而不区分是哪一个行人。

**应用场景**：可行驶区域检测、地面/障碍物识别、户外场景解析。

### 实例分割 (Instance Segmentation)

实例分割在语义分割的基础上，进一步区分同类别的不同个体。每个物体实例拥有独立的像素掩码 (Mask)。

**应用场景**：机器人抓取（区分堆叠物体的每个个体）、人群分析、多目标跟踪前处理。

### 全景分割 (Panoptic Segmentation)

全景分割统一了语义分割和实例分割：对"可数物体"（Things，如人、车）进行实例分割，对"不可数背景"（Stuff，如天空、地面）进行语义分割，从而对整个画面进行完整的像素级解析。


## 经典语义分割算法

### FCN (Fully Convolutional Network)

FCN [1] 是第一个端到端的像素级分类网络，将传统 CNN 中的全连接层替换为卷积层，从而输出与输入相同分辨率的预测图。它引入了**跳跃连接 (Skip Connection)**，融合不同分辨率的特征图以恢复空间细节。

FCN 的输出分辨率仍低于原图，需要上采样（双线性插值或转置卷积）恢复全分辨率。

### U-Net

U-Net [2] 采用对称的编码器-解码器（Encoder-Decoder）架构，编码器逐步下采样提取语义特征，解码器逐步上采样恢复空间分辨率，并通过跳跃连接将编码器的特征直接传递给解码器对应层。

$$
\text{特征融合：} F_{\text{dec}} = \text{Conv}([F_{\text{enc}}, F_{\text{up}}])
$$

U-Net 最初用于医学图像分割，因其在小数据集上的优秀表现而广泛应用。

### DeepLab 系列

DeepLab 系列的核心创新是**空洞卷积 (Atrous/Dilated Convolution)**，通过在卷积核中插入间隔来扩大感受野，而不降低分辨率：

$$
y[i] = \sum_k x[i + r \cdot k] \cdot w[k]
$$

其中 \(r\) 为空洞率 (Dilation Rate)。

**DeepLab v3+** [3] 在此基础上引入 ASPP (Atrous Spatial Pyramid Pooling) 模块，以不同空洞率并行提取多尺度特征，并结合轻量解码器恢复边界细节。这是目前语义分割领域最经典的基线之一。

### 基于 Transformer 的方法

近年来，Transformer 架构（原为 NLP 设计）被引入图像分割领域：

- **SegFormer** [4]：分层 Transformer 编码器 + 轻量级 MLP 解码器，在精度和速度之间取得良好平衡
- **Mask2Former** [5]：统一框架处理语义/实例/全景三类分割，以可学习查询（Learnable Query）通过交叉注意力提取目标特征

### 轻量级实时分割

嵌入式机器人平台（如 Jetson）上需要兼顾速度与精度：

| 模型 | 推理速度 | mIoU (Cityscapes) | 特点 |
|------|---------|-------------------|------|
| BiSeNetV2 | ~156 FPS (GTX 1080Ti) | 73.4% | 双分支：细节分支 + 语义分支 |
| PP-LiteSeg | ~273 FPS (GTX 1080Ti) | 76.0% | 轻量化骨干 + UAFM 注意力融合 |
| DDRNet | ~102 FPS (GTX 1080Ti) | 79.5% | 双分辨率网络，性能更强 |
| SegNet | 快速 | 较低 | 经典轻量架构，历史意义 |


## 实例分割算法

### Mask R-CNN

Mask R-CNN [6] 在 Faster R-CNN 目标检测框架上增加了一个并行的掩码预测分支。对每个候选区域（Region of Interest），掩码分支输出 \(K\) 个二值掩码（\(K\) 为类别数）。

关键改进是引入 **RoI Align** 替代 RoI Pooling，通过双线性插值避免了量化误差，使掩码预测更精确。

### SOLOv2

SOLOv2 [7] 摒弃了检测-分割的两阶段范式，直接预测实例掩码。它将图像划分为 \(S \times S\) 个网格，每个网格负责预测其中心落在该格内的实例的掩码。通过矩阵非极大值抑制（Matrix NMS）加速后处理，推理速度显著优于 Mask R-CNN。


## 全景分割算法

### Panoptic-DeepLab

Panoptic-DeepLab [8] 采用自底向上的方法：

1. **语义分割头**：预测每个像素的类别
2. **实例中心热图**：预测每个实例的中心点
3. **中心偏移量**：预测每个像素到其所属实例中心的偏移量

通过将像素聚类到最近的实例中心，得到最终的全景分割结果，无需 RPN（候选框生成）。

### Mask2Former（统一框架）

Mask2Former [5] 采用"Masked Attention"机制，将可学习的目标查询（Object Queries）限制在预测掩码的范围内进行注意力计算，提高了收敛速度和分割精度。通过更换训练标签，同一框架可处理三种分割任务。


## 评测指标

### 语义分割

- **像素精度 (Pixel Accuracy, PA)**：正确分类的像素数 / 总像素数
- **平均交并比 (Mean Intersection over Union, mIoU)**：各类别 IoU 的平均值

$$
\text{IoU}_c = \frac{|P_c \cap G_c|}{|P_c \cup G_c|}
$$

其中 \(P_c\) 为预测掩码，\(G_c\) 为真实掩码。mIoU 是语义分割最常用的评测指标。

- **边界 F1 分数 (Boundary F1, BF)**：衡量预测掩码与真实掩码在边界处的对齐程度

### 实例分割

- **AP（平均精度）**：在不同 IoU 阈值（0.5 ~ 0.95）下的平均精度，与目标检测类似，但 IoU 基于掩码而非边界框

### 全景分割

- **PQ（Panoptic Quality）**：

$$
\text{PQ} = \underbrace{\frac{\sum_{(p,g) \in \text{TP}} \text{IoU}(p,g)}{|\text{TP}|}}_{\text{SQ（分割质量）}} \times \underbrace{\frac{|\text{TP}|}{|\text{TP}| + \frac{1}{2}|\text{FP}| + \frac{1}{2}|\text{FN}|}}_{\text{RQ（识别质量）}}
$$


## 常用数据集

| 数据集 | 类型 | 类别数 | 图像数 | 特点 |
|--------|------|--------|--------|------|
| Cityscapes | 语义/实例 | 19/8 | 5000（精标注） | 城市街景，自动驾驶标准基准 |
| ADE20K | 语义/实例 | 150 | 25000 | 室内外场景，类别丰富 |
| PASCAL VOC | 语义/实例 | 20 | ~11000 | 经典目标检测和分割基准 |
| COCO | 全景/实例 | 80+53 | 118000 | 最常用综合视觉基准 |
| Mapillary Vistas | 语义 | 66 | 25000 | 多样化城市场景，高分辨率 |


## 与 ROS 集成

在 ROS 系统中，图像分割结果通常以以下方式发布：

```python
# 语义分割掩码（编码为灰度图，每个像素值为类别 ID）
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

bridge = CvBridge()
seg_msg = bridge.cv2_to_imgmsg(seg_mask.astype(np.uint8), encoding='mono8')
pub.publish(seg_msg)
```

常用的 ROS 图像分割包：

- **`semantic_segmentation_ros`**：封装 DeepLab 等模型，订阅相机话题并发布分割掩码
- **`mask_rcnn_ros`**：封装 Mask R-CNN，发布实例掩码和类别信息
- **`image_transport`**：高效的图像话题传输，支持压缩传输

## 参考资料

1. Long, J., Shelhamer, E., & Darrell, T. (2015). Fully Convolutional Networks for Semantic Segmentation. *CVPR*.
2. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI*.
3. Chen, L. C., et al. (2018). Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation (DeepLab v3+). *ECCV*.
4. Xie, E., et al. (2021). SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers. *NeurIPS*.
5. Cheng, B., et al. (2022). Masked-attention Mask Transformer for Universal Image Segmentation (Mask2Former). *CVPR*.
6. He, K., et al. (2017). Mask R-CNN. *ICCV*.
7. Wang, X., et al. (2020). SOLOv2: Dynamic and Fast Instance Segmentation. *NeurIPS*.
8. Cheng, B., et al. (2020). Panoptic-DeepLab: A Simple, Strong, and Fast Baseline for Bottom-Up Panoptic Segmentation. *CVPR*.
```

**Step 2: 验证文件已创建**

```bash
wc -l docs/cv/segmentation.md
```

预期：约 160 行

**Step 3: 提交**

```bash
git add docs/cv/segmentation.md
git commit -m "docs: add image segmentation article (cv/segmentation.md)"
```

---

### Task 2: 目标跟踪 `cv/tracking.md`

**Files:**
- Create: `docs/cv/tracking.md`

**Step 1: 写入文件**

完整内容见实施阶段（由子 agent 负责写作）。

核心章节：
- 跟踪任务定义与分类（SOT vs MOT）
- 相关滤波跟踪（MOSSE、KCF、CSRT）
- 孪生网络跟踪（SiamFC、SiamRPN++、SiamMask）
- Transformer 跟踪器（OSTrack、DropTrack）
- 多目标跟踪（SORT、DeepSORT、ByteTrack、StrongSORT）
- 评测指标（MOTA、MOTP、IDF1、HOTA）
- 机器人应用场景

**Step 2: 提交**

```bash
git add docs/cv/tracking.md
git commit -m "docs: add object tracking article (cv/tracking.md)"
```

---

### Task 3: 位姿估计 `cv/pose-estimation.md`

**Files:**
- Create: `docs/cv/pose-estimation.md`

核心章节：
- 物体 6DoF 位姿估计（旋转 SO(3) + 平移 ℝ³）
- 基于 RGB 的方法（PoseCNN、DPOD、GDR-Net）
- 基于 RGB-D 的方法（DenseFusion、FFB6D）
- 无 CAD 模型方法（FoundPose、Gen6D）
- 人体姿态估计（HRNet、ViTPose、VideoPose3D）
- 评测指标（ADD、ADD-S、PCK、MPJPE）
- 常用数据集（YCB-Video、LineMOD、COCO Keypoints）

**Step 2: 提交**

```bash
git add docs/cv/pose-estimation.md
git commit -m "docs: add pose estimation article (cv/pose-estimation.md)"
```

---

### Task 4: 三维视觉 `cv/3d-vision.md`

**Files:**
- Create: `docs/cv/3d-vision.md`

核心章节：
- 双目立体视觉（标定/校正/立体匹配 BM、SGM、RAFT-Stereo）
- 结构光与 ToF 深度传感
- 运动恢复结构（SfM：特征匹配 → 相机位姿 → 稀疏重建；COLMAP）
- 点云处理（格式、滤波、下采样、ICP 配准）
- 深度学习点云（PointNet、PointNet++、VoxelNet）
- 神经辐射场（NeRF）与 3D 高斯泼溅（3DGS）
- 工具（Open3D、PCL、CloudCompare）

**Step 2: 提交**

```bash
git add docs/cv/3d-vision.md
git commit -m "docs: add 3D vision article (cv/3d-vision.md)"
```

---

### Task 5: 传感器融合 `sensing/sensor-fusion.md`

**Files:**
- Create: `docs/sensing/sensor-fusion.md`

核心章节：
- 引言（单一传感器局限性）
- 融合层级（数据级/特征级/决策级）
- 贝叶斯估计框架
- 卡尔曼滤波（KF）推导（预测步 + 更新步）
- 扩展卡尔曼滤波（EKF，线性化非线性系统）
- 无迹卡尔曼滤波（UKF，无需 Jacobian）
- 粒子滤波（非参数贝叶斯，适合非高斯噪声）
- 典型应用（IMU+GPS 惯导、LiDAR+相机 3D 检测、VIO）
- 时间同步与空间外参标定（kalibr）
- ROS 工具（`robot_localization`、`imu_filter_madgwick`）

**Step 2: 提交**

```bash
git add docs/sensing/sensor-fusion.md
git commit -m "docs: add sensor fusion article (sensing/sensor-fusion.md)"
```

---

### Task 6: 深度学习在机器人中的应用 `learning/dl.md`

**Files:**
- Create: `docs/learning/dl.md`

核心章节：
- 引言（深度学习对机器人的革命性影响）
- 神经网络基础回顾（MLP、CNN、RNN/LSTM、Attention/Transformer）
- 端到端学习（直接从原始感知输入映射到控制输出）
- 模仿学习（行为克隆 BC、DAgger、隐式行为克隆）
- 基础模型在机器人中（RT-2、OpenVLA、π₀）
- 深度学习与经典控制融合（学习残差控制、学习代价函数）
- 部署优化（量化、剪枝、知识蒸馏、TensorRT、ONNX）
- 常用框架（PyTorch、JAX/Flax、LeRobot、Lerobot）

**Step 2: 提交**

```bash
git add docs/learning/dl.md
git commit -m "docs: add deep learning for robotics article (learning/dl.md)"
```

---

### Task 7: 数据集大全 `database/dataset.md`（大幅扩充）

**Files:**
- Modify: `docs/database/dataset.md`（从 14 行扩充至 ~300 行）

核心章节：
- 引言
- 计算机视觉数据集（ImageNet、COCO、Pascal VOC、Objects365）
- 图像分割数据集（Cityscapes、ADE20K）
- 人体姿态数据集（COCO Keypoints、Human3.6M、MPII）
- 自动驾驶数据集（KITTI、nuScenes、Waymo Open、Argoverse 2）
- 机器人操作数据集（YCB-Video、Open X-Embodiment、RH20T）
- 导航与 SLAM 数据集（TUM RGB-D、EuRoC MAV、NCLT）
- 强化学习数据集（D4RL、ManiSkill2）
- 数据集管理工具（Roboflow、FiftyOne、HuggingFace Datasets）

**Step 2: 提交**

```bash
git add docs/database/dataset.md
git commit -m "docs: massively expand dataset page (database/dataset.md)"
```

---

### Task 8: Spot 四足机器人档案 `database/spot.md`

**Files:**
- Create: `docs/database/spot.md`

核心章节：
- 引言
- 发展历程（从 BigDog 到 Spot 的演进）
- 技术规格（尺寸/重量/运动/传感器/电池/IP 等级）
- 软件平台（Spot SDK：Python API、有效负载、自主导航）
- 行业应用（工业巡检、建筑测量、危险环境、国防）
- 与 ROS/ROS 2 集成（`spot_ros` 包）
- 衍生产品（Spot Arm、Spot Enterprise）

**Step 2: 提交**

```bash
git add docs/database/spot.md
git commit -m "docs: add Boston Dynamics Spot profile (database/spot.md)"
```

---

### Task 9: Shell 脚本与机器人自动化 `linux/shell-scripting.md`

**Files:**
- Create: `docs/linux/shell-scripting.md`

核心章节：
- 引言（机器人启动自动化的必要性）
- Bash 核心语法（变量、条件、循环、函数、数组）
- 机器人常用脚本模式
  - 自动启动 ROS / ROS 2 节点
  - 环境检测与依赖校验（检查 ROS 环境、串口设备）
  - 日志轮转与数据记录
- systemd 服务化机器人程序（.service 文件编写）
- udev 规则固定串口名（解决 /dev/ttyUSB0 漂移问题）
- 实用脚本示例（开机自启、进程守护、自动重连）
- 参考资料

**Step 2: 提交**

```bash
git add docs/linux/shell-scripting.md
git commit -m "docs: add shell scripting for robotics article (linux/shell-scripting.md)"
```

---

### Task 10: 更新 `mkdocs.yml` 导航

**Files:**
- Modify: `mkdocs.yml`

**Step 1: 在 `nav:` 中做以下修改**

```yaml
# 视觉部分，在 object-detection.md 后添加：
- cv/segmentation.md
- cv/tracking.md
- cv/pose-estimation.md
- cv/3d-vision.md

# 感知部分，在 slam.md 后添加：
- sensing/sensor-fusion.md

# 学习部分，在 rl.md 后添加：
- learning/dl.md

# Linux 部分，在 commands.md 后添加：
- linux/shell-scripting.md

# 数据库 > 机器人图鉴，在 unitree-h1.md 后添加：
- database/spot.md
```

**Step 2: 本地验证构建**

```bash
mkdocs build --strict 2>&1 | tail -20
```

预期：无 WARNING，BUILD 成功

**Step 3: 提交**

```bash
git add mkdocs.yml
git commit -m "docs: update navigation for 9 new articles"
```

---

## 实施说明

- **并行执行**：Task 1-9 互相独立，可由 9 个子 agent 同时执行
- **串行依赖**：Task 10（更新导航）必须在所有文章文件创建后执行
- **验证**：每篇文章写完后用 `wc -l` 确认行数在预期范围内
- **格式校验**：`mkdocs build --strict` 作为最终验收标准
