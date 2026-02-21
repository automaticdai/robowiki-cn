# 目标检测 (Object Detection)

!!! note "引言"
    目标检测是计算机视觉中的核心任务之一，旨在识别图像中物体的位置（定位）和类别（分类）。随着深度学习的发展，目标检测算法经历了从传统方法到深度学习的转变，并在精度和速度方面取得了显著提升。

## 概述

目标检测任务需要同时完成两个子任务：

1. **定位 (Localization)**：确定物体在图像中的位置（通常用边界框表示）
2. **分类 (Classification)**：识别物体的类别

## 分类体系

### 按检测流程分类

#### 两阶段检测器 (Two-Stage Detectors)
先产生候选区域（Region Proposals），再对候选区域进行分类和回归。

**特点**：

- 精度高，但速度较慢
- 适合对精度要求高的场景

**代表算法**：

- **R-CNN** (Region-based CNN)
    - 开创性地将 CNN 应用于目标检测
    - 使用选择性搜索生成候选区域
- **Fast R-CNN**
    - 引入 ROI Pooling，共享卷积特征
    - 端到端训练，速度提升
- **Faster R-CNN**
    - 引入 RPN (Region Proposal Network) 生成候选区域
    - 实现真正的端到端检测
- **Mask R-CNN**
    - 在 Faster R-CNN 基础上增加实例分割分支
    - 同时完成检测和分割任务
- **Light-Head R-CNN**
    - 轻量级头部设计，提升速度
- **Cascade R-CNN**
    - 级联检测器，逐步提高检测质量
- **R-FCN** (Region-based Fully Convolutional Networks)
    - 全卷积设计，位置敏感得分图
- **FPN** (Feature Pyramid Network)
    - 特征金字塔网络，多尺度特征融合

#### 单阶段检测器 (One-Stage Detectors)
直接预测物体的类别和位置，无需候选区域生成。

**特点**：

- 速度快，适合实时应用
- 精度通常略低于两阶段方法

**代表算法**：

- **YOLO 系列** (You Only Look Once)
    - **YOLOv1**：首个单阶段实时检测器
    - **YOLOv2 (YOLO9000)**：引入 Anchor、多尺度训练
    - **YOLOv3**：多尺度预测、更好的小目标检测
    - **YOLOv4**：引入 CSP、PAN、Mosaic 数据增强
    - **YOLOv5**：工程化改进，易于部署
    - **YOLOv6/YOLOv7/YOLOv8**：持续优化版本
- **SSD** (Single Shot MultiBox Detector)
    - 多尺度特征图检测
    - 平衡速度和精度
    - **变体**：
        - **DSSD**：引入反卷积模块
        - **FSSD**：特征融合 SSD
        - **ESSD**：增强型 SSD
        - **MDSSD**：多方向 SSD
        - **Fire SSD**：轻量级设计
- **RetinaNet**
    - 引入 Focal Loss 解决类别不平衡问题
    - 单阶段检测器达到两阶段精度
- **CornerNet**
    - Anchor-free 方法，检测角点对
- **CenterNet**
    - 检测物体中心点和尺寸
- **FCOS** (Fully Convolutional One-Stage)
    - 像素级预测，无需 Anchor

### 按 Anchor 机制分类

#### Anchor-based 方法
使用预定义的 Anchor 框进行检测。

**代表算法**：

- Faster R-CNN
- YOLOv2/v3
- SSD
- RetinaNet

#### Anchor-free 方法
不依赖预定义的 Anchor，直接预测关键点或中心点。

**代表算法**：

- CornerNet
- CenterNet
- FCOS
- DETR (Detection Transformer)

### 按监督方式分类

#### 全监督学习 (Fully Supervised Learning)
使用完整的标注数据（边界框和类别）进行训练。

**代表算法**：

- 大多数主流检测算法（R-CNN、YOLO、SSD 等）

#### 弱监督学习 (Weakly Supervised Object Detection)
仅使用图像级标签（无边界框标注）进行训练。

**代表算法**：

- WSDDN
- OICR
- PCL
- CASD

#### 少样本学习 (Few-Shot Learning)
- **零样本检测 (ZSD, Zero-Shot Object Detection)**：检测训练时未见过的类别
- **单样本检测 (OSD, One-Shot Object Detection)**：仅使用一个样本进行检测

### 按应用场景分类

#### 实时检测
注重速度和实时性，适合视频流处理。

**代表算法**：

- YOLO 系列
- SSD
- MobileNet-SSD
- Pelee

#### 高精度检测
注重检测精度，适合离线处理。

**代表算法**：

- Faster R-CNN
- Cascade R-CNN
- Mask R-CNN

#### 3D 目标检测
检测三维空间中的物体。

**方法**：

- 基于点云的方法（PointRCNN、VoxelNet）
- 基于多视图的方法
- 基于 RGB-D 的方法

#### 小目标检测
专门针对小尺寸物体的检测。

**方法**：

- 多尺度特征融合
- 特征金字塔网络
- 高分辨率特征图

#### 密集场景检测
处理物体密集、遮挡严重的场景。

**方法**：

- NMS 改进（Soft-NMS、Softer-NMS）
- 后处理优化

### 按网络架构分类

#### 基于 CNN 的方法
使用卷积神经网络提取特征。

**代表算法**：

- 大多数传统检测算法

#### 基于 Transformer 的方法
使用 Transformer 架构进行检测。

**代表算法**：

- **DETR** (Detection Transformer)
    - 端到端检测，无需 NMS
    - 使用 Transformer 编码器-解码器
- **Deformable DETR**
    - 可变形注意力机制
- **Swin Transformer**
    - 分层 Transformer 用于检测

#### 轻量级网络
针对移动端和边缘设备优化。

**代表算法**：

- MobileNet-SSD
- Pelee
- YOLOv5s/v5n
- NanoDet

### 特殊技术和方法

#### 特征提取与融合
- **SPP-Net** (Spatial Pyramid Pooling)
    - 空间金字塔池化，处理不同尺寸输入
- **FPN** (Feature Pyramid Network)
    - 特征金字塔，多尺度特征融合
- **PANet** (Path Aggregation Network)
    - 路径聚合网络
- **BiFPN** (Bidirectional Feature Pyramid Network)
    - 双向特征金字塔

#### 后处理技术
- **NMS** (Non-Maximum Suppression)
    - 非极大值抑制，去除重复检测
- **Soft-NMS**
    - 软非极大值抑制
- **Softer-NMS**
    - 更柔和的 NMS
- **DIoU-NMS**
    - 基于距离的 NMS

#### 数据增强
- Mosaic 数据增强
- MixUp
- CutMix
- 自动增强策略

#### 损失函数改进
- **Focal Loss**：解决类别不平衡
- **IoU Loss**：直接优化 IoU
- **GIoU Loss**：广义 IoU 损失
- **DIoU/CIoU Loss**：考虑距离和形状的损失

## 性能评估指标

### 主要指标
- **mAP** (mean Average Precision)：平均精度均值
- **IoU** (Intersection over Union)：交并比
- **FPS** (Frames Per Second)：每秒处理帧数
- **FLOPs**：浮点运算次数
- **参数量**：模型参数数量

### 数据集
- **COCO**：Microsoft Common Objects in Context
- **PASCAL VOC**：Visual Object Classes
- **ImageNet**：大规模图像数据集
- **Open Images**：Google 开源数据集

## YOLO 系列版本对比

近年来 YOLO 系列持续迭代，从 YOLOv5 到 YOLOv10 以及基于 Transformer 的 RT-DETR，在 COCO 数据集上的表现持续提升。下表汇总了主流版本的关键参数（数据来自各自论文，输入分辨率为 640×640，推理设备为 V100 GPU）：

| 版本 | 年份 | 主要改进 | mAP (COCO) | FPS |
|------|------|---------|-----------|-----|
| YOLOv5 | 2020 | PyTorch 重写，模块化设计 | 50.7 | 140 |
| YOLOv7 | 2022 | ELAN，辅助训练头 | 55.9 | 161 |
| YOLOv8 | 2023 | 无锚框，Ultralytics API | 53.9 | 128 |
| YOLOv9 | 2024 | GELAN，PGI 可编程梯度信息 | 55.6 | — |
| YOLOv10 | 2024 | 无 NMS 架构，双重分配训练 | 54.4 | — |
| RT-DETR | 2023 | 端到端 Transformer | 54.8 | 114 |

**各版本关键技术说明**：

- **YOLOv7**：提出 E-ELAN（扩展高效层聚合网络），并使用辅助训练头（Auxiliary Head）在训练时提供额外监督，推理时去除；
- **YOLOv8**：由 Ultralytics 团队发布，统一了检测、分割、姿态估计等任务，使用无锚框（Anchor-free）预测头；
- **YOLOv9**：引入 GELAN（广义高效层聚合网络）和 PGI（可编程梯度信息），缓解深层网络中的信息瓶颈问题；
- **YOLOv10**：清华大学提出，采用双重分配（Dual Assignments）训练策略，推理时完全去除 NMS，实现真正的端到端推理；
- **RT-DETR**：百度提出的实时 DETR，结合 Transformer 与高效编码器，在实时速度下达到 DETR 级精度。

## 专题子页面

本专题包含以下子页面，涵盖从代码实战到前沿应用的完整内容：

| 页面 | 内容摘要 |
|------|---------|
| [实战代码与工程部署](object-detection-practical.md) | YOLOv8 完整使用流程、DETR 系列端到端检测器、机器人平台（Jetson / ROS 2）部署、数据标注与数据集管理 |
| [三维检测与高级应用](object-detection-3d-advanced.md) | 3D 目标检测（LiDAR / 相机 / 多模态融合）、开放词汇检测（GroundingDINO / OWL-ViT）、工业缺陷检测与异常检测 |

## 发展趋势

### 当前热点
1. **Transformer 在检测中的应用**
    - DETR 及其变体
    - Vision Transformer 用于检测
2. **端到端检测**
    - 去除 NMS 等后处理步骤
    - 更简洁的检测流程
3. **实时高精度检测**
    - 平衡速度和精度
    - 模型压缩与加速
4. **多模态检测**
    - 结合 RGB、深度、点云等信息
    - 跨模态学习
5. **自监督/半监督学习**
    - 减少标注需求
    - 利用无标注数据

### 未来方向
- 更高效的网络架构
- 更好的小目标检测
- 实时 3D 检测
- 视频目标检测
- 开放词汇检测

## 算法选择指南

### 精度优先
- **推荐**：Faster R-CNN、Cascade R-CNN、Mask R-CNN
- **适用场景**：离线处理、对精度要求高的应用

### 速度优先
- **推荐**：YOLOv5/v8、SSD、MobileNet-SSD
- **适用场景**：实时视频处理、移动端应用

### 平衡精度和速度
- **推荐**：RetinaNet、YOLOv4、FCOS
- **适用场景**：大多数实际应用

### 特殊需求
- **小目标检测**：FPN、PANet、高分辨率输入
- **3D 检测**：PointRCNN、VoxelNet、CenterPoint、BEVFusion
- **轻量级部署**：Pelee、NanoDet、YOLOv5n
- **开放词汇/零样本**：GroundingDINO、OWL-ViT
- **工业缺陷（少样本）**：PatchCore、CutPaste 异常检测方法

## 参考资料

1. [Awesome Object Detection](https://github.com/amusi/awesome-object-detection), GitHub
2. Redmon, J., et al. (2016). You Only Look Once: Unified, Real-Time Object Detection. *CVPR*.
3. Ren, S., et al. (2015). Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. *NIPS*.
4. Lin, T. Y., et al. (2017). Focal Loss for Dense Object Detection. *ICCV*.
5. Carion, N., et al. (2020). End-to-End Object Detection with Transformers. *ECCV*.
6. Jocher, G., et al. (2023). Ultralytics YOLOv8. [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
7. Wang, C. Y., et al. (2022). YOLOv7: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors. *CVPR 2023*.
8. Wang, A., et al. (2024). YOLOv10: Real-Time End-to-End Object Detection. *NeurIPS 2024*.
9. Zhao, Y., et al. (2023). DETRs Beat YOLOs on Real-time Object Detection. *CVPR 2024*.
10. Zhu, X., et al. (2020). Deformable DETR: Deformable Transformers for End-to-End Object Detection. *ICLR 2021*.
11. Zhang, H., et al. (2022). DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection. *ICLR 2023*.
12. Lang, A. H., et al. (2019). PointPillars: Fast Encoders for Object Detection from Point Clouds. *CVPR*.
13. Yin, T., et al. (2021). Center-based 3D Object Detection and Tracking. *CVPR*.
14. Liu, Z., et al. (2022). BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation. *ICRA 2023*.
15. Minderer, M., et al. (2022). Simple Open-Vocabulary Object Detection with Vision Transformers. *ECCV*.
16. Liu, S., et al. (2023). Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection. *arXiv*.
17. Roth, K., et al. (2022). Towards Total Recall in Industrial Anomaly Detection. *CVPR*.
18. [MVTec Anomaly Detection Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad), MVTec Software GmbH.
