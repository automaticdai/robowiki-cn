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

## Ultralytics YOLOv8 实战代码

Ultralytics 提供了统一的 Python 接口，极大降低了 YOLOv8 的使用门槛。以下代码覆盖从推理到训练的完整流程。

### 安装

```bash
pip install ultralytics
```

### 单张图像与视频推理

```python
from ultralytics import YOLO
import cv2

# 1. 加载预训练模型
# n=nano, s=small, m=medium, l=large, x=xlarge
model = YOLO('yolov8n.pt')

# 2. 单张图像推理
results = model('path/to/image.jpg')

# 3. 解析并可视化结果
for result in results:
    boxes = result.boxes.xyxy   # 边界框坐标 [x1, y1, x2, y2]
    scores = result.boxes.conf  # 置信度分数
    classes = result.boxes.cls  # 类别索引

    # 读取原始图像用于绘制
    frame = cv2.imread('path/to/image.jpg')
    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = map(int, box)
        label = f"{model.names[int(cls)]}: {score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Result', frame)
    cv2.waitKey(0)

# 4. 视频实时检测（摄像头）
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # stream=True 启用流式生成器，降低内存占用
    results = model(frame, stream=True, conf=0.5)
    for result in results:
        annotated = result.plot()  # 自动绘制边界框与标签
    cv2.imshow('YOLOv8 Detection', annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 自定义数据集训练

训练前需准备 YAML 格式的数据集配置文件 `custom_dataset.yaml`，内容示例如下：

```yaml
# custom_dataset.yaml
path: /data/robot_parts   # 数据集根目录
train: images/train       # 训练集图像路径（相对于 path）
val: images/val           # 验证集图像路径

nc: 3                     # 类别数量
names: ['bolt', 'nut', 'washer']  # 类别名称列表
```

然后启动训练：

```python
from ultralytics import YOLO

# 5. 自定义数据集训练
model = YOLO('yolov8n.pt')  # 从预训练权重微调
model.train(
    data='custom_dataset.yaml',  # 数据集配置文件
    epochs=100,
    imgsz=640,
    batch=16,
    device='cuda',          # 或 'cpu'
    project='robot_detection',
    name='exp1'
)

# 训练完成后评估
metrics = model.val()
print(f"mAP50: {metrics.box.map50:.3f}")
print(f"mAP50-95: {metrics.box.map:.3f}")
```

### 其他任务支持

YOLOv8 通过统一接口支持多种任务，只需切换模型文件后缀：

| 任务 | 模型后缀 | 说明 |
|------|---------|------|
| 目标检测 | `yolov8n.pt` | 输出边界框 |
| 实例分割 | `yolov8n-seg.pt` | 输出掩码 |
| 姿态估计 | `yolov8n-pose.pt` | 输出关键点 |
| 图像分类 | `yolov8n-cls.pt` | 输出类别概率 |

## DETR 端到端检测

### 核心思想

**DETR** (Detection Transformer，端到端目标检测 Transformer) 由 Facebook Research 于 2020 年提出，是首个完全去除 NMS 和手工设计 Anchor 的检测框架。其核心设计包括：

1. **二分图匹配损失 (Bipartite Matching Loss)**：训练时通过匈牙利算法（Hungarian Algorithm）在预测框集合与真实框集合之间寻找最优一对一匹配，从根本上消除了重复预测问题，因此推理时不需要 NMS；
2. **Transformer 编码器-解码器架构**：编码器处理 CNN 提取的图像特征序列，解码器以一组可学习的目标查询（Object Queries）为输入，通过交叉注意力与图像特征交互，直接输出固定数量的预测结果；
3. **并行预测**：所有预测同时生成，而非逐步迭代，符合集合预测的思想。

**DETR 的局限性**：收敛速度慢（需要训练 500 epoch 才能与 Faster R-CNN 媲美），对小目标检测效果较差。

### DETR 的后继者

**Deformable DETR**（2021）引入可变形注意力机制（Deformable Attention），每个查询只关注参考点周围少量采样点，将训练 epoch 从 500 降至 50，并改善了小目标性能。

**DINO**（2022，ICLR 2023 最佳论文候选）在 Deformable DETR 基础上引入对比去噪训练（Contrastive DeNoising Training）和混合查询选择（Mixed Query Selection），在 COCO 上达到 63.3 mAP，大幅超越同期单阶段检测器。

**RT-DETR**（Real-Time DETR，2023）由百度提出，针对实时性优化：

- 使用高效混合编码器（Efficient Hybrid Encoder），将 Transformer 编码器与 CNN 解耦，显著降低计算量；
- 支持灵活调整解码器查询数量，在速度与精度之间灵活权衡；
- 在 T4 GPU 上以 114 FPS 达到 54.8 mAP，首次实现 YOLO 级速度与 DETR 级精度的统一。

## 在机器人上的部署

### NVIDIA Jetson 上的 TensorRT 优化

NVIDIA Jetson 系列（Orin、Xavier、Nano）是机器人常用的边缘计算平台。TensorRT（张量运行时）通过层融合、精度校准等技术大幅提升推理速度。Ultralytics 原生支持导出 TensorRT 引擎：

```python
from ultralytics import YOLO

# 导出为 TensorRT 格式（需在目标 Jetson 设备上执行）
model = YOLO('yolov8n.pt')
model.export(
    format='engine',
    device=0,       # GPU 设备编号
    half=True,      # 启用 FP16 半精度优化
    imgsz=640,
    workspace=4     # TensorRT 构建时最大工作空间（GiB）
)

# 加载并使用 TensorRT 引擎推理
trt_model = YOLO('yolov8n.engine')
results = trt_model('image.jpg')
```

在 Jetson Orin NX 上，YOLOv8n 经 FP16 优化后推理速度可从原始 PyTorch 的约 30 FPS 提升至 100 FPS 以上。

### 与 ROS 2 集成

在机器人系统中，目标检测结果通常需要通过 ROS 2（机器人操作系统 2）发布给其他节点（如路径规划、机械臂控制）。标准消息类型为 `vision_msgs/Detection2DArray`。以下是一个简化的 ROS 2 检测节点示例：

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D
from cv_bridge import CvBridge
from ultralytics import YOLO


class YoloDetectorNode(Node):
    def __init__(self):
        super().__init__('yolo_detector')
        self.model = YOLO('yolov8n.engine')  # 使用 TensorRT 引擎
        self.bridge = CvBridge()

        # 订阅相机图像话题
        self.sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)

        # 发布检测结果话题
        self.pub = self.create_publisher(
            Detection2DArray, '/detections', 10)

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        results = self.model(frame, conf=0.5)

        det_array = Detection2DArray()
        det_array.header = msg.header

        for result in results:
            for box, score, cls in zip(
                    result.boxes.xyxy,
                    result.boxes.conf,
                    result.boxes.cls):
                det = Detection2D()
                x1, y1, x2, y2 = box.tolist()
                det.bbox.center.position.x = (x1 + x2) / 2
                det.bbox.center.position.y = (y1 + y2) / 2
                det.bbox.size_x = x2 - x1
                det.bbox.size_y = y2 - y1
                det_array.detections.append(det)

        self.pub.publish(det_array)


def main():
    rclpy.init()
    node = YoloDetectorNode()
    rclpy.spin(node)
    rclpy.shutdown()
```

启动节点：

```bash
ros2 run your_package yolo_detector
```

## 三维目标检测 (3D Object Detection)

### 应用背景

二维目标检测输出图像平面上的边界框，而机器人操作、自动驾驶等场景需要知道物体在三维空间中的位置、尺寸和朝向。三维目标检测输出的是三维边界框，通常表示为 \((x, y, z, l, w, h, \theta)\)，其中 \(\theta\) 为偏航角（yaw）。

### 基于激光雷达的方法

激光雷达（LiDAR）直接获取三维点云，天然适合三维检测任务：

- **VoxelNet**（2018）：将点云体素化（Voxelization）后逐体素提取特征，再送入 RPN 网络预测三维框；
- **PointPillars**（2019）：将点云按垂直列（Pillar）组织，使用简化的 PointNet 提取 Pillar 特征，展平为伪图像后使用 2D CNN 检测；推理速度快，工程实用性强；
- **CenterPoint**（2021）：将三维检测转化为鸟瞰图（BEV, Bird's Eye View）上的中心点检测，类似 2D CenterNet，支持多类别和速度预测，是自动驾驶领域的主流方案之一。

### 基于相机的方法

相机成本低、信息丰富，但深度信息需要从单目或多目图像中恢复：

- **FCOS3D**（2021）：在 FCOS 框架上扩展，直接从单目图像回归三维边界框的中心、尺寸和朝向；
- **ImVoxelNet**（2022）：将多视角图像特征反投影到三维体素网格中，在三维空间直接检测。

### 多模态融合方法

融合激光雷达点云与相机图像可以兼顾深度精度和语义丰富性：

- **BEVFusion**（2022，MIT 与 Horizon Robotics 分别提出）：将激光雷达点云特征和相机图像特征统一投影到鸟瞰图（BEV）空间后进行融合，再在 BEV 空间执行检测和分割。BEV 空间融合避免了透视畸变，便于多传感器对齐。

鸟瞰图（BEV）的概念：将三维场景从正上方俯视投影到二维平面，x 轴为左右，y 轴为前后，物体在 BEV 图中呈现俯视轮廓。自动驾驶中 BEV 感知已成为主流范式，因为下游的规划模块也在 BEV 坐标系中工作。

### 机器人拣选中的三维检测

在仓储机器人和工业机械臂场景中，三维检测用于确定待抓取物体的六自由度（6-DoF）位姿，包括位置 \((x, y, z)\) 和旋转（滚转 roll、俯仰 pitch、偏航 yaw）。常用流程：

1. 使用深度相机（RGB-D，如 Intel RealSense、Azure Kinect）获取彩色图像和深度图；
2. 用 YOLOv8 等检测器在 RGB 图中定位目标，获得二维边界框；
3. 将边界框内的深度像素转换为三维点云；
4. 对点云拟合三维边界框或使用 ICP（迭代最近点）配准 CAD 模型，得到精确位姿；
5. 将位姿发布给机械臂运动规划模块执行抓取。

## 开放词汇目标检测 (Open-Vocabulary Detection)

### 零样本检测

传统检测模型只能检测训练集中出现过的固定类别集合。**开放词汇目标检测**（Open-Vocabulary Detection，OVD）旨在使模型能够检测任意由自然语言描述的类别，无需针对新类别收集标注数据重新训练，即**零样本检测**（Zero-Shot Detection）。

这一能力对机器人尤为重要：操作人员可以用自然语言指令描述目标（"拿起红色杯子"），机器人无需预先知道"红色杯子"的训练样本即可定位并抓取。

### 基于 CLIP 的检测器

**CLIP**（Contrastive Language-Image Pre-training）通过大规模图像-文本对比学习，使视觉编码器和文本编码器嵌入到同一语义空间中。基于 CLIP 的检测器利用这一对齐的嵌入空间实现开放词汇识别：

- **OWL-ViT**（Open-World Localization with Vision Transformers，Google，2022）：将 CLIP 的 ViT 图像编码器与轻量级检测头结合，支持用文本或参考图像 patch 描述目标类别，实现零样本检测；
- **GroundingDINO**（2023）：将 DINO 检测器与 BERT 文本编码器结合，通过跨模态注意力实现文本短语与图像区域的精确对应，支持任意短语定位（Phrase Grounding）。与 SAM（Segment Anything Model）组合使用，可实现"文本描述 → 分割掩码"的完整流程。

### 机器人语言引导操作示例

```python
# 使用 GroundingDINO 实现语言引导目标定位（伪代码示意）
from groundingdino.util.inference import load_model, predict
import cv2

model = load_model("groundingdino_swint_ogc.pth")
image = cv2.imread("scene.jpg")

# 用自然语言描述目标
text_prompt = "red cup . blue screwdriver . metal bolt"

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=text_prompt,
    box_threshold=0.35,
    text_threshold=0.25
)
# boxes 中包含与每个短语对应的边界框
# 后续可将框内点云送入位姿估计模块
```

## 工业缺陷检测应用

### 典型场景

工业视觉检测（Industrial Visual Inspection）是目标检测最重要的落地场景之一：

- **PCB 缺陷检测**：检测印刷电路板上的短路、断路、针孔、毛刺等缺陷；
- **焊缝质量检测**：识别焊缝中的气孔、裂纹、夹渣等缺陷；
- **瓶盖/包装检测**：检测食品饮料生产线上的瓶盖破损、标签缺失等问题；
- **纺织品缺陷检测**：检测布料的跳线、污渍、破洞。

### 类别不平衡问题

工业缺陷数据集的核心挑战是**严重的类别不平衡**：正常样本数量远多于缺陷样本（比例可达 1000:1 乃至更高）。应对策略：

1. **Focal Loss**：降低易分类样本的损失权重，使模型聚焦于难分类的缺陷样本；
2. **过采样/欠采样**：对缺陷样本重复采样（过采样），或对正常样本随机丢弃（欠采样）；
3. **数据增强**：对少量缺陷样本进行旋转、翻转、亮度变换等增强，扩充数据量；
4. **合成数据**：使用 GAN（生成对抗网络）或扩散模型生成逼真的缺陷图像，填充样本不足。

### 异常检测方法

当缺陷样本极度稀少（甚至没有）时，可采用**异常检测**（Anomaly Detection）思路：只用正常样本训练模型学习正常分布，测试时偏离分布的区域被判定为异常：

- **PatchCore**（2022）：提取正常图像的局部 Patch 特征构建记忆库（Memory Bank），推理时计算测试 Patch 与记忆库的最近邻距离作为异常分数；无需任何缺陷样本，在 MVTec AD 数据集上达到 99.1% AUROC；
- **CutPaste**（2021）：通过随机裁剪并粘贴图像局部来自监督地生成"伪缺陷"，训练分类器区分正常和伪缺陷，从而学习正常特征表示。

### MVTec AD 数据集

工业异常检测领域最常用的基准数据集，包含 15 个类别（螺栓、皮革、瓷砖等），提供像素级异常掩码标注，常用 AUROC 和 PRO（Per-Region Overlap）指标评估。

## 标注工具与数据集准备

### 标注工具

高质量标注数据是检测模型性能的基础。常用开源/商业标注工具：

- **LabelImg**：轻量级开源工具，支持 PASCAL VOC（XML）和 YOLO（TXT）格式，适合小团队快速标注；
- **CVAT**（Computer Vision Annotation Tool）：Intel 开源的 Web 端标注平台，支持图像、视频、3D 点云标注，支持团队协作和半自动标注；
- **Roboflow**：云端数据集管理和标注平台，支持数据版本管理、增强预览和一键导出多种格式，对小型项目免费；
- **Label Studio**：开源的通用标注工具，支持图像、音频、文本等多模态数据。

### 数据集格式

不同框架使用不同的标注格式，迁移时注意转换：

**COCO 格式（JSON）**：所有标注存储在单个 JSON 文件中，包含 `images`、`annotations`、`categories` 三个列表。适合 Detectron2、MMDetection 等框架。

```json
{
  "images": [{"id": 1, "file_name": "img001.jpg", "width": 640, "height": 480}],
  "annotations": [
    {"id": 1, "image_id": 1, "category_id": 1,
     "bbox": [100, 150, 80, 60], "area": 4800, "iscrowd": 0}
  ],
  "categories": [{"id": 1, "name": "bolt"}]
}
```

**YOLO 格式（TXT）**：每张图像对应一个同名 `.txt` 文件，每行代表一个目标，格式为 `class_id cx cy w h`，坐标均归一化到 0-1。

```
# img001.txt
0 0.328125 0.395833 0.125000 0.125000
1 0.671875 0.604167 0.093750 0.083333
```

### 数据增强策略

合理的数据增强可显著提升模型泛化能力，尤其对小数据集效果明显：

- **几何变换**：随机水平/垂直翻转、随机旋转（±15°）、随机裁剪与缩放；
- **颜色变换**：亮度、对比度、饱和度、色调随机抖动（Color Jitter），随机灰度化；
- **混合增强**：
    - **Mosaic**（YOLOv4 提出）：将 4 张图像拼接为一张，丰富背景多样性，增强小目标检测；
    - **MixUp**：两张图像以随机权重线性插值，标签也按比例混合；
    - **CutMix**：将一张图像的随机矩形区域替换为另一张图像的对应区域；
- **针对工业检测的增强**：随机添加高斯噪声（模拟相机噪声）、随机运动模糊（模拟传送带运动）、随机透视变换（模拟相机角度偏差）。

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
