# 目标检测：实战代码与工程部署

!!! note "引言"
    本页面是[目标检测](object-detection.md)专题的工程实践部分，涵盖 Ultralytics YOLOv8 完整使用流程、DETR 系列端到端检测器的核心原理与代码、目标检测模型在机器人平台（NVIDIA Jetson / ROS 2）上的部署方法，以及数据标注与数据集管理工具的选型指南。


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

**典型 Jetson 平台推理性能参考（YOLOv8n，640×640 输入）：**

| 平台 | 功耗 | FP32 FPS | FP16 FPS | INT8 FPS |
|------|------|---------|---------|---------|
| Jetson Nano | 10W | ~8 | ~15 | ~25 |
| Jetson Xavier NX | 15W | ~45 | ~90 | ~130 |
| Jetson Orin NX 8G | 15W | ~30 | ~100 | ~160 |
| Jetson AGX Orin | 40W | ~80 | ~200 | ~350 |

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

### 典型机器人部署工作流

```
相机 → ROS 2 话题 (/camera/image_raw)
  ↓
YoloDetectorNode (GPU 推理)
  ↓
/detections 话题 (Detection2DArray)
  ↓
┌─────────────────┬──────────────────┐
│  路径规划节点    │   机械臂控制节点  │
│ (动态障碍物感知) │  (目标定位抓取)   │
└─────────────────┴──────────────────┘
```


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

### 数据集管理平台

| 工具 | 主要功能 | 特点 |
|------|----------|------|
| Roboflow | 版本控制、格式转换、在线标注、数据增强 | 一站式数据管道，对小项目免费 |
| FiftyOne | 数据集可视化、质量分析、标注错误排查 | 嵌入空间可视化，与 HuggingFace 集成 |
| Hugging Face Datasets | 机器人与视觉数据集托管 | LeRobot Dataset 标准格式，支持流式加载 |
| DVC | 大文件数据集版本管理 | 与 Git 协同工作 |


## 参考资料

1. Jocher, G., et al. (2023). Ultralytics YOLOv8. [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
2. Carion, N., et al. (2020). End-to-End Object Detection with Transformers. *ECCV*.
3. Zhu, X., et al. (2020). Deformable DETR: Deformable Transformers for End-to-End Object Detection. *ICLR 2021*.
4. Zhang, H., et al. (2022). DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection. *ICLR 2023*.
5. Zhao, Y., et al. (2023). DETRs Beat YOLOs on Real-time Object Detection. *CVPR 2024*.
6. [NVIDIA TensorRT 文档](https://docs.nvidia.com/deeplearning/tensorrt/)
7. [CVAT 项目主页](https://github.com/cvat-ai/cvat)
8. [Roboflow 平台](https://roboflow.com/)
