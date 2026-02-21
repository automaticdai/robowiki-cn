# 目标检测：三维检测与高级应用

!!! note "引言"
    本页面是[目标检测](object-detection.md)专题的进阶内容，涵盖三维目标检测（基于激光雷达、相机和多模态融合）、开放词汇目标检测（零样本检测）以及工业缺陷检测应用。这些方向是目标检测技术与机器人感知深度融合的前沿领域。


## 三维目标检测 (3D Object Detection)

### 应用背景

二维目标检测输出图像平面上的边界框，而机器人操作、自动驾驶等场景需要知道物体在三维空间中的位置、尺寸和朝向。三维目标检测输出的是三维边界框，通常表示为 \((x, y, z, l, w, h, \theta)\)，其中 \(\theta\) 为偏航角（yaw）。

| 输出维度 | 2D 检测 | 3D 检测 |
|---------|---------|---------|
| 位置 | \((x_{\min}, y_{\min}, x_{\max}, y_{\max})\) | \((x_c, y_c, z_c)\) 中心坐标 |
| 尺寸 | 宽 \(w\)、高 \(h\) | 长 \(l\)、宽 \(w\)、高 \(h\) |
| 方向 | — | 偏航角 \(\theta\)（可选 roll/pitch） |
| 传感器 | RGB 相机 | LiDAR、RGB-D、多模态融合 |

### 基于激光雷达的方法

激光雷达（LiDAR）直接获取三维点云，天然适合三维检测任务：

**VoxelNet**（2018）：将点云体素化（Voxelization）后逐体素提取特征，再送入 RPN 网络预测三维框。将无序点云划分为规则体素网格，每个非空体素用 VFE（Voxel Feature Encoding）层提取特征，然后通过 3D 卷积网络处理整体体素特征图。

**PointPillars**（2019）：将点云按垂直列（Pillar）组织，使用简化的 PointNet 提取 Pillar 特征，展平为伪图像后使用 2D CNN 检测。相比 VoxelNet 无需 3D 卷积，推理速度快（可达 62 FPS），工程实用性强，是自动驾驶工业部署中应用最广泛的方法之一。

**CenterPoint**（2021）：将三维检测转化为鸟瞰图（BEV, Bird's Eye View）上的中心点检测，类似 2D CenterNet。

- 预测阶段：在 BEV 特征图上预测目标中心热图、尺寸、方向和速度
- 细化阶段：以中心点为基准提取关键点特征，进一步精化边界框
- 支持多类别和速度预测，是自动驾驶领域的主流方案之一

```python
# 使用 OpenPCDet 框架进行点云三维检测推理示例
import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu

cfg_from_yaml_file('cfgs/kitti_models/pointpillar.yaml', cfg)

model = build_network(model_cfg=cfg.MODEL,
                      num_class=len(cfg.CLASS_NAMES),
                      dataset=dataset)
model.load_params_from_file(filename='pointpillar_7728.pth', to_cpu=False)
model.cuda()
model.eval()

with torch.no_grad():
    load_data_to_gpu(batch_dict)
    pred_dicts, _ = model.forward(batch_dict)
    # pred_dicts 包含 pred_boxes, pred_scores, pred_labels
```

### 基于相机的方法

相机成本低、信息丰富，但深度信息需要从单目或多目图像中恢复：

**FCOS3D**（2021）：在 FCOS 框架上扩展，直接从单目图像回归三维边界框的中心、尺寸和朝向。通过学习目标中心点在图像中的投影位置以及相对深度，结合相机内参恢复三维位置。适合无法配置 LiDAR 的低成本机器人场景。

**ImVoxelNet**（2022）：将多视角图像特征反投影到三维体素网格中，在三维空间直接检测。充分利用多视角的几何约束，无需 LiDAR 即可实现较好的三维检测效果。

**BEVDepth**（2022）：引入显式的深度监督（Depth Supervision），使网络在 Lift-Splat 框架中更精确地估计每个像素的深度概率分布，从而更准确地将图像特征展开到 BEV 空间。

### 多模态融合方法

融合激光雷达点云与相机图像可以兼顾深度精度和语义丰富性：

**BEVFusion**（2022，MIT 与 Horizon Robotics 分别提出）：将激光雷达点云特征和相机图像特征统一投影到鸟瞰图（BEV）空间后进行融合，再在 BEV 空间执行检测和分割。

- **相机分支**：Lift-Splat-Shoot 范式将图像特征提升到 BEV 空间
- **LiDAR 分支**：PointPillars 或 VoxelNet 提取 BEV 特征
- **融合层**：通道拼接 + 卷积融合两路 BEV 特征
- **BEV 空间融合避免了透视畸变**，便于多传感器对齐

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

**OWL-ViT**（Open-World Localization with Vision Transformers，Google，2022）：将 CLIP 的 ViT 图像编码器与轻量级检测头结合，支持用文本或参考图像 patch 描述目标类别，实现零样本检测；既可用文本描述（"a yellow cup"）也可用参考图像（one-shot）定位目标。

**GroundingDINO**（2023）：将 DINO 检测器与 BERT 文本编码器结合，通过跨模态注意力实现文本短语与图像区域的精确对应，支持任意短语定位（Phrase Grounding）。与 SAM（Segment Anything Model）组合使用，可实现"文本描述 → 分割掩码"的完整流程。

**OWLv2**（2023）：OWL-ViT 的升级版，采用自训练（Self-Training）策略在大规模检测数据上迭代训练，在零样本检测基准上显著超越 OWL-ViT，同时保持开放词汇能力。

### 机器人语言引导操作示例

```python
# 使用 GroundingDINO 实现语言引导目标定位
from groundingdino.util.inference import load_model, predict
import cv2

model = load_model("groundingdino_swint_ogc.pth")
image = cv2.imread("scene.jpg")

# 用自然语言描述目标（支持多目标，用 ". " 分隔）
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
print(f"检测到 {len(boxes)} 个目标：{phrases}")
```

结合 SAM（Segment Anything Model）实现从文本到分割掩码的完整流程：

```python
# GroundingDINO + SAM 完整管线（伪代码）
from segment_anything import SamPredictor, sam_model_registry

# 1. GroundingDINO 定位目标框
boxes, _, _ = predict(model, image, text_prompt="red cup", ...)

# 2. SAM 生成精细掩码
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
predictor = SamPredictor(sam)
predictor.set_image(image)

for box in boxes:
    masks, _, _ = predictor.predict(box=box.numpy(), multimask_output=False)
    # masks 为像素级二值掩码，可用于精确位姿估计
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

**PatchCore**（2022）：提取正常图像的局部 Patch 特征构建记忆库（Memory Bank），推理时计算测试 Patch 与记忆库的最近邻距离作为异常分数；无需任何缺陷样本，在 MVTec AD 数据集上达到 99.1% AUROC。

```python
# PatchCore 推理流程（简化示意）
import torch
import torchvision.models as models
from sklearn.neighbors import NearestNeighbors

# 1. 提取正常训练样本特征
backbone = models.wide_resnet50_2(pretrained=True)
# ... 逐层提取中间特征

# 2. 构建记忆库（使用贪婪核心集近似减小库大小）
memory_bank = coreset_reduction(all_features, ratio=0.1)
nbrs = NearestNeighbors(n_neighbors=1).fit(memory_bank)

# 3. 推理：计算测试样本最近邻距离
def infer(test_img):
    features = extract_features(test_img)
    distances, _ = nbrs.kneighbors(features)
    anomaly_score = distances.max()  # 最大距离作为异常分数
    return anomaly_score
```

**CutPaste**（2021）：通过随机裁剪并粘贴图像局部来自监督地生成"伪缺陷"，训练分类器区分正常和伪缺陷，从而学习正常特征表示。

**SimpleNet**（2023）：在 PatchCore 基础上引入可学习的特征适配层和简单的判别网络，进一步提升异常检测精度并降低推理延迟，适合工业实时部署。

### MVTec AD 数据集

工业异常检测领域最常用的基准数据集，由 MVTec Software GmbH 发布，包含 15 个类别（螺栓、皮革、瓷砖等工业物品），提供像素级异常掩码标注，常用 AUROC 和 PRO（Per-Region Overlap）指标评估。

| 类别 | 正常训练图数 | 缺陷类型数 | 测试图数 |
|------|-----------|-----------|---------|
| 螺栓 (Bolt) | 209 | 3 | 114 |
| 瓶子 (Bottle) | 209 | 3 | 83 |
| 皮革 (Leather) | 245 | 5 | 92 |
| 木材 (Wood) | 247 | 5 | 60 |
| ... | ... | ... | ... |
| **合计** | **3629** | **73** | **1725** |

**主流方法在 MVTec AD 上的 AUROC（图像级）对比：**

| 方法 | 年份 | AUROC (平均) | 特点 |
|------|------|------------|------|
| GaussianAD | 2020 | 86.8% | 特征空间高斯建模 |
| CutPaste | 2021 | 96.1% | 自监督伪缺陷生成 |
| PatchCore | 2022 | 99.1% | 记忆库 + 最近邻 |
| SimpleNet | 2023 | 99.6% | 可学习判别网络 |
| UniFormaly | 2023 | 99.8% | 统一异常检测框架 |


## 参考资料

1. Lang, A. H., et al. (2019). PointPillars: Fast Encoders for Object Detection from Point Clouds. *CVPR*.
2. Yin, T., et al. (2021). Center-based 3D Object Detection and Tracking. *CVPR*.
3. Liu, Z., et al. (2022). BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation. *ICRA 2023*.
4. Minderer, M., et al. (2022). Simple Open-Vocabulary Object Detection with Vision Transformers. *ECCV*.
5. Liu, S., et al. (2023). Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection. *arXiv*.
6. Roth, K., et al. (2022). Towards Total Recall in Industrial Anomaly Detection. *CVPR*.
7. Li, C. L., et al. (2023). SimpleNet: A Simple Network for Image Anomaly Detection and Localization. *CVPR*.
8. [MVTec Anomaly Detection Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad), MVTec Software GmbH.
9. [OpenPCDet: Open-source 3D Object Detection Toolbox](https://github.com/open-mmlab/OpenPCDet)
10. [GroundingDINO GitHub](https://github.com/IDEA-Research/GroundingDINO)
