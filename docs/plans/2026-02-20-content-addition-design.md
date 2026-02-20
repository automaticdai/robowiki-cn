# 云飞机器人中文维基 — 内容扩充设计文档

**日期**：2026-02-20
**作者**：Claude Code（AI 辅助）
**状态**：待确认

---

## 背景

当前 wiki 在以下方向存在明显内容缺口：

1. **cv/**：`index.md` 引用了图像分割、目标跟踪、位姿估计、三维视觉，但对应页面不存在
2. **sensing/**：缺少传感器融合（Sensor Fusion）专项页面
3. **learning/**：缺少深度学习在机器人中的应用专项页面
4. **database/dataset.md**：仅 14 行，内容极度单薄
5. **database/**：仅有 H1 等少数人形机器人详细档案，缺少四足机器人（Spot）
6. **linux/**：缺少 Shell 脚本/机器人自动化等实用内容

---

## 目标

新增 **8 篇文章**，大幅扩充 **1 篇**，同步更新 `mkdocs.yml` 导航。

---

## 新增文章清单

### 1. `cv/segmentation.md` — 图像分割

**主题**：语义分割、实例分割、全景分割在机器人视觉中的应用

**章节结构**：
- 引言（为什么机器人需要图像分割）
- 分割任务类型（语义 / 实例 / 全景）
- 经典算法（FCN、U-Net、DeepLab v3+）
- 实例分割（Mask R-CNN、SOLOv2）
- 全景分割（Panoptic-DeepLab）
- 轻量级实时分割（BiSeNet、PP-LiteSeg）
- 与 ROS 集成（`image_segmentation` 话题）
- 评测指标（mIoU、Dice、Boundary F1）
- 数据集（Cityscapes、ADE20K、COCO Panoptic）
- 参考资料

**规模**：~300 行 Markdown（含公式、代码块、表格）

---

### 2. `cv/tracking.md` — 目标跟踪

**主题**：视频序列中单目标与多目标的持续定位

**章节结构**：
- 引言（跟踪在机器人中的作用）
- 单目标跟踪（SOT）
  - 相关滤波（MOSSE、KCF、CSRT）
  - 孪生网络（SiamFC、SiamRPN、SiamMask）
  - Transformer 跟踪器（OSTrack、DropTrack）
- 多目标跟踪（MOT）
  - 检测-关联框架（SORT、DeepSORT、ByteTrack）
  - 端到端 MOT（MOTR、TrackFormer）
- 评测指标（MOTA、MOTP、IDF1、HOTA）
- 机器人应用（人员跟随、目标交互）
- 参考资料

**规模**：~280 行

---

### 3. `cv/pose-estimation.md` — 位姿估计

**主题**：物体 6DoF 位姿估计 + 人体姿态估计

**章节结构**：
- 引言
- 物体 6DoF 位姿估计
  - 问题定义（旋转 SO(3) + 平移 R³）
  - 基于 RGB 的方法（PoseCNN、DPOD、FoundPose）
  - 基于 RGB-D 的方法（DenseFusion、FFB6D）
  - 无 CAD 模型的方法（FoundPose、Gen6D）
  - 机器人抓取应用
- 人体姿态估计
  - 2D 姿态估计（HRNet、ViTPose）
  - 3D 姿态估计（VideoPose3D）
  - 机器人协作与人机交互应用
- 评测指标（ADD、ADD-S、PCK、MPJPE）
- 常用数据集（YCB-Video、LineMOD、COCO Keypoints）
- 参考资料

**规模**：~320 行

---

### 4. `cv/3d-vision.md` — 三维视觉

**主题**：从 2D 恢复三维结构，点云处理

**章节结构**：
- 引言
- 双目立体视觉（Stereo Vision）
  - 立体标定与校正
  - 立体匹配（BM、SGM、RAFT-Stereo）
  - 视差图转深度图
- 结构光与 ToF
- 运动恢复结构（Structure from Motion, SfM）
  - 特征匹配 → 相机位姿估计 → 稀疏重建
  - 软件工具（COLMAP、OpenMVG）
- 点云处理
  - 坐标系与格式（PCD、PLY、LAS）
  - 处理算法（滤波、下采样、法线估计、配准 ICP）
  - 深度学习（PointNet、PointNet++、VoxelNet）
- 神经辐射场（NeRF）与 3D 高斯泼溅
- 常用工具（Open3D、PCL、CloudCompare）
- 参考资料

**规模**：~340 行

---

### 5. `sensing/sensor-fusion.md` — 传感器融合

**主题**：多传感器数据融合理论与实践

**章节结构**：
- 引言（为什么需要传感器融合）
- 融合架构（低层 / 特征层 / 决策层）
- 概率框架
  - 贝叶斯估计
  - 卡尔曼滤波（KF）
  - 扩展卡尔曼滤波（EKF）
  - 无迹卡尔曼滤波（UKF）
  - 粒子滤波（PF）
- 典型融合场景
  - IMU + GPS（惯性导航）
  - LiDAR + 相机（3D 目标检测）
  - 视觉 + IMU（Visual-Inertial Odometry, VIO）
- 时间同步与空间标定
- ROS 中的传感器融合工具（`robot_localization`、`kalibr`）
- 参考资料

**规模**：~300 行（含公式推导）

---

### 6. `learning/dl.md` — 深度学习在机器人中的应用

**主题**：神经网络基础 + 机器人专属深度学习方法

**章节结构**：
- 引言
- 神经网络基础回顾（MLP、CNN、RNN/LSTM、Transformer）
- 机器人中的核心应用
  - 端到端学习（End-to-End Learning）
  - 模仿学习（Imitation Learning / Behavioral Cloning）
  - 基础模型在机器人中的应用（RT-2、OpenVLA）
- 深度学习与经典控制的融合
- 部署优化（量化、剪枝、TensorRT、ONNX）
- 常用框架与工具（PyTorch、JAX、LeRobot）
- 参考资料

**规模**：~280 行

---

### 7. `database/dataset.md` — 机器人与CV数据集大全（大幅扩充）

**现状**：14 行，几乎为空
**目标**：~300 行，系统整理各类数据集

**章节结构**：
- 引言
- 计算机视觉数据集
  - 目标检测（ImageNet、COCO、Pascal VOC、Objects365）
  - 语义分割（Cityscapes、ADE20K、KITTI Semantic）
  - 人体姿态（COCO Keypoints、MPII、Human3.6M）
- 自动驾驶数据集
  - KITTI、nuScenes、Waymo Open Dataset、Argoverse
- 机器人操作数据集
  - YCB-Video、OCID、Open X-Embodiment、RH20T
- 导航与 SLAM 数据集
  - TUM RGB-D、EuRoC MAV、NCLT
- 强化学习与仿真数据集
  - D4RL、ManiSkill
- 数据集搜索与管理工具
- 参考资料

---

### 8. `database/spot.md` — Boston Dynamics Spot 档案

**主题**：Spot 四足机器人详细档案

**章节结构**：
- 发展历程
- 技术规格（尺寸/重量/运动能力/传感器/电池）
- 软件平台（Spot SDK、Spot Core）
- 行业应用（工业巡检、建筑测量、危险环境）
- 与 ROS 集成
- 参考资料

**规模**：~130 行

---

### 9. `linux/shell-scripting.md` — Shell 脚本与机器人自动化

**主题**：机器人开发中常用的 Shell 脚本技术

**章节结构**：
- 引言（机器人启动自动化的意义）
- Bash 基础语法（变量、条件、循环、函数）
- 机器人常用脚本模式
  - 自动启动 ROS 节点
  - 环境检测与依赖校验
  - 日志轮转与数据备份
- systemd 服务化机器人程序
- 串口/网络设备自动配置（udev 规则）
- 实用脚本示例（开机自启、定时任务、进程守护）
- 参考资料

**规模**：~200 行

---

## mkdocs.yml 导航更新

```yaml
# 视觉部分新增
- 视觉:
  - cv/index.md
  - cv/object-detection.md
  - cv/segmentation.md      # 新增
  - cv/tracking.md          # 新增
  - cv/pose-estimation.md   # 新增
  - cv/3d-vision.md         # 新增

# 感知部分新增
- 感知:
  - sensing/index.md
  - sensing/sensors.md
  - sensing/depth-camera.md
  - sensing/slam.md
  - sensing/sensor-fusion.md  # 新增

# 学习部分新增
- 学习:
  - learning/ml.md
  - learning/rl.md
  - learning/dl.md            # 新增

# Linux部分新增
- Linux:
  - linux/index.md
  - linux/commands.md
  - linux/shell-scripting.md  # 新增

# 数据库机器人图鉴新增
- 机器人图鉴:
  - database/robots.md
  - database/nao.md
  - database/atlas.md
  - database/optimus.md
  - database/figure.md
  - database/asimo.md
  - database/digit.md
  - database/unitree-h1.md
  - database/spot.md          # 新增
```

---

## 写作规范

所有新文章遵循现有 wiki 规范：

- 文件名：英文小写连字符
- 首节：`!!! note "引言"` admonition
- 数学公式：行内 `\(...\)`，行间 `$$ ... $$`
- 标题：最多 4 级
- 末节：`## 参考资料`，尾有空行
- 语言：中文正式用语，首次出现缩写须标注英文全称

---

## 实施方式

**并行多智能体**：同时启动 9 个子 agent，各自独立写作一篇文章，最后统一更新 `mkdocs.yml`。

预计完成后新增约 **2400 行**高质量内容。
