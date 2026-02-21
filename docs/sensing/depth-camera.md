# 深度视觉相机

!!! note "引言"
    深度相机 (Depth Camera) 能够同时获取彩色图像和每个像素的深度信息，是机器人三维感知的重要工具。本页面介绍主流深度相机的技术原理、产品系列及其在机器人中的应用。


## 深度感知技术

深度相机根据获取深度信息的方式，主要分为以下三种技术路线：

- **结构光 (Structured Light)**：向场景投射已知的光学图案（如红外散斑或条纹），通过分析图案的变形程度计算深度。代表产品有 Microsoft Kinect v1 和 Intel RealSense D405。
- **飞行时间 (Time-of-Flight, ToF)**：发射调制的红外光，通过测量光往返时间或相位差来计算距离。代表产品有 Microsoft Kinect v2。ToF 的深度精度与距离大致成正比，误差随距离增加而增大。
- **主动红外双目 (Active IR Stereo)**：利用红外投射器增强纹理，通过双目立体匹配算法计算视差和深度。代表产品有 Intel RealSense D435 和 ZED 系列。这种方案在纹理丰富的场景中表现良好。


## 工作原理详解

### 结构光法（Structured Light）

结构光法的工作流程如下：红外投影仪向场景投射已知的编码图案（通常为散斑或条纹阵列），图案照射到三维物体表面后发生形变，由红外图像传感器捕获变形后的图案，再通过三角几何关系解算每个像素点的深度值。

散斑结构光（如初代 Kinect 和 RealSense D405）将一块随机散斑图案投射到场景中，通过模板匹配计算散斑偏移量，进而换算深度。这种方式对处理器算力要求适中，适合室内短距离（0.1 m～5 m）应用。

代表产品：Intel RealSense D415、D435，Orbbec Astra 系列，Microsoft Kinect v1。

**优势**：近距离精度高，计算较简单。

**局限**：强环境光（尤其是直射阳光中的红外成分）会干扰投影图案，导致室外使用效果差；多台结构光相机同时工作时会产生相互干扰。


### 飞行时间法（Time of Flight，ToF）

飞行时间法通过测量红外光脉冲从发射到反射回传感器的往返时间来计算距离。根据光速 \(c \approx 3 \times 10^8\) m/s，距离公式为：

$$d = \frac{c \cdot \Delta t}{2}$$

其中 \(\Delta t\) 为光脉冲的往返飞行时间，除以 2 是因为光走了来回两段距离。

现代 ToF 相机多采用连续调制光（CWAM-ToF），通过测量发射光与接收光之间的相位差来推算距离，而非直接测量单次脉冲时间，从而实现更高精度的连续深度图输出。

代表产品：Microsoft Azure Kinect DK、Intel RealSense L515、Orbbec Femto Mega。

**优势**：帧率高（可达 60 fps），适合快速运动场景；不依赖环境纹理；相机结构紧凑。

**局限**：多径效应（Multi-path Interference）在反射强的场景（玻璃、镜面）中会引入误差；多台 ToF 相机同时工作时会发生干扰；测量范围通常受相位模糊距离限制。


### 双目立体视觉（Stereo Vision）

双目立体视觉使用两个水平间隔固定距离（基线，Baseline，\(B\)）的相机，通过计算同一物体在左右图像中的像素位移（视差，Disparity，\(d\)）来恢复深度。

深度与视差的关系为：

$$Z = \frac{f \cdot B}{d}$$

其中 \(f\) 为相机焦距（单位：像素），\(B\) 为两相机光心之间的基线距离（单位：米），\(d\) 为左右图像中匹配点的像素差值。

由公式可知，视差越大对应的深度越小（物体越近）；基线越长，相同深度处的视差越大，测量精度越高，但近距离盲区也随之增大。

代表产品：Stereolabs ZED 系列（被动双目）、Intel RealSense D435（主动红外双目，投影仪增强纹理以改善低纹理场景）。

**优势**：被动双目不依赖主动光源，不受日光干扰，可室外使用；测量范围可达 20 m 以上。

**局限**：低纹理（白墙、均匀地面）场景中视差匹配失败；立体匹配算法计算量大，需要 GPU 或专用芯片加速。


### 激光三角测距（Laser Triangulation）

激光三角测距适用于近距离高精度场景（毫米至厘米级）。激光发射器发出一条激光线投射到物体表面，高分辨率相机从侧面观察激光线的位置变化，通过三角几何关系计算表面形貌。

这种原理广泛应用于工业3D扫描仪（如 Sick Ranger、Keyence 系列）和机械臂末端的精密抓取传感器。测量范围通常在 10 mm 至 500 mm，精度可达 0.01 mm 量级。


## Microsoft Kinect

Microsoft Kinect 是深度相机的先驱产品，最初为游戏主机开发，后来被广泛应用于机器人研究。

### Kinect v1

- **深度技术**：结构光 (Structured Light)
- **深度范围**：0.8m ~ 4.0m
- **深度分辨率**：640 x 480
- **RGB分辨率**：640 x 480 @ 30fps
- **接口**：USB 2.0 + 专用电源适配器
- **ROS支持**：通过 `openni_camera` 或 `freenect_stack` 驱动
- **特点**：首款大规模商用深度相机，推动了深度视觉研究的发展。在近距离（<1.2m）和远距离（>3.5m）处精度较差。已停产。

### Kinect v2

- **深度技术**：飞行时间 (ToF)
- **深度范围**：0.5m ~ 4.5m
- **深度分辨率**：512 x 424
- **RGB分辨率**：1920 x 1080 @ 30fps
- **接口**：USB 3.0
- **ROS支持**：通过 `iai_kinect2` 驱动
- **特点**：相比 v1 深度精度和分辨率均有显著提升，多人骨骼追踪能力增强。功耗较高，已停产。

### Azure Kinect DK

- **深度技术**：飞行时间 (ToF)
- **深度范围**：窄视场模式 0.25m ~ 5.46m，宽视场模式 0.25m ~ 2.88m
- **深度分辨率**：最高 1024 x 1024
- **RGB分辨率**：3840 x 2160 @ 30fps
- **接口**：USB 3.0 Type-C
- **附加传感器**：7麦克风阵列、IMU
- **ROS支持**：通过 `azure_kinect_ros_driver` 驱动
- **特点**：Microsoft 最新一代深度相机，集成 AI 加速芯片，支持身体追踪 (Body Tracking) 和空间锚点 (Spatial Anchors)。


## ZED Camera

ZED 系列由 Stereolabs 公司生产，基于主动红外双目视觉技术，以长测距范围和 SLAM 功能著称。

### ZED

- **深度技术**：被动双目立体视觉 (Passive Stereo)
- **深度范围**：0.5m ~ 20m
- **RGB分辨率**：最高 2208 x 1242 @ 15fps，或 1344 x 376 @ 100fps
- **基线距离**：120mm
- **接口**：USB 3.0
- **特点**：初代产品，深度范围远，适合室外场景。无内置 IMU。

### ZED Mini

- **深度技术**：被动双目立体视觉
- **深度范围**：0.15m ~ 12m
- **基线距离**：63mm
- **接口**：USB 3.0
- **附加传感器**：内置 IMU
- **特点**：体积更紧凑，基线更短适合近距离应用。内置 IMU 支持视觉惯性里程计 (VIO)。适合无人机和 AR/VR 头显。

### ZED 2

- **深度技术**：被动双目立体视觉 + 神经网络深度估计
- **深度范围**：0.2m ~ 20m
- **RGB分辨率**：最高 2208 x 1242 @ 15fps
- **基线距离**：120mm
- **接口**：USB 3.0
- **附加传感器**：IMU、气压计、磁力计
- **ROS支持**：通过 `zed-ros-wrapper` 或 `zed-ros2-wrapper` 驱动
- **特点**：引入 AI 增强的深度估计，支持物体检测与骨骼追踪，IP66 防护等级。还集成了 GPS 接口，可用于室外大范围导航。
- **产品页面**：<https://www.stereolabs.com/zed-2/>

### ZED 2i

- **深度技术**：被动双目立体视觉 + 神经网络深度估计
- **深度范围**：0.2m ~ 20m
- **附加传感器**：IMU、气压计、磁力计
- **接口**：USB 3.0 / GMSL2
- **特点**：工业级设计，IP66 防护，支持 GMSL2 接口（适配 NVIDIA 平台）。适合自动驾驶和室外机器人。


## Intel RealSense

Intel RealSense 系列是目前应用最广泛的深度相机之一，产品线覆盖从近距离精密测量到远距离环境感知的多种需求。

### RealSense D405

- **深度技术**：主动红外立体视觉 (Active IR Stereo)
- **深度范围**：0.07m ~ 0.7m
- **深度分辨率**：1280 x 720
- **RGB分辨率**：1280 x 720
- **接口**：USB 3.2 Type-C
- **ROS支持**：通过 `realsense2_camera` 驱动
- **特点**：专为近距离精密应用设计，最小工作距离仅 7cm。适合机械臂末端抓取、精密装配。

### RealSense D415

- **深度技术**：主动红外立体视觉
- **深度范围**：0.16m ~ 10m
- **深度分辨率**：1280 x 720
- **视场角 (FOV)**：深度 65° x 40°
- **接口**：USB 3.2 Type-C
- **特点**：窄视场角、高精度，适合需要精确深度测量的场景。采用滚动快门 (Rolling Shutter)。

### RealSense D435 / D435i / D435f

- **深度技术**：主动红外立体视觉
- **深度范围**：0.105m ~ 10m
- **深度分辨率**：1280 x 720 @ 90fps
- **视场角 (FOV)**：深度 87° x 58°
- **接口**：USB 3.2 Type-C
- **ROS支持**：通过 `realsense2_camera` 驱动
- **D435i 附加**：内置 BMI055 IMU，支持视觉惯性里程计
- **D435f 附加**：工业级设计，IP65 防护
- **特点**：广视场角、全局快门 (Global Shutter)，适合运动场景。是 RealSense 系列中最受欢迎的型号。

### RealSense D455

- **深度技术**：主动红外立体视觉
- **深度范围**：0.6m ~ 6m（最优范围），最远可达约 10m
- **基线距离**：95mm（比 D435 更长，深度精度更高）
- **深度分辨率**：1280 x 720
- **视场角 (FOV)**：深度 87° x 58°
- **接口**：USB 3.2 Type-C
- **附加传感器**：内置 BMI055 IMU
- **ROS支持**：通过 `realsense2_camera` 驱动
- **特点**：更长的基线提供更精确的深度估计，适合需要较远距离深度感知的场景。


## 主流深度相机对比

| 型号 | 技术 | 测量范围 | 精度（典型） | 室外使用 | 价格区间（参考） | 典型应用 |
|------|------|----------|-------------|---------|----------------|---------|
| Intel RealSense D435i | 主动红外双目 + IMU | 0.1 ~ 10 m | ±2% @ 2m | 受限 | ¥1500～2500 | 移动机器人、SLAM |
| Microsoft Azure Kinect DK | ToF + 结构光 | 0.25 ~ 5.46 m | ±1.1 mm @ 1m | 受限 | ¥2500～4000 | 身体追踪、AR |
| Orbbec Astra+ | 结构光 | 0.4 ~ 8 m | ±2 mm @ 1m | 受限 | ¥800～1500 | 低成本室内导航 |
| Stereolabs ZED 2 | 被动双目 + AI | 0.2 ~ 20 m | ±1% @ 5m | 支持 | ¥3000～5000 | 室外机器人、无人机 |
| Orbbec Femto Mega | ToF（iToF） | 0.25 ~ 9 m | ±5 mm | 有限 | ¥2000～3500 | Azure Kinect替代 |
| Intel RealSense D405 | 主动红外双目 | 0.07 ~ 0.7 m | <1% @ 0.5m | 不适合 | ¥1200～2000 | 机械臂末端抓取 |

> 注：价格随市场变动，仅供参考。室外使用能力主要受太阳红外光干扰影响，被动双目相机受影响最小。


## 型号对比

| 特性 | Kinect v2 | ZED 2 | D435i | D455 | D405 |
|------|-----------|-------|-------|------|------|
| 深度技术 | ToF | 双目 | 主动IR双目 | 主动IR双目 | 主动IR双目 |
| 深度范围 | 0.5~4.5m | 0.2~20m | 0.1~10m | 0.6~6m | 0.07~0.7m |
| 内置IMU | 否 | 是 | 是 | 是 | 否 |
| ROS支持 | iai_kinect2 | zed-ros-wrapper | realsense2_camera | realsense2_camera | realsense2_camera |
| 适用场景 | 室内交互 | 室外/远距离 | 通用/移动机器人 | 中远距离 | 近距离精密 |

Intel RealSense 完整型号对比：<https://www.intelrealsense.com/compare-depth-cameras/>


## 相机标定（Camera Calibration）

深度相机在出厂时已进行内部标定，但在高精度应用中，用户需要重新标定以获得最佳精度。相机标定的目的是确定相机的内参（焦距、主点坐标、畸变系数）和外参（旋转矩阵、平移向量）。

### 针孔相机模型

标准针孔相机模型（Pinhole Camera Model）将三维空间点 \([X, Y, Z]^T\) 投影到图像平面 \([u, v]^T\) 的关系为：

$$s\begin{bmatrix}u\\v\\1\end{bmatrix} = K\begin{bmatrix}R \mid t\end{bmatrix}\begin{bmatrix}X\\Y\\Z\\1\end{bmatrix}$$

其中 \(s\) 为尺度因子，\(\begin{bmatrix}R \mid t\end{bmatrix}\) 为 \(3 \times 4\) 外参矩阵，\(K\) 为 \(3 \times 3\) 内参矩阵（Intrinsic Matrix）：

$$K = \begin{bmatrix}f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1\end{bmatrix}$$

各参数含义如下：

- \(f_x, f_y\)：水平和垂直方向的焦距（单位：像素）
- \(c_x, c_y\)：主点坐标（光轴与图像平面的交点，单位：像素）

实际镜头还存在畸变，常用的畸变模型包含径向畸变系数 \(k_1, k_2, k_3\) 和切向畸变系数 \(p_1, p_2\)。

### 使用 OpenCV 进行标定

最常用的标定方法是棋盘格标定法（Checkerboard Calibration）。标定流程如下：

1. 打印一张已知格子尺寸的棋盘格图案（例如 9×6 个内角点）。
2. 从多个角度（至少10～20张，覆盖不同位置和倾斜角）拍摄棋盘格图像。
3. 使用 OpenCV 自动检测角点并求解内参。

```python
import cv2
import numpy as np
import glob

# 棋盘格参数：内角点数量（列数, 行数）
CHECKERBOARD = (9, 6)
square_size = 0.025  # 每格边长，单位：米

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

obj_points = []  # 三维世界坐标
img_points = []  # 二维图像坐标

images = glob.glob('calibration_images/*.png')
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    if ret:
        obj_points.append(objp)
        corners_refined = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        img_points.append(corners_refined)

# 执行标定
ret, K, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, gray.shape[::-1], None, None
)

print("内参矩阵 K:\n", K)
print("畸变系数:", dist_coeffs)
print("重投影误差 (RMS):", ret)
```

重投影误差（Root Mean Square Reprojection Error）小于 0.5 像素通常被认为是良好的标定结果。


## ROS 2 深度相机集成

### realsense-ros2 驱动

Intel RealSense 相机有官方的 ROS 2 驱动包 `realsense-ros`，支持 Humble、Iron 和 Jazzy 等版本。

```bash
# 安装驱动（Ubuntu 22.04 + ROS 2 Humble）
sudo apt install ros-humble-realsense2-camera

# 启动相机节点（默认启用深度图和彩色图）
ros2 launch realsense2_camera rs_launch.py \
    enable_depth:=true \
    enable_color:=true \
    align_depth.enable:=true
```

### 主要话题（Topics）

启动后，相机节点会发布以下主要话题：

| 话题名称 | 消息类型 | 说明 |
|----------|----------|------|
| `/camera/depth/image_rect_raw` | `sensor_msgs/Image` | 原始深度图（16位，单位：毫米） |
| `/camera/color/image_raw` | `sensor_msgs/Image` | 彩色RGB图像 |
| `/camera/depth/camera_info` | `sensor_msgs/CameraInfo` | 相机内参 |
| `/camera/depth/color/points` | `sensor_msgs/PointCloud2` | 已对齐的彩色点云 |
| `/camera/imu` | `sensor_msgs/Imu` | IMU数据（D435i/D455专有） |

### 同步 RGB-D 数据

在需要同时使用彩色图和深度图时，应使用 `message_filters` 进行时间同步：

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import message_filters

class RGBDSubscriber(Node):
    def __init__(self):
        super().__init__('rgbd_subscriber')
        color_sub = message_filters.Subscriber(
            self, Image, '/camera/color/image_raw'
        )
        depth_sub = message_filters.Subscriber(
            self, Image, '/camera/depth/image_rect_raw'
        )
        # 时间同步：允许最大0.1秒时间差
        ts = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub], queue_size=10, slop=0.1
        )
        ts.registerCallback(self.callback)

    def callback(self, color_msg, depth_msg):
        self.get_logger().info('收到同步 RGB-D 帧')
        # 在此处理彩色图和深度图
```

### 深度图转点云

使用 `depth_image_proc` 功能包可以将深度图转换为点云，也可以将深度图叠加到彩色图上生成 XYZRGB 点云：

```bash
# 安装 depth_image_proc
sudo apt install ros-humble-depth-image-proc

# 通过 launch 文件启动点云转换节点
ros2 run depth_image_proc point_cloud_xyz_node \
    --ros-args -r image_rect:=/camera/depth/image_rect_raw \
               -r camera_info:=/camera/depth/camera_info
```


## 点云处理（Point Cloud Processing）

深度相机输出的深度图可以转换为三维点云（Point Cloud），进而进行各种三维处理任务。Open3D 是目前最流行的点云处理库之一。

### Open3D 基础操作

```python
import open3d as o3d
import numpy as np

# 读取点云文件
pcd = o3d.io.read_point_cloud("cloud.pcd")
print(f"点云包含 {len(pcd.points)} 个点")

# 体素下采样：将点云稀疏化以减少计算量
voxel_size = 0.02  # 体素边长 2cm
pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)

# 法向量估计：用于曲面重建和光照计算
pcd_down.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=voxel_size * 2, max_nn=30
    )
)

# 可视化
o3d.visualization.draw_geometries([pcd_down])
```

### ICP 点云配准

迭代最近点（Iterative Closest Point，ICP）算法用于将两帧点云对齐，常用于位姿估计和三维重建：

```python
# ICP 精配准
threshold = 0.02  # 最大对应点距离
reg_icp = o3d.pipelines.registration.registration_icp(
    source_pcd, target_pcd, threshold,
    np.eye(4),  # 初始变换矩阵
    o3d.pipelines.registration.TransformationEstimationPointToPoint()
)
print("ICP 变换矩阵:\n", reg_icp.transformation)
print("配准误差:", reg_icp.inlier_rmse)
```

### 平面分割与物体检测

使用随机采样一致性（RANSAC）算法分割点云中的平面（如地面、桌面）：

```python
# RANSAC 平面分割
plane_model, inliers = pcd_down.segment_plane(
    distance_threshold=0.01,  # 内点距离阈值 1cm
    ransac_n=3,
    num_iterations=1000
)
[a, b, c, d] = plane_model
print(f"平面方程: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

# 分离平面内点和异点
inlier_cloud = pcd_down.select_by_index(inliers)
outlier_cloud = pcd_down.select_by_index(inliers, invert=True)

# 对异点（非平面部分）进行聚类，检测物体
labels = np.array(outlier_cloud.cluster_dbscan(
    eps=0.05, min_points=10, print_progress=False
))
print(f"检测到 {labels.max() + 1} 个物体簇")
```


## 深度图像处理技巧

### 深度图空洞填充

原始深度图中常存在空洞（无效深度像素），原因包括：反射表面、透明物体、遮挡区域以及传感器测量盲区。常用的填充策略有：

- **空间滤波（Spatial Filter）**：对邻域有效深度像素取加权平均填充空洞
- **时间滤波（Temporal Filter）**：利用相邻帧的深度信息补全当前帧空洞
- **孔洞填充（Hole Filling）**：用周围最近有效像素填充，适合小尺寸空洞

使用 pyrealsense2 直接应用 RealSense 内置后处理滤波器：

```python
import pyrealsense2 as rs
import numpy as np

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

# 创建后处理滤波器
spatial_filter = rs.spatial_filter()      # 空间平滑
temporal_filter = rs.temporal_filter()    # 时间平滑
hole_filling = rs.hole_filling_filter()   # 孔洞填充
align = rs.align(rs.stream.color)         # 深度对齐到彩色

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)   # 深度图对齐到彩色图坐标系

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # 应用后处理滤波
        depth_frame = spatial_filter.process(depth_frame)
        depth_frame = temporal_filter.process(depth_frame)
        depth_frame = hole_filling.process(depth_frame)

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        # depth_image 单位为毫米，dtype 为 uint16
finally:
    pipeline.stop()
```

### 深度图与彩色图对齐

由于深度传感器和彩色相机的光轴不同，需要将深度图变换到彩色相机坐标系（或反之）才能进行像素级的 RGB-D 对应。上方代码中的 `rs.align(rs.stream.color)` 即完成此操作，将每个深度像素重新投影到彩色相机坐标系中。


## 机器人抓取应用

深度相机与点云处理结合，是机器人抓取（Grasping）应用的核心技术栈。典型的物体抓取流水线如下：

1. **深度图采集**：深度相机获取工作台的 RGB-D 图像
2. **点云生成与预处理**：深度图转点云，去除地面点，聚类得到目标物体点云
3. **物体位姿估计**：通过模板匹配或深度学习方法估计物体的6自由度位姿（3D位置 + 旋转）
4. **抓取点规划**：根据物体形状和位姿规划夹爪的抓取位置和姿态
5. **运动规划与执行**：将抓取位姿转换为机械臂关节轨迹并执行

### 主要开源工具

- **AnyGrasp**：基于深度学习的通用6自由度抓取检测网络，输入点云输出抓取候选集，鲁棒性强，适合无序堆叠（Bin Picking）场景
- **GraspNet-1Billion**：大规模抓取数据集及配套基准测试，提供多种抓取检测方法的参考实现
- **GPD（Grasp Pose Detection）**：经典的抓取位姿检测方法，基于点云局部特征评分

### 料箱拣选（Bin Picking）工作流

料箱拣选指从无序堆放的零件堆中识别并抓取单个零件，是工业机器人的常见应用场景：

1. 俯视深度相机获取料箱点云
2. 点云分割识别单个零件
3. 抓取检测网络输出抓取位姿
4. 碰撞检查过滤不可行抓取
5. 机械臂执行最优抓取


## 室外使用注意事项

在室外或强光环境下使用深度相机时，需要注意以下问题：

### 结构光相机的局限

结构光相机（如 RealSense D435、Orbbec Astra）的工作原理依赖于投射红外图案。在室外强烈日光下，太阳辐射中的红外成分会淹没投影图案，导致深度测量完全失效或噪声极大。因此，**结构光相机通常不适合在室外或强烈阳光直射环境中使用**。

### ToF 相机的多相机干扰

多台 ToF 相机在同一场景中工作时，会发生相互干扰：一台相机发出的调制光被另一台的传感器接收，产生幻影深度值。解决方法包括：

- 错开不同相机的调制频率（若硬件支持）
- 在时间上交替触发各相机（Time Multiplexing）
- 物理遮挡隔离各相机的视野

### 被动双目与激光雷达的互补

对于室外远距离感知任务，推荐使用被动双目相机（如 ZED 2）或与激光雷达（LiDAR）结合：

- 被动双目不依赖主动光源，不受日光干扰，测量范围可达 20 m
- 激光雷达（如 Velodyne、Livox）可在任何光照条件下工作，且测量范围更远（100 m 以上），但水平分辨率较低，无法提供稠密深度图
- RGB-D + LiDAR 融合可同时获得稠密彩色点云和远距离精确测距


## 选型建议

在为机器人选择深度相机时，可参考以下原则：

- **室内近距离操作（抓取、装配）**：选择最小工作距离小、精度高的型号，如 RealSense D405
- **室内移动机器人导航**：选择广视场角、支持 IMU 的型号，如 RealSense D435i
- **室外远距离感知**：选择深度范围远的型号，如 ZED 2 或 ZED 2i
- **人体追踪与交互**：选择骨骼追踪功能完善的型号，如 Azure Kinect DK
- **与 ROS 集成**：优先选择 ROS 驱动成熟、社区支持良好的产品
- **强化学习与仿真**：优先考虑有 Python SDK 且驱动稳定的型号，便于在 Gym 环境中集成


## 参考资料

1. Intel RealSense 官方文档：<https://dev.intelrealsense.com/docs>
2. Stereolabs ZED 文档：<https://www.stereolabs.com/docs>
3. Microsoft Azure Kinect 文档：<https://learn.microsoft.com/en-us/azure/kinect-dk/>
4. Khoshelham, K. & Elberink, S. O. (2012). Accuracy and resolution of Kinect depth data for indoor mapping applications. *Sensors*, 12(2), 1437-1454.
5. Open3D 官方文档：<http://www.open3d.org/docs/release/>
6. GraspNet-1Billion：<https://graspnet.net/>
7. OpenCV 相机标定教程：<https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html>
8. Fischler, M. A. & Bolles, R. C. (1981). Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography. *Communications of the ACM*, 24(6), 381-395.
