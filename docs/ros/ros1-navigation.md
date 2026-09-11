# ROS 1 Navigation Stack

!!! note "引言"
    ROS Navigation Stack（导航栈）是 ROS 1 中移动机器人自主导航的标准框架，提供从原始传感器数据到底盘速度指令的完整处理流程。其核心是 move_base 节点：它维护全局与局部两层代价地图（Costmap），调用全局规划器生成从当前位置到目标点的路径，再由局部规划器结合实时障碍物生成速度指令，同时依赖 AMCL 在已知地图中完成粒子滤波定位。本页面介绍导航栈的组成、各模块的参数配置与调试要点。


## ROS Navigation Stack

ROS Navigation Stack（导航栈）是ROS 1中用于移动机器人自主导航的完整软件框架，提供从原始传感器数据到速度指令的完整处理流程。

### 整体架构

```
传感器输入                    导航栈核心                    执行输出
────────────                ───────────────                ────────────
/scan          ──────►  ┌─────────────────────┐
/odom          ──────►  │       move_base      │  ──────►  /cmd_vel
/map           ──────►  │                      │
               ──────►  │  ┌───────────────┐  │
/initialpose   ──────►  │  │     AMCL      │  │
                         │  │  (定位节点)    │  │
/tf (odom→     ──────►  │  └───────────────┘  │
   base_link)            │                      │
                         │  ┌───────────────┐  │
                         │  │  全局代价地图  │  │
                         │  │ (global_cost- │  │
                         │  │    map)       │  │
                         │  └───────────────┘  │
                         │                      │
                         │  ┌───────────────┐  │
                         │  │  局部代价地图  │  │
                         │  │ (local_cost-  │  │
                         │  │    map)       │  │
                         │  └───────────────┘  │
                         │                      │
                         │  ┌───────────────┐  │
                         │  │  全局路径规划  │  │
                         │  │  (Dijkstra/A*)│  │
                         │  └───────────────┘  │
                         │                      │
                         │  ┌───────────────┐  │
                         │  │  局部路径规划  │  │
                         │  │    (DWA)      │  │
                         │  └───────────────┘  │
                         └─────────────────────┘
```

`move_base` 是导航栈的核心节点，订阅`/move_base_simple/goal`（或通过actionlib接收导航目标），协调全局规划、局部规划和代价地图，输出速度指令`/cmd_vel`。

### 全局路径规划：Dijkstra 与 A*

全局规划器（`global_planner`）在已知的静态全局地图上规划从当前位置到目标的完整路径。ROS Navigation Stack提供两种经典算法：

**Dijkstra算法**（默认）：从起点出发，逐步扩展最低代价节点，保证找到最短路径，但计算代价较高（时间复杂度 \(O(V \log V)\)，\(V\) 为节点数）。

**A*算法**：在Dijkstra基础上引入启发函数（heuristic）加速搜索。常用欧氏距离启发函数：

$$h(n) = \sqrt{(x_n - x_{goal})^2 + (y_n - y_{goal})^2}$$

A*通过优先扩展估计总代价 \(f(n) = g(n) + h(n)\) 最小的节点来引导搜索方向，其中 \(g(n)\) 为从起点到当前节点的实际代价。在启发函数可接受（不高估实际代价）时，A*同样能保证最优路径，且通常比Dijkstra快得多。

在`move_base`中选择全局规划器：

```yaml
# global_planner_params.yaml
base_global_planner: "navfn/NavfnROS"   # 使用Dijkstra
# 或
base_global_planner: "global_planner/GlobalPlanner"  # 可配置A*或Dijkstra
```

### 局部路径规划：动态窗口法（DWA）

局部规划器（默认使用`dwa_local_planner`）在局部地图上实时规避动态障碍物，生成平滑可执行的速度指令。

DWA（Dynamic Window Approach，动态窗口法）的核心思想是：在机器人当前可达的速度空间中采样速度指令 \((v, \omega)\)（线速度和角速度），通过综合代价函数评分，选择最优速度对执行。

DWA代价函数为：

$$J(v, \omega) = \alpha \cdot \text{heading}(v, \omega) + \beta \cdot \text{dist}(v, \omega) + \gamma \cdot \text{velocity}(v, \omega)$$

其中：

- \(\text{heading}(v, \omega)\)：航向代价，衡量轨迹终点朝向与目标方向的偏差，偏差越小代价越低
- \(\text{dist}(v, \omega)\)：障碍物距离代价，衡量轨迹与最近障碍物的距离，距离越大代价越低
- \(\text{velocity}(v, \omega)\)：速度代价，鼓励机器人保持较高前进速度以提高效率
- \(\alpha, \beta, \gamma\)：各项代价的权重系数，通过参数文件调整

DWA在每个控制周期（通常50~100 ms）内重新采样和评估，能够快速响应环境变化，适合动态场景。

DWA关键参数（`dwa_local_planner_params.yaml`）：

```yaml
DWAPlannerROS:
  # 速度和加速度限制
  max_vel_x: 0.5          # 最大线速度 (m/s)
  min_vel_x: 0.0
  max_vel_theta: 1.0      # 最大角速度 (rad/s)
  min_vel_theta: -1.0
  acc_lim_x: 2.5          # 线加速度上限 (m/s²)
  acc_lim_theta: 3.2      # 角加速度上限 (rad/s²)

  # 代价函数权重
  path_distance_bias: 32.0    # 跟随全局路径的权重（对应heading）
  goal_distance_bias: 24.0    # 朝向目标的权重
  occdist_scale: 0.01         # 障碍物距离权重（对应dist）

  # 前向仿真参数
  sim_time: 1.7               # 前向仿真时间 (s)
  vx_samples: 3               # 线速度采样数
  vtheta_samples: 20          # 角速度采样数
```


### 代价地图（Costmaps）

代价地图将环境中的障碍物信息转换为机器人可以使用的代价值（0~254，0表示自由空间，254表示致命障碍），用于路径规划时的代价计算。

**全局代价地图（global_costmap）**：基于已知的静态地图构建，覆盖整个任务环境，用于全局路径规划。一般不频繁更新。

**局部代价地图（local_costmap）**：以机器人为中心的滑动窗口地图（如4×4米），融合实时传感器数据（激光雷达、超声波等）频繁更新，用于局部避障。

两种代价地图共享相同的配置框架，主要区别在于范围和更新频率：

```yaml
# global_costmap_params.yaml
global_costmap:
  global_frame: map            # 参考坐标系
  robot_base_frame: base_link
  update_frequency: 1.0        # 更新频率 (Hz)，全局地图更新较慢
  publish_frequency: 0.5
  static_map: true             # 基于静态地图初始化
  rolling_window: false        # 不使用滑动窗口
  inflation_radius: 0.55       # 障碍物膨胀半径 (m)
  cost_scaling_factor: 10.0   # 膨胀代价衰减系数

# local_costmap_params.yaml
local_costmap:
  global_frame: odom
  robot_base_frame: base_link
  update_frequency: 5.0        # 实时更新
  publish_frequency: 2.0
  static_map: false
  rolling_window: true         # 使用以机器人为中心的滑动窗口
  width: 4.0                   # 局部地图宽度 (m)
  height: 4.0                  # 局部地图高度 (m)
  resolution: 0.05             # 地图分辨率 (m/格)
  inflation_radius: 0.55
```


### AMCL：自适应蒙特卡洛定位

AMCL（Adaptive Monte Carlo Localization，自适应蒙特卡洛定位）是 ROS Navigation Stack 中的标准定位算法，基于粒子滤波 (Particle Filter) 实现在已知地图中的概率性定位。

**基本原理**：AMCL维护一组粒子（Particles），每个粒子代表机器人可能的位置和姿态 \((x, y, \theta)\) 以及对应的权重。算法分三个步骤循环执行：

1. **预测步骤（Motion Model）**：根据里程计数据 \(\Delta x, \Delta y, \Delta\theta\) 更新每个粒子的位置，加入运动噪声以表示里程计不确定性
2. **更新步骤（Sensor Model）**：利用激光雷达扫描数据计算每个粒子的似然权重 \(w_i \propto p(\mathbf{z} | \mathbf{x}_i, \mathbf{m})\)，即在粒子所表示的位置上观测到当前激光数据的概率
3. **重采样步骤（Resampling）**：按权重重采样粒子集，权重高的粒子被多次复制，低权重粒子被淘汰，使粒子集向高概率区域聚集

**自适应粒子数**：AMCL通过KLD采样（Kullback-Leibler Divergence Sampling）动态调整粒子数量：定位不确定性高时增加粒子数（最多数千个）；定位收敛后减少粒子数（最少数十个），节省计算资源。

机器人位置估计由粒子加权均值给出：

$$\hat{\mathbf{x}} = \sum_{i=1}^{N} w_i \mathbf{x}_i$$

AMCL的协方差矩阵表示定位的不确定性，发布于`/amcl_pose`话题（`geometry_msgs/PoseWithCovarianceStamped`类型）。

AMCL关键参数：

```yaml
# amcl_params.yaml
amcl:
  # 粒子数范围
  min_particles: 500
  max_particles: 2000

  # 运动模型噪声参数（差速驱动模型）
  odom_model_type: diff          # 差速驱动里程计模型
  odom_alpha1: 0.2               # 旋转运动引起的旋转噪声
  odom_alpha2: 0.2               # 平移运动引起的旋转噪声
  odom_alpha3: 0.8               # 平移运动引起的平移噪声
  odom_alpha4: 0.2               # 旋转运动引起的平移噪声

  # 激光传感器模型参数
  laser_model_type: likelihood_field  # 似然场模型
  laser_max_range: 12.0
  laser_min_range: 0.1
  laser_max_beams: 60            # 每次更新使用的激光束数量

  # 坐标系
  odom_frame_id: odom
  base_frame_id: base_link
  global_frame_id: map

  # 初始位姿不确定性（在/initialpose未发布前）
  initial_pose_x: 0.0
  initial_pose_y: 0.0
  initial_pose_a: 0.0            # 初始偏航角
```


## 参考资料

1. ROS Wiki, [navigation](http://wiki.ros.org/navigation)
2. ROS Wiki, [move_base](http://wiki.ros.org/move_base)
3. ROS Wiki, [amcl](http://wiki.ros.org/amcl)
4. E. Marder-Eppstein et al., "The Office Marathon: Robust Navigation in an Indoor Office Environment," *ICRA*, 2010.
5. [导航概述](../planning/navigation.md)
6. [代价地图](../planning/navigation-costmap.md)
7. [局部规划器](../planning/navigation-local-planners.md)
