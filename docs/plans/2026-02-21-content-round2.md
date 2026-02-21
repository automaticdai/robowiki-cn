# Wiki Content Round 2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 大幅扩充 5 篇仿真文章 + 新增 2 篇控制理论 + 新增 3 篇机器人档案 + 扩充 2 篇数据库页面，共 12 项并行任务。

**Architecture:** 每项任务独立写入对应 Markdown 文件，遵循现有 wiki 规范。仿真文章在原文件上大幅扩充，控制文章需新建子目录，机器人档案新建文件，最后统一更新 mkdocs.yml。

**Tech Stack:** MkDocs + Material Theme, Markdown, MathJax (LaTeX)

---

### Task 1: 扩充 Gazebo 仿真文章
**Files:** Modify `docs/simulation/gazebo.md` (94→350+ lines)
- 新增：SDF/URDF 建模、world 文件结构、传感器插件（相机/激光雷达/IMU）、launch 文件、与 ROS 2 集成、`gz sim` 命令行、性能调优

### Task 2: 扩充 MuJoCo 仿真文章
**Files:** Modify `docs/simulation/mujoco.md` (93→350+ lines)
- 新增：MJCF XML 建模详解、关节/执行器/传感器配置、Python API（mujoco 包）、MJX GPU 加速、Gymnasium/IsaacGym 集成、强化学习工作流

### Task 3: 扩充 Webots 仿真文章
**Files:** Modify `docs/simulation/webots.md` (90→280+ lines)
- 新增：场景树编辑器、Python/C++ 控制器编写、传感器节点、ROS 2 webots_ros2 接口、多机器人场景

### Task 4: 扩充 Unity 仿真文章
**Files:** Modify `docs/simulation/unity.md` (90→250+ lines)
- 新增：ML-Agents 框架（训练流程、环境配置、Python API）、ROS# 集成、Unity Perception（合成数据生成）、sim-to-real

### Task 5: 扩充 Unreal Engine 仿真文章
**Files:** Modify `docs/simulation/unreal.md` (90→250+ lines)
- 新增：AirSim/Cosys-AirSim、ROS 2 rclUE 插件、Chaos Physics、光线追踪传感器仿真、Isaac Sim 对比

### Task 6: 新增 LQR 控制文章
**Files:** Create `docs/control/lqr/lqr.md`
- 内容：线性二次型调节器（LQR）完整推导、Riccati 方程、最优性条件、Python/scipy 实现、LQG（LQR + 卡尔曼滤波）、与 PID 对比

### Task 7: 新增自适应控制文章
**Files:** Create `docs/control/adaptive/adaptive.md`
- 内容：自适应控制动机、参考模型自适应控制（MRAC）、MIT 规则、Lyapunov 稳定性分析、自整定 PID、增益调度

### Task 8: 新增 Pepper 机器人档案
**Files:** Create `docs/database/pepper.md`
- 内容：发展历程、技术规格（表格）、NAOqi SDK、情感识别与交互、商业部署案例、与 ROS 集成

### Task 9: 新增 Apollo 机器人档案
**Files:** Create `docs/database/apollo.md`
- 内容：Apptronik 公司背景、Apollo 规格、NASA 合作、商业化计划、技术特点

### Task 10: 新增 Walker S 机器人档案
**Files:** Create `docs/database/walker-s.md`
- 内容：优必选（UBTECH）背景、Walker S 规格、工业场景应用、与国内竞品对比

### Task 11: 扩充机器人企业页面
**Files:** Modify `docs/database/companies.md` (88→300+ lines)
- 新增：全球主要机器人企业详细介绍（波士顿动力、宇树、优必选、智元、傅利叶、Figure AI、Agility、Apptronik 等），含企业规模/融资/核心产品/技术路线表格

### Task 12: 扩充全球机器人实验室页面
**Files:** Modify `docs/database/labs.md` (58→250+ lines)
- 新增：MIT CSAIL、Stanford AI Lab、CMU Robotics、ETH ASL、UCB BAIR、Imperial Dyson、Oxford Robotics 等，含研究方向/代表成果/著名校友表格

---

## mkdocs.yml 更新（Task 完成后执行）

```yaml
# 控制部分新增
- 控制:
  - control/index.md
  - control/modelling/modelling.md
  - control/modelling/state-space.md
  - control/pid/pid.md
  - control/lqr/lqr.md        # 新增
  - control/mpc/mpc.md
  - control/nn/nn.md
  - control/adaptive/adaptive.md  # 新增

# 数据库机器人图鉴新增
- 机器人图鉴:
  - database/robots.md
  - ...（现有档案）
  - database/pepper.md     # 新增
  - database/apollo.md     # 新增
  - database/walker-s.md   # 新增
```
