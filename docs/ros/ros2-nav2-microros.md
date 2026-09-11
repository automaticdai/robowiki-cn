# Nav2 导航框架与 micro-ROS

!!! note "引言"
    Nav2 与 micro-ROS 代表了 ROS 2 向上与向下两个方向的延伸。Nav2 是 ROS 1 Navigation Stack 的完整重写，用行为树（Behavior Tree）替代原先硬编码的状态机，把规划器、控制器、恢复行为都变成可插拔的生命周期节点，从而支持更复杂的任务编排。micro-ROS 则把 ROS 2 下探到微控制器：通过 Micro XRCE-DDS 这一精简中间件，STM32、ESP32 等 MCU 可以直接作为 ROS 2 节点参与通信，无需自定义串口协议与转换节点。本页面介绍两者的架构与配置方式。


## Nav2导航框架

Nav2（Navigation 2）是ROS 2的官方导航框架，为移动机器人提供自主导航能力，是ROS 1 Navigation Stack的完整重写。

### 架构概述

Nav2采用行为树（Behavior Tree）驱动的分层架构：

```
用户目标
    ↓
BT Navigator（行为树导航器）
    ├── Planner Server（全局规划器）
    │     └── NavFn / Smac Planner
    ├── Controller Server（局部控制器）
    │     └── DWB Controller / MPPI Controller
    ├── Smoother Server（路径平滑器）
    └── Recovery Server（恢复行为）
          └── Spin / Back Up / Wait
```

### 关键组件

- **AMCL (Adaptive Monte Carlo Localization)**：基于粒子滤波的自适应蒙特卡洛定位，利用激光雷达数据在已知地图上进行机器人位置估计
- **costmap_2d**：代价地图，分为全局代价地图（用于全局路径规划）和局部代价地图（用于实时避障），支持多种图层（静态层、障碍物层、膨胀层）
- **NavFn Planner**：基于Dijkstra或A*算法的全局路径规划器
- **DWB Controller**：动态窗口法局部控制器，在代价地图上实时计算速度指令
- **BT Navigator**：使用行为树组织整个导航流程，通过XML配置文件定义导航逻辑

### 行为树示例

Nav2的导航逻辑通过行为树XML文件配置，以下是一个简化示例：

```xml
<root main_tree_to_execute="MainTree">
  <BehaviorTree ID="MainTree">
    <RecoveryNode number_of_retries="6" name="NavigateRecovery">
      <PipelineSequence name="NavigateWithReplanning">
        <RateController hz="1.0">
          <ComputePathToPose goal="{goal}" path="{path}" planner_id="GridBased"/>
        </RateController>
        <FollowPath path="{path}" controller_id="FollowPath"/>
      </PipelineSequence>
      <ReactiveFallback name="RecoveryFallback">
        <GoalUpdated/>
        <RoundRobin name="RecoveryActions">
          <Sequence name="ClearingActions">
            <ClearEntireCostmap name="ClearLocalCostmap-Context"
              service_name="local_costmap/clear_entirely_local_costmap"/>
          </Sequence>
          <Spin spin_dist="1.57"/>
          <Wait wait_duration="5"/>
          <Back up backup_dist="0.15" backup_speed="0.025"/>
        </RoundRobin>
      </ReactiveFallback>
    </RecoveryNode>
  </BehaviorTree>
</root>
```

### 启动导航

以TurtleBot3为例启动Nav2完整导航栈：

```bash
# 安装TurtleBot3软件包（Humble版本）
sudo apt install ros-humble-turtlebot3-navigation2 ros-humble-turtlebot3-gazebo

# 设置机器人型号
export TURTLEBOT3_MODEL=waffle

# 启动Gazebo仿真
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

# 在另一个终端启动Nav2（包含AMCL定位和导航服务）
ros2 launch turtlebot3_navigation2 navigation2.launch.py \
    use_sim_time:=True \
    map:=/path/to/map.yaml

# 在RViz中使用"2D Pose Estimate"设置初始位置，然后使用"Nav2 Goal"发送目标
```

## micro-ROS

micro-ROS将ROS 2的核心功能移植到资源受限的微控制器（MCU）上，使嵌入式设备能够直接参与ROS 2通信网络，无需中间转换层。

### 核心概念

micro-ROS使用Micro XRCE-DDS（eXtremely Resource Constrained Environments DDS）作为通信中间件，这是DDS协议的轻量级实现。MCU上的micro-ROS节点通过**micro-ROS Agent**桥接到标准ROS 2网络：

```
[MCU: STM32 / ESP32 / Arduino]
    micro-ROS库
        ↕ 串口 / UDP / USB
[Linux主机: micro-ROS Agent]
        ↕ DDS
[ROS 2网络]
    标准ROS 2节点
```

### 支持硬件

| 硬件平台 | 连接方式 | 备注 |
| --- | --- | --- |
| STM32系列 | 串口、USB | 通过FreeRTOS或ThreadX集成，工业场景首选 |
| ESP32 | Wi-Fi（UDP）、串口 | 无线连接，适合移动场景 |
| Arduino Due | 串口 | 基础支持，资源较紧张 |
| Raspberry Pi Pico | 串口、USB | 低成本选择，支持FreeRTOS |

### Arduino风格代码示例

以下示例在ESP32或Arduino Due上以固定频率发布里程计数据：

```cpp
#include <micro_ros_arduino.h>
#include <rcl/rcl.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>
#include <std_msgs/msg/int32.h>

rcl_publisher_t publisher;
std_msgs__msg__Int32 msg;
rclc_executor_t executor;
rclc_support_t support;
rcl_allocator_t allocator;
rcl_node_t node;
rcl_timer_t timer;

void timer_callback(rcl_timer_t * timer, int64_t last_call_time)
{
    (void) last_call_time;
    if (timer != NULL) {
        rcl_publish(&publisher, &msg, NULL);
        msg.data++;
    }
}

void setup()
{
    // 通过串口连接micro-ROS Agent
    set_microros_transports();

    allocator = rcl_get_default_allocator();

    // 初始化micro-ROS支持结构
    rclc_support_init(&support, 0, NULL, &allocator);

    // 创建节点
    rclc_node_init_default(&node, "micro_ros_arduino_node", "", &support);

    // 创建发布者
    rclc_publisher_init_default(
        &publisher,
        &node,
        ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Int32),
        "micro_ros_arduino_node_publisher"
    );

    // 创建定时器（100ms周期）
    const unsigned int timer_timeout = 100;
    rclc_timer_init_default(&timer, &support, RCL_MS_TO_NS(timer_timeout), timer_callback);

    // 创建执行器
    rclc_executor_init(&executor, &support.context, 1, &allocator);
    rclc_executor_add_timer(&executor, &timer);

    msg.data = 0;
}

void loop()
{
    // 处理一次执行器事件
    rclc_executor_spin_some(&executor, RCL_MS_TO_NS(100));
}
```

### 启动micro-ROS Agent

micro-ROS Agent是运行在Linux主机上的桥接程序，负责在MCU和ROS 2网络之间转发消息。

**串口模式**（适用于USB转串口连接）：

```bash
# 安装micro-ROS Agent（通过snap）
snap install micro-ros-agent

# 或通过Docker运行
docker run -it --rm \
    -v /dev:/dev \
    --privileged \
    microros/micro-ros-agent:humble \
    serial --dev /dev/ttyUSB0 -b 115200
```

**UDP模式**（适用于ESP32 Wi-Fi连接）：

```bash
# 监听UDP端口8888
docker run -it --rm \
    --net=host \
    microros/micro-ros-agent:humble \
    udp4 --port 8888
```

### 典型使用场景

在移动机器人系统中，STM32微控制器常承担底层驱动任务：

- **发布话题**：`/odom`（里程计）、`/imu/data`（IMU数据）
- **订阅话题**：`/cmd_vel`（速度指令）

STM32通过串口连接到运行ROS 2的上位机（Jetson Nano或树莓派），micro-ROS Agent负责透明转发，上位机的导航和感知节点无需感知底层通信细节。

## 参考资料

1. Nav2 Documentation. https://docs.nav2.org/
2. S. Macenski, F. Martín, R. White, and J. Ginés Clavero, "The Marathon 2: A Navigation System," *IROS*, 2020.
3. micro-ROS Documentation. https://micro.ros.org/
4. [导航概述](../planning/navigation.md)
5. [行为树](../planning/behavior-trees.md)
6. [实时操作系统](../rtos/index.md)
