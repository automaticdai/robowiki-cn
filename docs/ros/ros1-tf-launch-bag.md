# ROS 1 坐标变换、启动与数据录制

!!! note "引言"
    在完成单个节点的编写之后，把多个节点组织成一个可运行、可调试、可复现的系统，依赖三件工具：TF2 维护机器人各坐标系之间随时间变化的变换关系，使传感器数据可以在任意坐标系之间转换；roslaunch 用一个 XML 文件描述整套系统的节点、参数与命名空间；rosbag 则把运行时的话题数据完整录制下来，供离线回放与算法回测。本页面介绍这三者的用法与常见陷阱。


## TF2 坐标变换

### 概述

TF2（Transform Library 2）是ROS 1中管理坐标系变换的核心库，用于追踪机器人系统中各坐标系随时间变化的空间关系。TF2维护一棵坐标系树，树中每条边代表两个坐标系之间的变换关系（包含平移和旋转），并记录变换的历史（默认缓存10秒）。

常见的坐标系包括：`world`（世界坐标系）、`odom`（里程计坐标系）、`base_link`（机器人基坐标系）、`base_footprint`（地面投影）以及各传感器坐标系（如`camera_link`、`laser_link`）。

### TransformBroadcaster（Python 示例）

`TransformBroadcaster` 用于向TF树广播坐标变换：

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import tf2_ros
import geometry_msgs.msg
import math

def broadcast_tf():
    rospy.init_node('tf_broadcaster')

    # 创建广播器
    broadcaster = tf2_ros.TransformBroadcaster()

    rate = rospy.Rate(50)  # 以50 Hz广播变换，保持TF树更新

    while not rospy.is_shutdown():
        # 构造变换消息
        t = geometry_msgs.msg.TransformStamped()

        # 时间戳必须使用当前ROS时间
        t.header.stamp = rospy.Time.now()
        # 父坐标系：变换的参考坐标系
        t.header.frame_id = 'base_link'
        # 子坐标系：被描述的坐标系
        t.child_frame_id = 'camera_link'

        # 平移分量（单位：米）
        # camera_link相对于base_link，前方0.1 m、上方0.2 m
        t.transform.translation.x = 0.1
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.2

        # 旋转分量（四元数表示）
        # 此处为无旋转（单位四元数）
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        # 广播变换
        broadcaster.sendTransform(t)

        rate.sleep()

if __name__ == '__main__':
    broadcast_tf()
```


### TransformListener 与 lookup_transform（Python 示例）

`TransformListener` 用于查询坐标系之间的变换：

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import tf2_ros
import tf2_geometry_msgs
import geometry_msgs.msg

def listen_tf():
    rospy.init_node('tf_listener')

    # 创建TF缓冲区，存储最近的变换历史
    tf_buffer = tf2_ros.Buffer()
    # 创建监听器，自动填充缓冲区
    listener = tf2_ros.TransformListener(tf_buffer)

    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        try:
            # 查询从'base_link'到'camera_link'的变换
            # 参数：目标坐标系、源坐标系、查询时刻（rospy.Time(0)表示最新可用变换）
            # 最后一个参数是超时时间
            trans = tf_buffer.lookup_transform(
                'base_link',
                'camera_link',
                rospy.Time(0),
                rospy.Duration(1.0)
            )

            tx = trans.transform.translation.x
            ty = trans.transform.translation.y
            tz = trans.transform.translation.z
            rospy.loginfo('camera_link 相对于 base_link：(%.3f, %.3f, %.3f)',
                          tx, ty, tz)

        except tf2_ros.LookupException as e:
            # 请求的坐标系在TF树中不存在
            rospy.logwarn('LookupException：%s', str(e))
        except tf2_ros.ConnectivityException as e:
            # 两个坐标系之间没有连通路径
            rospy.logwarn('ConnectivityException：%s', str(e))
        except tf2_ros.ExtrapolationException as e:
            # 请求的时刻超出了TF缓冲区的时间范围
            rospy.logwarn('ExtrapolationException：%s', str(e))

        rate.sleep()

if __name__ == '__main__':
    listen_tf()
```


### static_transform_publisher 命令

对于固定不变的坐标系变换（如传感器安装位置），无需编写节点，直接使用`static_transform_publisher`命令即可：

```bash
# 格式：static_transform_publisher x y z yaw pitch roll 父坐标系 子坐标系 发布频率
# 以下命令发布激光雷达相对于机器人底盘的固定变换
rosrun tf static_transform_publisher 0.15 0.0 0.3 0.0 0.0 0.0 \
    base_link laser_link 100

# 使用四元数格式（x y z qx qy qz qw）
rosrun tf static_transform_publisher 0.1 0.0 0.2 0.0 0.0 0.0 1.0 \
    base_link camera_link 100
```

在launch文件中使用`static_transform_publisher`更为常见，参见 roslaunch 章节中的示例。


## roslaunch 文件

roslaunch 是 ROS 1 中同时启动多个节点、设置参数的标准工具。Launch 文件使用 XML 格式编写，扩展名为`.launch`。

### 完整 Launch 文件示例

以下是一个功能完整的 launch 文件，涵盖了常用的所有标签：

```xml
<?xml version="1.0" encoding="UTF-8"?>
<launch>
  <!-- =====================================================================
       参数声明（arg）：类似函数参数，支持命令行覆盖
       用法：roslaunch my_pkg demo.launch use_sim:=true robot_name:=robot2
       ===================================================================== -->
  <arg name="use_sim"     default="false" doc="是否使用仿真时钟" />
  <arg name="robot_name"  default="robot1" doc="机器人命名空间" />
  <arg name="map_file"    default="$(find my_pkg)/maps/default.yaml" />

  <!-- =====================================================================
       参数设置（param）：向参数服务器写入单个参数
       ===================================================================== -->
  <param name="use_sim_time" value="$(arg use_sim)" />
  <param name="robot_description"
         command="$(find xacro)/xacro $(find my_pkg)/urdf/robot.urdf.xacro" />

  <!-- =====================================================================
       批量参数加载（rosparam）：从YAML文件加载参数组
       ===================================================================== -->
  <rosparam file="$(find my_pkg)/config/navigation_params.yaml"
            command="load" />

  <!-- =====================================================================
       节点启动（node）：启动单个ROS节点
       pkg        ：软件包名
       type       ：可执行文件名（对Python脚本即为脚本文件名）
       name       ：节点在ROS图中的名称（覆盖代码中的init_node名称）
       output     ：日志输出目标，"screen"输出到终端，"log"输出到文件
       respawn    ：节点崩溃后是否自动重启
       required   ：若为true，节点退出时关闭整个launch
       launch-prefix：在节点命令前添加前缀，用于调试（如 "xterm -e" 或 "gdb -ex run --args"）
       ===================================================================== -->
  <node pkg="map_server" type="map_server" name="map_server"
        args="$(arg map_file)"
        output="screen"
        respawn="false"
        required="false" />

  <!-- =====================================================================
       话题重映射（remap）：将节点内部话题名映射到外部话题名
       ===================================================================== -->
  <node pkg="my_pkg" type="camera_node.py" name="camera_node" output="screen">
    <!-- 将节点内部的"/image_raw"重映射为"/camera/image_raw" -->
    <remap from="/image_raw" to="/camera/image_raw" />
    <!-- 节点私有参数 -->
    <param name="image_width"  value="640" />
    <param name="image_height" value="480" />
    <param name="fps"          value="30" />
  </node>

  <!-- =====================================================================
       静态坐标变换发布
       ===================================================================== -->
  <node pkg="tf" type="static_transform_publisher" name="base_to_laser"
        args="0.15 0.0 0.3 0.0 0.0 0.0 base_link laser_link 100" />

  <!-- =====================================================================
       分组（group）：为一组节点设置公共命名空间或条件
       ===================================================================== -->
  <group ns="$(arg robot_name)">
    <node pkg="my_pkg" type="controller_node" name="controller"
          output="screen">
      <param name="max_vel" value="1.0" />
    </node>

    <node pkg="robot_state_publisher" type="robot_state_publisher"
          name="robot_state_publisher" output="screen" />
  </group>

  <!-- =====================================================================
       条件包含：根据参数决定是否启动某个节点或包含某个文件
       ===================================================================== -->
  <group if="$(arg use_sim)">
    <include file="$(find gazebo_ros)/launch/empty_world.launch">
      <arg name="use_sim_time" value="true" />
      <arg name="paused"       value="false" />
    </include>
  </group>

  <!-- =====================================================================
       包含其他launch文件（include）
       ===================================================================== -->
  <include file="$(find amcl)/examples/amcl_diff.launch" />

  <!-- RViz可视化 -->
  <node pkg="rviz" type="rviz" name="rviz" output="screen"
        args="-d $(find my_pkg)/rviz/default.rviz" />

</launch>
```


### Launch 文件常用标签速查表

| 标签 | 关键属性 | 说明 |
| --- | --- | --- |
| `<node>` | `pkg`, `type`, `name`, `output`, `respawn`, `required` | 启动一个节点 |
| `<param>` | `name`, `value`, `type`, `command` | 设置单个参数 |
| `<rosparam>` | `file`, `command`, `ns` | 批量加载/保存YAML参数 |
| `<arg>` | `name`, `default`, `value`, `doc` | 声明可配置参数 |
| `<remap>` | `from`, `to` | 重映射话题/服务名称 |
| `<group>` | `ns`, `if`, `unless` | 分组并设置命名空间或条件 |
| `<include>` | `file` | 包含另一个launch文件 |
| `<env>` | `name`, `value` | 设置环境变量 |


## rosbag 使用指南

rosbag 是 ROS 1 中用于记录和回放话题数据的工具，生成的文件扩展名为`.bag`。Bag 文件记录了消息的内容和时间戳，是数据采集、离线调试和算法回测的核心工具。

### 数据记录

```bash
# 记录所有话题（数据量大，谨慎使用）
rosbag record -a

# 记录指定话题，并指定输出文件名（不含扩展名）
rosbag record -O my_dataset /camera/image_raw /laser/scan /odom

# 自动分割文件：每500 MB或每300秒创建一个新文件
rosbag record -a --split --size 500 --duration 300

# 排除特定话题（使用正则表达式）
rosbag record -a -x "/camera/image_raw|/diagnostics"

# 限制每个话题的缓冲区大小（MB），防止内存溢出
rosbag record -a --buffsize 256
```


### 数据回放

```bash
# 基本回放（以录制时的实际速率）
rosbag play my_dataset.bag

# 以2倍速回放
rosbag play -r 2.0 my_dataset.bag

# 循环回放（适合调试订阅者节点）
rosbag play -l my_dataset.bag

# 发布仿真时钟（配合use_sim_time=true使用）
# 使依赖时间的节点（如TF、滤波器）与bag文件时间同步
rosbag play --clock my_dataset.bag

# 从指定时刻开始回放（跳过前60秒）
rosbag play -s 60 my_dataset.bag

# 只回放指定话题
rosbag play my_dataset.bag /camera/image_raw /odom

# 暂停后手动步进（回放开始后按空格暂停，s键单步）
rosbag play --pause my_dataset.bag
```


### 信息查看与校验

```bash
# 查看bag文件的元数据（话题列表、消息数量、持续时间等）
rosbag info my_dataset.bag

# 检查bag文件完整性
rosbag check my_dataset.bag

# 压缩bag文件（支持bz2和lz4格式）
rosbag compress --lz4 my_dataset.bag

# 过滤bag文件：只保留特定话题或时间范围内的消息
# 以下命令创建只包含/odom话题的新bag文件
rosbag filter my_dataset.bag filtered.bag "topic == '/odom'"

# 合并多个bag文件
rosbag play first.bag second.bag
```


### Python API 读取 Bag 文件

使用`rosbag` Python API可以在脚本中读取和处理 bag 文件数据：

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线读取bag文件并提取里程计数据示例
"""
import rosbag
import rospy

bag_path = '/path/to/my_dataset.bag'

# 打开bag文件（使用with语句确保文件正确关闭）
with rosbag.Bag(bag_path, 'r') as bag:
    # 打印基本信息
    print('Bag文件信息：')
    print('  开始时间：', bag.get_start_time())
    print('  结束时间：', bag.get_end_time())
    print('  话题列表：', list(bag.get_type_and_topic_info().topics.keys()))

    # 遍历指定话题的消息
    # read_messages返回(topic, msg, t)三元组，t为rospy.Time类型
    odom_data = []
    for topic, msg, t in bag.read_messages(topics=['/odom']):
        timestamp = t.to_sec()  # 转换为浮点秒数
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        odom_data.append((timestamp, x, y))

    print('共读取里程计消息 {} 条'.format(len(odom_data)))

    # 同时遍历多个话题
    for topic, msg, t in bag.read_messages(
            topics=['/camera/image_raw', '/laser/scan']):
        if topic == '/laser/scan':
            # 处理激光扫描数据
            ranges = msg.ranges
            angle_min = msg.angle_min
            # ... 进一步处理
            pass
        elif topic == '/camera/image_raw':
            # 处理图像数据（通常配合cv_bridge使用）
            height = msg.height
            width = msg.width
            # ... 进一步处理
            pass
```


## 参考资料

1. ROS Wiki, [tf2](http://wiki.ros.org/tf2)
2. ROS Wiki, [roslaunch/XML](http://wiki.ros.org/roslaunch/XML)
3. ROS Wiki, [rosbag](http://wiki.ros.org/rosbag)
4. ROS REP-103, *Standard Units of Measure and Coordinate Conventions*. https://ros.org/reps/rep-0103.html
5. ROS REP-105, *Coordinate Frames for Mobile Platforms*. https://ros.org/reps/rep-0105.html
6. [空间变换与位姿表示](../kinematics/spatial-transformation.md)
