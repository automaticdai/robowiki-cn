# ROS 1 URDF 机器人建模

!!! note "引言"
    统一机器人描述格式（Unified Robot Description Format, URDF）是 ROS 1 中描述机器人三维结构的标准 XML 格式，定义连杆与关节的层次关系、视觉外观、碰撞几何与惯性参数。URDF 是整个 ROS 生态的公共输入：robot_state_publisher 依据它发布 TF 树，RViz 依据它渲染模型，Gazebo 依据它生成物理实体，MoveIt 依据它规划运动。由于原生 URDF 缺乏参数化能力，实际工程中几乎总是配合 xacro 宏语言使用。本页面介绍 URDF 的完整语法与 xacro 实践。


## URDF 机器人建模

统一机器人描述格式 (Unified Robot Description Format，URDF) 是ROS 1中描述机器人三维结构的标准XML格式。URDF定义了机器人的运动学结构（连杆和关节的层次关系）、视觉外观、碰撞几何体和惯性参数。

### URDF 基本结构

```xml
<?xml version="1.0" encoding="UTF-8"?>
<robot name="my_robot">

  <!-- =====================================================================
       连杆（link）：机器人的刚体部件
       每个link包含三个可选子元素：visual、collision、inertial
       ===================================================================== -->

  <!-- 世界坐标系虚拟连杆（固定机器人时使用） -->
  <link name="world" />

  <!-- 机器人底盘 -->
  <link name="base_link">
    <!-- 视觉几何体：用于RViz显示 -->
    <visual>
      <origin xyz="0 0 0.05" rpy="0 0 0" />
      <geometry>
        <!-- 支持box、cylinder、sphere、mesh四种几何类型 -->
        <box size="0.4 0.3 0.1" />
      </geometry>
      <material name="grey">
        <color rgba="0.5 0.5 0.5 1.0" />
      </material>
    </visual>

    <!-- 碰撞几何体：用于物理仿真（通常简化为基本形状） -->
    <collision>
      <origin xyz="0 0 0.05" rpy="0 0 0" />
      <geometry>
        <box size="0.4 0.3 0.1" />
      </geometry>
    </collision>

    <!-- 惯性参数：用于动力学仿真 -->
    <inertial>
      <origin xyz="0 0 0.05" rpy="0 0 0" />
      <!-- 质量（kg） -->
      <mass value="5.0" />
      <!-- 惯性张量（kg·m²），对称矩阵的上三角元素 -->
      <inertia ixx="0.04" ixy="0.0" ixz="0.0"
               iyy="0.04" iyz="0.0"
               izz="0.08" />
    </inertial>
  </link>

  <!-- 左前轮 -->
  <link name="left_front_wheel">
    <visual>
      <origin xyz="0 0 0" rpy="1.5708 0 0" />
      <geometry>
        <cylinder radius="0.08" length="0.04" />
      </geometry>
      <material name="black">
        <color rgba="0.1 0.1 0.1 1.0" />
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="1.5708 0 0" />
      <geometry>
        <cylinder radius="0.08" length="0.04" />
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5" />
      <inertia ixx="0.001" ixy="0.0" ixz="0.0"
               iyy="0.001" iyz="0.0"
               izz="0.002" />
    </inertial>
  </link>

  <!-- 激光雷达（使用mesh网格模型） -->
  <link name="laser_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <!-- mesh文件路径使用package://协议 -->
        <mesh filename="package://my_robot_description/meshes/lidar.dae" />
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.07" />
      </geometry>
    </collision>
    <inertial>
      <mass value="0.2" />
      <inertia ixx="0.0001" ixy="0" ixz="0"
               iyy="0.0001" iyz="0"
               izz="0.0002" />
    </inertial>
  </link>

  <!-- =====================================================================
       关节（joint）：连接两个连杆，定义运动类型和约束
       parent：父连杆，child：子连杆
       ===================================================================== -->

  <!-- 世界坐标系到底盘的固定关节 -->
  <joint name="world_to_base" type="fixed">
    <parent link="world" />
    <child link="base_link" />
    <origin xyz="0 0 0" rpy="0 0 0" />
  </joint>

  <!-- 左前轮关节（连续旋转关节） -->
  <joint name="left_front_wheel_joint" type="continuous">
    <parent link="base_link" />
    <child link="left_front_wheel" />
    <!-- 关节原点相对于父连杆坐标系的位置 -->
    <origin xyz="0.15 0.17 0" rpy="0 0 0" />
    <!-- 旋转轴（Y轴） -->
    <axis xyz="0 1 0" />
    <!-- 动力学参数（摩擦和阻尼） -->
    <dynamics damping="0.1" friction="0.0" />
  </joint>

  <!-- 激光雷达安装关节（固定关节） -->
  <joint name="base_to_laser" type="fixed">
    <parent link="base_link" />
    <child link="laser_link" />
    <origin xyz="0.15 0.0 0.18" rpy="0 0 0" />
  </joint>

</robot>
```


### 关节类型速查表

| 类型 | 说明 | 是否需要limits |
| --- | --- | --- |
| `fixed` | 固定关节，无自由度，两连杆刚性连接 | 否 |
| `revolute` | 旋转关节，绕轴旋转，有角度范围限制 | 是 |
| `continuous` | 连续旋转关节，绕轴无限旋转（如车轮） | 否 |
| `prismatic` | 滑动关节，沿轴线平移，有位移范围限制 | 是 |
| `planar` | 平面关节，在平面内平移和旋转 | 是 |
| `floating` | 浮动关节，6个自由度（很少使用） | 否 |

对于`revolute`和`prismatic`关节，必须声明`<limit>`元素：

```xml
<joint name="arm_shoulder_joint" type="revolute">
  <parent link="arm_base" />
  <child link="upper_arm" />
  <origin xyz="0 0 0.1" rpy="0 0 0" />
  <axis xyz="0 1 0" />
  <limit lower="-1.57" upper="1.57"
         effort="10.0" velocity="1.0" />
</joint>
```


### xacro 宏示例

xacro（XML Macros）是URDF的扩展格式，支持变量、宏、数学计算和条件语句，大幅减少重复代码：

```xml
<?xml version="1.0" encoding="UTF-8"?>
<robot name="my_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- 定义属性（变量） -->
  <xacro:property name="wheel_radius" value="0.08" />
  <xacro:property name="wheel_width"  value="0.04" />
  <xacro:property name="wheel_mass"   value="0.5"  />

  <!-- 定义宏：可复用的连杆+关节组合 -->
  <xacro:macro name="wheel" params="name parent x_pos y_pos">

    <link name="${name}_wheel">
      <visual>
        <origin xyz="0 0 0" rpy="1.5708 0 0" />
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}" />
        </geometry>
      </visual>
      <collision>
        <origin xyz="0 0 0" rpy="1.5708 0 0" />
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}" />
        </geometry>
      </collision>
      <inertial>
        <mass value="${wheel_mass}" />
        <!-- xacro支持内联数学表达式 -->
        <inertia
          ixx="${wheel_mass * (3 * wheel_radius**2 + wheel_width**2) / 12}"
          ixy="0" ixz="0"
          iyy="${wheel_mass * (3 * wheel_radius**2 + wheel_width**2) / 12}"
          iyz="0"
          izz="${wheel_mass * wheel_radius**2 / 2}" />
      </inertial>
    </link>

    <joint name="${name}_wheel_joint" type="continuous">
      <parent link="${parent}" />
      <child link="${name}_wheel" />
      <origin xyz="${x_pos} ${y_pos} 0" rpy="0 0 0" />
      <axis xyz="0 1 0" />
    </joint>

  </xacro:macro>

  <link name="base_link">
    <!-- ... -->
  </link>

  <!-- 使用宏实例化四个车轮，避免重复代码 -->
  <xacro:wheel name="left_front"  parent="base_link" x_pos=" 0.15" y_pos=" 0.17" />
  <xacro:wheel name="right_front" parent="base_link" x_pos=" 0.15" y_pos="-0.17" />
  <xacro:wheel name="left_rear"   parent="base_link" x_pos="-0.15" y_pos=" 0.17" />
  <xacro:wheel name="right_rear"  parent="base_link" x_pos="-0.15" y_pos="-0.17" />

</robot>
```

使用xacro处理文件：

```bash
# 将xacro转换为标准URDF
xacro robot.urdf.xacro > robot.urdf

# 在launch文件中直接使用xacro生成robot_description参数
# <param name="robot_description" command="$(find xacro)/xacro $(find my_pkg)/urdf/robot.urdf.xacro" />
```


### joint_state_publisher 与 robot_state_publisher

这两个节点配合使用，将URDF模型的关节状态发布到TF树，是机器人可视化和运动规划的基础：

- **`joint_state_publisher`**：读取`robot_description`参数中的URDF，以固定频率（默认10 Hz）向`/joint_states`话题发布所有非固定关节的状态。在没有真实硬件时，`joint_state_publisher_gui`还提供滑块界面手动设置各关节角度，方便调试URDF。

- **`robot_state_publisher`**：订阅`/joint_states`话题，结合URDF中的运动学结构，利用正向运动学计算所有连杆的坐标系变换，并广播到TF2树。

典型launch文件片段：

```xml
<!-- 加载URDF到参数服务器 -->
<param name="robot_description"
       command="$(find xacro)/xacro $(find my_pkg)/urdf/robot.urdf.xacro" />

<!-- 发布关节状态（无真实硬件时使用GUI版本） -->
<node pkg="joint_state_publisher_gui" type="joint_state_publisher_gui"
      name="joint_state_publisher" output="screen" />

<!-- 将关节状态转换为TF变换并广播 -->
<node pkg="robot_state_publisher" type="robot_state_publisher"
      name="robot_state_publisher" output="screen">
  <param name="publish_frequency" value="50.0" />
</node>
```


## 参考资料

1. ROS Wiki, [urdf](http://wiki.ros.org/urdf)
2. ROS Wiki, [xacro](http://wiki.ros.org/xacro)
3. ROS Wiki, [robot_state_publisher](http://wiki.ros.org/robot_state_publisher)
4. [正运动学](../kinematics/forward-kinematics.md)
5. [Gazebo 仿真](../simulation/gazebo.md)
6. [MoveIt](../manipulation/moveit.md)
