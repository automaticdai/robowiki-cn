# ROS 1

!!! note "引言"
    ROS 1是ROS的第一代版本，自2010年首个正式发行版Box Turtle发布以来，已成为学术研究和机器人原型开发的事实标准。ROS 1以其成熟的工具链、丰富的软件包和庞大的社区支持著称。尽管ROS 2已逐步成为主流，ROS 1仍然在众多现有项目中发挥着重要作用。

## 概述

ROS 1最初由Willow Garage公司开发，后由Open Source Robotics Foundation (OSRF) 维护。它主要面向学术研究和单机器人系统的开发，在Ubuntu Linux上有最好的支持。ROS 1的最后一个版本是Noetic Ninjemys（2020年发布），支持至2025年5月。

ROS 1的设计目标是降低机器人软件开发的门槛，通过提供标准化的通信机制、丰富的工具和大量可复用的软件包，使开发者能够快速构建功能丰富的机器人系统。

## 核心概念

### ROS Master

ROS Master是ROS 1架构的核心组件，负责管理节点之间的命名和注册服务。所有节点在启动时都需要向Master注册，Master负责维护节点的查找表，使节点能够相互发现和建立通信。通过`roscore`命令启动Master。

ROS Master是ROS 1的一个关键特征，同时也是其局限性之一。如果Master崩溃，整个系统的通信将会中断。

### 节点 (Nodes)

节点是ROS 1中执行计算的基本进程。ROS 1鼓励开发者创建大量小型、功能单一的节点，而不是少量庞大的多功能进程。这种设计使系统具有更好的模块化特性，便于调试和复用。

### 话题与消息 (Topics & Messages)

话题 (Topics) 是ROS 1中最常用的通信方式，采用发布-订阅 (publish-subscribe) 模式。发布者 (Publisher) 向一个命名话题发送消息，订阅者 (Subscriber) 从该话题接收消息。通信是异步的，发布者和订阅者之间无需知道对方的存在。

消息 (Messages) 是话题传输的数据结构，使用`.msg`文件定义。常见的消息类型包括：

- `std_msgs`：标准基础类型，如`String`、`Int32`、`Float64`
- `geometry_msgs`：几何相关消息，如`Twist`（速度）、`Pose`（位姿）
- `sensor_msgs`：传感器数据，如`Image`（图像）、`LaserScan`（激光扫描）、`PointCloud2`（点云）
- `nav_msgs`：导航相关消息，如`Odometry`（里程计）、`Path`（路径）

### 服务 (Services)

服务 (Services) 提供同步的请求-响应通信机制，使用`.srv`文件定义请求和响应的数据结构。与话题不同，服务是一对一的通信，适用于需要立即获得结果的场景。

### 参数服务器 (Parameter Server)

参数服务器 (Parameter Server) 是一个集中式的键值存储系统，允许节点在运行时存储和检索配置参数。参数服务器由ROS Master维护，支持整数、浮点数、字符串、布尔值、列表和字典等数据类型。

## catkin构建系统

catkin是ROS 1的官方构建系统，基于CMake开发。它使用**工作空间** (workspace) 来组织和编译ROS软件包。

典型的catkin工作空间结构如下：

```
catkin_ws/
├── src/              # 源代码目录
│   ├── CMakeLists.txt
│   ├── package_1/
│   │   ├── CMakeLists.txt
│   │   ├── package.xml
│   │   └── src/
│   └── package_2/
├── build/            # 编译中间文件
├── devel/            # 开发空间（编译产物）
└── logs/             # 日志文件
```

常用的catkin命令包括：

- `catkin_make`：编译整个工作空间
- `catkin build`：逐包编译（由catkin_tools提供，更灵活）
- `catkin_create_pkg`：创建新的ROS软件包

## 常用工具

ROS 1提供了大量实用的命令行和图形化工具：

- **roscore**：启动ROS Master、参数服务器和rosout日志节点
- **rosrun**：运行单个ROS节点，如`rosrun turtlesim turtlesim_node`
- **roslaunch**：通过`.launch`文件同时启动多个节点并设置参数
- **rostopic**：查看、发布和监控话题消息
- **rosservice**：调用和查看ROS服务
- **rosparam**：获取和设置参数服务器中的参数
- **rosbag**：记录和回放话题数据，用于数据采集和离线分析
- **RViz**：三维可视化工具，用于显示传感器数据和机器人模型
- **rqt**：基于Qt的模块化图形工具，包含rqt_graph（节点关系图）、rqt_plot（数据绘图）、rqt_console（日志查看）等插件

## 常用软件包

ROS 1拥有数千个社区贡献的软件包，以下是一些最常用的核心包：

| 软件包 | 功能描述 |
| --- | --- |
| `navigation` | 移动机器人自主导航，包含路径规划、定位和避障 |
| `moveit` | 机械臂运动规划、抓取和操作 |
| `gmapping` | 基于激光雷达的SLAM建图 |
| `amcl` | 自适应蒙特卡洛定位 (Adaptive Monte Carlo Localization) |
| `tf` | 坐标变换管理 |
| `robot_state_publisher` | 发布机器人关节状态到tf树 |
| `rviz` | 三维可视化 |
| `gazebo_ros` | Gazebo仿真环境的ROS接口 |
| `image_transport` | 图像传输和压缩 |
| `pcl_ros` | 点云库 (Point Cloud Library) 的ROS封装 |

## ROS 1版本

| 版本代号                                                     | 发布日期            | Logo                                                         | 教程海龟的图标                                               | 终止维护日期             |
| ------------------------------------------------------------ | ------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------ |
| [ROS Noetic Ninjemys](http://wiki.ros.org/noetic) (**Recommended**) | May 23rd, 2020      | ![Noetic Ninjemys](https://raw.githubusercontent.com/ros-infrastructure/artwork/master/distributions/noetic.png) | ![https://raw.githubusercontent.com/ros/ros_tutorials/noetic-devel/turtlesim/images/noetic.png](https://raw.githubusercontent.com/ros/ros_tutorials/noetic-devel/turtlesim/images/noetic.png) | May, 2025 (Focal EOL)    |
| [ROS Melodic Morenia](http://wiki.ros.org/melodic)           | May 23rd, 2018      | [![Melodic Morenia](https://raw.githubusercontent.com/ros-infrastructure/artwork/master/distributions/melodic_with_bg.png)](http://wiki.ros.org/melodic) | ![Melodic Morenia](https://raw.githubusercontent.com/ros/ros_tutorials/melodic-devel/turtlesim/images/melodic.png) | May, 2023 (Bionic EOL)   |
| [ROS Lunar Loggerhead](http://wiki.ros.org/lunar)            | May 23rd, 2017      | [![Lunar Loggerhead](https://raw.githubusercontent.com/ros-infrastructure/artwork/master/distributions/lunar_with_bg.png)](http://wiki.ros.org/lunar) | ![Lunar Loggerhead](https://raw.githubusercontent.com/ros/ros_tutorials/lunar-devel/turtlesim/images/lunar.png) | May, 2019                |
| [ROS Kinetic Kame](http://wiki.ros.org/kinetic)              | May 23rd, 2016      | [![Kinetic Kame](https://raw.githubusercontent.com/ros-infrastructure/artwork/master/distributions/kinetic.png)](http://wiki.ros.org/kinetic) | ![Kinetic Kame](https://raw.github.com/ros/ros_tutorials/kinetic-devel/turtlesim/images/kinetic.png) | April, 2021 (Xenial EOL) |
| [ROS Jade Turtle](http://wiki.ros.org/jade)                  | May 23rd, 2015      | [![Jade Turtle](http://i.imgur.com/99oTyT5.png)](http://wiki.ros.org/jade) | ![Jade Turtle](https://raw.github.com/ros/ros_tutorials/jade-devel/turtlesim/images/jade.png) | May, 2017                |
| [ROS Indigo Igloo](http://wiki.ros.org/indigo)               | July 22nd, 2014     | [![I-turtle](http://i.imgur.com/YBCUixi.png)](http://wiki.ros.org/indigo) | ![I-turtle](https://raw.github.com/ros/ros_tutorials/indigo-devel/turtlesim/images/indigo.png) | April, 2019 (Trusty EOL) |
| [ROS Hydro Medusa](http://wiki.ros.org/hydro)                | September 4th, 2013 | [![H-turtle](http://i.imgur.com/xvfZPAo.png)](http://wiki.ros.org/hydro) | ![H-turtle](https://raw.github.com/ros/ros_tutorials/hydro-devel/turtlesim/images/hydro.png) | May, 2015                |
| [ROS Groovy Galapagos](http://wiki.ros.org/groovy)           | December 31, 2012   | [![G-turtle](http://www.ros.org/images/groovygalapagos-320w.jpg)](http://wiki.ros.org/groovy) | ![G-turtle](https://raw.github.com/ros/ros_tutorials/groovy-devel/turtlesim/images/groovy.png) | July, 2014               |
| [ROS Fuerte Turtle](http://wiki.ros.org/fuerte)              | April 23, 2012      | [![F-turtle](http://www.ros.org/images/fuerte-320w.jpg)](http://wiki.ros.org/fuerte) | ![F-turtle](https://raw.github.com/ros/ros_tutorials/groovy-devel/turtlesim/images/fuerte.png) | --                       |
| [ROS Electric Emys](http://wiki.ros.org/electric)            | August 30, 2011     | [![E-turtle](http://www.ros.org/news/resources/2011/electric_640w.png)](http://wiki.ros.org/electric) | ![E-turtle](https://raw.github.com/ros/ros_tutorials/groovy-devel/turtlesim/images/electric.png) | --                       |
| [ROS Diamondback](http://wiki.ros.org/diamondback)           | March 2, 2011       | [![D-turtle](http://ros.org/images/wiki/diamondback_posterLo-240w.jpg)](http://wiki.ros.org/diamondback) | ![D-turtle](https://raw.github.com/ros/ros_tutorials/groovy-devel/turtlesim/images/diamondback.png) | --                       |
| [ROS C Turtle](http://wiki.ros.org/cturtle)                  | August 2, 2010      | [![C-turtle](http://ros.org/images/wiki/cturtle.jpg)](http://wiki.ros.org/cturtle) | ![C-turtle](https://raw.github.com/ros/ros_tutorials/groovy-devel/turtlesim/images/sea-turtle.png) | --                       |
| [ROS Box Turtle](http://wiki.ros.org/boxturtle)              | March 2, 2010       | [![B-turtle](http://ros.org/wiki/boxturtle?action=AttachFile&do=get&target=Box_Turtle.320.png)](http://wiki.ros.org/boxturtle) | ![B-turtle](https://raw.github.com/ros/ros_tutorials/groovy-devel/turtlesim/images/box-turtle.png) | --                       |

## 安装指南

ROS 1通常安装在Ubuntu Linux上，每个ROS 1版本对应特定的Ubuntu版本：

- **Noetic**：Ubuntu 20.04 (Focal Fossa)
- **Melodic**：Ubuntu 18.04 (Bionic Beaver)
- **Kinetic**：Ubuntu 16.04 (Xenial Xerus)

安装的基本步骤包括：配置软件源 (sources.list)、添加密钥 (keys)、通过`apt`安装ROS、初始化`rosdep`以及配置环境变量。详细安装教程请参考[ROS Wiki安装页面](http://wiki.ros.org/ROS/Installation)。

## ROS 1的局限性

尽管ROS 1在学术领域取得了巨大成功，但随着机器人技术向工业化和商业化发展，其设计上的一些局限性逐渐显现，这也是ROS 2被开发的直接原因：

- **单点故障** (Single Point of Failure)：ROS Master是系统的单一故障点，一旦崩溃将导致整个系统通信中断
- **缺乏实时性支持** (No Real-time Support)：ROS 1的通信层不支持实时性保障，无法满足工业控制等对时间敏感的应用需求
- **安全性不足** (Lack of Security)：ROS 1没有内置的认证和加密机制，不适合部署在开放网络环境中
- **仅支持Linux**：ROS 1主要支持Ubuntu Linux，在其他操作系统上的支持有限
- **不支持多机器人系统** (No Native Multi-robot Support)：缺乏对多机器人协作场景的原生支持
- **嵌入式支持有限**：对资源受限的嵌入式平台支持不够完善

这些局限性促使社区开发了ROS 2，以更好地满足工业级和商业级机器人应用的需求。

## C++ 节点编程（roscpp）

roscpp是ROS 1的C++客户端库，是构建高性能ROS节点的首选方式。它提供了对ROS通信原语的完整封装，并与catkin构建系统紧密集成。

### 发布者节点（talker.cpp）

以下是一个完整的C++发布者节点示例，它以10 Hz的频率向`/chatter`话题发布字符串消息：

```cpp
// talker.cpp
#include <ros/ros.h>
#include <std_msgs/String.h>
#include <sstream>

int main(int argc, char **argv)
{
    // 初始化ROS节点，节点名称为"talker"
    // 节点名称必须唯一，不能包含斜杠
    ros::init(argc, argv, "talker");

    // 创建节点句柄 (NodeHandle)
    // NodeHandle是与ROS系统进行交互的主要入口点
    // 第一个NodeHandle实例的创建会初始化该节点
    ros::NodeHandle nh;

    // 创建发布者，向"/chatter"话题发布std_msgs::String类型的消息
    // 第二个参数是消息队列长度：若消息发布速度超过传输速度，队列将缓冲消息
    ros::Publisher chatter_pub = nh.advertise<std_msgs::String>("chatter", 1000);

    // 创建Rate对象，控制循环频率为10 Hz
    ros::Rate loop_rate(10);

    int count = 0;
    // ros::ok()在节点正常运行时返回true
    // 以下情况会使其返回false：收到SIGINT信号（Ctrl+C）、
    // 另一个同名节点启动、ros::shutdown()被调用
    while (ros::ok())
    {
        // 构造消息对象
        std_msgs::String msg;
        std::stringstream ss;
        ss << "hello world " << count;
        msg.data = ss.str();

        // 打印日志信息（同时输出到终端和/rosout话题）
        ROS_INFO("%s", msg.data.c_str());

        // 发布消息
        chatter_pub.publish(msg);

        // 处理回调队列（对于仅发布的节点，此处可省略，但保留是良好实践）
        ros::spinOnce();

        // 按照指定频率休眠，使循环保持在10 Hz
        loop_rate.sleep();

        ++count;
    }

    return 0;
}
```


### 订阅者节点（listener.cpp）

以下是对应的C++订阅者节点，它接收`/chatter`话题上的消息并打印：

```cpp
// listener.cpp
#include <ros/ros.h>
#include <std_msgs/String.h>

// 回调函数：每当收到新消息时被调用
// 参数使用ConstPtr（即boost::shared_ptr<const T>）以避免不必要的拷贝
void chatterCallback(const std_msgs::String::ConstPtr& msg)
{
    ROS_INFO("I heard: [%s]", msg->data.c_str());
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "listener");

    ros::NodeHandle nh;

    // 创建订阅者，订阅"/chatter"话题
    // 参数依次为：话题名、队列长度、回调函数
    ros::Subscriber sub = nh.subscribe("chatter", 1000, chatterCallback);

    // ros::spin()进入事件循环，持续等待并处理回调
    // 此调用会阻塞，直到节点关闭
    ros::spin();

    return 0;
}
```


### CMakeLists.txt构建配置

在软件包的`CMakeLists.txt`中添加以下内容以编译上述节点：

```cmake
cmake_minimum_required(VERSION 3.0.2)
project(my_ros_package)

# 查找catkin及所需的组件包
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  message_generation
)

# 声明catkin软件包（供其他包依赖时使用）
catkin_package(
  CATKIN_DEPENDS roscpp std_msgs message_runtime
)

# 添加头文件搜索路径
include_directories(
  ${catkin_INCLUDE_DIRS}
)

# 声明可执行文件并指定源文件
add_executable(talker src/talker.cpp)
add_executable(listener src/listener.cpp)

# 链接catkin库（包含roscpp、std_msgs等）
target_link_libraries(talker ${catkin_LIBRARIES})
target_link_libraries(listener ${catkin_LIBRARIES})

# 确保消息头文件在编译前生成（若有自定义消息）
add_dependencies(talker ${${PROJECT_NAME}_EXPORTED_TARGETS}
                        ${catkin_EXPORTED_TARGETS})
add_dependencies(listener ${${PROJECT_NAME}_EXPORTED_TARGETS}
                          ${catkin_EXPORTED_TARGETS})
```


### NodeHandle、ros::spin() 与 ros::spinOnce()

**NodeHandle（节点句柄）**是节点与ROS系统交互的核心对象。它负责管理节点的资源，包括发布者、订阅者、服务、定时器等。NodeHandle支持命名空间机制：

- `ros::NodeHandle nh`：使用节点的全局命名空间（`/`）
- `ros::NodeHandle nh("~")`：使用节点的私有命名空间（`/node_name/`），适合存放节点私有参数
- `ros::NodeHandle nh("sensors")`：使用相对命名空间（`/sensors/`）

**ros::spin()** 进入一个阻塞式事件循环，持续处理到来的消息回调，直到节点关闭。适合订阅者节点或任务驱动型节点。

**ros::spinOnce()** 处理一次当前回调队列中的所有待处理回调，然后立即返回。适合在主循环中需要同时处理其他逻辑的发布者节点：

```cpp
// 使用spinOnce的典型模式
ros::Rate rate(50);
while (ros::ok()) {
    // 用户逻辑：计算控制量、更新状态等
    doControl();

    // 处理一次回调（如更新传感器数据）
    ros::spinOnce();

    rate.sleep();
}
```

需要注意：若回调处理时间过长，而`spinOnce()`调用间隔过大，消息队列可能溢出，导致旧消息被丢弃。


## Python 节点编程（rospy）

rospy是ROS 1的Python客户端库，使用纯Python实现，接口简洁，适合快速原型开发、脚本编写和算法验证。

### Python 发布者示例

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from std_msgs.msg import String

def talker():
    # 初始化节点，anonymous=True会在节点名末尾附加随机数，
    # 从而允许同时运行多个同名节点（常用于测试）
    rospy.init_node('talker', anonymous=True)

    # 创建发布者
    pub = rospy.Publisher('chatter', String, queue_size=10)

    # 创建Rate对象，设定循环频率为10 Hz
    rate = rospy.Rate(10)

    count = 0
    # rospy.is_shutdown()在节点收到关闭信号时返回True
    while not rospy.is_shutdown():
        msg = String()
        msg.data = 'hello world {}'.format(count)

        rospy.loginfo(msg.data)
        pub.publish(msg)

        count += 1
        # Rate.sleep()会自动补偿回调和计算耗时，
        # 确保实际循环频率尽量接近设定值
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
```


### Python 订阅者示例

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from std_msgs.msg import String

def callback(msg):
    # msg是接收到的消息对象，类型为std_msgs.msg.String
    rospy.loginfo('I heard: %s', msg.data)

def listener():
    rospy.init_node('listener', anonymous=True)

    # 创建订阅者
    rospy.Subscriber('chatter', String, callback)

    # rospy.spin()阻塞当前线程直到节点关闭
    # 与roscpp不同，rospy的回调在独立线程中执行，
    # spin()仅用于防止主线程退出
    rospy.spin()

if __name__ == '__main__':
    listener()
```


### Python 服务端与客户端

服务 (Service) 适用于需要立即返回结果的请求-响应场景。以下示例使用`std_srvs/SetBool`服务类型。

**服务端（server）：**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from std_srvs.srv import SetBool, SetBoolResponse

def handle_set_bool(req):
    """
    服务回调函数，req为请求对象，包含.data字段（bool类型）
    必须返回对应的Response对象
    """
    if req.data:
        rospy.loginfo('收到请求：开启')
        result_msg = '已开启'
    else:
        rospy.loginfo('收到请求：关闭')
        result_msg = '已关闭'

    # 构造并返回响应
    return SetBoolResponse(success=True, message=result_msg)

def server_node():
    rospy.init_node('set_bool_server')

    # 注册服务：服务名、服务类型、回调函数
    srv = rospy.Service('set_bool', SetBool, handle_set_bool)
    rospy.loginfo('服务 set_bool 已就绪')

    rospy.spin()

if __name__ == '__main__':
    server_node()
```

**客户端（client）：**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from std_srvs.srv import SetBool, SetBoolRequest

def client_node():
    rospy.init_node('set_bool_client')

    # 等待服务可用，超时前会阻塞
    # 若省略timeout参数则永久等待
    rospy.wait_for_service('set_bool', timeout=5.0)

    try:
        # 创建服务代理（ServiceProxy），调用方式如同本地函数
        set_bool = rospy.ServiceProxy('set_bool', SetBool)

        # 构造请求并调用服务（同步阻塞，直到收到响应）
        req = SetBoolRequest(data=True)
        resp = set_bool(req)

        rospy.loginfo('服务返回：success=%s, message=%s',
                      resp.success, resp.message)
    except rospy.ServiceException as e:
        rospy.logerr('服务调用失败：%s', str(e))
    except rospy.ROSException as e:
        rospy.logerr('等待服务超时：%s', str(e))

if __name__ == '__main__':
    client_node()
```


### Rate.sleep() 的自动补偿机制

`rospy.Rate` 的 `sleep()` 方法会追踪上次调用的实际时间，并自动补偿由回调处理或计算引入的额外延迟。例如，设定频率为10 Hz（即周期100 ms），若某次循环耗时120 ms，则下次`sleep()`会缩短休眠时间以弥补超时，从而使长期平均频率尽量稳定在10 Hz。

若实际耗时超过一个完整周期，`sleep()`会立即返回（不休眠）并给出警告，同时将下次计时基准重置为当前时间，避免连续超时导致的"追赶"效应。


## 动作（Actions）

### actionlib 库简介

`actionlib` 是 ROS 1 中用于执行**长时任务**的通信机制。与服务不同，动作调用是异步的，支持在任务执行过程中：

- 发送**反馈** (Feedback)：持续向客户端报告任务进度
- 支持**取消** (Cancel)：客户端可在任务完成前中止任务
- 获取最终**结果** (Result)：任务完成后返回结果

典型使用场景包括：机器人导航（移动到目标点）、机械臂轨迹执行、拍照等需要数秒乃至数分钟的操作。

### .action 文件格式

动作定义文件（`.action`）由三个部分组成，用 `---` 分隔：

```
# 文件名：Fibonacci.action（位于 action/ 目录下）

# 目标（Goal）：客户端发送给服务端的请求
int32 order
---
# 结果（Result）：任务完成时服务端返回给客户端的最终结果
int32[] sequence
---
# 反馈（Feedback）：任务执行过程中服务端周期性发送的中间状态
int32[] partial_sequence
```

catkin 构建系统会根据 `.action` 文件自动生成相应的 C++ 和 Python 消息类型。

### SimpleActionServer（Python 示例）

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import actionlib
# 假设包名为 actionlib_tutorials，消息由 .action 文件生成
from actionlib_tutorials.msg import FibonacciAction, FibonacciFeedback, FibonacciResult

class FibonacciServer:
    def __init__(self):
        # 创建SimpleActionServer
        # execute_cb：每次收到新Goal时调用的回调函数
        # auto_start=False：手动调用start()以避免竞态条件
        self.server = actionlib.SimpleActionServer(
            'fibonacci',
            FibonacciAction,
            execute_cb=self.execute_cb,
            auto_start=False
        )
        self.server.start()
        rospy.loginfo('Fibonacci动作服务端已启动')

    def execute_cb(self, goal):
        rospy.loginfo('收到目标：order=%d', goal.order)
        feedback = FibonacciFeedback()
        result = FibonacciResult()

        sequence = [0, 1]
        feedback.partial_sequence = sequence
        success = True

        for i in range(2, goal.order):
            # 检查是否有取消请求
            if self.server.is_preempt_requested():
                rospy.loginfo('任务被取消')
                self.server.set_preempted()
                success = False
                break

            sequence.append(sequence[-1] + sequence[-2])
            feedback.partial_sequence = sequence

            # 发布中间反馈
            self.server.publish_feedback(feedback)
            rospy.sleep(0.5)  # 模拟计算耗时

        if success:
            result.sequence = sequence
            rospy.loginfo('任务完成，结果长度：%d', len(result.sequence))
            # 标记任务成功并返回结果
            self.server.set_succeeded(result)

if __name__ == '__main__':
    rospy.init_node('fibonacci_server')
    server = FibonacciServer()
    rospy.spin()
```


### SimpleActionClient（Python 示例）

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import actionlib
from actionlib_tutorials.msg import FibonacciAction, FibonacciGoal

def feedback_cb(feedback):
    rospy.loginfo('反馈：当前序列长度 = %d', len(feedback.partial_sequence))

def main():
    rospy.init_node('fibonacci_client')

    # 创建动作客户端，连接到"fibonacci"动作服务端
    client = actionlib.SimpleActionClient('fibonacci', FibonacciAction)

    # 等待服务端启动（最多等待5秒）
    rospy.loginfo('等待动作服务端...')
    client.wait_for_server(timeout=rospy.Duration(5.0))
    rospy.loginfo('已连接到动作服务端')

    # 构造并发送目标
    goal = FibonacciGoal(order=10)
    # send_goal是非阻塞的，可传入回调函数
    client.send_goal(goal, feedback_cb=feedback_cb)

    # 阻塞等待结果，设置超时时间
    finished = client.wait_for_result(rospy.Duration(30.0))

    if finished:
        state = client.get_state()
        result = client.get_result()
        rospy.loginfo('任务状态：%d，结果：%s', state, result.sequence)
    else:
        rospy.logwarn('任务超时，主动取消')
        client.cancel_goal()

if __name__ == '__main__':
    main()
```


### 动作 vs 服务的选择

| 特性 | 服务（Service） | 动作（Action） |
| --- | --- | --- |
| 通信方式 | 同步（阻塞调用） | 异步（非阻塞） |
| 执行时长 | 短暂（毫秒级） | 任意时长（秒至分钟） |
| 中间反馈 | 不支持 | 支持 |
| 取消功能 | 不支持 | 支持 |
| 典型场景 | 参数查询、模式切换 | 导航、轨迹执行、抓取 |

一般原则：若任务执行时间超过100 ms或需要进度反馈，优先选择动作而非服务。


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

1. ROS Wiki, [ROS Distributions](http://wiki.ros.org/Distributions)
2. [ROS/Introduction](http://wiki.ros.org/ROS/Introduction), ROS Wiki
3. [catkin/conceptual_overview](http://wiki.ros.org/catkin/conceptual_overview), ROS Wiki
4. [ROS/Installation](http://wiki.ros.org/ROS/Installation), ROS Wiki
5. [roscpp Overview](http://wiki.ros.org/roscpp/Overview), ROS Wiki
6. [rospy Overview](http://wiki.ros.org/rospy/Overview), ROS Wiki
7. [actionlib](http://wiki.ros.org/actionlib), ROS Wiki
8. [tf2](http://wiki.ros.org/tf2), ROS Wiki
9. [roslaunch/XML](http://wiki.ros.org/roslaunch/XML), ROS Wiki
10. [rosbag/Commandline](http://wiki.ros.org/rosbag/Commandline), ROS Wiki
11. [rosbag/Code API](http://wiki.ros.org/rosbag/Code%20API), ROS Wiki
12. [URDF/XML/robot](http://wiki.ros.org/urdf/XML/robot), ROS Wiki
13. [xacro](http://wiki.ros.org/xacro), ROS Wiki
14. [navigation/Tutorials](http://wiki.ros.org/navigation/Tutorials), ROS Wiki
15. [dwa_local_planner](http://wiki.ros.org/dwa_local_planner), ROS Wiki
16. [amcl](http://wiki.ros.org/amcl), ROS Wiki
17. Thrun S, Burgard W, Fox D. Probabilistic Robotics. MIT Press, 2005.
18. Fox D, Burgard W, Dellaert F, Thrun S. Monte Carlo Localization: Efficient Position Estimation for Mobile Robots. AAAI, 1999.
19. Marder-Eppstein E, Berger E, Foote T, et al. The Office Marathon: Robust Navigation in an Indoor Office Environment. ICRA, 2010.
