# ROS 1 节点编程

!!! note "引言"
    节点（Node）是 ROS 1 中最小的可执行单元。ROS 1 提供两套官方客户端库：roscpp 面向性能敏感的实时任务，直接映射底层通信机制，编译为原生可执行文件；rospy 以纯 Python 实现，接口简洁，适合快速原型与算法验证。两者在概念上完全一致，差别集中在回调调度、多线程模型与生命周期管理的写法上。本页面给出两套库的完整节点编写模板与关键 API 说明。


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


## 参考资料

1. ROS Wiki, [roscpp](http://wiki.ros.org/roscpp)
2. ROS Wiki, [rospy](http://wiki.ros.org/rospy)
3. ROS Wiki, [Writing a Simple Publisher and Subscriber](http://wiki.ros.org/ROS/Tutorials)
4. [ROS 1 通信机制](ros1-communication.md) —— 话题、服务、动作的概念与自定义消息
5. [ROS 1 工作空间与软件包](ros1-workspace.md) —— CMakeLists.txt 与 package.xml 配置
