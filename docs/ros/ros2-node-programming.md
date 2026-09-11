# ROS 2 节点编程

!!! note "引言"
    ROS 2 的客户端库建立在统一的 rcl（ROS Client Library）核心之上，rclpy 与 rclcpp 分别是它的 Python 与 C++ 绑定，因此两者的 API 命名与生命周期语义高度一致。与 ROS 1 相比最显著的变化是节点以面向对象方式组织：用户继承 `Node` 基类，在构造函数中创建发布者、订阅者与定时器，由执行器（Executor）统一调度回调。本页面给出两套库的节点编写模板、执行器模型与参数处理方式。


## rclpy Python节点编程

rclpy是ROS 2的Python客户端库，封装了底层的rcl（ROS Client Library）接口，是编写Python节点的标准方式。

### 节点初始化与执行器

在任何rclpy程序中，必须首先调用`rclpy.init()`初始化ROS 2上下文，并在程序退出前调用`rclpy.shutdown()`释放资源：

```python
import rclpy

rclpy.init(args=None)   # 初始化，可传入命令行参数
# ... 创建节点并使用
rclpy.shutdown()        # 清理资源
```

**spin函数**控制节点的事件循环：

- `rclpy.spin(node)`：阻塞式运行，持续处理回调直到节点被关闭，适合大多数场景
- `rclpy.spin_once(node, timeout_sec=0)`：处理一次回调后立即返回，适合需要在主循环中穿插其他逻辑的场景
- `rclpy.spin_until_future_complete(node, future)`：运行直到指定的Future完成，常用于服务客户端等待响应

**执行器 (Executor)** 管理回调的调度方式：

- `SingleThreadedExecutor`：所有回调在单一线程中顺序执行，这是默认行为，适合不需要并发的节点
- `MultiThreadedExecutor`：回调可在多个线程中并发执行，适合包含耗时回调（如图像处理）的节点，需配合`ReentrantCallbackGroup`或`MutuallyExclusiveCallbackGroup`使用

```python
from rclpy.executors import MultiThreadedExecutor

executor = MultiThreadedExecutor(num_threads=4)
executor.add_node(node_a)
executor.add_node(node_b)
executor.spin()
```

### 发布者节点

下面是一个完整的发布者节点示例，以固定频率发布字符串消息：

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MinimalPublisher(Node):
    """最简发布者节点示例。"""

    def __init__(self):
        super().__init__('minimal_publisher')
        # 创建发布者：消息类型、话题名称、队列深度
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        # 创建定时器：定时周期（秒）、回调函数
        timer_period = 0.5  # 以2 Hz频率发布
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'发布: "{msg.data}"')
        self.i += 1


def main(args=None):
    rclpy.init(args=args)
    node = MinimalPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### 订阅者节点

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MinimalSubscriber(Node):
    """最简订阅者节点示例。"""

    def __init__(self):
        super().__init__('minimal_subscriber')
        # 创建订阅者：消息类型、话题名称、回调函数、队列深度
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10
        )
        # 防止Python垃圾回收订阅对象
        self.subscription

    def listener_callback(self, msg):
        self.get_logger().info(f'收到: "{msg.data}"')


def main(args=None):
    rclpy.init(args=args)
    node = MinimalSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### 服务端与客户端

**服务端**使用`create_service`注册回调函数，当客户端发送请求时自动调用：

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts


class AddTwoIntsServer(Node):

    def __init__(self):
        super().__init__('add_two_ints_server')
        self.srv = self.create_service(
            AddTwoInts,
            'add_two_ints',
            self.add_two_ints_callback
        )

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(
            f'接收请求: a={request.a}, b={request.b} -> 返回: {response.sum}'
        )
        return response


def main(args=None):
    rclpy.init(args=args)
    node = AddTwoIntsServer()
    rclpy.spin(node)
    rclpy.shutdown()
```

**服务客户端**使用`create_client`，并通过`call_async`发送异步请求：

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts


class AddTwoIntsClient(Node):

    def __init__(self):
        super().__init__('add_two_ints_client')
        self.client = self.create_client(AddTwoInts, 'add_two_ints')
        # 等待服务上线
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('等待服务上线...')

    def send_request(self, a, b):
        request = AddTwoInts.Request()
        request.a = a
        request.b = b
        # 发送异步请求，返回Future对象
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()


def main(args=None):
    rclpy.init(args=args)
    client = AddTwoIntsClient()
    result = client.send_request(3, 5)
    client.get_logger().info(f'结果: {result.sum}')
    client.destroy_node()
    rclpy.shutdown()
```

## rclcpp C++节点编程

rclcpp是ROS 2的C++客户端库，提供与rclpy相对应的C++接口，通常用于对性能要求更高的场景。

### 发布者节点

C++节点通过继承`rclcpp::Node`类来实现，并使用智能指针管理资源：

```cpp
#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

using namespace std::chrono_literals;

class MinimalPublisher : public rclcpp::Node
{
public:
    MinimalPublisher()
    : Node("minimal_publisher"), count_(0)
    {
        // 创建发布者：话题名称、队列深度
        publisher_ = this->create_publisher<std_msgs::msg::String>("topic", 10);
        // 创建定时器：周期、回调（绑定成员函数）
        timer_ = this->create_wall_timer(
            500ms,
            std::bind(&MinimalPublisher::timer_callback, this)
        );
    }

private:
    void timer_callback()
    {
        auto message = std_msgs::msg::String();
        message.data = "Hello, world! " + std::to_string(count_++);
        RCLCPP_INFO(this->get_logger(), "发布: '%s'", message.data.c_str());
        publisher_->publish(message);
    }

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    size_t count_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    // make_shared自动管理节点生命周期
    rclcpp::spin(std::make_shared<MinimalPublisher>());
    rclcpp::shutdown();
    return 0;
}
```

### 订阅者节点

```cpp
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

using std::placeholders::_1;

class MinimalSubscriber : public rclcpp::Node
{
public:
    MinimalSubscriber()
    : Node("minimal_subscriber")
    {
        subscription_ = this->create_subscription<std_msgs::msg::String>(
            "topic",
            10,
            std::bind(&MinimalSubscriber::topic_callback, this, _1)
        );
    }

private:
    void topic_callback(const std_msgs::msg::String & msg) const
    {
        RCLCPP_INFO(this->get_logger(), "收到: '%s'", msg.data.c_str());
    }

    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MinimalSubscriber>());
    rclcpp::shutdown();
    return 0;
}
```

**关键模式说明**：

- `RCLCPP_INFO(logger, fmt, ...)`：输出INFO级别日志，类似的宏还有`RCLCPP_WARN`、`RCLCPP_ERROR`、`RCLCPP_DEBUG`
- `std::make_shared<T>()`：创建共享指针，ROS 2中节点和大多数资源都通过共享指针管理
- `std::bind(&Class::method, this, _1)`：将成员函数绑定为回调，`_1`代表回调的第一个参数占位符

## 参考资料

1. ROS 2 Documentation, [rclpy API](https://docs.ros.org/en/humble/p/rclpy/)
2. ROS 2 Documentation, [rclcpp API](https://docs.ros.org/en/humble/p/rclcpp/)
3. ROS 2 Documentation, [Writing a simple publisher and subscriber](https://docs.ros.org/en/humble/Tutorials/)
4. [动作与组件节点](ros2-actions-components.md)
5. [QoS 与 DDS](ros2-qos-dds.md)
