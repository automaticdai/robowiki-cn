# ROS 2 动作与组件节点

!!! note "引言"
    话题与服务之外，ROS 2 还提供两项面向工程落地的机制。动作（Action）针对导航、轨迹执行这类耗时任务，在服务的请求-响应之上增加了持续反馈与中途取消能力，其底层由若干话题与服务组合实现。组件节点（Component Node）则解决性能问题：把多个节点加载进同一进程的容器中，消息通过进程内指针传递而非序列化后过网络栈，对相机图像与点云等大数据量话题可显著降低延迟与 CPU 占用。本页面介绍两者的实现方式。


## 动作（Actions）

动作 (Actions) 是ROS 2中适用于长时间运行任务的通信机制，结合了服务（请求/响应）和话题（持续反馈）的特点，并支持任务取消。典型应用场景包括导航到目标点、执行机械臂轨迹等。

### .action文件格式

动作接口定义在`.action`文件中，包含三个部分，用`---`分隔：

```
# Fibonacci.action
# 目标（Goal）：客户端发送给服务端
int32 order
---
# 结果（Result）：任务完成后服务端返回给客户端
int32[] sequence
---
# 反馈（Feedback）：任务进行中服务端持续发送给客户端
int32[] partial_sequence
```

### 动作服务端

```python
import time
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from action_tutorials_interfaces.action import Fibonacci


class FibonacciActionServer(Node):

    def __init__(self):
        super().__init__('fibonacci_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            self.execute_callback
        )

    def execute_callback(self, goal_handle):
        self.get_logger().info(f'执行目标: order={goal_handle.request.order}')

        feedback_msg = Fibonacci.Feedback()
        feedback_msg.partial_sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            # 检查是否收到取消请求
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('目标已取消')
                return Fibonacci.Result()

            # 计算下一个斐波那契数
            feedback_msg.partial_sequence.append(
                feedback_msg.partial_sequence[i] + feedback_msg.partial_sequence[i - 1]
            )
            self.get_logger().info(f'反馈: {feedback_msg.partial_sequence}')
            # 发布中间反馈
            goal_handle.publish_feedback(feedback_msg)
            time.sleep(1)

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.partial_sequence
        return result
```

### 动作客户端

```python
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from action_tutorials_interfaces.action import Fibonacci


class FibonacciActionClient(Node):

    def __init__(self):
        super().__init__('fibonacci_action_client')
        self._action_client = ActionClient(self, Fibonacci, 'fibonacci')

    def send_goal(self, order):
        goal_msg = Fibonacci.Goal()
        goal_msg.order = order

        self._action_client.wait_for_server()

        # 异步发送目标，注册反馈回调
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        # 目标被服务端接受/拒绝时触发
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('目标被拒绝')
            return
        self.get_logger().info('目标已接受')
        # 注册结果回调
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'收到反馈: {feedback.partial_sequence}')

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'最终结果: {result.sequence}')
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    client = FibonacciActionClient()
    client.send_goal(10)
    rclpy.spin(client)
```

## 组件（Component）节点

组件节点（Component Nodes）是ROS 2推荐的进程内通信方案，允许将多个节点加载到同一个进程（容器）中运行，通过绕过序列化和网络栈，显著降低大数据量通信（如相机图像、点云）的延迟和CPU占用。

### 进程内通信的优势

- **零拷贝传输**：对于支持的消息类型，消息数据不需要序列化和反序列化，直接通过指针共享
- **降低延迟**：消除了网络栈的开销，延迟可从毫秒级降至微秒级
- **减少CPU占用**：特别是对于高频大消息（1080p图像约6 MB/帧），效果显著

### 定义组件节点

组件节点与普通节点的代码几乎完全相同，唯一区别是需要在文件末尾注册组件：

```cpp
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "std_msgs/msg/string.hpp"

namespace composition
{

class Talker : public rclcpp::Node
{
public:
    explicit Talker(const rclcpp::NodeOptions & options)
    : Node("talker", options)
    {
        publisher_ = this->create_publisher<std_msgs::msg::String>("chatter", 10);
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(500),
            [this]() {
                auto msg = std_msgs::msg::String();
                msg.data = "Hello, component!";
                RCLCPP_INFO(this->get_logger(), "发布: '%s'", msg.data.c_str());
                publisher_->publish(msg);
            }
        );
    }

private:
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace composition

// 注册组件节点，使其可被动态加载
RCLCPP_COMPONENTS_REGISTER_NODE(composition::Talker)
```

CMakeLists.txt中还需要添加组件注册和库构建配置：

```cmake
add_library(talker_component SHARED src/talker.cpp)
rclcpp_components_register_node(talker_component
    PLUGIN "composition::Talker"
    EXECUTABLE talker_node
)
```

### 动态加载组件

使用`ros2 component`命令在运行时动态加载组件到容器进程：

```bash
# 启动一个空的组件容器进程
ros2 run rclcpp_components component_container

# 在另一个终端，将Talker组件加载进容器
ros2 component load /ComponentManager composition composition::Talker

# 列出容器中已加载的组件
ros2 component list

# 卸载组件（使用组件ID）
ros2 component unload /ComponentManager 1
```

### 在Launch文件中使用组件

通过Launch文件将多个组件加载到同一容器，是推荐的生产部署方式：

```python
from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    container = ComposableNodeContainer(
        name='my_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            ComposableNode(
                package='composition',
                plugin='composition::Talker',
                name='talker'
            ),
            ComposableNode(
                package='composition',
                plugin='composition::Listener',
                name='listener'
            ),
        ],
        output='screen',
    )

    return LaunchDescription([container])
```

启动后，Talker和Listener运行在同一进程内，消息通过共享内存传递，相比跨进程通信性能大幅提升。

## 参考资料

1. ROS 2 Documentation, [Understanding actions](https://docs.ros.org/en/humble/Tutorials/Beginner-CLI-Tools/Understanding-ROS2-Actions/)
2. ROS 2 Documentation, [Composing multiple nodes in a single process](https://docs.ros.org/en/humble/Tutorials/Intermediate/Composition.html)
3. ROS 2 Design, *Intra-Process Communication*. https://design.ros2.org/
4. [ROS 2 节点编程](ros2-node-programming.md)
5. [QoS 与 DDS](ros2-qos-dds.md)
