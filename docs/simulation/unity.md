# Unity

- 官方网站：https://unity.com/
- Unity Robotics Hub：https://github.com/Unity-Technologies/Unity-Robotics-Hub
- 物理引擎：PhysX (内置) / Havok Physics
- 许可：个人版免费 / 专业版收费

!!! note "引言"
    Unity是全球使用最广泛的游戏引擎之一，近年来在机器人仿真和工业数字孪生 (Digital Twin) 领域的应用迅速增长。Unity Technologies专门成立了Robotics团队，推出了Unity Robotics Hub等工具集，致力于将Unity打造为机器人开发和验证的重要平台。Unity的易用性、跨平台能力和丰富的资源生态使其成为机器人仿真领域的有力竞争者。

## Unity 在机器人仿真中的优势

Unity作为游戏引擎进入机器人仿真领域，带来了多方面的独特优势：

- **易用性 (Usability)**：Unity编辑器界面直观，学习曲线比Unreal Engine更平缓，适合非游戏开发背景的机器人研究人员
- **跨平台部署 (Cross-Platform Deployment)**：支持Windows、Linux、macOS以及移动设备和Web平台
- **C# 编程**：使用C#语言进行开发，语法简洁，开发效率高
- **丰富的资产商店 (Asset Store)**：提供大量三维模型、场景、材质等资源，可快速构建仿真环境


## Unity Robotics Hub

Unity Robotics Hub是Unity官方推出的机器人开发工具集合，提供了将Unity与机器人开发生态连接的核心组件：

- **URDF Importer**：将URDF格式的机器人模型导入Unity场景，自动创建关节结构和碰撞体
- **ROS-TCP-Connector**：Unity侧的ROS通信组件，通过TCP连接与ROS系统交换消息
- **ROS-TCP-Endpoint**：ROS侧的通信节点，负责将ROS消息转发给Unity
- **示例项目**：提供机械臂抓取 (Pick-and-Place)、导航、SLAM等完整示例


## ROS-Unity 集成

Unity与ROS/ROS 2的集成通过TCP通信桥接实现。该架构支持双向消息传递：

- Unity仿真中的传感器数据（相机图像、激光雷达点云、IMU数据等）可以以ROS消息格式发布
- ROS侧的控制指令可以传递到Unity中驱动仿真机器人
- 支持自定义ROS消息类型
- 通信延迟低，适合实时控制回路的仿真

这种集成方式使得在ROS中开发的算法可以直接在Unity仿真环境中进行测试和验证。


## ML-Agents 工具包

Unity ML-Agents Toolkit是Unity官方推出的机器学习工具包 (Machine Learning Toolkit)，为在Unity环境中训练智能体 (Agent) 提供了完整的框架：

- **训练算法**：内置PPO (Proximal Policy Optimization)、SAC (Soft Actor-Critic) 等主流强化学习算法
- **模仿学习 (Imitation Learning)**：支持通过人类演示进行行为克隆 (Behavioral Cloning) 和GAIL
- **课程学习 (Curriculum Learning)**：支持逐步增加任务难度的训练策略
- **多智能体训练 (Multi-Agent Training)**：支持协作和对抗场景下的多智能体同时训练
- **Python API**：通过Python接口与PyTorch等深度学习框架集成

在机器人领域，ML-Agents可用于训练机械臂操作、移动机器人导航、多机器人协调等任务。


## 传感器仿真

Unity中可以实现多种机器人传感器的仿真：

- **相机 (Camera)**：利用Unity的渲染管线生成RGB图像、深度图和法线图
- **激光雷达 (LiDAR)**：通过射线投射 (Raycasting) 模拟二维和三维激光雷达扫描
- **IMU**：基于Unity物理引擎的刚体数据模拟加速度计和陀螺仪
- **接触传感器 (Contact Sensor)**：利用物理引擎的碰撞回调检测接触事件
- **GPS**：基于仿真世界坐标系模拟全局定位数据

Unity Perception包还提供了语义分割 (Semantic Segmentation)、实例分割 (Instance Segmentation) 和边界框 (Bounding Box) 标注的自动生成功能。


## 数字孪生 (Digital Twin)

Unity在数字孪生领域的应用日益广泛。通过高质量的三维渲染和实时数据集成，Unity可以创建工厂、仓库和其他工业环境的数字镜像：

- 实时可视化机器人的运行状态和传感器数据
- 在数字孪生中测试新的控制策略和工作流程
- 与云平台集成，实现远程监控和分析
- 支持与PLC (可编程逻辑控制器) 和工业通信协议对接


## HDRP 高清渲染管线

Unity的高清渲染管线 (HDRP, High Definition Render Pipeline) 为机器人仿真提供了高质量的视觉效果：

- **光线追踪 (Ray Tracing)**：支持实时光线追踪反射和全局光照
- **体积效果 (Volumetric Effects)**：模拟雾、烟尘等大气效果
- **后处理效果 (Post-Processing)**：运动模糊、景深、色调映射等
- **物理光照单位 (Physical Light Units)**：使用真实世界的光照参数配置场景

HDRP渲染的高保真图像可用于训练计算机视觉模型，有效缩小仿真到真实的差距 (Sim-to-Real Gap)。


## ML-Agents 框架

Unity ML-Agents Toolkit 是 Unity 官方推出的机器人与游戏 AI 强化学习框架，允许直接在 Unity 场景中训练智能体。

### 安装

```bash
pip install mlagents
# Unity 侧需安装 ML-Agents Unity Package（通过 Package Manager）
```

### 核心概念

- **Agent（智能体）**：继承 `Unity.MLAgents.Agent` 的 C# 组件，负责收集观测、执行动作和计算奖励
- **Behavior（行为）**：定义智能体的策略名称和动作/观测空间
- **Environment（环境）**：Unity 场景即为训练环境，可通过 `--num-envs` 参数启动多个并行实例

### 自定义 Agent 示例（C#）

```csharp
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using UnityEngine;

public class RobotAgent : Agent
{
    [SerializeField] private Rigidbody rb;
    [SerializeField] private Transform target;
    private float previousDistance;

    public override void OnEpisodeBegin()
    {
        // 随机重置机器人位置
        rb.velocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;
        transform.localPosition = new Vector3(Random.Range(-4f, 4f), 0.5f, Random.Range(-4f, 4f));
        target.localPosition = new Vector3(Random.Range(-4f, 4f), 0.5f, Random.Range(-4f, 4f));
        previousDistance = Vector3.Distance(transform.localPosition, target.localPosition);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // 观测：自身位置、目标位置、自身速度（共 9 维）
        sensor.AddObservation(transform.localPosition);
        sensor.AddObservation(target.localPosition);
        sensor.AddObservation(rb.velocity);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        // 连续动作：前后、左右移动
        float forceX = actions.ContinuousActions[0];
        float forceZ = actions.ContinuousActions[1];
        rb.AddForce(new Vector3(forceX, 0, forceZ) * 10f);

        // 奖励：靠近目标
        float currentDistance = Vector3.Distance(transform.localPosition, target.localPosition);
        float reward = (previousDistance - currentDistance) * 0.1f;
        AddReward(reward);
        previousDistance = currentDistance;

        // 到达目标
        if (currentDistance < 0.5f)
        {
            AddReward(1.0f);
            EndEpisode();
        }

        // 超出边界惩罚
        if (Mathf.Abs(transform.localPosition.x) > 5f || Mathf.Abs(transform.localPosition.z) > 5f)
        {
            AddReward(-1.0f);
            EndEpisode();
        }
    }

    // 手动控制（调试用）
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var actions = actionsOut.ContinuousActions;
        actions[0] = Input.GetAxis("Horizontal");
        actions[1] = Input.GetAxis("Vertical");
    }
}
```

### 训练配置（YAML）

```yaml
behaviors:
  RobotAgent:
    trainer_type: ppo
    hyperparameters:
      batch_size: 1024
      buffer_size: 10240
      learning_rate: 3.0e-4
      beta: 5.0e-3          # 熵系数（鼓励探索）
      epsilon: 0.2           # PPO 裁剪参数
      lambd: 0.95            # GAE Lambda
      num_epoch: 3
    network_settings:
      normalize: true
      hidden_units: 256
      num_layers: 2
    reward_signals:
      extrinsic:
        gamma: 0.99
        strength: 1.0
    max_steps: 5000000
    time_horizon: 64
    summary_freq: 10000
```

```bash
# 启动训练（Unity 编辑器中按 Play）
mlagents-learn config/robot_config.yaml --run-id=robot_run_01

# 多进程并行（需构建可执行文件）
mlagents-learn config/robot_config.yaml --run-id=parallel_run --num-envs=8 \
  --env=./Build/RobotEnv
```

### TensorBoard 监控

```bash
tensorboard --logdir results/robot_run_01
```


## Unity Perception：合成数据生成

Unity Perception 包专为生成带标注的合成训练数据设计，可快速生成大量目标检测/分割数据：

- **Randomizer 系统**：自动随机化物体位姿、材质、光照、相机角度、遮挡情况
- **标注格式**：自动生成 COCO 格式的 JSON 标注文件（边界框、语义分割、关键点）
- **Domain Randomization**：通过丰富的域随机化减小 sim-to-real gap

```csharp
// 添加光照随机化器
using UnityEngine.Perception.Randomization.Randomizers;

[AddRandomizerMenu("Perception/Light Randomizer")]
public class LightRandomizer : Randomizer
{
    public FloatParameter intensity = new FloatParameter { value = new UniformSampler(0.5f, 2.0f) };

    protected override void OnIterationStart()
    {
        var light = FindObjectOfType<Light>();
        light.intensity = intensity.Sample();
    }
}
```


## ROS# 与 ROS 2 集成

ROS# 是连接 Unity 与 ROS/ROS 2 的中间件，通过 WebSocket 桥接实现双向通信：

```bash
# 安装 ROS TCP Connector（Unity Package Manager）
# 在 ROS 2 侧安装端点
pip install ros-tcp-endpoint
ros2 run ros_tcp_endpoint default_server_endpoint --ros-args -p ROS_IP:=127.0.0.1
```

Unity 侧代码示例（发布机器人命令）：

```csharp
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Geometry;

public class RobotCommander : MonoBehaviour
{
    ROSConnection ros;
    const string topicName = "/cmd_vel";

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<TwistMsg>(topicName);
    }

    void Update()
    {
        var msg = new TwistMsg();
        msg.linear.x = 0.5;
        msg.angular.z = 0.1;
        ros.Publish(topicName, msg);
    }
}
```


## Sim-to-Real 策略

Unity 中常用的 sim-to-real 技术：

| 技术 | 说明 | 效果 |
|------|------|------|
| 域随机化（Domain Randomization） | 随机化物理参数、外观、光照 | 提升策略鲁棒性 |
| 域适应（Domain Adaptation） | 将仿真图像风格迁移到真实图像风格 | 视觉任务迁移 |
| 真实数据微调 | 用少量真实数据对预训练策略进行微调 | 最终性能校准 |
| 系统辨识（System Identification） | 用真实硬件数据标定仿真物理参数 | 减小动力学差距 |


## 参考资料

- [Unity Robotics Hub GitHub](https://github.com/Unity-Technologies/Unity-Robotics-Hub)
- [Unity ML-Agents文档](https://unity-technologies.github.io/ml-agents/)
- [Unity Perception包](https://github.com/Unity-Technologies/com.unity.perception)
- [Unity HDRP文档](https://docs.unity3d.com/Packages/com.unity.render-pipelines.high-definition@latest)
- Juliani, A., Berges, V. P., Teng, E., et al. (2018). Unity: A general platform for intelligent agents. *arXiv preprint arXiv:1809.02627*.
- [Unity ML-Agents 官方文档](https://unity-technologies.github.io/ml-agents/)
- [Unity Perception 文档](https://docs.unity3d.com/Packages/com.unity.perception@0.11/manual/index.html)
- [ROS# GitHub](https://github.com/siemens/ros-sharp)
- Juliani, A., et al. (2020). Unity: A General Platform for Intelligent Agents. *arXiv:1809.02627*.
