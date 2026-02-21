# SoftBank Pepper

!!! note "引言"
    Pepper 是由 SoftBank Robotics（软银机器人）于 2014 年发布的全球首款面向大众市场的社交人形机器人（Social Humanoid Robot）。它具备情绪感知与表达能力，能够通过语音、肢体动作和胸部平板电脑与人类进行自然交互。Pepper 的设计初衷并非执行体力劳动，而是作为陪伴者、迎宾员和教育辅助角色活跃于零售、医疗、教育等场景。自发布以来，Pepper 已部署于全球 70 余个国家，成为商业服务机器人领域最具代表性的产品之一。


## 发展历程

Pepper 的诞生源于 SoftBank Robotics 前身——法国公司 Aldebaran Robotics（阿德巴兰机器人）的长期研究积累。以下为 Pepper 的主要里程碑：

- **2012 年**：SoftBank Robotics（原 Aldebaran Robotics）正式启动 Pepper 项目研发，目标是将 NAO 机器人的技术平台商业化，打造一款能够进入公共场所的社交机器人。

- **2014 年 6 月**：Pepper 在日本正式对外发布，定价约 198,000 日元（约合 1,800 美元），并额外收取每月约 24,600 日元的保险及云服务费用。发布会上，软银创始人孙正义亲自出席，引发全球广泛关注。

- **2015 年 6 月**：Pepper 开始向日本消费者和企业开放销售，首批 1,000 台在上线后不足一分钟内售罄。同年，Pepper 开始出现在软银门店担任导购角色。

- **2016 年**：SoftBank Robotics 推出面向企业的 **Pepper for Biz** 版本，提供更完善的企业级支持和定制化服务包，陆续部署到零售店、银行、医疗机构及会展中心。

- **2018 年**：Pepper 进入欧美市场，包括英国、法国、美国等多个国家开展商业合作。

- **2019 年**：全球累计销量突破 27,000 台，部署覆盖 70 余个国家和地区，成为全球部署规模最大的社交人形机器人。

- **2021 年**：受新冠疫情冲击及商业模式挑战影响，SoftBank 宣布暂停 Pepper 的大规模生产，同时削减 Pepper 相关团队人员，引发业界对服务机器人商业化可持续性的广泛讨论。官方未宣布完全停产，但生产规模大幅收缩。

### 与 NAO 的关系

Pepper 与 NAO 同属 SoftBank Robotics 产品线，共享同一套软件生态系统——NAOqi OS。可以将 Pepper 理解为 NAO 的放大商业化版本：NAO 身高 58 cm，主要面向研究和教育市场；Pepper 身高 120 cm，配备轮式底盘和胸部平板电脑，更适合在公共场所与真实用户交互。两款机器人的开发工具链和 API 高度兼容，开发者可将为 NAO 编写的代码以较低成本迁移至 Pepper。


## 技术规格

| 参数 | 规格 |
|------|------|
| 身高 | 120 cm |
| 体重 | 28 kg |
| 移动底座 | 三向全向轮（Omnidirectional Wheels），最高速度 3 km/h |
| 自由度（DoF） | 20 DoF：头部 2 + 手臂 3×2 + 手 1×2 + 腰部 1 + 腿部 3 |
| 胸部屏幕 | 10.1 英寸触摸屏平板电脑（Android 系统） |
| 电池续航 | 待机约 12 小时，交互使用约 5–8 小时 |
| 充电时间 | 约 2.5 小时（满充） |
| 处理器 | Intel Atom（主控），NAOqi OS 运行于机身内嵌计算单元 |
| 内存 | 4 GB RAM |
| 存储 | 8 GB 闪存 + 可扩展 microSD |
| 无线通信 | Wi-Fi 802.11 a/b/g/n，以太网（维护用） |
| 工作温度 | 0–40°C |
| 防护等级 | 室内使用，非防水设计 |


## 感知系统

Pepper 配备了多模态感知系统（Multimodal Perception System），能够从视觉、听觉、触觉和测距等多个维度感知周围环境。

### 视觉传感器

- **头部 2D 摄像头**：2 个 720p RGB 摄像头（前向），用于人脸检测、表情识别及人体追踪。
- **胸部 3D 深度相机**：搭载华硕 Xtion（Asus Xtion）深度传感器，提供深度图像，支持手势识别和三维空间感知。
- **前额 RGB 相机**：1 个高分辨率摄像头，辅助近场人脸识别。

### 听觉传感器

- **麦克风阵列**：头部内置 4 个定向麦克风（Directional Microphone），支持噪声抑制（Noise Suppression）和声源定位（Sound Source Localization），可在嘈杂环境中识别说话方向并提高语音识别准确率。

### 触觉传感器

- **头部触摸传感器**：3 个电容式触摸传感器，分布于头顶前、中、后三处。
- **手部触摸传感器**：每只手 3 个，共 6 个，用于检测握手和触摸交互。

### 测距与避障传感器

- **激光雷达（LiDAR）**：底盘前方 2 个激光传感器，扫描角度各 270°，用于地面障碍物检测和路径规划。
- **声纳（Sonar）**：前后各 1 个，共 2 个超声波传感器，用于近距离（0.1–0.8 m）障碍物检测，与激光雷达互补。
- **红外传感器**：底部 6 个红外传感器，检测台阶和坡道，防止跌落。

### 惯性测量单元

- **IMU（惯性测量单元，Inertial Measurement Unit）**：机身内置 2 个 IMU，各含三轴陀螺仪（Gyroscope）和三轴加速度计（Accelerometer），用于姿态估计和运动稳定控制。


## 软件平台

### NAOqi OS

NAOqi OS 是 SoftBank Robotics 为 NAO 和 Pepper 开发的基于 Linux 的专有操作系统。它提供了统一的机器人应用编程接口（API），屏蔽了底层硬件差异，使开发者可以专注于上层行为逻辑的开发。NAOqi OS 采用模块化架构，各功能以 ALModule 的形式注册到名为 NAOqi Broker 的消息总线上，模块间通过远程过程调用（RPC）通信。

### NAOqi SDK

NAOqi SDK（Software Development Kit）支持多种编程语言：

- **Python**：最常用，适合快速原型开发
- **C++**：适合性能敏感的底层控制
- **Java**：通过 Android 平板集成扩展功能
- **ROS**：通过桥接驱动包实现集成（详见"与 ROS 集成"一节）

### Choregraphe

Choregraphe 是 SoftBank Robotics 官方提供的图形化编程工具（Graphical Programming Tool）。用户可通过拖拽行为模块（Behavior Box）并连接数据流来编写机器人行为，无需编写代码。Choregraphe 内置仿真视图，可在不连接实体机器人的情况下预览动作效果。

### ALModule 系统

NAOqi API 以 ALModule 为核心组织，常用模块包括：

| 模块名 | 功能 |
|--------|------|
| `ALTextToSpeech` | 语音合成（Text-to-Speech） |
| `ALSpeechRecognition` | 语音识别（Automatic Speech Recognition） |
| `ALMotion` | 关节运动控制 |
| `ALVideoDevice` | 摄像头图像获取 |
| `ALFaceDetection` | 人脸检测与追踪 |
| `ALBasicAwareness` | 自主感知与注意力管理 |
| `ALLeds` | 全身 LED 灯控制 |
| `ALTabletService` | 胸部平板电脑控制 |

### Python SDK 示例

以下示例展示了通过 `qi` 框架连接 Pepper 并执行基本交互操作：

```python
import qi

app = qi.Application()
app.start()
session = app.session

# 连接到 Pepper
session.connect("tcp://192.168.1.100:9559")

# 语音合成
tts = session.service("ALTextToSpeech")
tts.setLanguage("Chinese")
tts.say("你好，我是 Pepper！")

# 情绪表达（通过胸部 LED 和动作）
leds = session.service("ALLeds")
leds.fadeRGB("ChestLeds", 0.0, 1.0, 0.0, 0.5)  # 绿色，0.5秒

# 自主生活（轻微的自主动作）
awareness = session.service("ALBasicAwareness")
awareness.setEnabled(True)
awareness.setStimulusDetectionEnabled("Sound", True)

# 人脸检测与跟踪
face_detection = session.service("ALFaceDetection")
face_detection.subscribe("MyApp", 500, 0.0)
```


## 情感计算与社交交互

Pepper 的核心设计理念是情感计算（Affective Computing），即机器人能够感知、理解并表达情绪，从而与人类建立更自然的社会性连接。

### 情绪识别

- **面部表情分析**：利用头部摄像头捕捉人脸图像，通过面部动作编码系统（FACS，Facial Action Coding System）识别喜悦、惊讶、愤怒、悲伤等基本情绪。
- **语音情感分析**：对用户语音的语调（Intonation）、语速（Speech Rate）和音量进行实时分析，推断当前情绪状态，并相应调整交互策略。

### 自主情绪系统

Pepper 本身维护一套内部情绪状态机，包含好奇（Curious）、高兴（Happy）、不安（Uncomfortable）等状态。这些状态会随交互内容、时间流逝和环境刺激自动变化，并影响 Pepper 的肢体动作幅度、语速和 LED 颜色表达。这一机制使 Pepper 在长时间交互中显得更具生命感，而非机械地执行固定脚本。

### 近场交互

Pepper 的 **ALBasicAwareness** 模块能够持续扫描 3 米范围内的人体，一旦检测到有人靠近，便主动转向并发出问候。该模块整合了声音刺激检测、人体移动追踪和视觉显著性分析，形成完整的注意力管理（Attention Management）机制。

### 多通道输出

Pepper 通过以下通道综合表达情绪和意图：

- **语音**：语调和语速随情绪状态动态调整
- **肢体动作**：手臂、头部和腰部协同完成表情性动作（Expressive Motion）
- **LED 灯光**：眼部、耳部和胸部 LED 以颜色和节奏传递情绪信号
- **胸部平板电脑**：显示动态表情图案、信息卡片或引导界面


## 行业应用

Pepper 的商业部署覆盖多个行业，以下为典型案例：

### 零售业

万事达卡（Mastercard）、雀巢（Nestlé）等品牌将 Pepper 部署为门店迎宾机器人和产品介绍机器人。Pepper 能够主动向顾客介绍促销信息、回答常见问题，并引导顾客前往目标区域，有效提升门店的科技感和顾客体验。

### 银行

日本三菱 UFJ 银行（MUFG Bank）在多家网点引入 Pepper，用于接待客户、引导排队叫号，并解答业务咨询，缓解前台人员压力。

### 医疗与养老

部分日本养老院将 Pepper 部署为老人陪伴机器人，提供聊天互动、音乐播放和轻度认知训练游戏。在心理疏导场景中，Pepper 温和的外形和情绪表达有助于降低老人的孤独感和焦虑情绪。

### 教育

SoftBank Robotics 向日本、英国等国的中小学提供 Pepper，用于编程教育和 STEM（科学、技术、工程、数学）课程。学生可通过 Choregraphe 或 Python SDK 为 Pepper 编写行为脚本，在实践中学习逻辑思维和机器人基础知识。

### 航空

比利时布鲁塞尔航空（Brussels Airlines）将 Pepper 部署于机场，为旅客提供值机指引、航班信息查询和登机口导航服务，探索机场无人化服务的可行性。


## 与 ROS 集成

ROS（机器人操作系统，Robot Operating System）社区为 Pepper 提供了较为完善的驱动和工具包支持，使研究人员能够在标准 ROS 生态中使用 Pepper。

### 主要软件包

| 包名 | 说明 |
|------|------|
| `naoqi_driver` | ROS 1 桥接驱动，将 NAOqi API 封装为标准 ROS 话题（Topics）和服务（Services） |
| `pepper_robot` | 提供 Pepper 的 URDF（统一机器人描述格式，Unified Robot Description Format）模型，支持在 RViz 中可视化机器人状态 |
| `pepper_meshes` | Pepper 的三维网格模型文件，配合 URDF 使用 |
| `naoqi_bridge` | 社区维护的 ROS 2 适配层，提供基本的话题桥接功能 |
| `pepper_moveit_config` | MoveIt! 运动规划配置，支持手臂轨迹规划 |

### 典型 ROS 话题

通过 `naoqi_driver` 启动后，可订阅以下常用话题：

```
/pepper_robot/camera/front/image_raw    # 前置摄像头图像
/pepper_robot/camera/depth/image_raw   # 深度相机图像
/pepper_robot/laser                    # 激光雷达扫描数据
/pepper_robot/sonar/front              # 前方声纳距离
/joint_states                          # 全身关节状态
/pepper_robot/imu/base                 # IMU 数据
```

### 注意事项

- `naoqi_driver` 主要支持 ROS 1（Melodic/Noetic），ROS 2 支持依赖社区包，功能尚不完整。
- NAOqi OS 与 ROS 运行在同一网络时，需确保 ROS_IP 和 ROS_MASTER_URI 配置正确。
- 部分高级功能（如 ALBasicAwareness）在 ROS 桥接模式下需通过 NAOqi Python SDK 单独调用。


## 局限性与争议

尽管 Pepper 在社交机器人领域具有里程碑意义，但其商业化历程也暴露出若干值得关注的问题：

### AI 能力受限

Pepper 的自然语言理解和情绪识别能力在实验室演示中往往表现出色，但在真实部署环境中，受限于噪声、光线变化和用户行为多样性，识别准确率显著下降。大量商业部署中，Pepper 实际执行的是预设脚本（Scripted Interaction）而非真正的自主对话，与宣传的"智能交互"存在明显差距。

### 停产风波

2021 年 SoftBank 宣布暂停 Pepper 大规模生产，引发业界对服务机器人商业模式的深刻反思。分析人士指出，Pepper 的高售价、高维护成本和有限的实际价值产出，使企业客户难以实现清晰的投资回报（ROI，Return on Investment）。这一事件被视为第一波消费级/商业级社交机器人浪潮退潮的标志性信号。

### 维护成本

企业用户反映，Pepper 的硬件故障率较高，关节和感知传感器的维修费用不低。加之 SoftBank 强制捆绑的云服务订阅合同，实际总拥有成本（TCO，Total Cost of Ownership）远超初始购置价格。

### 隐私问题

Pepper 持续采集用户的面部图像和语音数据用于云端处理，在欧盟《通用数据保护条例》（GDPR，General Data Protection Regulation）框架下面临合规压力，部分机构因此限制了 Pepper 的部署范围。


## 参考资料

- [SoftBank Robotics 官网 — Pepper 产品页](https://www.softbankrobotics.com/emea/en/pepper)
- Pandey, A. K., & Gelin, R. (2018). A Mass-Produced Sociable Humanoid Robot: Pepper. *IEEE Robotics & Automation Magazine*, 25(3), 40–48.
- [NAOqi SDK 官方文档](http://doc.aldebaran.com/2-5/index.html)
- [naoqi_driver — ROS Wiki](http://wiki.ros.org/naoqi_driver)
- Bohus, D., & Horvitz, E. (2014). Managing Human-Robot Engagement with Forecasts and... *Proceedings of SIGDIAL 2014*.
- [pepper_robot — GitHub (ros-naoqi)](https://github.com/ros-naoqi/pepper_robot)
