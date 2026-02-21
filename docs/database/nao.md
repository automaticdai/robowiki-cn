# NAO

!!! note "引言"
    NAO（发音近似英文"Now"）是由法国 Aldebaran 机器人公司研发的自主可编程仿人机器人（Humanoid Robot）。凭借其开放的软件平台、丰富的传感器系统和亲切的外形设计，NAO 已成为全球高校、科研机构和中小学教育中使用最广泛的仿人机器人平台之一。自 2008 年起，NAO 还是 RoboCup 标准平台联赛（Standard Platform League, SPL）的唯一指定平台，持续推动自主机器人技术的发展。


## 发展历程

### Aldebaran 公司的创立与 NAO 项目起源

Aldebaran 机器人公司（Aldebaran Robotics）于 2005 年由布鲁诺·梅松尼尔（Bruno Maisonnier）在法国巴黎创立，致力于研发面向日常生活的人形服务机器人。公司名称取自猎户座中最亮的恒星毕宿五（Aldebaran），寓意在机器人领域点亮新星。

NAO 项目实际上早于公司正式成立，于 2004 年便已开始立项研究。最初的设计目标是打造一款造价合理、易于编程、具备完整感知与运动能力的小型仿人机器人，以填补当时市场上仿人机器人平台过于昂贵或过于简陋的空白。

### RoboCup 2007：AIBO 的落幕与 NAO 的崛起

2006—2007 年间，RoboCup 标准平台联赛（SPL）面临一场重要的平台更迭。索尼公司（Sony）宣布停止生产 AIBO 机器狗，导致 SPL 赛事失去原有的统一硬件平台。RoboCup 组委会随即开始寻找替代方案，并公开征集符合赛事要求的仿人机器人平台。

Aldebaran 携 NAO 原型机参加了 2007 年 RoboCup 的评选竞争，最终击败多个竞争对手，获得 SPL 官方认可，成为 2008 年起 SPL 赛事的唯一指定平台。这一里程碑事件极大地推动了 NAO 的知名度，也为 Aldebaran 带来了大量来自全球高校和研究机构的订单。

### NAO V3 与 V4 时代（2010—2012）

2010 年，NAO V3（也称 NAO V3R）正式亮相，搭配改进的机械结构和更稳定的 NAOqi 软件框架，被应用于当年的 RoboCup 世界杯。同年夏天，NAO 在上海世博会日本馆以多机协同舞蹈表演惊艳全场，引发国际媒体的广泛关注。

2011—2012 年间，Aldebaran 推出了 NAO V4（亦称 NAO Next Gen）。这一版本带来了两项重要硬件升级：

- 搭载高清摄像头（HD Camera），分辨率显著提升，极大改善了机器人的视觉感知能力；
- CPU 性能大幅提升，为更复杂的实时图像处理算法提供了算力基础。

2010 年 10 月，东京大学中村实验室一次性采购了 30 台 NAO 机器人，计划将其开发为实验室助理，这也是当时单笔规模最大的高校采购订单之一。

### NAO V5 时代与 Intel Atom 处理器（2014）

2014 年，Aldebaran 推出了 NAO V5，这是迄今为止 NAO 平台规格提升幅度最大的一次迭代。NAO V5 的核心改进在于将处理器升级为 Intel Atom（E3845，四核 1.91 GHz），显著增强了板载计算能力，使机器人能够在不依赖外部服务器的情况下，在本地运行更复杂的感知与决策算法。

同年，Aldebaran 还正式推出了 Pepper 机器人，定位于商业服务场景，与 NAO 形成互补的产品线布局。

### 软银收购与品牌重塑（2015）

2015 年，日本软银集团（SoftBank Group）完成了对 Aldebaran 机器人公司的全资收购，Aldebaran 随即更名为软银机器人公司（SoftBank Robotics）。这一收购使 NAO 和 Pepper 获得了软银庞大商业网络和资金的支持，进一步扩大了两款机器人的全球市场覆盖范围。

### NAO6 的发布（2018 至今）

2018 年，SoftBank Robotics 发布了目前仍在量产销售的 NAO6 版本。NAO6 在 V5 基础上进行了多项优化：

- 电池容量提升至 48.6 Wh，续航时间延长至约 60—90 分钟；
- 新增背部 LED 指示灯阵列，改善了状态反馈体验；
- 软件平台持续迭代，NAOqi OS 趋于成熟稳定；
- 摄像头升级为 OV5640，支持最高 2560×1920 分辨率。

### United Robotics Group 时代（2021 至今）

2021 年，SoftBank Robotics 将 NAO 和 Pepper 的业务剥离，转让给德国工业机器人联合集团 United Robotics Group（URG）。URG 于 2021 年完成收购，并在欧洲继续推进 NAO 平台的商业化与教育应用拓展。NAO 至今仍在生产销售，在全球已部署超过 13,000 台，用户遍及 70 余个国家。


## 技术规格

### NAO6 主要参数

NAO 第六代（NAO6）的主要技术参数如下：

| 参数 | 规格 |
|------|------|
| 身高 | 57.4 cm |
| 体重 | 5.48 kg |
| 自由度（DOF） | 25 个（头部 2、手臂 2×5、手部 2×1、腿部 2×5、髋部 1） |
| 处理器 | Intel Atom E3845 四核 1.91 GHz |
| 内存 | 4 GB DDR3 RAM |
| 存储 | 32 GB eMMC |
| 电池 | 48.6 Wh 锂离子电池，续航约 60—90 分钟 |
| 操作系统 | NAOqi OS（基于 Linux） |
| 无线网络 | Wi-Fi 802.11 a/b/g/n（2.4 GHz 和 5 GHz 双频） |
| 有线网络 | 以太网（Ethernet）接口 |
| 扬声器 | 2 个立体声扬声器 |

### 传感器系统

NAO 搭载了丰富的传感器系统，覆盖视觉、听觉、触觉、平衡等多个感知通道：

- **视觉（Vision）**：两个高清摄像头（OV5640），分辨率可达 2560×1920，分别位于前额和嘴部，支持不同俯仰角视角
- **听觉（Audio）**：头部四个麦克风，支持声源定位（Sound Source Localization, SSL）
- **触觉（Touch）**：头部三个电容式触摸传感器，双手各一个触摸传感器，双脚各一个接触传感器
- **惯性测量单元（IMU）**：包含三轴加速度计（Accelerometer）和三轴陀螺仪（Gyroscope），用于姿态估计
- **足底力传感器（FSR）**：每只脚底四个力敏电阻（Force Sensitive Resistor），用于检测压力分布和重心位置
- **红外（Infrared）**：两个红外发射器和接收器，用于近距离障碍物检测和与其他 NAO 通信
- **超声波声纳（Sonar）**：两个超声波传感器，测量距离范围约 0.25—2.55 m，用于障碍物检测

### 各版本对比

下表列出了 NAO 主要版本的关键规格变化，反映了平台迭代的技术演进脉络：

| 对比项 | NAO V3.3 | NAO V4（Next Gen） | NAO V5 | NAO6 |
|--------|----------|-------------------|--------|------|
| 发布年份 | 2010 | 2011/2012 | 2014 | 2018 |
| 处理器 | AMD Geode 500 MHz | Intel Atom Z530 1.6 GHz（单核） | Intel Atom E3845 1.91 GHz（四核） | Intel Atom E3845 1.91 GHz（四核） |
| 内存 | 256 MB | 1 GB DDR2 | 4 GB DDR3 | 4 GB DDR3 |
| 摄像头 | VGA（640×480） | HD（1280×960） | HD（1280×960） | HD（2560×1920） |
| 电池容量 | 21.6 Wh | 21.6 Wh | 36.4 Wh | 48.6 Wh |
| 自由度 | 21 | 25 | 25 | 25 |
| RoboCup SPL | 是（2010） | 是（2012—2014） | 是（2015—2017） | 是（2018 至今） |


## 编程与开发

### NAOqi 软件框架

NAOqi（发音："nao-ki"）是 NAO 机器人的核心软件框架（Software Framework），以分布式模块化架构为基础，提供了一套统一的 API 接口。NAOqi 支持跨语言调用——开发者可以使用 Python、C++ 或 Java 通过相同的接口控制机器人。

NAOqi 的核心架构基于代理（Broker）和模块（Module）的概念：每个功能模块向代理注册，代理维护所有可用模块的目录，客户端程序通过代理发现并调用模块的方法。这种设计使得模块既可以运行在机器人本体上，也可以运行在远程计算机上，远程调用通过网络透明地完成。

主要内置模块包括：

- **ALMotion**：运动控制模块，负责关节角度控制、步态行走、笛卡尔空间运动、跌倒保护等
- **ALTextToSpeech（TTS）**：语音合成模块，支持 20 余种语言，可调节语速、音调、音量
- **ALSpeechRecognition（ASR）**：语音识别模块，基于关键词列表匹配，支持在线和离线识别
- **ALVideoDevice**：视觉设备模块，管理摄像头参数和图像数据流
- **ALFaceDetection**：人脸检测模块，基于 Haar 特征级联分类器（Haar Cascade Classifier）
- **ALSonar**：超声波声纳模块，用于环境障碍物距离测量
- **ALMemory**：共享内存模块，作为模块间事件和数据传递的中枢
- **ALRobotPosture**：姿态管理模块，支持预设姿态切换（如站立、蹲下、坐下）
- **ALNavigation**：导航模块，支持基于地图的自主导航

### NAOqi SDK 代码示例

以下展示了通过 Python 调用 NAOqi SDK 控制 NAO 机器人的典型用法。NAOqi Python SDK 通过 TCP/IP 协议与机器人通信，只需知道机器人的 IP 地址和端口（默认 9559）即可连接。

#### 基础连接与语音控制

```python
from naoqi import ALProxy

# 连接到 NAO 机器人（替换为实际 IP 地址）
ROBOT_IP = "192.168.1.100"
ROBOT_PORT = 9559

# 创建语音合成代理
tts = ALProxy("ALTextToSpeech", ROBOT_IP, ROBOT_PORT)

# 让 NAO 说中文（需确保安装了中文语音包）
tts.setLanguage("Chinese")
tts.say("你好，我是NAO机器人！")
```

#### 运动控制

```python
from naoqi import ALProxy

ROBOT_IP = "192.168.1.100"
ROBOT_PORT = 9559

# 创建运动控制代理
motion = ALProxy("ALMotion", ROBOT_IP, ROBOT_PORT)
posture = ALProxy("ALRobotPosture", ROBOT_IP, ROBOT_PORT)

# 唤醒机器人（释放关节刚度并准备运动）
motion.wakeUp()

# 切换到站立姿态
posture.goToPosture("StandInit", 0.5)

# 向前行走 0.5 米（x=前后, y=左右, theta=旋转角度，单位均为米/弧度）
motion.moveTo(0.5, 0, 0)

# 单关节角度控制：让头部偏转 0.5 弧度，速度分数为 0.1（最大速度的 10%）
motion.setAngles("HeadYaw", 0.5, 0.1)

# 同时控制多个关节
joint_names = ["LShoulderPitch", "LElbowRoll"]
joint_angles = [1.0, -0.5]   # 单位：弧度
motion.setAngles(joint_names, joint_angles, 0.2)

# 行走结束后让机器人回到休息姿态
posture.goToPosture("Crouch", 0.5)
motion.rest()
```

#### 人脸检测

```python
from naoqi import ALProxy
import time

ROBOT_IP = "192.168.1.100"
ROBOT_PORT = 9559

# 创建人脸检测代理和内存代理
face_detection = ALProxy("ALFaceDetection", ROBOT_IP, ROBOT_PORT)
memory = ALProxy("ALMemory", ROBOT_IP, ROBOT_PORT)
tts = ALProxy("ALTextToSpeech", ROBOT_IP, ROBOT_PORT)

# 启用人脸检测，设置检测频率为 10 Hz
face_detection.subscribe("FaceDetectionExample", 500, 0.0)

print("开始人脸检测，持续 10 秒...")
start_time = time.time()

while time.time() - start_time < 10:
    # 从共享内存读取人脸检测结果
    face_data = memory.getData("FaceDetected")

    if face_data and len(face_data) >= 2:
        face_info_array = face_data[1]
        if len(face_info_array) >= 1:
            face_info = face_info_array[0]
            # face_info[0] 包含形状信息（位置、大小）
            shape_info = face_info[0]
            print(f"检测到人脸！位置：alpha={shape_info[1]:.2f}, beta={shape_info[2]:.2f}")
            tts.say("我看到你了！")
    time.sleep(0.5)

# 取消订阅
face_detection.unsubscribe("FaceDetectionExample")
```

#### 语音识别

```python
from naoqi import ALProxy
import time

ROBOT_IP = "192.168.1.100"
ROBOT_PORT = 9559

asr = ALProxy("ALSpeechRecognition", ROBOT_IP, ROBOT_PORT)
memory = ALProxy("ALMemory", ROBOT_IP, ROBOT_PORT)
tts = ALProxy("ALTextToSpeech", ROBOT_IP, ROBOT_PORT)

# 设置识别语言为英文（中文识别需安装对应语言包）
asr.setLanguage("English")

# 定义待识别的关键词列表
vocabulary = ["hello", "goodbye", "sit down", "stand up", "dance"]
asr.setVocabulary(vocabulary, False)  # False 表示不使用单词边界检测

# 启动语音识别
asr.subscribe("SpeechRecognitionExample")
print("语音识别已启动，请说出关键词...")

try:
    for _ in range(20):
        time.sleep(0.5)
        result = memory.getData("WordRecognized")
        if result and result[1] > 0.4:  # 置信度阈值 0.4
            recognized_word = result[0]
            confidence = result[1]
            print(f"识别到：'{recognized_word}'，置信度：{confidence:.2f}")

            if recognized_word == "hello":
                tts.say("Hello! Nice to meet you!")
            elif recognized_word == "dance":
                tts.say("Let me dance for you!")
finally:
    asr.unsubscribe("SpeechRecognitionExample")
```

### Choregraphe 图形化编程环境

Choregraphe 是 Aldebaran 提供的跨平台图形化编程环境（Graphical Programming Environment），支持在 Windows、macOS 和 Linux 上运行。其核心设计理念是"所见即所得"——用户通过在画布上拖拽预制的功能模块（Box），并用连接线将输入输出端口相连，即可为 NAO 编写完整的行为程序。

Choregraphe 的主要功能包括：

- **行为编辑器（Behavior Editor）**：以流程图形式可视化编排机器人的感知-决策-执行流程
- **动作时间线（Motion Timeline）**：通过关键帧（Keyframe）编辑机器人的动作序列，类似动画制作软件
- **3D 仿真视图（3D Simulation View）**：在虚拟环境中预览机器人的动作，无需实体机器人即可开发调试
- **Python 脚本嵌入**：每个模块内部可嵌入任意 Python 代码，支持复杂逻辑实现
- **NAO 远程连接**：一键连接真实 NAO 机器人，实时传输和执行行为程序

### NAO Academy 在线学习平台

NAO Academy 是 SoftBank Robotics 官方提供的在线学习平台（Online Learning Platform），提供系统化的 NAO 编程课程，内容涵盖：

- Choregraphe 基础操作教程
- NAOqi Python SDK 编程指南
- 传感器数据读取与处理
- 运动控制与步态编程
- 视觉感知与人脸识别应用

平台提供视频教程、交互式练习题和项目案例，适合从零基础到高级开发者的不同学习需求。


## 步态控制与双足行走

### ZMP 步态控制原理

NAO 的双足行走基于零力矩点（Zero Moment Point, ZMP）理论。ZMP 是地面反力（Ground Reaction Force）和重力合力的作用点，当 ZMP 位于支撑多边形（Support Polygon）内部时，机器人保持动态稳定；当 ZMP 超出支撑多边形时，机器人将发生倾倒。

NAOqi 的 ALMotion 模块通过预测性 ZMP 控制器（Predictive ZMP Controller）实现稳定行走：

$$
\ddot{x}_{ZMP}(t) = \ddot{x}_{CoM}(t) - \frac{g}{z_{CoM}} \left( x_{CoM}(t) - x_{ZMP}(t) \right)
$$

其中 \(x_{CoM}\) 为质心（Center of Mass, CoM）水平位置，\(z_{CoM}\) 为质心高度，\(g\) 为重力加速度。控制器通过规划质心轨迹，使 ZMP 始终保持在支撑多边形内。

### 全向行走能力

NAO 具备全向行走（Omnidirectional Walking）能力，可以在平坦地面上沿任意方向运动，包括：

- **前进/后退**：沿 \(x\) 轴方向移动
- **侧移**：沿 \(y\) 轴方向横向移动
- **原地旋转**：绕竖直轴 \(z\) 旋转
- **斜向行走**：上述运动的任意组合

在 NAOqi 中，全向行走通过 `motion.move(x_vel, y_vel, theta_vel)` 接口控制，其中三个参数分别对应前后速度（m/s）、侧移速度（m/s）和旋转角速度（rad/s）。

### 步态参数调节

NAOqi 允许开发者通过 `motion.getMoveConfig()` 和 `motion.setMoveConfig()` 接口调节步态参数，主要可调参数包括：

| 参数 | 说明 | 典型范围 |
|------|------|----------|
| `MaxStepX` | 单步最大前进距离 | 0—0.08 m |
| `MaxStepY` | 单步最大侧移距离 | 0—0.16 m |
| `MaxStepTheta` | 单步最大旋转角 | 0—0.52 rad |
| `MaxStepFrequency` | 步频 | 0—1（归一化） |
| `StepHeight` | 抬脚高度 | 0.01—0.04 m |
| `TorsoWx` | 躯干横滚角 | ±5° |
| `TorsoWy` | 躯干俯仰角 | ±5° |

### 跌倒检测与自主起身

NAO 的 IMU 持续监测机器人的姿态角。当检测到躯干倾斜角超过安全阈值时，ALMotion 模块会触发跌倒保护（Fall Protection）机制：

1. **跌倒预测**：加速度计和陀螺仪数据融合，估计当前姿态和角速度
2. **保护动作**：在机器人倒地前，主动放松部分关节刚度（Stiffness），减少碰撞冲击
3. **倒地检测**：接触传感器和加速度计确认倒地状态
4. **自主起身（Self-Righting）**：NAOqi 内置自主起身算法，可识别机器人处于面朝上（仰卧）还是面朝下（俯卧）状态，并执行对应的起身动作序列

自主起身是 NAO 在 RoboCup 比赛中的重要能力，因为比赛中机器人频繁发生碰撞和跌倒。


## RoboCup 标准平台联赛深度分析

### 联赛概述与规则

RoboCup 标准平台联赛（Standard Platform League, SPL）是 RoboCup 中规则最严格的联赛之一：所有参赛队伍必须使用完全相同的硬件平台（NAO），仅允许修改软件。这一规则消除了硬件差异带来的不公平性，使竞争完全聚焦于软件算法的优劣。

比赛采用 5 对 5 的全自主足球对抗形式（5v5 Autonomous Soccer），场地为 9m×6m 的标准足球场（带有白色线条和球门）。机器人全程自主运行，不允许任何人工干预——裁判通过无线网络（Wi-Fi）向机器人发送游戏状态（比赛开始、暂停、点球等），机器人根据游戏状态自主决策行动。

### 核心技术挑战

SPL 涉及的技术挑战涵盖机器人学的多个核心领域：

**球检测（Ball Detection）**

早期 SPL 使用橙色足球，依赖颜色阈值分割（Color Thresholding）。现代 SPL 改用标准黑白花纹足球，要求机器人使用基于深度学习的物体检测算法（Object Detection），在复杂光照条件下实现鲁棒的实时球检测。光照变化（自然光/人工光混合、阴影）是最大的技术挑战之一。

**自定位（Self-Localization）**

机器人通过识别场地线条（Field Lines）、球门柱（Goalpost）、角标（Corner Marks）等地标，结合扩展卡尔曼滤波器（Extended Kalman Filter, EKF）实现自身位置和朝向的实时估计。场地线条提供了丰富的约束，但线段匹配本质上存在多义性（例如中线与边线难以区分），需要融合历史轨迹信息消除歧义。

**多机器人协作（Multi-Robot Coordination）**

5 台机器人通过 Wi-Fi 局域网广播消息，共享各自的位置估计、球位置估计和当前行为意图。团队通过分布式角色分配（Role Assignment）算法协调行动——距球最近的机器人承担"进攻者"角色，其他机器人根据场上局势分配守门员、防守等角色。

**实时运动规划（Real-Time Motion Planning）**

机器人需要在移动的动态环境中规划路径，绕过场上的对手和队友，同时保持对球和球门的朝向控制。常用方法包括势场法（Potential Field）和基于抽样的规划算法（Sampling-Based Planning）。

### 著名参赛队伍

**B-Human（不来梅大学，德国）**

不来梅大学（Universität Bremen）的 B-Human 队是 SPL 历史上最成功的参赛队伍之一，曾多次获得 SPL 世界冠军。B-Human 开发了一套完整的开源软件框架（B-Human Framework），并在每年 RoboCup 结束后将代码开源发布，成为全球众多 SPL 参赛队和研究者的重要参考。

B-Human 框架的主要组成部分包括：

- **视觉模块**：基于图像块分类和边缘检测的球检测、场地线条检测
- **自定位模块**：基于蒙特卡洛定位（Monte Carlo Localization, MCL）和卡尔曼滤波的混合定位框架
- **行为模块**：基于分层状态机（Hierarchical State Machine）和行为树（Behavior Tree）的决策框架
- **运动模块**：自研步态发生器（Gait Generator），步行性能优于 NAOqi 默认步态

**rUNSWift（新南威尔士大学，澳大利亚）**

澳大利亚新南威尔士大学（UNSW）的 rUNSWift 队同样是 SPL 的多届冠军，也长期保持开源传统。rUNSWift 在计算机视觉和感知领域有深厚积累，其视觉系统在处理复杂光照环境方面表现出色。

**其他著名队伍**

- **NaoDevils**（多特蒙德理工大学，德国）
- **HTWK Robots**（莱比锡理工大学，德国）
- **Nao-Team HTWK**（莱比锡，德国）
- **UPennalizers**（宾夕法尼亚大学，美国）


## 主要能力

### 语音交互

NAO 支持 20 余种语言的语音合成（Text-to-Speech, TTS）和基于关键词列表的语音识别（Speech Recognition）。NAOqi 的 ALTextToSpeech 模块支持调节语速、音调、音量，并支持使用 SSML（Speech Synthesis Markup Language）标记控制停顿、重音等细节。

对于更高级的自然语言理解（Natural Language Understanding, NLU），开发者通常将 NAO 与外部 NLU 服务（如 Dialogflow 或本地运行的大语言模型）结合使用，NAO 负责语音输入输出，外部服务负责语义理解和对话管理。

### 视觉感知

NAO 的双摄像头系统支持以下视觉任务：

- **人脸检测与识别（Face Detection & Recognition）**：NAOqi 内置 ALFaceDetection 模块，支持识别已注册的面孔
- **物体颜色检测（Color Detection）**：基于 HSV 颜色空间的颜色分割
- **地标识别（Landmark Recognition）**：识别特定的 ArUco 标记或 NAO 专有视觉标记
- **二维码识别（QR Code Detection）**：结合 ZBar 等库实现二维码解码
- **OpenCV 集成**：开发者可将摄像头数据传入 OpenCV 进行自定义图像处理

### 自主行走

NAO 具备全向双足步行能力，在 RoboCup 比赛和日常使用中均有良好表现。受限于板载算力和 NAOqi 步态算法的保守设计，NAO 的默认行走速度约为 0.35 m/s，经过优化的自定义步态（如 B-Human 框架的步态）可达 0.5 m/s 以上。


## 在教育和科研中的应用

### STEM 教育平台生态

NAO 已被全球超过 70 个国家的数百所高校和中学引入课程体系，形成了较为完善的教育生态：

**课程材料**：多家教育机构和第三方出版商提供了面向不同年级的 NAO 编程课程包，涵盖从 Scratch 式图形编程到 Python 面向对象编程的多个层次。

**Choregraphe 教学**：Choregraphe 的图形化编程模式降低了编程门槛，使学生无需掌握编程语言即可控制机器人完成复杂行为，非常适合 K-12 阶段的初步机器人教育。

**竞赛平台**：除 RoboCup SPL 外，多项国际机器人竞赛（如 RoboCup@Home Education、SPL Jr.）以 NAO 为平台，为学生提供了将课堂知识转化为竞技实践的机会。

**大学实验室应用**：在大学阶段，NAO 被广泛用于人机交互（Human-Robot Interaction, HRI）、认知科学、计算机视觉、运动控制等方向的实验课程和毕业设计项目。

### 自闭症辅助治疗

自闭症谱系障碍（Autism Spectrum Disorder, ASD）儿童的社交能力训练是 NAO 最具影响力的应用方向之一。相比人类治疗师，机器人在 ASD 治疗中具有独特优势：行为高度可预测和一致、互动节奏可精确控制、不会因儿童的非典型行为而产生情绪反应。

**Kaspar 机器人 vs NAO**

英国赫特福德大学（University of Hertfordshire）开发的 Kaspar 机器人是专门为 ASD 治疗设计的平台——外形刻意简化，表情固定，以减少 ASD 儿童面对复杂人类表情时的认知负担。相比之下，NAO 外形更加完整，具备更丰富的肢体语言和 LED 情绪表达能力，在社交参与度上通常表现更好。

两种机器人在 ASD 治疗中各有侧重：Kaspar 专注于减少社交焦虑和引导初步社交接触；NAO 则更适合进行更复杂的社交技能训练，如轮流对话、情绪识别和协作游戏。

**具体治疗方案**

基于 NAO 的 ASD 辅助治疗（Robot-Assisted Therapy, RAT）常用方案包括：

- **联合注意训练（Joint Attention Training）**：NAO 通过指向、凝视和语言引导儿童注意力，训练儿童在社交情境中跟随他人的注意力方向——这是 ASD 儿童普遍缺乏的基础社交能力
- **情绪识别练习（Emotion Recognition Exercise）**：NAO 展示面部 LED 颜色变化和肢体语言，配合语音讲解，帮助儿童识别和命名基本情绪（快乐、悲伤、愤怒、惊讶等）
- **社交故事演练（Social Story Role Play）**：NAO 扮演对话伙伴，与儿童共同演练特定的社交场景（如打招呼、请求帮助、轮流说话）

**临床研究成果**

多项随机对照试验（Randomized Controlled Trial, RCT）证实了基于 NAO 的 ASD 治疗的有效性：

- 2016 年，来自比利时 KU Leuven 大学的研究团队发现，经过 10 周基于 NAO 的联合注意训练后，参与儿童的联合注意行为显著改善，效果优于等量的传统治疗；
- 2018 年，斯坦福大学（Stanford University）的研究表明，ASD 儿童在与 NAO 互动时表现出更多社交凝视（Social Gaze）和情感表达，而在与同龄人或成人互动时则有所回避；
- 多项元分析（Meta-Analysis）综合约 20 项独立研究，总体认为机器人辅助治疗在提升 ASD 儿童联合注意、情绪识别和社交参与度方面具有中等到较强的效应量。


## NAO 与现代 AI 的结合

### ChatGPT 与大语言模型集成

随着大语言模型（Large Language Model, LLM）的快速普及，研究者和开发者开始将 NAO 与 ChatGPT 等 LLM 服务结合，赋予 NAO 更自然的对话能力。

典型架构如下：

1. NAO 通过麦克风采集用户语音，交由云端语音识别服务（如 Google Speech-to-Text 或 Azure Speech）转为文字
2. 文字输入传送给 OpenAI API（ChatGPT / GPT-4）或本地部署的 LLM（如 Llama、Qwen）进行自然语言理解和回复生成
3. LLM 返回的文字回复通过 NAO 的 TTS 模块转化为语音输出
4. 根据回复内容，NAO 可同步执行相应的肢体动作（如挥手、点头）以增强交互自然感

这种架构使 NAO 从基于关键词列表的简单问答机器人升级为具备开放域对话能力的社交机器人，大幅扩展了 NAO 在教育、接待和陪伴领域的应用潜力。

### 基于深度学习的计算机视觉升级

原生 NAOqi 的视觉模块依赖传统的图像处理算法（颜色阈值、Haar 特征）。现代开发者通过以下方式为 NAO 引入深度学习视觉能力：

- **轻量级神经网络本地部署**：利用 MobileNet、YOLO-Nano 等轻量级模型在 NAO 的 Intel Atom 处理器上本地推理，实现实时物体检测（每秒 5—15 帧）
- **云端推理卸载（Cloud Inference Offloading）**：将图像通过 Wi-Fi 传至外部 GPU 服务器进行推理，再将结果返回 NAO，适用于对延迟不敏感的任务
- **ROS2 集成**：通过 ROS2（Robot Operating System 2）桥接，将 NAO 纳入更完整的机器人软件生态，调用丰富的 ROS2 视觉包（如 OpenVINO、ONNX Runtime 节点）

### 云端语音识别升级

NAOqi 内置的离线语音识别基于有限词汇表的关键词匹配，识别精度有限。当前主流升级方案是将 NAO 的音频流接入云端语音识别 API（如百度语音、科大讯飞、Google STT、Whisper），实现：

- 不受词汇表限制的连续语音识别
- 更高的噪声环境下识别鲁棒性
- 多语言、多方言支持

中文用户通常选用百度语音识别 API 或科大讯飞实时语音 API，二者均提供 Python SDK，可较方便地与 NAOqi 框架集成。


## 与同类教育机器人对比

以下表格对比了 NAO 与其他主要教育/研究型机器人平台的关键特性：

| 对比项 | NAO | Pepper | Furhat | Misty II |
|--------|-----|--------|--------|----------|
| 研发公司 | Aldebaran / URG | SoftBank Robotics | Furhat Robotics | Misty Robotics |
| 发布年份 | 2008 | 2014 | 2014 | 2018 |
| 形态 | 全身仿人 | 轮式仿人 | 社交机器人头部 | 轮式移动机器人 |
| 身高 | 57.4 cm | 120 cm | 约 40 cm（桌面型） | 35.6 cm |
| 自由度（DOF） | 25 | 20 | 颈部 3 DOF + 面部投影 | 约 8 DOF |
| 双足行走 | 是 | 否（轮式） | 否（固定底座） | 否（轮式） |
| 主要用途 | 科研/教育/RoboCup | 商业接待/情绪交互 | 社会科学/对话研究 | 教育/研究/开发 |
| 编程接口 | NAOqi（Python/C++）/ROS | NAOqi / ROS | FurhatOS SDK | Misty SDK（REST/JS） |
| 面部表情 | LED 眼部 | LED 眼部 + 情感引擎 | 投影高清人脸 | LCD 面部显示屏 |
| 参考价格（USD） | ~$9,000—16,000 | ~$25,000—30,000 | ~$10,000 | ~$2,000（停产） |
| 开源生态 | 较丰富（RoboCup 社区） | 一般 | 一般 | 较丰富（已开源） |

各平台侧重点明显不同：NAO 凭借双足行走能力和成熟的 RoboCup 社区生态，在运动控制研究和机器人教育领域占据独特优势；Pepper 更适合商业前台和情绪交互场景；Furhat 专注于面部表情逼真度和社会科学实验；Misty II 则以相对低廉的价格面向 K-12 编程教育市场。


## 参考资料

1. [NAO](https://zh.wikipedia.org/zh-hans/NAO), 维基百科
2. [NAO6 Documentation](http://doc.aldebaran.com/2-8/home-nao.html), SoftBank Robotics
3. [RoboCup Standard Platform League](https://spl.robocup.org/), RoboCup
4. [B-Human Team Report 2023](https://b-human.de/downloads/publications/2023/TeamReport2023.pdf), B-Human, Universität Bremen
5. [rUNSWift Team Report](https://cgi.cse.unsw.edu.au/~robocup/2019site/reports/), rUNSWift, UNSW Sydney
6. [NAOqi Python SDK Documentation](http://doc.aldebaran.com/2-8/dev/python/index.html), SoftBank Robotics
7. [NAO Academy](https://www.softbankrobotics.com/emea/en/nao), SoftBank Robotics
8. Scassellati B, Admoni H, Matarić M. Robots for Use in Autism Research. *Annual Review of Biomedical Engineering*, 2012, 14: 275-294.
9. Shamsuddin S, et al. Humanoid Robot NAO: Review of Control and Motion Exploration. *IEEE International Conference on Control System, Computing and Engineering*, 2011.
10. Vochten M, De Winter J, Deconinck S. Robot-assisted therapy for children with ASD: A systematic review. *Computers in Human Behavior*, 2021.
11. [United Robotics Group — NAO](https://www.united-robotics-group.com/en/robots/nao), United Robotics Group
