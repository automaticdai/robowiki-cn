# 通信接口与人形机器人标准

!!! note "引言"
    通信与接口标准决定了不同厂商的部件能否组成一个系统：工业现场总线（EtherCAT、CANopen CiA 402、PROFINET）规定了实时控制数据的传输方式，OPC UA Robotics 面向工厂信息层的互通，而 ROS REP 系列虽非正式国际标准，却在开源生态中承担了事实规范的角色（如 REP-103 的单位与坐标系约定、REP-105 的坐标系树）。人形机器人作为新兴品类，其标准体系仍在快速成形，中国与国际标准化机构均已启动相关工作。本页面整理这两类标准的现状。


## 分类 V. 通信与接口标准

机器人系统内部各模块之间，以及机器人与外部系统之间的通信需要遵循标准化协议，以保证实时性、可靠性和互操作性。以下是机器人领域常用的通信与接口标准。

| 标准名称             | 描述                                                                    |
|----------------------|-------------------------------------------------------------------------|
| EtherCAT (IEC 61158) | 实时工业以太网通信协议，广泛用于机器人关节伺服控制                       |
| CANopen (CiA 301)    | 基于 CAN 总线的设备通信协议，用于关节控制器和传感器接口                  |
| OPC-UA (IEC 62541)   | 工业物联网数据通信标准，用于机器人与 MES/SCADA 系统的数据交换及数字孪生  |
| ROS REP-2000         | ROS 2 目标平台和策略，定义各 ROS 2 发行版支持的操作系统和硬件平台        |
| MTConnect (ANSI/MTC1)| 制造设备数据采集标准，用于机器人与生产线监控系统的集成                    |
| MQTT (ISO/IEC 20922) | 轻量级消息传输协议，用于机器人云端连接和远程监控                          |

### EtherCAT：实时工业以太网

EtherCAT（Ethernet for Control Automation Technology）由德国倍福（Beckhoff Automation）开发，现已被纳入 IEC 61158 工业现场总线标准体系。EtherCAT 采用"飞行处理"（Processing on the Fly）技术，使得数据帧在经过每个从站节点时无需等待完整接收即可被处理并转发，实现了极低的通信延迟（通常在 1 ms 以内）和高度的时间确定性（Determinism）。

在机器人领域，EtherCAT 是目前主流工业机器人和协作机器人关节伺服驱动器（Servo Driver）通信的事实标准。使用 EtherCAT 的代表性机器人系统包括 KUKA、ABB 等品牌的工业机器人控制器，以及采用 EtherCAT 主站（Master）的开源机器人控制框架（如 ethercat_master、SOEM 等）。

### CANopen：嵌入式设备总线协议

CANopen 基于控制器局域网络（Controller Area Network，CAN）总线，由 CAN in Automation（CiA）组织制定，核心规范为 CiA 301。CANopen 定义了设备对象字典（Object Dictionary）、过程数据对象（Process Data Object，PDO）和服务数据对象（Service Data Object，SDO）等机制，为不同厂商的设备提供了统一的通信接口。

在机器人领域，CANopen 常用于连接关节控制器（Joint Controller）、力矩传感器（Torque Sensor）、编码器（Encoder）等低层设备，尤其在成本敏感的中小型机械臂和移动机器人中应用广泛。CiA 402 是 CANopen 的电机驱动设备子协议，定义了速度模式、位置模式、力矩模式等标准控制模式。

### OPC-UA：工业物联网通信标准

OPC 统一架构（OPC Unified Architecture，OPC-UA，IEC 62541）是一种平台无关的工业数据通信标准，提供安全、可靠的机器间（Machine-to-Machine，M2M）通信。与传统 OPC 不同，OPC-UA 不依赖 Windows COM/DCOM 技术，可运行于嵌入式 Linux、实时操作系统（RTOS）甚至微控制器上。

在智能制造（Smart Manufacturing）背景下，OPC-UA 被 RAMI 4.0（工业 4.0 参考架构模型）和 IIC（工业互联网联盟）列为工业物联网（Industrial Internet of Things，IIoT）的推荐通信标准。机器人通过 OPC-UA 服务器将状态数据（位置、速度、温度、错误代码等）暴露给上层制造执行系统（Manufacturing Execution System，MES）、数字孪生（Digital Twin）平台和云端分析服务。


## 分类 VI. 人形机器人相关标准

人形机器人（Humanoid Robot）作为近年来机器人领域发展最快的方向之一，其标准化工作仍处于起步阶段。现有标准的制定速度远落后于技术发展速度，这既是挑战，也为从业者参与标准制定提供了机会。

### 现有标准的适用性

目前，人形机器人在安全性方面主要参考以下已有标准，但均需要结合人形机器人的特殊性加以解释和补充：

- **ISO 13482（B 型：身体辅助机器人）**：人形机器人若用于辅助人体运动（如老年人助行）可适用此分类，但标准的许多具体要求是针对轮椅机器人等传统护理设备制定的，对于全身运动控制的人形机器人适用性有限。
- **ISO/TR 23482-1 和 ISO/TR 23482-2**：作为 ISO 13482 的应用指南，提供了服务机器人安全测试方法和应用场景指导，可为人形机器人安全测试方案的设计提供参考框架。
- **ISO/TS 15066（PFL 模式）**：若人形机器人手臂被用于与人协作的工业任务，PFL 模式的力限制要求和身体部位接触力阈值表同样适用。
- **IEC 61508 / EN ISO 13849-1**：人形机器人的关节驱动控制系统、安全停止功能等安全相关控制部件，需要按照功能安全标准进行设计和认证。

### 标准的空白与挑战

当前针对人形机器人的标准化工作面临以下主要挑战：

**动态稳定性（Dynamic Stability）**：传统机器人标准主要针对固定底座机器人或轮式移动机器人，缺乏针对双足步行（Bipedal Walking）机器人动态平衡失效（如跌倒）风险的评估方法。

**全身运动安全**：人形机器人具有数十个自由度，其运动空间大、速度快，与单臂工业机器人相比，人机碰撞的几何复杂度和风险评估难度大幅提升。

**意图感知与决策安全**：随着人形机器人开始搭载大型语言模型（Large Language Model，LLM）和具身智能（Embodied Intelligence）系统，机器人的行为决策不再完全可预测，传统基于确定性逻辑的安全标准面临根本性挑战。

**电池安全**：大容量锂电池组的热失控（Thermal Runaway）风险、跌倒时的结构安全等问题，需要新的测试方法和安全设计规范。

ISO/TC 299 已于 2023 年前后开始讨论专门针对人形机器人的新标准项目，预计未来几年内将陆续出现针对人形机器人测试方法、安全要求和性能评估的专项标准。


## 参考资料

1. ROS. *ROS Enhancement Proposals (REP)*. https://ros.org/reps/rep-0000.html
2. OPC Foundation. *OPC UA for Robotics (Companion Specification)*. https://opcfoundation.org/
3. EtherCAT Technology Group. *EtherCAT 规范与 CiA 402 驱动子协议*. https://www.ethercat.org/
4. [机器人领域行业标准总览](standard.md)
5. [通信总线](../hardware/communication-buses.md)
