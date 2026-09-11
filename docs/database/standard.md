# 机器人领域行业标准

!!! note "引言"
    行业标准（Industry Standard）是机器人工程师日常工作中不可回避的重要参考依据。标准的存在解决了多个核心问题：首先是**互操作性**（Interoperability），不同厂商生产的机器人部件、传感器和控制器需要能够协同工作，统一的接口与术语定义使之成为可能；其次是**安全认证**（Safety Certification），工业机器人、服务机器人在进入市场前必须通过相应安全标准的认证，否则无法合法销售；第三是**采购要求**（Procurement Requirements），大型制造企业在招标时通常明确要求供应商的产品符合特定标准；第四是**监管合规**（Regulatory Compliance），欧盟机械指令（Machinery Directive）、美国职业安全与健康管理局（OSHA）法规等均直接引用机器人安全标准。

    对于机器人工程师而言，标准的影响贯穿整个产品开发周期。在设计阶段，坐标系定义（ISO 9787）和术语规范（ISO 8373）确保团队内部沟通无歧义；在测试阶段，性能评估标准（ISO 9283）提供了可重复的测试方法，使不同实验室的测试结果具有可比性；在产品上市前，安全标准（ISO 10218、ISO 13482）规定了必须满足的最低安全要求，协作机器人标准（ISO/TS 15066）则定义了人机共工的具体边界条件。忽视这些标准，不仅可能导致产品认证失败，还可能引发严重的安全事故和法律责任。

    本列表整理了国际上常用的机器人相关标准，涵盖 ISO、IEC、ASTM、VDI 等主要标准化机构发布的文件，以及 ROS REP 等开源平台的约定规范。列表按照对于标准的需求类型划分，而非机器人系统的种类或标准的制定机构。


## 标准化机构介绍

机器人领域的标准由多个国际和地区性机构制定，了解这些机构有助于工程师找到权威的参考文件。

### ISO/TC 299（国际标准化组织机器人技术委员会）

国际标准化组织（International Organization for Standardization，ISO）是全球最具影响力的标准制定机构。其下设的第 299 技术委员会（Technical Committee 299）专门负责机器人领域的标准制定工作，前身为 ISO/TC 184/SC 2。TC 299 的工作范围涵盖工业机器人、服务机器人、协作机器人、移动机器人等各类机器人系统。ISO 标准通常需要成员国投票通过后正式发布，代表了全球范围内的最广泛共识。

ISO 标准可通过 ISO 官网（iso.org）购买，部分标准也可通过各国国家标准机构获取。在中国，ISO 标准往往会被等同采用（等同采用标志为"IDT"）或修改采用（"MOD"）转化为 GB/T 国家标准，由国家市场监督管理总局负责发布。

### IEC（国际电工委员会）

国际电工委员会（International Electrotechnical Commission，IEC）负责电气、电子和相关技术领域的国际标准化工作。在机器人领域，IEC 的贡献主要集中在功能安全（Functional Safety）、工业通信协议和电气安全等方面，例如 IEC 61508（功能安全基础标准）、IEC 62541（OPC-UA 通信标准）和 IEC 61158（工业以太网通信）。IEC 与 ISO 在机器人领域存在密切合作，部分标准以 ISO/IEC 联合发布。

### ASTM International

ASTM International 前身为美国材料与测试协会（American Society for Testing and Materials），现为全球性标准组织。其 F45 委员会专注于无人驾驶系统（Driverless Automatic Guided Vehicles，DAGV）和自主工业车辆（Autonomous Industrial Vehicles，A-IVs）的标准制定。ASTM 标准在北美制造业中被广泛采用，尤其是仓储物流和 AGV（Automated Guided Vehicle，自动导引车）领域。

### VDI（德国工程师协会）

德国工程师协会（Verein Deutscher Ingenieure，VDI）是德国最具影响力的工程技术学会，发布的 VDI 指南（Richtlinie）在自动导引车系统（AGVS）领域具有重要地位，尤其在欧洲制造业中被广泛参考。VDI 标准侧重于工程实践指导，许多 AGV 系统的设计规范、接口定义和经济效益评估方法均源自 VDI 系列文件。

### ROS REP（ROS 增强提案）

ROS 增强提案（ROS Enhancement Proposal，REP）是 ROS（Robot Operating System，机器人操作系统）开源社区的技术规范文件，类似于 Python 的 PEP 或 IETF 的 RFC。REP 定义了 ROS 生态系统内的约定俗成，包括坐标系方向、单位规范、话题命名等。虽然 REP 不具有法律约束力，但在 ROS/ROS 2 开发社区中具有极高的权威性，遵循 REP 规范是保证代码与第三方包兼容的基本前提。


## 标准分类总览

本章按「工程师遇到的需求类型」而非制定机构对标准分类，共六类：

| 分类 | 回答什么问题 | 代表标准 | 所在页面 |
|------|--------------|----------|----------|
| I. 术语公约 | 这个词在行业里到底指什么？坐标轴怎么命名？ | ISO 8373、ISO 9787、ISO 19649 | [术语与性能评估](standard-terminology-performance.md) |
| II. 性能评估 | 标称精度怎么测？不同厂商数据能否比较？ | ISO 9283、ISO 18646、ASTM F45 | [术语与性能评估](standard-terminology-performance.md) |
| III. 系统安全 | 机器人能否与人共处？需要什么防护？ | ISO 10218、ISO/TS 15066、ISO 13482 | [安全与功能安全](standard-safety.md) |
| IV. 功能安全 | 控制系统失效时会发生什么？如何定级？ | IEC 61508、ISO 13849、IEC 62061 | [安全与功能安全](standard-safety.md) |
| V. 通信与接口 | 不同厂商的部件如何互联互通？ | EtherCAT、CiA 402、OPC UA、ROS REP | [通信接口与人形标准](standard-interface-humanoid.md) |
| VI. 人形机器人 | 这一新兴品类目前有哪些规范？ | GB/T 相关标准、IEEE 与 ISO 在研项目 | [通信接口与人形标准](standard-interface-humanoid.md) |

在实际产品开发中，这六类标准的使用时机不同：设计阶段主要参考 I 与 V，测试阶段依赖 II，认证阶段则由 III 与 IV 决定能否上市。


## 标准获取渠道

了解标准的获取途径对于工程师快速找到权威文本至关重要。以下是主要的标准获取渠道：

### 官方购买渠道

- **ISO 官网（iso.org）**：所有 ISO 标准的权威发布平台，可直接购买 PDF 版本。部分标准提供免费预览（通常为前几页）。ISO 标准价格通常在 100-300 瑞士法郎之间。
- **IEC 官网（iec.ch）**：IEC 系列标准的官方来源，提供与 ISO 联合发布标准的查询和购买服务。
- **ASTM 官网（astm.org）**：ASTM 系列标准的官方来源，支持按单份标准或打包订阅的方式购买。
- **VDI 官网（vdi.de）**：VDI 指南文件的官方来源，部分文件提供德语和英语双语版本。

### 国家标准机构

- **中国（SAC，国家标准化管理委员会）**：通过全国标准信息公共服务平台（std.samr.gov.cn）可以查询和购买 GB/T 国家标准。许多 ISO 标准被等同采用为 GB/T 标准，价格远低于直接购买 ISO 原版。例如 ISO 10218-1 对应的国标为 GB 11291.1，ISO 8373 对应 GB/T 12643。
- **美国（ANSI）**：通过 ANSI 网上商店（webstore.ansi.org）可购买美国采用的 ISO/IEC 标准（ANSI/ISO/IEC 联合发布版）。
- **欧洲（CEN/CENELEC）**：欧盟协调标准（EN 系列）通过各成员国国家标准机构（如德国 DIN、英国 BSI、法国 AFNOR）购买。

### 免费资源

- **ROS REP 文档（ros.org/reps）**：所有 REP 文件均免费在线访问，这是 ROS 开发者最常查阅的标准资源。
- **标准草案（Draft Standards）**：部分标准在正式发布前会公开征求意见，草案（Draft International Standard，DIS 或 Final Draft International Standard，FDIS）有时可免费获取。
- **ISO 免费标准计划**：ISO 设有面向发展中国家和学术机构的免费或优惠访问计划（如 RIDES 计划），相关高校和研究机构可申请。
- **学术图书馆**：许多高校图书馆订阅了 IHS Markit、Techstreet 或 BSI Knowledge 等标准数据库，在校师生可免费访问。


## 本章内容导览

| 页面 | 主要内容 |
|------|---------|
| [机器人领域行业标准](standard.md) | 标准化机构介绍、六大分类总览、标准获取渠道 |
| [术语与性能评估标准](standard-terminology-performance.md) | ISO 8373 术语、ISO 9787 坐标系、ISO 9283 性能测试方法 |
| [安全与功能安全标准](standard-safety.md) | ISO 10218、ISO/TS 15066、ISO 13482、IEC 61508 与 PL/SIL 定级 |
| [通信接口与人形机器人标准](standard-interface-humanoid.md) | 现场总线、OPC UA、ROS REP 约定、人形机器人在研标准 |
| [功能安全](../hardware/functional-safety.md) | 功能安全工程实践 |
| [协作机器人安全](../hardware/safety-collaborative-robot.md) | 人机协作的力与速度限制 |


## 参考资料

[1] [机器人领域行业标准汇总 - 云飞机器人实验室](https://www.yfworld.com/?p=5753)

[2] [ISO/TC 299 Robotics - ISO](https://www.iso.org/committee/5915511.html)

[3] [REP 103 - Standard Units of Measure and Coordinate Conventions - ROS](https://www.ros.org/reps/rep-0103.html)

[4] [REP 105 - Coordinate Frames for Mobile Platforms - ROS](https://www.ros.org/reps/rep-0105.html)

[5] [REP 120 - Coordinate Frames for Humanoid Robots - ROS](https://www.ros.org/reps/rep-0120.html)

[6] [ISO/TS 15066:2016 - Robots and robotic devices — Collaborative robots](https://www.iso.org/standard/62996.html)

[7] [IEC 61508 - Functional Safety of E/E/PE Safety-related Systems](https://www.iec.ch/functionalsafety/)

[8] [EtherCAT Technology Group - ethercat.org](https://www.ethercat.org)

