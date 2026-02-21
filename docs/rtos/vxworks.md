# VxWorks

!!! note "引言"
    VxWorks是全球安全认证等级最高的商用实时操作系统之一，广泛应用于航空航天、国防军工、轨道交通、工业自动化和医疗设备等对可靠性要求极高的领域。从NASA的火星探测车到波音787客机的航电系统，VxWorks在众多关键任务系统中发挥着核心作用。

VxWorks操作系统是美国WindRiver公司于1983年设计开发的一种嵌入式实时操作系统（RTOS），是Tornado嵌入式开发环境的关键组成部分。良好的持续发展能力、高性能的内核以及友好的用户开发环境，在嵌人式实时操作系统领域逐渐占据一席之地。

VxWorks具有可裁剪微内核结构；高效的任务管理；灵活的任务间通讯；微秒级的中断处理；支持POSIX 1003．1b实时扩展标准；支持多种物理介质及标准的、完整的TCP/IP网络协议等。然而其价格昂贵。由于操作系统本身以及开发环境都是专有的，价格一般都比较高，通常需花费10万元人民币以上才能建起一个可用的开发环境，对每一个应用一般还要另外收取版税。一般不通供源代码，只提供二进制代码。由于它们都是专用操作系统，需要专门的技术人员掌握开发技术和维护，所以软件的开发和维护成本都非常高。支持的硬件数量有限。


## Wind微内核架构

VxWorks的核心是Wind微内核（Wind Microkernel），这是一个高度优化的实时内核，具有以下架构特点：

- **微内核设计**：内核仅包含最基本的功能，如任务调度、中断处理、任务间通信和同步机制。文件系统、网络协议栈等扩展功能以可选组件的形式提供。
- **可裁剪性（Scalability）**：Wind内核支持高度裁剪，开发者可以根据目标系统的需求选择性地包含或排除功能组件。最小配置下，VxWorks的ROM占用可以控制在数百KB级别。
- **确定性调度**：Wind内核提供微秒级的中断响应时间和确定性的任务切换时间，满足硬实时应用的需求。

VxWorks的系统架构从底层到顶层依次为：

1. **板级支持包（Board Support Package, BSP）**：封装与具体硬件板卡相关的初始化代码和驱动程序。
2. **Wind微内核**：实现核心的调度、同步和通信功能。
3. **系统组件**：包括文件系统、网络协议栈、设备驱动框架等。
4. **应用程序**：运行在内核之上的用户应用。


## 任务管理

VxWorks的任务管理机制是其实时性能的基础。

### 任务调度

VxWorks采用基于优先级的抢占式调度算法，支持256个优先级等级（0为最高优先级，255为最低优先级）。调度器始终运行优先级最高的就绪任务。在同一优先级内，可选择时间片轮转（Round-Robin）调度。

### 任务状态

VxWorks中的任务状态包括：

- **就绪态（Ready）**：任务等待处理器资源。
- **挂起态（Pended）**：任务等待某个资源（如信号量、消息队列）。
- **延时态（Delayed）**：任务在等待一个定时器到期。
- **挂起+延时态（Pended + Delayed）**：任务同时等待资源和定时器。
- **暂停态（Suspended）**：任务被显式暂停，通常用于调试。

### 任务间通信

VxWorks提供了丰富的任务间通信（Inter-Task Communication, ITC）机制：

- **信号量（Semaphore）**：支持二值信号量、计数信号量和互斥信号量。互斥信号量内置优先级继承机制。
- **消息队列（Message Queue）**：支持变长消息的FIFO传递，可选择紧急消息插入队列头部。
- **管道（Pipe）**：基于虚拟I/O设备的任务间通信方式，兼容标准的read/write接口。
- **共享内存（Shared Memory）**：多处理器系统中的通信方式。
- **信号（Signal）**：类UNIX的异步通知机制。


## 任务API详解

### taskSpawn — 创建任务

`taskSpawn()` 是VxWorks中创建任务的核心函数，其完整原型为：

```c
TASK_ID taskSpawn(
    char   *name,        /* 任务名称字符串 */
    int     priority,    /* 优先级：0（最高）～ 255（最低） */
    int     options,     /* 选项标志，如 VX_FP_TASK 启用浮点 */
    size_t  stackSize,   /* 任务栈大小（字节） */
    FUNCPTR entryPt,     /* 任务入口函数指针 */
    _Vx_usr_arg_t arg1,  /* 传给入口函数的参数 1～10 */
    /* ... arg2 ~ arg10 */
    _Vx_usr_arg_t arg10
);
```

参数说明：

| 参数 | 说明 |
|------|------|
| `name` | 任务名，用于调试和 `taskNameToId()` 查询 |
| `priority` | 0 为最高，255 为最低；实时控制任务通常设为 50～100 |
| `options` | `VX_FP_TASK`（启用浮点单元）；`VX_NO_STACK_FILL`（不初始化栈） |
| `stackSize` | 典型值 4096～65536 字节，不足会导致栈溢出 |
| `entryPt` | 任务函数，签名为 `int myFunc(int a1, …, int a10)` |
| `arg1～10` | 最多10个整数参数传递给入口函数 |

返回值为 `TASK_ID`，若创建失败则返回 `TASK_ID_ERROR`。

```c
#include <vxWorks.h>
#include <taskLib.h>
#include <sysLib.h>
#include <stdio.h>

/* 任务入口函数：周期性打印心跳 */
int controlTask(int periodMs, int arg2, int arg3,
                int arg4, int arg5, int arg6,
                int arg7, int arg8, int arg9, int arg10)
{
    int ticksPerMs = sysClkRateGet() / 1000;

    while (1)
    {
        printf("控制周期 %d ms\n", periodMs);

        /* taskDelay 参数单位为系统时钟节拍（tick） */
        /* sysClkRateGet() 返回每秒节拍数，通常为 100～1000 */
        taskDelay(periodMs * ticksPerMs);
    }
    return OK;
}

void spawnExample(void)
{
    TASK_ID tid;

    /* 创建优先级100、栈4KB的控制任务，周期10ms */
    tid = taskSpawn(
        "tControl",          /* 任务名 */
        100,                 /* 优先级 */
        VX_FP_TASK,          /* 选项：允许浮点运算 */
        4096,                /* 栈大小 */
        (FUNCPTR)controlTask,/* 入口函数 */
        10,                  /* arg1: 周期 10ms */
        0, 0, 0, 0,
        0, 0, 0, 0, 0
    );

    if (tid == TASK_ID_ERROR)
    {
        printf("任务创建失败！\n");
        return;
    }

    printf("任务ID: 0x%x\n", tid);

    /* 延时100ms：sysClkRateGet()/10 个节拍 */
    taskDelay(sysClkRateGet() / 10);
}
```

### 任务控制API

```c
/* 挂起与恢复 */
taskSuspend(tid);    /* 将任务置于暂停态，常用于调试 */
taskResume(tid);     /* 恢复被挂起的任务 */

/* 优先级动态调整 */
taskPrioritySet(tid, 80);    /* 临时提升优先级 */
taskPriorityGet(tid, &prio); /* 查询当前优先级 */

/* 安全删除 */
taskDelete(tid);             /* 立即删除任务并回收资源 */
taskDeleteForce(tid);        /* 强制删除（跳过安全钩子） */

/* 查询系统中所有任务 */
taskIdListGet(idList, 50);   /* 获取最多50个任务ID */
taskInfoGet(tid, &info);     /* 获取任务详细信息结构体 */
```


## 任务间通信（IPC）

### 消息队列（Message Queue）

消息队列（`MSG_Q_ID`）是VxWorks中最常用的任务间通信手段，支持任意数量的发送者和接收者，先进先出（FIFO）或优先级排序。

```c
#include <msgQLib.h>

#define MAX_MSGS   10
#define MSG_SIZE   64

typedef struct {
    int  type;
    char data[60];
} MyMsg;

MSG_Q_ID gQueue;

/* 初始化：通常在系统启动时调用 */
void initQueue(void)
{
    gQueue = msgQCreate(
        MAX_MSGS,          /* 队列最大消息数 */
        sizeof(MyMsg),     /* 每条消息的最大字节数 */
        MSG_Q_FIFO         /* 排队方式：FIFO 或 MSG_Q_PRIORITY */
    );

    if (gQueue == MSG_Q_ID_NULL)
        printf("消息队列创建失败\n");
}

/* 生产者任务：发送传感器数据 */
int producerTask(int a1, int a2, int a3, int a4, int a5,
                 int a6, int a7, int a8, int a9, int a10)
{
    MyMsg msg;
    int count = 0;

    while (1)
    {
        msg.type = 1;
        snprintf(msg.data, sizeof(msg.data), "传感器数据 #%d", count++);

        /* NO_WAIT = 非阻塞；WAIT_FOREVER = 无限等待 */
        if (msgQSend(gQueue, (char *)&msg, sizeof(msg),
                     WAIT_FOREVER, MSG_PRI_NORMAL) != OK)
        {
            printf("发送失败\n");
        }

        taskDelay(sysClkRateGet() / 100); /* 10ms 周期 */
    }
    return OK;
}

/* 消费者任务：接收并处理数据 */
int consumerTask(int a1, int a2, int a3, int a4, int a5,
                 int a6, int a7, int a8, int a9, int a10)
{
    MyMsg msg;

    while (1)
    {
        /* 阻塞等待消息 */
        if (msgQReceive(gQueue, (char *)&msg, sizeof(msg),
                        WAIT_FOREVER) != ERROR)
        {
            printf("收到[类型%d]: %s\n", msg.type, msg.data);
        }
    }
    return OK;
}
```

### 信号量（Semaphore）

VxWorks提供三种信号量，各有适用场景：

| 类型 | 创建函数 | 典型用途 |
|------|----------|----------|
| 二值信号量（Binary Semaphore） | `semBCreate()` | 任务同步、中断通知 |
| 计数信号量（Counting Semaphore） | `semCCreate()` | 资源池管理 |
| 互斥信号量（Mutual Exclusion Semaphore） | `semMCreate()` | 临界区保护，支持优先级继承 |

```c
#include <semLib.h>

SEM_ID gBinSem;   /* 二值信号量 */
SEM_ID gMutex;    /* 互斥信号量 */
SEM_ID gCntSem;   /* 计数信号量 */

void initSemaphores(void)
{
    /* 二值信号量：初始为空（SEM_EMPTY）*/
    gBinSem = semBCreate(SEM_Q_FIFO, SEM_EMPTY);

    /* 互斥信号量：自动启用优先级继承（SEM_INVERSION_SAFE）*/
    gMutex = semMCreate(SEM_Q_PRIORITY | SEM_INVERSION_SAFE);

    /* 计数信号量：初始计数为5（代表5个空闲缓冲区）*/
    gCntSem = semCCreate(SEM_Q_FIFO, 5);
}

/* 生产者-消费者示例（使用互斥量保护共享缓冲区） */
static int sharedBuffer[256];
static int bufHead = 0, bufTail = 0;

void produceData(int value)
{
    semTake(gMutex, WAIT_FOREVER);   /* 加锁 */
    sharedBuffer[bufHead] = value;
    bufHead = (bufHead + 1) % 256;
    semGive(gMutex);                 /* 解锁 */
    semGive(gBinSem);                /* 通知消费者有新数据 */
}

void consumeData(void)
{
    semTake(gBinSem, WAIT_FOREVER);  /* 等待生产者通知 */
    semTake(gMutex, WAIT_FOREVER);   /* 加锁 */
    int value = sharedBuffer[bufTail];
    bufTail = (bufTail + 1) % 256;
    semGive(gMutex);                 /* 解锁 */
    printf("消费数据: %d\n", value);
}
```


## 中断处理

### intConnect — 注册中断服务例程

VxWorks使用 `intConnect()` 将中断向量与中断服务例程（Interrupt Service Routine, ISR）绑定：

```c
#include <intLib.h>
#include <iv.h>

SEM_ID isrSyncSem;

/* ISR：运行在中断上下文，限制极严格 */
void myISR(int arg)
{
    /*
     * ISR 内的限制（必须遵守）：
     *   - 禁止调用任何可能阻塞的函数（semTake、msgQReceive 等）
     *   - 禁止调用 printf（内部有互斥锁）
     *   - 禁止分配/释放内存（malloc/free）
     *   - 可调用 semGive、msgQSend（非阻塞方式）
     */

    /* 通过二值信号量将中断事件延迟到任务上下文处理 */
    semGive(isrSyncSem);
}

/* 延迟中断处理任务（Deferred Interrupt Processing） */
int deferredHandler(int a1, int a2, int a3, int a4, int a5,
                    int a6, int a7, int a8, int a9, int a10)
{
    while (1)
    {
        /* 等待ISR发出的信号量 */
        semTake(isrSyncSem, WAIT_FOREVER);

        /* 在任务上下文中安全地执行复杂处理 */
        printf("中断事件已处理\n");
        /* 可以调用任意API，包括内存分配、文件操作等 */
    }
    return OK;
}

void setupInterrupt(void)
{
    /* 创建初始为空的二值信号量 */
    isrSyncSem = semBCreate(SEM_Q_FIFO, SEM_EMPTY);

    /* 注册ISR到向量号 INUM_TO_IVEC(5) */
    intConnect(INUM_TO_IVEC(5), (VOIDFUNCPTR)myISR, 0);

    /* 启用该中断 */
    intEnable(5);

    /* 创建延迟处理任务，优先级设为较高的10 */
    taskSpawn("tIrqHandler", 10, 0, 4096,
              (FUNCPTR)deferredHandler,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
}
```

### ISR设计原则

中断服务例程在VxWorks中运行于特殊的中断上下文，遵守以下规则是避免系统崩溃的关键：

1. **越短越好**：ISR中只做最少量的工作，复杂逻辑推迟到任务中。
2. **禁止阻塞**：任何可能阻塞调用者的API（`semTake(WAIT_FOREVER)`、`msgQReceive` 等）绝对禁止使用。
3. **禁止打印**：`printf` 内部使用互斥锁，在ISR中调用会死锁。可用 `logMsg()` 替代（它将消息发送到日志任务）。
4. **禁止动态内存**：`malloc`/`free` 不是可重入的，在ISR中调用行为未定义。
5. **允许的操作**：`semGive()`、以 `NO_WAIT` 模式调用的 `msgQSend()`、读写硬件寄存器。


## 内存保护

VxWorks在内存管理方面提供了多层次的保护机制：

- **实时进程模型（Real-Time Process, RTP）**：VxWorks 6.0引入的用户态进程模型，每个RTP拥有独立的地址空间，通过MMU实现内存隔离。一个RTP的崩溃不会影响内核和其他RTP。
- **内核任务模式**：传统的VxWorks任务运行在内核态，所有任务共享同一地址空间，效率最高但缺少隔离保护。
- **内存分区保护**：内核可以为不同的任务组分配独立的内存分区，防止越界访问。
- **栈溢出检测**：内核可以在任务栈的边界设置保护页（Guard Page），检测栈溢出错误。
- **内存保护单元（Memory Protection Unit, MPU）**：在不含MMU的微控制器上，VxWorks可借助MPU实现基本的内存区域保护，防止任务访问非授权内存区域。


## 网络协议栈

VxWorks内置了完整的网络协议栈，支持：

- **TCP/IP协议族**：包括IPv4、IPv6、TCP、UDP、ICMP、ARP等核心协议。
- **路由协议**：RIP、OSPF等。
- **应用层协议**：HTTP、FTP、TFTP、SNMP、SSH、DNS等。
- **网络安全**：IPsec、SSL/TLS、防火墙等。
- **工业协议**：支持EtherNet/IP、PROFINET等工业以太网协议。
- **高可用性**：VRRP（虚拟路由冗余协议）和快速故障切换。


## 开发工具

### Tornado开发环境

Tornado是VxWorks早期版本的集成开发环境（IDE），包括：

- **编译器和构建工具**：基于GNU工具链的交叉编译环境。
- **目标服务器（Target Server）**：连接主机开发环境和目标板的桥梁，支持通过以太网、串口等方式与目标板通信。
- **Shell**：交互式命令行工具，可在运行的VxWorks系统上直接调用函数和查看系统状态。
- **调试器（CrossWind）**：支持源码级调试、断点设置、单步执行等功能。

### Wind River Workbench

Wind River Workbench是基于Eclipse的新一代IDE，取代了Tornado，提供了更现代化的开发体验：

- 图形化的项目管理和构建配置
- 集成的系统查看器（System Viewer），可实时查看任务调度、中断和事件
- 代码分析和性能优化工具
- 支持VxWorks和Wind River Linux的统一开发
- 集成版本控制和团队协作功能


## 安全认证

VxWorks是获得安全认证最多的商用RTOS之一，这使其成为安全关键系统的首选：

| 认证标准 | 适用领域 | 等级 |
|----------|----------|------|
| DO-178C | 航空航天 | DAL A（最高等级） |
| IEC 61508 | 工业安全 | SIL 3 |
| ISO 26262 | 汽车电子 | ASIL D |
| IEC 62304 | 医疗设备 | Class C |
| EN 50128 | 轨道交通 | SIL 4 |

Wind River提供的安全认证版本称为**VxWorks Cert**，该版本在标准VxWorks基础上经过严格的代码审查、测试和文档化流程，满足各行业的安全认证要求。

### VxWorks 7 Safety Critical版本

VxWorks 7专门推出了Safety Critical配置，具备以下增强特性：

- **静态内存分配**：禁用动态内存分配，消除内存碎片和不确定性延迟。
- **MPU强制隔离**：即使在不含MMU的处理器上，也通过MPU强制隔离任务地址空间。
- **确定性内核路径**：内核关键路径经过最坏情况执行时间（Worst-Case Execution Time, WCET）分析和认证。
- **完整的认证工件**：提供安全需求追溯矩阵、DO-178C软件生命周期数据（SLD）、IEC 61508功能安全手册等文档。
- **DO-178C DAL A支持**：是极少数通过民用航空最高安全等级（Design Assurance Level A）认证的RTOS之一。


## VxWorks在机器人中的应用

### 工业机器人控制器

工业机器人对实时性和可靠性的要求与航空航天系统相当。VxWorks凭借其微秒级的确定性响应时间，在工业机器人领域有着深厚的历史积累：

- **KUKA**：德国KUKA机器人历史上的控制器系统曾使用VxWorks作为实时核心，运行关节伺服控制环（周期约1ms），保证机械臂的轨迹精度。
- **ABB**：ABB工业机器人的部分历史平台同样基于VxWorks，运行IRC5控制器的实时任务。
- **通用工业控制器**：许多PLC（可编程逻辑控制器）和运动控制卡的固件使用VxWorks，实现确定性的多轴同步控制。

VxWorks在工业机器人中的关键优势：
1. 抢占式调度保证伺服环的周期性执行，抖动（Jitter）通常在微秒量级。
2. 优先级继承防止优先级反转导致的控制失效。
3. 丰富的现场总线协议支持（EtherCAT、PROFIBUS、CANopen）。

### 手术机器人

手术机器人是对实时性、可靠性和安全认证要求最严格的机器人应用之一：

- **达芬奇手术系统（da Vinci Surgical System）**：Intuitive Surgical公司的达芬奇系统是全球最广泛使用的微创手术机器人。其运动控制子系统对延迟极度敏感——任何毫秒级的抖动都会传导为手术器械的位置误差，威胁手术安全。
- **高可靠性需求**：手术机器人控制器必须在多年连续运行中保证零意外重启，VxWorks的内存保护和故障隔离机制满足这一需求。
- **安全认证对接**：VxWorks的IEC 62304（医疗设备软件）认证支持大幅简化了手术机器人的FDA（美国食品药品监督管理局）申报流程。

### 航天应用

VxWorks在航天领域的应用是其品牌影响力的重要来源：

- **火星探路者号（Mars Pathfinder, 1997）**：首个使用VxWorks的火星任务。着陆后曾因优先级反转（Priority Inversion）导致系统频繁重启。JPL工程师格伦·里夫斯（Glenn Reeves）通过远程上传补丁，启用了互斥信号量的优先级继承选项，最终解决问题。这一事件成为实时系统领域的经典案例。
- **勇气号（Spirit）和机遇号（Opportunity）火星车（2004）**：两辆火星车均运行VxWorks，分别在火星表面工作了3年和14年以上，远超90天的设计寿命。
- **好奇号（Curiosity）火星车（2012）**：延续使用VxWorks，并采用了RAD750辐射加固处理器。

### 自动驾驶与ADAS

随着汽车行业的智能化转型，VxWorks在ADAS（高级驾驶辅助系统）和自动驾驶领域也有布局：

- **传感器融合ECU**：部分Tier 1供应商的摄像头/雷达融合控制器运行VxWorks，处理实时感知数据。
- **ISO 26262 ASIL D认证**：VxWorks 7的Safety Critical版本通过ASIL D认证，满足自动驾驶功能安全要求。
- **VxWorks Hypervisor**：允许在同一SoC上同时运行安全关键的ADAS任务（VxWorks）和信息娱乐任务（Linux/Android），通过硬件虚拟化实现严格隔离。

### 核电站控制

核电站仪控系统（Instrumentation and Control, I&C）是最严苛的安全关键应用之一：

- 反应堆控制系统要求在极端恶劣环境下连续运行数十年，且单次故障不得影响安全功能。
- VxWorks的IEC 61508 SIL 3认证使其具备进入核电仪控系统的资质。
- 部分核电站的非安全级DCS（分散控制系统）采用VxWorks作为实时执行环境。


## 与Linux实时扩展对比

| 特性 | VxWorks | Linux + Preempt-RT | FreeRTOS |
|------|---------|-------------------|---------|
| **架构** | 商用微内核RTOS | 宏内核 + 实时补丁 | 商用/开源轻量级RTOS |
| **最坏情况延迟** | 微秒级，经认证 | 几十微秒，不确定 | 微秒级，依赖配置 |
| **安全认证** | DO-178C/IEC 61508/ISO 26262/IEC 62304 全覆盖 | 无官方认证（需第三方） | 部分版本有IEC 61508认证 |
| **内存占用** | 数百KB～数MB | 数MB起步 | 数KB～数十KB |
| **开源程度** | 闭源（部分组件开源） | 完全开源 | 开源（MIT协议） |
| **成本** | 高（许可证费+版税） | 免费（工程支持付费） | 免费 |
| **生态系统** | 丰富的工业中间件 | 极丰富的开源生态 | 轻量，社区生态 |
| **多核支持** | SMP/AMP，经认证 | SMP，良好 | 有限SMP支持 |
| **典型应用** | 航空、医疗、国防 | 工业控制、软实时 | 消费电子、IoT |
| **技术支持** | Wind River官方支持 | 社区 + 商业支持 | 社区 + 商业支持 |


## 知名应用案例

VxWorks在全球众多关键系统中得到应用，以下是一些著名案例：

- **NASA火星探测**：火星探路者号（Mars Pathfinder, 1997）、勇气号（Spirit）和机遇号（Opportunity）火星车、好奇号（Curiosity）火星车均运行VxWorks。火星探路者号在着陆后曾遭遇优先级反转（Priority Inversion）导致的系统重启问题，最终通过远程启用优先级继承机制解决。
- **波音787梦想客机**：航电系统中的多个关键子系统运行VxWorks，负责飞行控制和机载系统管理。
- **SpaceX**：Dragon飞船的部分航电系统使用VxWorks。
- **轨道交通**：全球众多高速铁路和地铁的列车控制系统采用VxWorks。
- **网络设备**：众多企业级路由器和交换机的控制平面运行VxWorks。


## VxWorks 7

VxWorks 7是Wind River于2014年发布的最新一代VxWorks平台，引入了多项重要更新：

- **模块化架构**：采用分层的模块化设计，内核、中间件和应用层可以独立更新和部署。
- **容器支持**：支持在VxWorks上运行OCI兼容的容器，便于部署和管理应用。
- **安全增强**：内置安全启动（Secure Boot）、安全存储和加密框架。
- **多核支持优化**：改进了对称多处理（SMP）和非对称多处理（AMP）的支持。
- **开发者体验**：支持C++14/17标准，改进了POSIX兼容性，提供了更多开源软件包的移植（如Python、OpenSSL等）。
- **云连接**：集成了MQTT、AMQP等物联网协议，支持与AWS、Azure等云平台的连接。


## 参考资料

1. Wind River Systems, *VxWorks Programmer's Guide*.
2. Wind River Systems, *VxWorks 7 Product Overview*.
3. Glenn Reeves, "What Really Happened on Mars," JPL, 1997.
4. [Wind River VxWorks官方网站](https://www.windriver.com/products/vxworks)
5. Wind River Systems, *VxWorks Kernel Programmer's Guide: Tasks*.
6. IEC 61508, *Functional Safety of Electrical/Electronic/Programmable Electronic Safety-related Systems*, 2010.
7. RTCA DO-178C, *Software Considerations in Airborne Systems and Equipment Certification*, 2011.
8. ISO 26262, *Road Vehicles — Functional Safety*, 2018.
