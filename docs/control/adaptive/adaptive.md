# 自适应控制

!!! note "引言"
    经典控制器（如 PID、线性二次型调节器 Linear Quadratic Regulator, LQR）在设计时通常假设系统模型已知且参数固定不变。然而，实际机器人系统的参数会随时间发生变化：抓取物体时负载突然变化、关节摩擦系数因磨损而漂移、水下机器人随水流和深度变化而动力学特性改变。自适应控制（Adaptive Control）能够在线估计并补偿这些未知的时变参数，使控制器在不确定环境中仍能保持良好性能。


## 自适应控制的动机

### 机器人系统中的参数不确定性

实际工程应用中，以下几类典型场景会导致系统参数发生显著变化：

**负载变化**：工业机械臂在抓取不同重量的工件时，末端执行器的等效质量可能从几百克变化到数十千克。对于固定增益的 PID 控制器，同一组参数在轻载和重载时的控制效果会相差悬殊——轻载时可能响应过快产生振荡，重载时响应又过于迟缓。

**关节摩擦漂移**：机械臂的关节摩擦系数并非常数。随着使用时间延长，润滑油老化、轴承磨损会使摩擦系数发生漂移，造成控制精度下降，尤其影响低速运动时的跟踪性能。

**环境动力学变化**：
- 水下机器人（Autonomous Underwater Vehicle, AUV）在不同水深处的流体阻力不同，在不同水流速度下所受的附加质量（Added Mass）也会变化。
- 无人机（Unmanned Aerial Vehicle, UAV）在不同风速、不同海拔（空气密度不同）下的气动特性显著不同。
- 康复外骨骼（Exoskeleton）需要适应不同患者的肌肉力量和运动意图。

### 自适应控制与鲁棒控制的区别

面对不确定性，控制工程界有两种主要对策：

| 特性 | 自适应控制 | 鲁棒控制（Robust Control） |
|------|-----------|--------------------------|
| 核心思想 | 在线"学习"，识别并消除不确定性 | 设计足够大的稳定裕量，容忍不确定性 |
| 对不确定性的处理 | 动态补偿，参数估计后修正控制律 | 静态保守设计，保证最坏情况下稳定 |
| 性能 | 不确定性消除后可达到理想性能 | 保守设计导致名义工况下性能有所牺牲 |
| 典型方法 | MRAC、自整定 PID、间接自适应 | \(H_\infty\) 控制、滑模控制 |
| 适用场景 | 参数缓慢漂移或已知不确定性结构 | 快速时变扰动或不确定性结构未知 |

鲁棒控制通过"大量裕量"来容忍不确定性，代价是在标称（Nominal）工况下的性能并非最优。自适应控制则通过持续"学习"逐步消除不确定性，理论上可以在不确定参数被准确辨识后达到与已知系统相近的控制性能。


## 参考模型自适应控制

模型参考自适应控制（Model Reference Adaptive Control, MRAC）是自适应控制中最经典的方法之一。其核心思想是：设计一个稳定的**参考模型**来描述期望的系统行为，然后设计自适应律，驱使实际系统的响应趋近于参考模型的响应。

### 系统描述

**参考模型**描述期望的理想动态：

$$\dot{x}_m = A_m x_m + B_m r$$

其中 \(A_m\) 为稳定矩阵（所有特征值具有负实部），\(r\) 为参考输入，\(x_m\) 为参考状态。

**实际系统**含有未知参数 \(\theta\)：

$$\dot{x} = A x + B\theta u$$

其中 \(\theta \in \mathbb{R}^p\) 为未知参数向量（如末端负载、摩擦系数）。

**控制目标**：设计控制律 \(u\) 和参数更新律，使跟踪误差

$$e = x - x_m \to 0 \quad (t \to \infty)$$

### 直接 MRAC 结构

直接 MRAC（Direct MRAC）直接在线调整控制器增益，而不先辨识系统参数。控制律形式为：

$$u = \theta^T(t) \phi(x, r)$$

其中 \(\phi(x, r)\) 为由状态 \(x\) 和参考输入 \(r\) 构成的回归向量（Regressor），\(\theta(t)\) 为时变控制增益，由自适应律实时更新。

### MIT 规则

MIT 规则（MIT Rule）是最早提出的 MRAC 参数更新方法，基于梯度下降思想。定义代价函数：

$$J(\theta) = \frac{1}{2} e^2$$

沿代价函数下降方向更新参数：

$$\dot{\theta} = -\gamma \frac{\partial J}{\partial \theta} = -\gamma e \frac{\partial e}{\partial \theta}$$

其中 \(\gamma > 0\) 为**自适应增益**（Adaptive Gain），控制参数更新的快慢。\(\gamma\) 越大，参数更新越快，但过大的 \(\gamma\) 会导致参数振荡甚至不稳定。

MIT 规则简单直观，但**无法从理论上保证全局稳定性**，在大信号或快速变化场景下可能失稳。


## Lyapunov 稳定性方法

为克服 MIT 规则稳定性无法保证的缺陷，可以基于李雅普诺夫（Lyapunov）稳定性理论推导参数更新律，从而在理论上保证闭环系统的全局渐近稳定性。

### Lyapunov 函数设计

选取如下正定 Lyapunov 函数：

$$V(e, \tilde{\theta}) = \frac{1}{2} e^T P e + \frac{1}{2\gamma} \tilde{\theta}^T \tilde{\theta}$$

其中：
- \(e = x - x_m\) 为跟踪误差
- \(\tilde{\theta} = \theta^* - \theta\) 为参数误差（\(\theta^*\) 为未知真实参数）
- \(P = P^T \succ 0\) 为正定矩阵，满足李雅普诺夫方程 \(A_m^T P + P A_m = -Q\)（\(Q \succ 0\)）
- \(\gamma > 0\) 为自适应增益

### 参数更新律推导

对 \(V\) 求时间导数，经推导得：

$$\dot{V} = -\frac{1}{2} e^T Q e + \tilde{\theta}^T \left( \frac{1}{\gamma} \dot{\tilde{\theta}} + e^T P B \phi(x, r) \right)$$

令第二项为零，得到使 \(\dot{V} \leq 0\) 的参数更新律：

$$\dot{\theta} = \gamma e^T P B \phi(x, r)$$

### 稳定性结论

在满足持续激励（Persistent Excitation, PE）条件时，可证明：
- 闭环系统的跟踪误差 \(e \to 0\)（\(t \to \infty\)）
- 自适应参数 \(\theta(t)\) 有界，但**不一定收敛到真实值** \(\theta^*\)

参数不收敛到真实值这一现象称为**参数漂移（Parameter Drift）**。持续激励条件要求参考输入 \(r\) 足够"丰富"（包含足够多的频率成分），才能保证参数的唯一辨识性和收敛性。


## 间接自适应控制

间接自适应控制（Indirect Adaptive Control）将控制问题分解为两个子问题：**参数辨识**和**控制器设计**，采用"先估计、后控制"的流程。

### 基本流程

1. **参数估计**：利用在线辨识方法（如递推最小二乘 Recursive Least Squares, RLS）实时估计系统参数 \(\hat{A}\)、\(\hat{B}\)。
2. **控制器计算**：基于当前参数估计，重新设计 LQR 或极点配置（Pole Placement）控制器。
3. **确定性等价原则（Certainty Equivalence Principle）**：将估计值 \(\hat{\theta}\) 当作真实值代入控制器计算，忽略估计误差的影响。

### 带遗忘因子的递推最小二乘

标准递推最小二乘（RLS）对所有历史数据等权，无法追踪时变参数。引入**遗忘因子**（Forgetting Factor）\(\lambda \in (0.95, 1]\) 使旧数据的权重指数衰减：

$$\hat{\theta}_{k+1} = \hat{\theta}_k + K_k (y_k - \phi_k^T \hat{\theta}_k)$$

$$K_k = P_k \phi_k \left( \lambda + \phi_k^T P_k \phi_k \right)^{-1}$$

$$P_{k+1} = \frac{1}{\lambda} \left( P_k - K_k \phi_k^T P_k \right)$$

其中：
- \(\hat{\theta}_k\) 为第 \(k\) 步的参数估计
- \(\phi_k\) 为回归向量（由输入、输出历史数据构成）
- \(y_k\) 为当前测量输出
- \(K_k\) 为卡尔曼增益（Kalman Gain）
- \(P_k\) 为估计误差协方差矩阵

\(\lambda\) 越小，对参数变化的追踪越灵敏，但对测量噪声也越敏感；\(\lambda = 1\) 退化为标准 RLS。实际应用中通常取 \(\lambda \in [0.97, 0.99]\)。


## 自整定 PID

自整定 PID（Self-Tuning PID）通过自动辨识系统特性来整定 PID 参数，是工业界应用最广泛的自适应控制方法。相较于理论上复杂的 MRAC，自整定 PID 易于工程实现，适合大多数过程控制场景。

### 继电器反馈整定

继电器反馈整定（Relay Feedback Tuning）由 Åström 和 Hägglund 于 1984 年提出，是目前商业 PID 控制器自整定功能的主流实现方式。

**基本原理**：
1. 在控制回路中临时引入一个**继电器（滞环控制器）**，代替原有 PID 控制器。
2. 继电器迫使系统产生等幅振荡（Sustained Oscillation）。
3. 测量振荡的**临界增益** \(K_u\)（Ultimate Gain）和**振荡周期** \(T_u\)（Ultimate Period）。
4. 按 Ziegler-Nichols 规则计算 PID 参数：

$$K_p = 0.6\, K_u, \quad T_i = 0.5\, T_u, \quad T_d = 0.125\, T_u$$

整定完成后，继电器被计算得到的 PID 控制器取代，系统恢复正常控制。

### 基于阶跃响应的整定实现

```python
import numpy as np


def relay_feedback_tuning(system_step_response_data):
    """简化版继电器整定（从阶跃响应估计参数）

    参数
    ----
    system_step_response_data : tuple
        (t, y) — 时间数组与对应阶跃响应输出数组

    返回
    ----
    dict
        包含 Kp、Ti、Td 的 PID 参数字典
    """
    t, y = system_step_response_data
    y_final = y[-1]

    # 找到 63.2% 响应时间（一阶系统时间常数估计）
    idx_63 = np.argmax(y >= 0.632 * y_final)
    tau = t[idx_63]

    # 找到响应延迟（10% 响应时间）
    idx_10 = np.argmax(y >= 0.1 * y_final)
    L = t[idx_10]

    K = y_final  # 系统静态增益

    # Ziegler-Nichols 整定规则（基于时间常数和延迟）
    Kp = (1.2 * tau) / (K * L)
    Ti = 2.0 * L
    Td = 0.5 * L

    return {'Kp': Kp, 'Ti': Ti, 'Td': Td}
```

该函数通过测量一阶近似的时间常数 \(\tau\) 和纯延迟 \(L\)，用 Ziegler-Nichols 公式估算 PID 参数。适合响应曲线接近一阶加纯延迟（FOPDT）模型的工业过程。


## 增益调度

增益调度（Gain Scheduling）是一种实用的自适应策略：在多个预先确定的**工作点**（Operating Point）分别设计控制器，然后根据一个或多个**调度变量**（Scheduling Variable，如飞行速度、机械臂关节角）在线切换或插值控制增益。

增益调度不依赖在线参数估计，计算负担小，在工业和航空领域（如飞行控制系统）中有广泛应用。其局限在于：工作点之间的插值性能依赖设计者经验，且切换本身可能引入瞬态响应。

### 实现示例

```python
class GainScheduledController:
    """基于速度调度的 PID 增益表控制器"""

    def __init__(self):
        # 不同工作点对应的 PID 增益表
        self.schedule = {
            'low_speed':  {'Kp': 1.0, 'Ki': 0.1,  'Kd': 0.05},
            'mid_speed':  {'Kp': 0.8, 'Ki': 0.08, 'Kd': 0.04},
            'high_speed': {'Kp': 0.5, 'Ki': 0.05, 'Kd': 0.02},
        }

    def get_gains(self, speed):
        """根据当前速度插值获取增益

        参数
        ----
        speed : float
            当前调度变量（速度，单位 m/s）

        返回
        ----
        dict
            插值后的 PID 增益字典
        """
        if speed < 2.0:
            return self.schedule['low_speed']
        elif speed < 5.0:
            alpha = (speed - 2.0) / 3.0  # 线性插值系数
            low = self.schedule['low_speed']
            mid = self.schedule['mid_speed']
            return {k: (1 - alpha) * low[k] + alpha * mid[k] for k in low}
        else:
            return self.schedule['high_speed']
```

### 增益调度的适用条件

增益调度成立的前提是：调度变量的变化速度**远慢于**控制系统的闭环带宽，即系统在每个工作点附近的停留时间足够长，使控制器有时间稳定下来。若调度变量变化过快，需要额外分析切换稳定性（如使用多 Lyapunov 函数方法）。


## 机器人应用

自适应控制在机器人领域的应用覆盖从工业机械臂到特种机器人的广泛场景。

### 空间机械臂

空间站的机械臂（如加拿大臂 Canadarm）在捕获卫星或移动宇航员时，末端载荷的质量可能从零变化到数吨。由于太空中无法精确预知载荷惯量，固定增益控制器难以适应如此大范围的参数变化。采用 MRAC 方案，控制器可以在抓取后数秒内自动估计负载惯量并更新控制律，避免振荡或失稳。

### 软体机器人

软体机器人（Soft Robots）采用硅胶、橡胶等柔顺材料，其刚度具有强烈的非线性和时变性（如气压驱动器的刚度随充气量变化）。传统刚体动力学模型不再适用，自整定控制器或基于神经网络的自适应方案可以实时补偿材料非线性，实现精准的力/位置控制。

### 水下机器人

水下机器人（AUV）的流体动力学模型极为复杂：附加质量、水动力阻尼系数随深度、温度、盐度以及自身速度而变化。间接自适应控制方案通过 RLS 在线辨识水动力参数，在轨迹跟踪任务中相较固定参数控制器可将跟踪误差降低 30%～50%。

### 康复机器人

康复机器人（Rehabilitation Robot）需要针对每位患者的肌肉力量、运动障碍程度和主动参与意愿进行个性化适配。基于肌电信号（Electromyography, EMG）的自适应阻抗控制（Adaptive Impedance Control）可以实时调整机器人的刚度和阻尼，在辅助运动的同时鼓励患者主动发力，从而提升康复效果。

### 典型方法与场景对应

| 应用场景 | 主要不确定性 | 推荐方法 |
|----------|------------|---------|
| 工业机械臂变负载 | 末端惯量、摩擦系数 | 直接 MRAC、间接 RLS |
| 无人机变风速 | 气动阻力、推力系数 | 增益调度、自整定 PID |
| 水下机器人 | 附加质量、水动力阻尼 | 间接自适应 + RLS |
| 康复外骨骼 | 患者肌力、运动意图 | 自适应阻抗控制 |
| 软体机器人 | 材料非线性、气压特性 | 神经网络自适应 |


## 参考资料

- Åström, K. J., & Wittenmark, B. (2008). *Adaptive Control* (2nd ed.). Dover Publications.
- Ioannou, P. A., & Sun, J. (2012). *Robust Adaptive Control*. Dover Publications.
- Slotine, J.-J. E., & Li, W. (1991). *Applied Nonlinear Control*. Prentice Hall.
- Åström, K. J., & Hägglund, T. (1984). Automatic tuning of simple regulators with specifications on phase and amplitude margins. *Automatica*, 20(5), 645–651.
