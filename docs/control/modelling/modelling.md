# 控制系统建模

!!! note "引言"
    系统建模（System Modeling）是控制系统设计的基础。一个核心原则是：**你只能控制你能建模的系统**。无论是设计PID控制器、状态反馈控制律，还是基于模型预测控制（Model Predictive Control，MPC），都需要一个准确的数学模型来描述系统的动态行为。

    数学模型将物理世界中的机械、电气、热力学等现象转化为可以用计算工具分析和设计的方程组。模型的精度直接决定了控制器的性能上限：模型误差越大，控制器的鲁棒性要求越高，性能越难保证。

    本文介绍控制系统建模的四种主要表示形式——微分方程、差分方程、传递函数和状态空间——并通过直流电机、倒立摆和移动机器人等具体示例展示建模的完整流程。


系统可以用以下方式之一来描述：

- 微分方程 (Differential Equation)
- 差分方程 (Difference Equation)
- 传递函数 (Transfer Function)
- 状态空间 (State Space)


## 微分方程 (Differential Equation)

微分方程是描述连续时间动态系统最直接的方式。它直接来源于物理定律（牛顿定律、基尔霍夫定律、热力学定律等），具有明确的物理意义。

### 基本形式

一般 \(n\) 阶线性常系数常微分方程的形式为：

$$
a_n \frac{d^n y}{dt^n} + a_{n-1} \frac{d^{n-1} y}{dt^{n-1}} + \cdots + a_1 \frac{dy}{dt} + a_0 y = b_m \frac{d^m u}{dt^m} + \cdots + b_0 u
$$

其中 \(y(t)\) 是系统输出，\(u(t)\) 是系统输入。

### 物理解释

微分方程中各项通常对应系统中的能量存储或耗散机制：

- **惯性项**（含最高阶导数）：对应能量存储，如质量的动能、电感储能
- **阻尼项**（含一阶导数）：对应能量耗散，如阻尼力、电阻热耗散
- **刚度/恢复力项**（零阶项）：对应势能存储，如弹簧弹性势能、电容储能

### 弹簧-质量-阻尼系统

弹簧-质量-阻尼系统（Mass-Spring-Damper System）是控制理论中最经典的机械系统示例。

设质量块质量为 \(m\)，弹簧刚度系数为 \(k\)，阻尼系数为 \(c\)，外力为 \(F(t)\)，位移为 \(x(t)\)，由牛顿第二定律得：

$$
m\ddot{x} + c\dot{x} + kx = F(t)
$$

各项的物理含义：

- \(m\ddot{x}\)：惯性力，质量乘以加速度
- \(c\dot{x}\)：阻尼力，与速度成正比（方向相反）
- \(kx\)：弹簧恢复力，与位移成正比（方向相反）
- \(F(t)\)：外部激励力

这是一个二阶系统，需要两个初始条件 \(x(0)\) 和 \(\dot{x}(0)\) 才能唯一确定解。

### 直流电机模型

直流电机（DC Motor）是机器人系统中最常见的执行机构，其电气和机械方程分别为：

**电气方程**（基于基尔霍夫电压定律）：

$$
L\frac{di}{dt} + Ri = V - K_e\omega
$$

其中：
- \(L\)：电枢电感（Armature Inductance），单位 H
- \(R\)：电枢电阻（Armature Resistance），单位 Ω
- \(i\)：电枢电流，单位 A
- \(V\)：输入电压，单位 V
- \(K_e\)：反电动势系数（Back-EMF Constant），单位 V·s/rad
- \(\omega\)：电机转速，单位 rad/s

**机械方程**（基于牛顿第二定律的转动形式）：

$$
J\dot{\omega} = K_t i - B\omega
$$

其中：
- \(J\)：转动惯量（Moment of Inertia），单位 kg·m²
- \(K_t\)：转矩常数（Torque Constant），单位 N·m/A
- \(B\)：粘性摩擦系数（Viscous Friction Coefficient），单位 N·m·s/rad

这两个耦合的微分方程完整描述了直流电机的动态特性。


## 差分方程 (Difference Equation)

差分方程（Difference Equation）用于描述离散时间系统，是连续微分方程经过离散化后的形式，也是数字控制器在计算机上实现的基础。

### 基本形式

一阶差分方程：

$$
x_{k} = ax_{k-1} + bu_{k-1}
$$

一般 \(n\) 阶线性差分方程：

$$
y_k + a_1 y_{k-1} + \cdots + a_n y_{k-n} = b_0 u_k + b_1 u_{k-1} + \cdots + b_m u_{k-m}
$$

其中下标 \(k\) 表示第 \(k\) 个采样时刻，\(T_s\) 为采样周期（Sampling Period）。

### 与连续系统的关系

连续微分方程可以通过不同方法转化为差分方程：

| 方法 | 连续微分 → 差分近似 | 特点 |
|------|---------------------|------|
| 前向欧拉（Forward Euler） | \(\dot{x} \approx \frac{x_{k+1}-x_k}{T_s}\) | 简单，可能不稳定 |
| 后向欧拉（Backward Euler） | \(\dot{x} \approx \frac{x_k - x_{k-1}}{T_s}\) | 较稳定，有相位误差 |
| 双线性变换（Bilinear/Tustin） | \(s \leftarrow \frac{2}{T_s}\frac{z-1}{z+1}\) | 保持频率响应特性 |
| 零阶保持（Zero-Order Hold，ZOH） | 精确离散化 | 最精确 |


## 传递函数 (Transfer Function)

传递函数是在拉普拉斯域（Laplace Domain）描述线性时不变（Linear Time-Invariant，LTI）系统输入输出关系的方法。对于初始条件为零的系统，传递函数定义为输出的拉普拉斯变换与输入的拉普拉斯变换之比：

$$
G(s) = \frac{Y(s)}{U(s)}
$$

### 多项式形式 (Polynomial Form)

$$
G(s) = \frac{b_m s^m + b_{m-1}s^{m-1} + \cdots + b_1 s + b_0}{a_n s^n + a_{n-1}s^{n-1} + \cdots + a_1 s + a_0}
$$

对于物理可实现系统，要求分子阶次不超过分母阶次，即 \(m \leq n\)。

### 零极点形式 (Poles and Zeros)

$$
G(s) = K \frac{(s-z_m)(s-z_{m-1})\cdots(s-z_1)}{(s-p_n)(s-p_{n-1})\cdots(s-p_1)}
$$

其中 \(K\) 为增益，\(z_i\) 为零点（Zeros），\(p_i\) 为极点（Poles）。

### 极点与零点的物理意义

**极点（Poles）**是使传递函数分母为零的 \(s\) 值，决定系统的自然响应（Natural Response）：

- 实数负极点 \(p = -\sigma\)：对应指数衰减模态 \(e^{-\sigma t}\)，系统稳定
- 实数正极点 \(p = +\sigma\)：对应指数增长模态，系统不稳定
- 共轭复数极点 \(p = -\sigma \pm j\omega_d\)：对应衰减振荡 \(e^{-\sigma t}\sin(\omega_d t)\)
- 纯虚数极点 \(p = \pm j\omega_0\)：对应等幅振荡，临界稳定

**零点（Zeros）**是使传递函数分子为零的 \(s\) 值，影响系统对特定频率的响应：

- 零点可以抵消极点（若两者重合，称为极零相消）
- 右半平面零点（非最小相位，Non-Minimum Phase）会导致系统响应出现初始反向（Undershoot）

### 标准二阶系统

控制理论中最重要的参考模型是标准二阶系统（Standard Second-Order System）：

$$
H(s) = \frac{\omega_n^2}{s^2 + 2\zeta\omega_n s + \omega_n^2}
$$

参数含义：

- \(\omega_n\)：无阻尼自然频率（Undamped Natural Frequency），单位 rad/s
- \(\zeta\)：阻尼比（Damping Ratio），无量纲

阻尼比决定系统的响应特性：

| 阻尼比范围 | 系统类型 | 阶跃响应特征 |
|------------|----------|--------------|
| \(\zeta = 0\) | 无阻尼（Undamped） | 等幅振荡 |
| \(0 < \zeta < 1\) | 欠阻尼（Underdamped） | 衰减振荡，有超调 |
| \(\zeta = 1\) | 临界阻尼（Critically Damped） | 无超调，最快无振荡收敛 |
| \(\zeta > 1\) | 过阻尼（Overdamped） | 无超调，响应较慢 |

其极点为：

$$
p_{1,2} = -\zeta\omega_n \pm \omega_n\sqrt{\zeta^2 - 1}
$$

欠阻尼情况下（\(0 < \zeta < 1\)），极点为共轭复数：

$$
p_{1,2} = -\zeta\omega_n \pm j\omega_n\sqrt{1-\zeta^2} = -\sigma \pm j\omega_d
$$

其中 \(\omega_d = \omega_n\sqrt{1-\zeta^2}\) 称为有阻尼自然频率（Damped Natural Frequency）。

### 直流电机的闭环传递函数

将直流电机的电气方程和机械方程进行拉普拉斯变换：

电气方程：\((Ls + R)I(s) = V(s) - K_e\Omega(s)\)

机械方程：\((Js + B)\Omega(s) = K_t I(s)\)

消去 \(I(s)\)，得到从电压 \(V(s)\) 到转速 \(\Omega(s)\) 的开环传递函数：

$$
G(s) = \frac{\Omega(s)}{V(s)} = \frac{K_t}{(Ls+R)(Js+B) + K_eK_t}
$$

展开分母：

$$
G(s) = \frac{K_t}{LJs^2 + (LB+RJ)s + (RB + K_eK_t)}
$$

这是一个标准二阶系统，可以与 \(H(s)\) 的形式对比，提取 \(\omega_n\) 和 \(\zeta\)。

### 波特图简介

波特图（Bode Plot）是频率响应分析的主要工具，包含两个图：

- **幅频特性**：纵轴为增益（单位 dB），横轴为频率（对数坐标）
- **相频特性**：纵轴为相角（度），横轴为频率（对数坐标）

通过波特图可以直观判断：

- **增益裕度（Gain Margin）**：系统在相角为 -180° 时允许的额外增益，衡量稳定裕量
- **相位裕度（Phase Margin）**：系统在增益为 0 dB 时距离 -180° 的相位余量
- **带宽（Bandwidth）**：增益下降 3 dB 时对应的频率，衡量系统响应速度


## 状态空间 (State Space)

状态空间（State Space）表示是一种更为通用的系统描述方式，可以处理多输入多输出（Multi-Input Multi-Output，MIMO）系统和非线性系统，是现代控制理论的基础。

![](assets/markdown-img-paste-2017041221520164.png)

### 标准形式

$$
\begin{align}
\dot{x}(t) &= Ax(t) + Bu(t) \\\\
y(t) &= Cx(t) + Du(t)
\end{align}
$$

矩阵含义：

- \(\mathbf{x}\)：状态向量（State Vector），维度为 \(n \times 1\)，包含系统内部的完整信息
- \(\mathbf{A}\)：系统矩阵（System Matrix），维度为 \(n \times n\)，描述状态间的相互作用
- \(\mathbf{B}\)：输入矩阵（Input Matrix），维度为 \(n \times m\)，描述输入对状态的影响（\(m\) 为输入数）
- \(\mathbf{C}\)：输出矩阵（Output Matrix），维度为 \(p \times n\)，描述哪些状态被测量（\(p\) 为输出数）
- \(\mathbf{D}\)：前馈矩阵（Feedforward Matrix），维度为 \(p \times m\)，描述输入直接影响输出的部分

**重要性质**：传递函数的极点就是系统矩阵 \(\mathbf{A}\) 的特征值（Eigenvalues）。

传递函数与状态空间的关系：

$$
G(s) = C(sI - A)^{-1}B + D
$$

### 能控性与能观性

**能控性（Controllability）**：能否通过选择输入 \(u(t)\) 在有限时间内将系统从任意初始状态转移到任意目标状态。

能控性矩阵（Controllability Matrix）：

$$
\mathcal{C} = \begin{bmatrix} B & AB & A^2B & \cdots & A^{n-1}B \end{bmatrix}
$$

若 \(\mathcal{C}\) 满秩（rank \(= n\)），则系统完全能控。能控性是状态反馈控制（如极点配置、LQR）的必要条件。

**能观性（Observability）**：能否仅通过观测输出 \(y(t)\) 在有限时间内唯一确定系统的初始状态（进而确定所有状态）。

能观性矩阵（Observability Matrix）：

$$
\mathcal{O} = \begin{bmatrix} C \\ CA \\ CA^2 \\ \vdots \\ CA^{n-1} \end{bmatrix}
$$

若 \(\mathcal{O}\) 满秩（rank \(= n\)），则系统完全能观。能观性是设计状态观测器（如卡尔曼滤波器，Kalman Filter）的必要条件。

### 直流电机状态空间模型

选择状态变量 \(\mathbf{x} = \begin{bmatrix} i \\ \omega \end{bmatrix}\)（电流和角速度），输入 \(u = V\)（电压），输出 \(y = \omega\)（转速）：

由电气方程 \(L\frac{di}{dt} = V - Ri - K_e\omega\)，得：

$$
\dot{i} = -\frac{R}{L}i - \frac{K_e}{L}\omega + \frac{1}{L}V
$$

由机械方程 \(J\dot{\omega} = K_t i - B\omega\)，得：

$$
\dot{\omega} = \frac{K_t}{J}i - \frac{B}{J}\omega
$$

写成矩阵形式：

$$
\begin{bmatrix} \dot{i} \\ \dot{\omega} \end{bmatrix} = \begin{bmatrix} -R/L & -K_e/L \\ K_t/J & -B/J \end{bmatrix} \begin{bmatrix} i \\ \omega \end{bmatrix} + \begin{bmatrix} 1/L \\ 0 \end{bmatrix} V
$$

$$
y = \begin{bmatrix} 0 & 1 \end{bmatrix} \begin{bmatrix} i \\ \omega \end{bmatrix}
$$

即：

$$
A = \begin{bmatrix} -R/L & -K_e/L \\ K_t/J & -B/J \end{bmatrix}, \quad B = \begin{bmatrix} 1/L \\ 0 \end{bmatrix}, \quad C = \begin{bmatrix} 0 & 1 \end{bmatrix}, \quad D = \begin{bmatrix} 0 \end{bmatrix}
$$

### 倒立摆状态空间模型

倒立摆（Inverted Pendulum）系统的状态变量选为小车位置 \(x\)、小车速度 \(\dot{x}\)、摆角 \(\theta\)（以垂直向上为零点）、摆角速率 \(\dot{\theta}\)：

$$
\mathbf{x} = \begin{bmatrix} x \\ \dot{x} \\ \theta \\ \dot{\theta} \end{bmatrix}
$$

在直立平衡点（\(\theta = 0\)）附近线性化后，状态方程为：

$$
\dot{\mathbf{x}} = \begin{bmatrix} 0 & 1 & 0 & 0 \\ 0 & 0 & -\frac{mg}{M} & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & \frac{(M+m)g}{Ml} & 0 \end{bmatrix} \mathbf{x} + \begin{bmatrix} 0 \\ \frac{1}{M} \\ 0 \\ -\frac{1}{Ml} \end{bmatrix} F
$$

其中 \(M\) 为小车质量，\(m\) 为摆杆质量，\(l\) 为摆杆长度，\(g\) 为重力加速度，\(F\) 为施加在小车上的水平力。


## 本章内容导览

建模章节按「模型表示 → 数学工具 → 离散与辨识 → 状态空间分析 → 示例」的顺序组织：

| 页面 | 主要内容 |
|------|---------|
| [控制系统建模](modelling.md) | 微分方程、差分方程、传递函数、状态空间四种模型表示及其互换 |
| [建模数学基础](modelling-foundations.md) | 拉普拉斯变换、方框图代数、信号流图、非线性系统线性化 |
| [离散化与系统辨识](modelling-discrete-identification.md) | z 变换、ZOH/Tustin 离散化、参数辨识方法与模型验证 |
| [状态空间分析](state-space.md) | 能控性、能观性、Kalman 分解、状态观测器与分离原理 |
| [建模示例与工具实践](modelling-examples.md) | 直流电机与倒立摆完整建模、Python/MATLAB 代码、函数速查 |
| [建模参考资料](modelling-full-reference.md) | 推荐教材、在线课程、软件工具对比、建模最佳实践 |


## 参考资料

1. Control Tutorials, [Inverted Pendulum: Digital Controller Design](http://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=ControlDigital), University of Michigan
2. K. Ogata, *Modern Control Engineering*, 5th ed., Prentice Hall, 2010.
3. R. C. Dorf and R. H. Bishop, *Modern Control Systems*, 13th ed., Pearson, 2017.
4. G. F. Franklin, J. D. Powell, and A. Emami-Naeini, *Feedback Control of Dynamic Systems*, 8th ed., Pearson, 2019.
5. L. Ljung, *System Identification: Theory for the User*, 2nd ed., Prentice Hall, 1999.
6. Python Control Systems Library, [python-control.readthedocs.io](https://python-control.readthedocs.io/)
7. SciPy Signal Processing, [docs.scipy.org/doc/scipy/reference/signal.html](https://docs.scipy.org/doc/scipy/reference/signal.html)

