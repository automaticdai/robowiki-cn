# 建模示例与工具实践

!!! note "引言"
    建模方法只有落到具体系统上才会变得清晰。本页面给出直流电机、倒立摆、移动机器人等典型对象从物理规律到传递函数与状态空间模型的完整推导，随后以 Python `control` 库和 MATLAB Control System Toolbox 演示如何把这些模型转化为可仿真、可分析的代码——阶跃响应、波特图、极点零点、能控性判定与离散化验证。页末附常用 MATLAB 建模函数速查。


## 典型系统建模示例

### DC 电机完整建模

#### 物理建模

直流电机由电气子系统和机械子系统组成，通过电磁耦合：

**步骤 1：建立物理方程**

电气回路（基尔霍夫电压定律）：

$$
V(t) = L\frac{di}{dt} + Ri + e_{back}
$$

其中反电动势（Back Electromotive Force）\(e_{back} = K_e\omega\)，所以：

$$
L\frac{di}{dt} = V - Ri - K_e\omega
$$

机械转动（牛顿第二定律，转动形式）：

$$
J\frac{d\omega}{dt} = \tau_{motor} - \tau_{friction} = K_t i - B\omega
$$

**步骤 2：整理状态方程**

选状态 \(\mathbf{x} = [i, \omega]^T\)，输入 \(u = V\)：

$$
\frac{d}{dt}\begin{bmatrix} i \\ \omega \end{bmatrix} = \underbrace{\begin{bmatrix} -R/L & -K_e/L \\ K_t/J & -B/J \end{bmatrix}}_{A} \begin{bmatrix} i \\ \omega \end{bmatrix} + \underbrace{\begin{bmatrix} 1/L \\ 0 \end{bmatrix}}_{B} V
$$

**步骤 3：选择输出**

若输出为角速度：\(C = [0, 1]\)，\(D = [0]\)

若输出为位置角 \(\theta\)，需增加一个积分状态，状态扩展为 \(\mathbf{x} = [i, \omega, \theta]^T\)，在 \(A\) 矩阵末行添加 \([0, 1, 0]\)。

**步骤 4：验证能控性**

$$
\mathcal{C} = [B, AB] = \begin{bmatrix} 1/L & -R/L^2 \\ 0 & K_t/(JL) \end{bmatrix}
$$

行列式 \(\det(\mathcal{C}) = \frac{K_t}{JL^2} \neq 0\)（在正常参数下），系统完全能控。

### 倒立摆建模

#### 非线性方程推导

设小车质量为 \(M\)，摆杆质量为 \(m\)，摆杆长度为 \(l\)（质心到铰接点距离），小车位置为 \(x\)，摆角为 \(\theta\)（从竖直向上量起），外力为 \(F\)。

利用拉格朗日（Lagrangian）方法得到非线性运动方程：

$$
(M + m)\ddot{x} + ml\ddot{\theta}\cos\theta - ml\dot{\theta}^2\sin\theta = F
$$

$$
ml^2\ddot{\theta} + ml\ddot{x}\cos\theta - mgl\sin\theta = 0
$$

#### 在平衡点线性化

在平衡点 \(\theta_0 = 0\)，\(\dot{\theta}_0 = 0\)，\(\dot{x}_0 = 0\)，\(F_0 = 0\) 处，令 \(\sin\theta \approx \theta\)，\(\cos\theta \approx 1\)，\(\dot{\theta}^2 \approx 0\)：

$$
(M + m)\ddot{x} + ml\ddot{\theta} = F
$$

$$
ml^2\ddot{\theta} + ml\ddot{x} - mgl\theta = 0
$$

整理后得线性状态空间模型（状态 \(\mathbf{x} = [x, \dot{x}, \theta, \dot{\theta}]^T\)）：

$$
A = \begin{bmatrix} 0 & 1 & 0 & 0 \\ 0 & 0 & \frac{-mg}{M} & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & \frac{(M+m)g}{Ml} & 0 \end{bmatrix}, \quad B = \begin{bmatrix} 0 \\ \frac{1}{M} \\ 0 \\ \frac{-1}{Ml} \end{bmatrix}
$$

注意矩阵 \(A\) 有正实部特征值（对应不稳定的倒立平衡），必须通过主动控制稳定。

### 移动机器人运动学建模

差速驱动机器人（Differential Drive Robot）是移动机器人中最常见的结构。设机器人在二维平面内运动，位姿（Pose）为 \(\mathbf{q} = [x, y, \theta]^T\)，其中 \((x, y)\) 为位置，\(\theta\) 为朝向角。

#### 运动学模型

控制输入为线速度 \(v\) 和角速度 \(\omega\)，运动学方程为：

$$
\dot{x} = v\cos\theta
$$

$$
\dot{y} = v\sin\theta
$$

$$
\dot{\theta} = \omega
$$

写成向量形式：

$$
\begin{bmatrix} \dot{x} \\ \dot{y} \\ \dot{\theta} \end{bmatrix} = \begin{bmatrix} \cos\theta & 0 \\ \sin\theta & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} v \\ \omega \end{bmatrix}
$$

#### 从轮速到速度

设左右轮线速度分别为 \(v_L\) 和 \(v_R\)，轮距为 \(d\)，则：

$$
v = \frac{v_R + v_L}{2}, \quad \omega = \frac{v_R - v_L}{d}
$$

#### 模型特点

差速驱动模型是非完整约束（Nonholonomic Constraint）系统：机器人不能横向移动（侧移），即在任意时刻满足约束：

$$
\dot{x}\sin\theta - \dot{y}\cos\theta = 0
$$

非完整约束使得路径规划和控制比完整约束系统更为复杂。


## Python 代码示例

### 直流电机仿真

```python
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# 直流电机参数
R = 1.0    # 电枢电阻 (Ohm)
L = 0.5    # 电枢电感 (H)
Ke = 0.01  # 反电动势系数 (V·s/rad)
Kt = 0.01  # 转矩常数 (N·m/A)
J = 0.01   # 转动惯量 (kg·m²)
B = 0.1    # 粘性摩擦系数 (N·m·s/rad)

# 状态空间矩阵，状态 x = [电流 i, 角速度 ω]
A = np.array([[-R/L, -Ke/L],
              [Kt/J, -B/J]])
B_mat = np.array([[1/L],
                  [0]])
C = np.array([[0, 1]])  # 输出角速度
D = np.array([[0]])

# 构建状态空间系统
sys = signal.StateSpace(A, B_mat, C, D)

# 阶跃响应仿真
t, y = signal.step(sys)

plt.figure(figsize=(8, 4))
plt.plot(t, y, 'b-', linewidth=2)
plt.xlabel('时间 (s)')
plt.ylabel('角速度 (rad/s)')
plt.title('直流电机阶跃响应')
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"A 矩阵特征值（即极点）: {np.linalg.eigvals(A)}")
```

### 能控性与能观性分析

```python
import numpy as np

def controllability_matrix(A, B):
    """计算能控性矩阵"""
    n = A.shape[0]
    cols = [B]
    for i in range(1, n):
        cols.append(np.linalg.matrix_power(A, i) @ B)
    return np.hstack(cols)

def observability_matrix(A, C):
    """计算能观性矩阵"""
    n = A.shape[0]
    rows = [C]
    for i in range(1, n):
        rows.append(C @ np.linalg.matrix_power(A, i))
    return np.vstack(rows)

# 直流电机参数（沿用上方定义）
R, L, Ke, Kt, J, B = 1.0, 0.5, 0.01, 0.01, 0.01, 0.1

A = np.array([[-R/L, -Ke/L],
              [Kt/J, -B/J]])
B_mat = np.array([[1/L], [0]])
C = np.array([[0, 1]])

C_mat = controllability_matrix(A, B_mat)
O_mat = observability_matrix(A, C)

print(f"能控性矩阵秩: {np.linalg.matrix_rank(C_mat)} (系统阶数 n={A.shape[0]})")
print(f"能观性矩阵秩: {np.linalg.matrix_rank(O_mat)} (系统阶数 n={A.shape[0]})")
```

### 传递函数与波特图

```python
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# 直流电机传递函数参数
R, L, Ke, Kt, J, B = 1.0, 0.5, 0.01, 0.01, 0.01, 0.1

# 分子分母多项式系数
num = [Kt]
den = [L*J, L*B + R*J, R*B + Ke*Kt]

sys_tf = signal.TransferFunction(num, den)

# 波特图
w, mag, phase = signal.bode(sys_tf)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
ax1.semilogx(w, mag)
ax1.set_ylabel('幅值 (dB)')
ax1.set_title('直流电机波特图')
ax1.grid(True, which='both')

ax2.semilogx(w, phase)
ax2.set_xlabel('频率 (rad/s)')
ax2.set_ylabel('相角 (°)')
ax2.grid(True, which='both')

plt.tight_layout()
plt.show()
```


## Matlab 函数参考

### 传递函数 (Transfer Function)

```matlab
s = tf('s')
G = feedback(G_plant, H_sensor)   % 闭环传递函数
G = zpk(sys)                       % 转换为零极点增益形式
G = zpk([zeros], [poles], gain)    % 直接定义零极点增益形式
```

### 零极点 (Poles and Zeros)

查找 SISO 或 MIMO 系统的极点：

```matlab
pole(sys)      % 计算极点
zero(sys)      % 计算零点
pzplot(sys)    % 绘制零极点图
```

### 状态空间 (State Space)

```matlab
sys = ss(A, B, C, D)         % 连续时间状态空间模型
sys = ss(A, B, C, D, Ts)     % 离散时间状态空间模型（采样周期 Ts）
sys_ss = ss(sys_tf)          % 从传递函数转换为状态空间
Wc = ctrb(A, B)              % 能控性矩阵
Wo = obsv(A, C)              % 能观性矩阵
sys_d = c2d(sys_c, Ts, 'zoh') % 连续转离散（ZOH 方法）
```

### 系统分析 (System Analysis)

```matlab
linearSystemAnalyzer(G, T1, T2)  % 图形化线性系统分析工具
step(sys)                         % 阶跃响应
impulse(sys)                      % 冲激响应
bode(sys)                         % 波特图
nyquist(sys)                      % 奈奎斯特图
margin(sys)                       % 增益裕度和相位裕度
```


## 参考资料

1. Control Tutorials for MATLAB and Simulink, University of Michigan. http://ctms.engin.umich.edu/CTMS/
2. Python Control Systems Library 文档. https://python-control.readthedocs.io/
3. MathWorks, *Control System Toolbox Documentation*. https://www.mathworks.com/help/control/
4. [控制系统建模](modelling.md)
5. [建模数学基础](modelling-foundations.md)
6. [离散化与系统辨识](modelling-discrete-identification.md)
