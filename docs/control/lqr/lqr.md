# 线性二次型调节器

!!! note "引言"
    线性二次型调节器（Linear Quadratic Regulator，LQR）是最优控制理论（Optimal Control Theory）的核心算法之一。它通过最小化一个二次型代价函数（Quadratic Cost Function）来设计状态反馈控制器，在保证系统稳定性的同时兼顾控制性能与能量消耗之间的权衡。LQR 广泛应用于航空航天、机器人运动控制、自动驾驶等高性能控制场景，是现代控制理论中最具实用价值的工具之一。


## 问题定义

### 线性系统模型

LQR 基于线性时不变（Linear Time-Invariant，LTI）系统：

$$\dot{\mathbf{x}} = A\mathbf{x} + B\mathbf{u}$$

$$\mathbf{y} = C\mathbf{x}$$

其中：

- \(\mathbf{x} \in \mathbb{R}^n\) 为系统状态向量
- \(\mathbf{u} \in \mathbb{R}^m\) 为控制输入向量
- \(\mathbf{y} \in \mathbb{R}^p\) 为系统输出向量
- \(A \in \mathbb{R}^{n \times n}\) 为系统矩阵
- \(B \in \mathbb{R}^{n \times m}\) 为输入矩阵
- \(C \in \mathbb{R}^{p \times n}\) 为输出矩阵

对于非线性系统，通常在工作点附近进行线性化（Linearization），得到近似的线性模型后再应用 LQR。

### 无限时域代价函数

无限时域（Infinite Horizon）LQR 问题的目标是寻找最优控制律 \(\mathbf{u}^*(t)\)，使得以下代价函数最小化：

$$J = \int_0^\infty \left(\mathbf{x}^T Q \mathbf{x} + \mathbf{u}^T R \mathbf{u}\right) dt$$

其中：

- \(Q \in \mathbb{R}^{n \times n}\)，\(Q \succeq 0\) 为状态权重矩阵（半正定），惩罚状态偏差
- \(R \in \mathbb{R}^{m \times m}\)，\(R \succ 0\) 为控制输入权重矩阵（正定），惩罚控制量

代价函数的直觉意义：第一项 \(\mathbf{x}^T Q \mathbf{x}\) 衡量系统状态偏离平衡点的程度，第二项 \(\mathbf{u}^T R \mathbf{u}\) 衡量控制能量的消耗。两项之和对时间积分，反映全程的综合代价。

**前提条件**：系统 \((A, B)\) 需满足能控性（Controllability），即能控性矩阵

$$\mathcal{C} = \begin{bmatrix} B & AB & A^2B & \cdots & A^{n-1}B \end{bmatrix}$$

满秩；系统 \((A, \sqrt{Q})\) 需满足能观性（Observability）以保证解的存在性。


## 最优控制律推导

### 代数 Riccati 方程

利用变分法（Calculus of Variations）或动态规划（Dynamic Programming）中的 Hamilton-Jacobi-Bellman 方程，可以证明上述优化问题的解存在且唯一（在能控、能观条件下），最优值函数具有二次型形式：

$$V(\mathbf{x}) = \mathbf{x}^T P \mathbf{x}$$

其中正定矩阵 \(P \succ 0\) 满足代数 Riccati 方程（Algebraic Riccati Equation，ARE）：

$$A^T P + P A - P B R^{-1} B^T P + Q = 0$$

代数 Riccati 方程是 LQR 的核心。求解该方程可以得到唯一正定解 \(P\)，进而确定最优控制律。

### 最优状态反馈控制律

最优控制律为线性状态反馈形式：

$$\mathbf{u}^* = -K\mathbf{x}$$

其中最优反馈增益矩阵（Feedback Gain Matrix）\(K\) 为：

$$K = R^{-1}B^T P$$

### 闭环系统稳定性

施加最优控制律后，闭环系统（Closed-Loop System）矩阵为：

$$A_{cl} = A - BK = A - BR^{-1}B^T P$$

LQR 的重要性质：只要 \((A, B)\) 能控且 \((A, \sqrt{Q})\) 能观，LQR 设计的闭环系统必然渐近稳定（Asymptotically Stable），即 \(A_{cl}\) 的所有特征值实部均为负数。这一性质来源于最优性本身，是 LQR 相比经验调参方法的重要优势。

此外，LQR 还具有良好的鲁棒性（Robustness）保证：对于单输入系统，相位裕度（Phase Margin）至少为 \(\pm 60°\)，增益裕度（Gain Margin）至少为 \([0.5, +\infty)\)。


## Q 和 R 矩阵的选择

Q 和 R 矩阵的选择是 LQR 设计中最关键的工程环节，直接决定控制器的性能。

### Bryson 法则

Bryson 法则（Bryson's Rule）是工程中最常用的经验选取方法，基于各状态量和控制量的允许最大偏差进行归一化：

$$Q_{ii} = \frac{1}{x_{i,\max}^2}, \quad R_{jj} = \frac{1}{u_{j,\max}^2}$$

其中 \(x_{i,\max}\) 是第 \(i\) 个状态变量的允许最大偏差，\(u_{j,\max}\) 是第 \(j\) 个控制输入的允许最大值。非对角元素通常置零，即取对角矩阵。

Bryson 法则的优点是物理意义明确，且量纲统一，避免不同物理量之间因单位差异导致的比例失调。

### 调参直觉

- **增大 \(Q\) 的某个对角元 \(Q_{ii}\)**：对第 \(i\) 个状态的偏差惩罚增大，控制器将更积极地调节该状态，调节速度加快，代价是控制量增大，能量消耗增加。
- **增大 \(R\)**：控制量受到更严格惩罚，控制器趋于保守，输出更小的控制信号，代价是调节速度变慢，稳态收敛时间延长。
- **\(Q/R\) 比值**：决定系统性能与能量消耗之间的权衡，该比值越大系统响应越积极。

### 迭代调参流程

1. 用 Bryson 法则得到初始 \(Q\)、\(R\)
2. 求解 ARE，计算 \(K\) 和闭环极点
3. 仿真验证阶跃响应，评估超调量、调节时间
4. 根据仿真结果调整 \(Q\)、\(R\) 并重复步骤 2-3


## Python 实现

### 依赖库安装

```bash
pip install numpy scipy python-control matplotlib
```

### 倒立摆系统建模

以倒立摆小车（Cart-Pole）系统为例。状态向量为 \(\mathbf{x} = [x, \dot{x}, \theta, \dot{\theta}]^T\)，分别表示小车位置、小车速度、摆角、摆角速度；控制输入为施加在小车上的水平力 \(u\)。

在平衡点（\(\theta = 0\)）处线性化得到系统矩阵：

```python
import numpy as np
from scipy import linalg
import control  # pip install control

# 倒立摆系统矩阵（线性化后）
# 状态: [x, x_dot, theta, theta_dot]
m = 0.1   # 摆杆质量 [kg]
M = 0.5   # 小车质量 [kg]
L = 0.3   # 摆杆半长 [m]
g = 9.81  # 重力加速度 [m/s^2]

A = np.array([
    [0, 1,        0,    0],
    [0, 0, -m*g/M,     0],
    [0, 0,        0,    1],
    [0, 0, (M+m)*g/(M*L), 0]
])

B = np.array([[0], [1/M], [0], [-1/(M*L)]])

# 权重矩阵（Bryson 法则）
Q = np.diag([1.0, 1.0, 10.0, 1.0])  # 角度偏差权重最大
R = np.array([[0.01]])               # 控制输入权重

# 方法 1：scipy Riccati 求解器
P = linalg.solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P
print("LQR 增益 K:", K)

# 方法 2：python-control 库（更方便）
sys = control.ss(A, B, np.eye(4), np.zeros((4,1)))
K2, S, E = control.lqr(sys, Q, R)
print("闭环极点:", E)

# 验证闭环稳定性
A_cl = A - B @ K
eigenvalues = np.linalg.eigvals(A_cl)
print("闭环特征值:", eigenvalues)
# 所有实部应为负数（稳定）
```

### 关键函数说明

- `linalg.solve_continuous_are(A, B, Q, R)`：求解连续时间代数 Riccati 方程，返回正定矩阵 \(P\)
- `control.lqr(sys, Q, R)`：返回增益矩阵 \(K\)、Riccati 解 \(S\) 和闭环极点 \(E\)
- 两种方法结果等价，`python-control` 封装更简洁


## 仿真验证

### 闭环时域响应仿真

```python
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def cart_pole_dynamics(t, x, K, A, B):
    u = -K @ x
    u = np.clip(u, -10, 10)  # 限幅
    dxdt = A @ x + B.flatten() * u[0]
    return dxdt

# 初始状态：小车位移 0.1m，摆角 0.1rad
x0 = [0.1, 0.0, 0.1, 0.0]
t_span = (0, 5)
t_eval = np.linspace(0, 5, 500)

sol = solve_ivp(
    cart_pole_dynamics, t_span, x0,
    args=(K, A, B), t_eval=t_eval, method='RK45'
)

plt.figure(figsize=(12, 4))
labels = ['小车位置 x [m]', '小车速度 [m/s]', '摆角 θ [rad]', '摆角速度 [rad/s]']
for i, label in enumerate(labels):
    plt.subplot(1, 4, i+1)
    plt.plot(sol.t, sol.y[i])
    plt.xlabel('时间 [s]')
    plt.ylabel(label)
    plt.grid(True)
plt.tight_layout()
plt.savefig('lqr_cartpole.png', dpi=150)
```

### 仿真结果解读

- 正确设计的 LQR 控制器应使所有状态量从初始扰动出发，在有限时间内收敛至零（平衡点）
- 摆角 \(\theta\) 的收敛速度受 \(Q_{33}\) 控制，增大该值可加快摆角调节
- 控制量 \(u\) 应保持在物理允许范围内，若超出需适当增大 \(R\)
- 建议同时绘制控制输入 \(u(t)\) 曲线以评估能量消耗


## LQG 控制

### 状态不可完全测量的问题

实际系统中，全状态测量通常不可行。以倒立摆为例，速度量 \(\dot{x}\) 和 \(\dot{\theta}\) 往往通过传感器噪声较大或难以直接测量。线性二次型高斯控制（Linear Quadratic Gaussian，LQG）控制将 LQR 与卡尔曼滤波器（Kalman Filter）相结合，解决含噪声测量下的最优控制问题：

$$\mathbf{u} = -K\hat{\mathbf{x}}$$

其中 \(\hat{\mathbf{x}}\) 是卡尔曼滤波器对系统状态的最优估计。

### 分离定理

分离定理（Separation Principle）指出：LQR 增益 \(K\) 的设计和卡尔曼滤波器增益 \(L\) 的设计相互独立，可以分别优化后合并。这一性质大幅简化了 LQG 控制器的设计流程。

系统模型（含过程噪声和测量噪声）：

$$\dot{\mathbf{x}} = A\mathbf{x} + B\mathbf{u} + \mathbf{w}, \quad \mathbf{w} \sim \mathcal{N}(0, Q_n)$$

$$\mathbf{y} = C\mathbf{x} + \mathbf{v}, \quad \mathbf{v} \sim \mathcal{N}(0, R_n)$$

### LQG Python 实现

```python
# LQG 设计
# 过程噪声协方差 Qn，测量噪声协方差 Rn
Qn = np.eye(4) * 0.01
Rn = np.eye(2) * 0.1
C = np.array([[1,0,0,0],[0,0,1,0]])  # 仅能观测位置和角度

# 卡尔曼滤波器增益
L_T, _, _ = control.lqr(A.T, C.T, Qn, Rn)
L = L_T.T  # 观测器增益

# 闭环系统（含观测器）
# 观测器极点应比控制器极点快 3-5 倍
```

卡尔曼滤波器的状态估计方程为：

$$\dot{\hat{\mathbf{x}}} = A\hat{\mathbf{x}} + B\mathbf{u} + L(\mathbf{y} - C\hat{\mathbf{x}})$$

其中观测器增益 \(L\) 通过对偶 Riccati 方程确定。


## 与 PID 控制的对比

LQR 与比例-积分-微分（Proportional-Integral-Derivative，PID）控制是工程中最常用的两类控制器，各有优劣：

| 特性 | PID | LQR |
|------|-----|-----|
| 设计方法 | 经验调参 | 最优化（代价函数） |
| 系统类型 | SISO 为主 | MIMO 原生支持 |
| 约束处理 | 手动限幅 | 通过 R 矩阵间接 |
| 需要系统模型 | 否（经验调参） | 是（需要 A、B 矩阵） |
| 计算复杂度 | 极低 | 中等（离线求解 Riccati）|
| 稳定性保证 | 无（依赖调参） | 有（最优性 → 稳定性）|
| 典型应用 | 温控、工业过程 | 飞行器、机械臂、平衡车 |

**选择建议**：

- 对于单输入单输出（SISO）、模型未知或调参资源充裕的场景，PID 仍是首选
- 对于多输入多输出（MIMO）、有明确系统模型、对性能有较高要求的场景，LQR 更具优势
- 两者并非互斥：部分工程实践中在 LQR 基础上叠加积分项以消除稳态误差


## 机器人应用示例

LQR 在机器人领域有广泛的实际应用，以下为典型案例：

### 平衡车与倒立摆

平衡车（两轮自平衡机器人）是 LQR 最经典的演示案例。以摆角和摆角速度作为主要状态，设计 LQR 控制器使机器人保持直立。相比 PID，LQR 可同时考虑位置漂移和姿态稳定的耦合关系。

### 四旋翼飞行器

四旋翼飞行器（Quadrotor）姿态控制是 LQR 的重要应用场景。在悬停点附近线性化，以滚转角、俯仰角、偏航角及其角速度为状态，四个电机转速为控制输入，设计 LQR 实现姿态稳定控制。

### 机械臂关节控制

机械臂（Robotic Arm）的多关节协调控制本质上是 MIMO 控制问题。LQR 可以在考虑各关节动力学耦合的情况下，统一设计反馈增益矩阵，实现各关节的协调运动。

### 腿式机器人步态稳定

腿式机器人（Legged Robot）的步态稳定控制通常采用线性倒立摆（Linear Inverted Pendulum，LIP）模型对质心动力学进行线性化，然后应用 LQR 设计稳定控制器。该方法在双足机器人和四足机器人的运动控制中均有应用。

### 典型设计流程总结

1. 建立系统非线性动力学模型
2. 在目标工作点线性化，得到 \((A, B)\) 矩阵
3. 验证能控性
4. 用 Bryson 法则初始化 \(Q\)、\(R\)
5. 求解 ARE，计算 \(K\)
6. 仿真验证，根据结果迭代调整 \(Q\)、\(R\)
7. 在实际系统上进行测试与微调


## 参考资料

- Kirk, D. E. (1970). *Optimal Control Theory: An Introduction*. Dover Publications.
- Underactuated Robotics, MIT 6.832, R. Tedrake. [在线课程](https://underactuated.mit.edu/)
- Franklin, G. F., Powell, J. D., & Emami-Naeini, A. (2014). *Feedback Control of Dynamic Systems* (7th ed.). Pearson.
- [python-control 库文档](https://python-control.readthedocs.io/)
