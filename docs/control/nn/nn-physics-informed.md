# 物理信息与算子学习控制

!!! note "引言"
    传统神经网络控制器把系统当作黑箱，完全依赖数据拟合输入输出映射。物理信息与算子学习方法则把已知的物理规律重新注入网络：物理信息神经网络（Physics-Informed Neural Network, PINN）把微分方程残差写进损失函数，神经微分方程（Neural Ordinary Differential Equation, Neural ODE）直接学习连续时间动力学，Koopman 算子则在升维空间中把非线性系统化为线性系统，从而复用成熟的线性控制理论。三者共同的目标是用更少的数据换取更好的泛化能力与可分析性。


## 物理信息神经网络（PINN）

物理信息神经网络（Physics-Informed Neural Networks, PINN）由Raissi等人于2019年提出，其核心思想是在神经网络的损失函数中嵌入物理方程约束，使网络的预测结果自动满足已知的物理定律。

### 损失函数设计

PINN的总损失函数由数据拟合损失和物理残差损失两部分组成：

$$L = L_{\text{data}} + \lambda L_{\text{physics}}$$

其中 \(\lambda\) 为权衡系数，控制物理约束的强度。对于机器人动力学学习任务：

- **数据损失** \(L_{\text{data}}\)：使网络输出与测量数据吻合。
- **物理损失** \(L_{\text{physics}}\)：惩罚网络预测违反牛顿第二定律 \(F = ma\) 的程度。

$$L_{\text{physics}} = \left\| m\ddot{q} - \tau + G(q) + C(q,\dot{q})\dot{q} \right\|^2$$

其中 \(q\) 为关节角度，\(\tau\) 为关节力矩，\(G(q)\) 为重力项，\(C(q,\dot{q})\) 为科里奥利矩阵。

### PyTorch实现示例

```python
import torch
import torch.nn as nn

class PINN(nn.Module):
    """物理信息神经网络：学习机器人关节动力学"""
    def __init__(self, state_dim=4, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),   nn.Tanh(),
            nn.Linear(hidden, 1)         # 输出：预测加速度
        )

    def forward(self, x):
        return self.net(x)


def physics_residual(model, q, dq, tau, mass=1.0):
    """
    计算物理残差：F = ma  =>  tau = m * ddq
    使用 torch.autograd.grad 计算输出对输入的梯度
    """
    q   = q.requires_grad_(True)
    dq  = dq.requires_grad_(True)

    # 网络预测加速度
    state = torch.cat([q, dq], dim=-1)
    ddq_pred = model(state)

    # 物理约束残差：tau - m*ddq = 0
    residual = tau - mass * ddq_pred
    return residual


def pinn_loss(model, q, dq, tau, ddq_measured, lam=1.0):
    state = torch.cat([q, dq], dim=-1)
    ddq_pred = model(state)

    # 数据损失：与测量加速度比较
    loss_data = nn.functional.mse_loss(ddq_pred, ddq_measured)

    # 物理损失：动力学方程残差
    residual = physics_residual(model, q, dq, tau)
    loss_phys = (residual ** 2).mean()

    return loss_data + lam * loss_phys
```

与纯数据驱动方法相比，PINN在训练数据稀少时仍能保持良好的外推性能，因为物理约束限制了网络的假设空间，防止过拟合。


## 神经微分方程（Neural ODE）

神经微分方程（Neural Ordinary Differential Equation, Neural ODE）由Chen等人于2018年提出，将神经网络与常微分方程求解器（ODE Solver）结合，形成连续深度模型。

### 基本原理

传统残差网络（ResNet）的前向传播可以写成：

$$\mathbf{h}_{t+1} = \mathbf{h}_t + f_\theta(\mathbf{h}_t, t)$$

Neural ODE将离散的层堆叠推广为连续的微分方程：

$$\dot{x} = f_\theta(x, t), \quad x(t_0) = x_0$$

网络输出通过ODE求解器（如Runge-Kutta方法）在时间上积分得到：

$$x(t_1) = x(t_0) + \int_{t_0}^{t_1} f_\theta(x(t), t)\, dt$$

### 轨迹预测应用

Neural ODE天然适合建模机器人轨迹（位置随时间的连续演化）。以下使用 `torchdiffeq` 库实现潜在ODE（Latent ODE）进行轨迹预测：

```python
import torch
import torch.nn as nn
from torchdiffeq import odeint  # pip install torchdiffeq

class ODEFunc(nn.Module):
    """定义潜在空间中的动力学 dx/dt = f(x, t)"""
    def __init__(self, latent_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.Tanh(),
            nn.Linear(64, 64),         nn.Tanh(),
            nn.Linear(64, latent_dim)
        )

    def forward(self, t, x):
        # t 是标量时间，x 是 [batch, latent_dim]
        return self.net(x)


class LatentODE(nn.Module):
    """潜在ODE：编码器将轨迹压缩到潜在空间，ODE在其中演化"""
    def __init__(self, obs_dim=2, latent_dim=16):
        super().__init__()
        self.encoder  = nn.Linear(obs_dim, latent_dim)
        self.ode_func = ODEFunc(latent_dim)
        self.decoder  = nn.Linear(latent_dim, obs_dim)

    def forward(self, x0, t_span):
        """
        x0:     初始观测值 [batch, obs_dim]
        t_span: 时间点序列 [T]
        返回:   预测轨迹   [T, batch, obs_dim]
        """
        z0 = self.encoder(x0)                    # 编码到潜在空间
        zt = odeint(self.ode_func, z0, t_span)   # ODE积分
        return self.decoder(zt)                   # 解码回观测空间


# 使用示例：预测机器人末端执行器的2D轨迹
model  = LatentODE(obs_dim=2, latent_dim=16)
t_span = torch.linspace(0, 2.0, 50)     # 预测2秒，共50个时间点
x0     = torch.tensor([[0.5, 0.3]])     # 初始位置
traj   = model(x0, t_span)             # 预测轨迹 [50, 1, 2]
```

Neural ODE的主要优势在于：参数数量与积分步数无关（深度自适应），且天然支持不规则采样时间序列。


## Koopman算子与线性化

Koopman算子理论（Koopman Operator Theory）提供了将非线性系统转化为等价线性系统的数学框架，为将深度学习与经典线性控制理论（如LQR）结合提供了途径。

### 基本思想

对于非线性离散动力系统 \(x_{k+1} = f(x_k)\)，Koopman算子 \(\mathcal{K}\) 作用于可观测函数（Observable Function）\(\phi(x)\) 上：

$$\mathcal{K}\phi(x) = \phi(f(x))$$

选取合适的可观测函数集 \(\{\phi_1, \phi_2, \ldots, \phi_N\}\)，可以将非线性系统在**提升空间**（Lifted Space）中表示为线性系统：

$$z_{k+1} = K z_k, \quad z_k = \phi(x_k)$$

其中 \(K\) 是有限维的线性矩阵，可通过最小二乘拟合。

### 深度Koopman网络

深度Koopman（Deep Koopman）使用神经网络学习非线性编码器 \(\phi_\theta\)，将状态提升到更易于线性化的高维特征空间：

```python
class DeepKoopman(nn.Module):
    """
    深度Koopman网络
    encoder: 非线性提升 x -> z（升维）
    K:       线性动力学矩阵（可训练）
    decoder: 投影回原始状态空间 z -> x
    """
    def __init__(self, state_dim=4, lift_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ELU(),
            nn.Linear(64, lift_dim)
        )
        # 线性Koopman矩阵 K（核心线性假设）
        self.K = nn.Linear(lift_dim, lift_dim, bias=False)

        self.decoder = nn.Sequential(
            nn.Linear(lift_dim, 64), nn.ELU(),
            nn.Linear(64, state_dim)
        )

    def forward(self, x):
        z      = self.encoder(x)       # 提升
        z_next = self.K(z)             # 线性演化
        x_next = self.decoder(z_next)  # 投影回原空间
        return x_next, z, z_next


def koopman_loss(model, x, x_next):
    """Koopman损失 = 预测损失 + 线性一致性损失"""
    x_pred, z, z_next = model(x)
    _, z_true, _      = model(x_next)

    loss_pred  = nn.functional.mse_loss(x_pred, x_next)  # 一步预测
    loss_linear = nn.functional.mse_loss(z_next, z_true)  # 线性性约束
    return loss_pred + 0.5 * loss_linear
```

### 与LQR结合

Koopman线性化后，可以在提升空间中直接应用线性二次型调节器（Linear Quadratic Regulator, LQR）。LQR在提升空间 \(z\) 中设计最优线性反馈增益 \(K_{LQR}\)，再将控制律映射回原始状态空间，从而实现对非线性系统的近最优控制。


## 参考资料

1. M. Raissi, P. Perdikaris, and G. E. Karniadakis, "Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear Partial Differential Equations," *Journal of Computational Physics*, vol. 378, pp. 686-707, 2019.
2. R. T. Q. Chen, Y. Rubanova, J. Bettencourt, and D. Duvenaud, "Neural Ordinary Differential Equations," *NeurIPS*, 2018.
3. B. O. Koopman, "Hamiltonian Systems and Transformation in Hilbert Space," *PNAS*, vol. 17, no. 5, pp. 315-318, 1931.
4. B. Lusch, J. N. Kutz, and S. L. Brunton, "Deep Learning for Universal Linear Embeddings of Nonlinear Dynamics," *Nature Communications*, vol. 9, 4950, 2018.
5. S. L. Brunton and J. N. Kutz, *Data-Driven Science and Engineering: Machine Learning, Dynamical Systems, and Control*, Cambridge University Press, 2019.
