# 策略学习与安全控制

!!! note "引言"
    当控制目标难以写成解析的代价函数，却容易由人类或既有控制器示范时，策略学习（Policy Learning）提供了一条直接从示范数据中获得控制器的路径。本页覆盖从行为克隆（Behavioral Cloning）到扩散策略（Diffusion Policy）的模仿学习方法、以控制障碍函数（Control Barrier Function, CBF）为代表的安全学习控制，以及端到端视觉运动策略，最后给出训练神经网络控制器的工程实践要点。


## 模仿学习控制（Imitation Learning）

模仿学习（Imitation Learning）从专家演示数据中学习控制策略，无需手动设计奖励函数。

### 行为克隆（Behavioral Cloning, BC）

行为克隆（Behavioral Cloning, BC）将模仿学习转化为监督学习问题：直接将专家的（状态，动作）对作为训练数据，训练策略网络 \(\pi_\theta(a|s)\)。

$$L_{BC} = \mathbb{E}_{(s,a) \sim \mathcal{D}_{expert}}\left[\|\pi_\theta(s) - a\|^2\right]$$

BC的主要缺陷是**协变量偏移（Covariate Shift）**：训练数据与测试时智能体实际访问的状态分布不同——智能体在部署时遇到的轻微偏差会被策略放大，最终导致越来越大的错误积累。

### DAgger算法

数据集聚合（Dataset Aggregation, DAgger）算法（Ross et al., 2011）是BC的改进版，通过迭代地让策略与专家交互来解决协变量偏移问题：

```
DAgger 算法:
1. 用专家数据训练初始策略 π_1（等同于BC）
2. 对于每次迭代 i = 1, 2, ..., N:
   a. 用当前策略 π_i 在环境中运行，收集访问的状态 s_1, s_2, ...
   b. 请专家对这些状态标注最优动作: a* = π_expert(s)
   c. 将新的 (s, a*) 对加入数据集: D = D ∪ {(s, a*)}
   d. 在聚合数据集 D 上重新训练策略 π_{i+1}
```

DAgger的收敛性有理论保证：模型误差不随时间累积。主要局限是需要专家在线参与（或交互式仿真器），在真实机器人演示中成本较高。

### PyTorch行为克隆训练示例

```python
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class BCPolicy(nn.Module):
    """行为克隆策略网络（MLP）"""
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, act_dim)
        )

    def forward(self, obs):
        return self.net(obs)


def train_bc(obs_data, act_data, obs_dim, act_dim,
             epochs=100, batch_size=256, lr=3e-4):
    """
    obs_data: [N, obs_dim] 专家观测
    act_data: [N, act_dim] 专家动作
    """
    dataset    = TensorDataset(obs_data, act_data)
    loader     = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    policy     = BCPolicy(obs_dim, act_dim)
    optimizer  = torch.optim.AdamW(policy.parameters(), lr=lr,
                                   weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
                     optimizer, T_max=epochs)
    loss_fn    = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for obs_batch, act_batch in loader:
            pred = policy(obs_batch)
            loss = loss_fn(pred, act_batch)

            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪：防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            avg = total_loss / len(loader)
            print(f"Epoch {epoch+1}/{epochs}  Loss: {avg:.4f}  "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")

    return policy
```


## 扩散策略（Diffusion Policy）

扩散策略（Diffusion Policy）由Chi等人于2023年提出，将去噪扩散概率模型（Denoising Diffusion Probabilistic Model, DDPM）用于机器人操控策略学习，在处理多模态动作分布方面展现出超越传统行为克隆的显著优势。

### 核心思想

传统行为克隆（BC）使用均方误差损失，本质上拟合动作的条件均值，当专家示例存在多种等价解法（多模态）时，均值会落在不可行区域（平均多个模态）。

扩散策略将策略学习建模为**去噪过程**：

1. **前向过程（加噪）**：对真实专家动作 \(a_0\) 逐步加高斯噪声，得到纯噪声 \(a_T\)。
2. **反向过程（去噪）**：训练网络 \(\epsilon_\theta(a_t, t, o)\) 预测每一步的噪声，以观测 \(o\) 为条件，从纯噪声迭代恢复出动作 \(a_0\)。

策略在推理时从高斯噪声出发，经过 \(K\) 步去噪生成动作序列。由于每次采样路径不同，可以自然表达多模态分布（每种抓取姿态对应一个模态）。

### 相对于BC的优势

| 特性 | 行为克隆 (BC) | 扩散策略 |
|------|-------------|---------|
| 多模态动作分布 | 无法表达（均值塌缩） | 天然支持 |
| 动作精度 | 中等 | 高（迭代精化） |
| 推理速度 | 极快（单次前向） | 较慢（多步去噪） |
| 训练复杂度 | 简单（MSE损失） | 中等（DDPM损失） |
| 对超参数敏感性 | 中等 | 较低 |

扩散策略在Columbia的机器人操控实验中，相比BC和LSTM-GMM（高斯混合模型）将任务成功率提升了显著幅度，尤其在需要精确把握物体或处理多解情况时优势明显。


## 安全学习控制（Safe Learning）

在实际机器人部署中，确保学习控制器满足安全约束（如避免碰撞、关节限位）至关重要。控制障碍函数（Control Barrier Function, CBF）提供了一种严格的安全保证框架，可与神经网络控制器结合使用。

### 控制障碍函数（CBF）

给定安全集合 \(\mathcal{C} = \{x : h(x) \geq 0\}\)，CBF \(h(x)\) 要求系统轨迹在 \(\mathcal{C}\) 内保持**前向不变性（Forward Invariance）**：

$$\dot{h}(x, u) = \frac{\partial h}{\partial x} f(x, u) \geq -\alpha(h(x))$$

其中 \(\alpha(\cdot)\) 是一个K类函数（class-K function，严格递增且 \(\alpha(0)=0\)），常取 \(\alpha(h) = \gamma h\)（线性）。

**直觉理解**：当系统接近安全边界（\(h(x) \to 0\)）时，约束要求 \(\dot{h}\) 越来越小（允许趋向边界的速率递减），从而阻止系统穿越边界。

### CLF-CBF 二次规划

将控制李雅普诺夫函数（Control Lyapunov Function, CLF）用于稳定性、CBF用于安全性，两者通过**二次规划（Quadratic Program, QP）**统一求解：

$$u^* = \arg\min_{u} \|u - u_{ref}\|^2$$

$$\text{s.t.} \quad \dot{V}(x, u) \leq -\lambda V(x) \quad \text{（CLF稳定性约束）}$$

$$\qquad\quad \dot{h}(x, u) \geq -\gamma h(x) \quad \text{（CBF安全约束）}$$

$$\qquad\quad u_{min} \leq u \leq u_{max} \quad \text{（输入约束）}$$

其中 \(u_{ref}\) 是神经网络策略的输出，QP在满足安全约束的前提下对其进行最小修正。

```python
import torch
import cvxpy as cp
import numpy as np

def cbf_qp_filter(u_ref, x, h_func, dh_dx, f_func, gamma=1.0):
    """
    CBF安全过滤器：对神经网络控制输出 u_ref 进行安全修正
    h_func:  CBF函数 h(x)，安全集合 h(x) >= 0
    dh_dx:   h 对 x 的梯度（行向量）
    f_func:  系统动力学 f(x, u)，需满足 dh/dt >= -gamma*h
    """
    u_dim   = len(u_ref)
    u       = cp.Variable(u_dim)

    # CBF约束：Lf_h + Lg_h * u >= -gamma * h(x)
    h_val   = float(h_func(x))
    Lf_h    = float(dh_dx @ f_func(x, np.zeros(u_dim)))
    Lg_h    = dh_dx @ np.eye(u_dim)  # 简化：直接输入影响

    constraints = [Lg_h @ u >= -gamma * h_val - Lf_h]

    # 目标：最小化对参考控制的修正
    objective = cp.Minimize(cp.sum_squares(u - u_ref))
    prob      = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP, warm_start=True)

    if prob.status == cp.OPTIMAL:
        return u.value
    else:
        return u_ref  # 求解失败时回退到原始控制
```


## 端到端学习控制

端到端（End-to-End）学习控制指直接从原始感知输入（如摄像头图像、雷达点云）输出控制命令，中间不依赖手工设计的感知、规划等模块。

### 视觉运动策略（Visuomotor Policy）

Levine等人2016年提出的端到端视觉运动策略（Visuomotor Policy）将卷积神经网络（CNN）与运动规划网络连接，直接从RGB图像预测机器人关节力矩：

$$u = \pi_\theta(\text{image}, \text{robot\_state})$$

现代端到端控制通常采用以下架构：

1. **视觉编码器**：预训练的视觉Transformer（ViT）或ResNet提取图像特征。
2. **状态融合**：将图像特征与机器人本体状态（关节角度、速度）拼接。
3. **策略头**：MLP或Transformer输出动作序列。

### 仿真到真实迁移（Sim-to-Real Transfer）

由于在真实机器人上收集大规模训练数据成本极高，端到端控制器通常在仿真中训练，然后迁移到真实机器人。

**域随机化（Domain Randomization）** 是最常用的仿真到真实迁移技术：在训练时随机扰动仿真参数（物体颜色、纹理、光照、摩擦系数、传感器噪声等），迫使策略网络学习对这些变化鲁棒的特征：

| 随机化类别 | 随机化参数示例 |
|-----------|-------------|
| 视觉外观 | 纹理、颜色、光照方向、相机位置噪声 |
| 物理参数 | 质量、摩擦系数、关节阻尼、弹性系数 |
| 传感器噪声 | 图像噪声、关节编码器噪声、延迟 |
| 任务参数 | 物体初始位置、目标位置的随机扰动 |

### 局限性

端到端学习控制目前的主要挑战：

- **可解释性差**：网络内部决策过程不透明，难以诊断故障原因。
- **分布外泛化（Out-of-Distribution Generalization）**：对训练分布外的场景（新纹理、新光照）可能完全失败。
- **数据效率**：收敛通常需要数百万次仿真交互，样本效率远低于基于模型的方法。
- **安全性保证缺失**：端到端网络很难提供碰撞避免等硬约束的形式化保证。


## 训练实践

### 数据收集与课程学习

高质量的训练数据是神经网络控制器性能的基础。数据收集策略：

- **课程学习（Curriculum Learning）**：从简单任务开始训练，逐步增加难度。例如，先训练机器人在平地行走，再引入斜坡、台阶等障碍。
- **多样化场景覆盖**：收集覆盖尽可能多的状态空间的数据，避免策略在边界状态上失效。
- **数据增强**：对观测数据进行随机裁剪、颜色抖动、噪声添加等增强，提升泛化能力。

### 完整MLP策略训练流程

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class MLPPolicy(nn.Module):
    """带LayerNorm的MLP策略网络，适合机器人控制任务"""
    def __init__(self, obs_dim, act_dim, hidden_dim=256, n_layers=3):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for _ in range(n_layers):
            layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ]
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, act_dim))
        layers.append(nn.Tanh())  # 动作归一化到 [-1, 1]
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        return self.net(obs)


def train_policy(obs_data, act_data, obs_dim, act_dim,
                 epochs=200, batch_size=512,
                 lr=3e-4, weight_decay=1e-4):
    """
    完整训练流程：AdamW优化器 + 余弦退火学习率 + 梯度裁剪
    """
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy  = MLPPolicy(obs_dim, act_dim).to(device)

    dataset = TensorDataset(obs_data.to(device), act_data.to(device))
    loader  = DataLoader(dataset, batch_size=batch_size,
                         shuffle=True, pin_memory=False)

    # AdamW：带权重衰减的Adam，比L2正则化更稳定
    optimizer = torch.optim.AdamW(policy.parameters(),
                                  lr=lr, weight_decay=weight_decay)

    # 余弦退火：学习率从 lr 平滑衰减到 0，避免震荡
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs, eta_min=1e-6)

    loss_fn = nn.MSELoss()
    best_loss = float('inf')

    for epoch in range(epochs):
        policy.train()
        epoch_loss = 0.0

        for obs_batch, act_batch in loader:
            pred = policy(obs_batch)
            loss = loss_fn(pred, act_batch)

            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪：防止梯度爆炸，max_norm=1.0 是常用设置
            torch.nn.utils.clip_grad_norm_(policy.parameters(),
                                           max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(loader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(policy.state_dict(), "best_policy.pt")

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs}  "
                  f"Loss: {avg_loss:.5f}  "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")

    print(f"训练完成，最优损失: {best_loss:.5f}")
    return policy


def evaluate_policy(policy, env, n_episodes=20):
    """在仿真环境中评估策略，返回平均回报"""
    policy.eval()
    total_reward = 0.0

    with torch.no_grad():
        for _ in range(n_episodes):
            obs, done = env.reset(), False
            ep_reward = 0.0
            while not done:
                obs_t  = torch.FloatTensor(obs).unsqueeze(0)
                action = policy(obs_t).squeeze(0).numpy()
                obs, reward, done, _ = env.step(action)
                ep_reward += reward
            total_reward += ep_reward

    return total_reward / n_episodes
```

### 仿真到真实迁移技术汇总

| 技术 | 描述 | 适用场景 |
|------|------|----------|
| **域随机化** | 随机化仿真中的物理和视觉参数 | 视觉策略、操控任务 |
| **系统辨识** | 精确测量并匹配真实系统参数 | 精密控制任务 |
| **适应层** | 部署时用少量真实数据微调网络末层 | 快速适应新环境 |
| **域自适应** | 对齐仿真和真实数据的特征分布 | 视觉传感器差异 |
| **课程学习** | 渐进增加仿真难度 | 复杂运动技能 |
| **残差学习** | 在传统控制器上叠加神经网络残差 | 有先验模型的系统 |


## 参考资料

1. S. Ross, G. Gordon, and D. Bagnell, "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning," *AISTATS*, 2011.
2. C. Chi, S. Feng, Y. Du, et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion," *RSS*, 2023.
3. A. D. Ames, X. Xu, J. W. Grizzle, and P. Tabuada, "Control Barrier Function Based Quadratic Programs for Safety Critical Systems," *IEEE Transactions on Automatic Control*, vol. 62, no. 8, pp. 3861-3876, 2017.
4. S. Levine, C. Finn, T. Darrell, and P. Abbeel, "End-to-End Training of Deep Visuomotor Policies," *Journal of Machine Learning Research*, vol. 17, no. 39, pp. 1-40, 2016.
5. J. Hwangbo et al., "Learning Agile and Dynamic Motor Skills for Legged Robots," *Science Robotics*, vol. 4, no. 26, 2019.
6. T. Zhang, Z. McCarthy, O. Jow, et al., "Deep Imitation Learning for Complex Manipulation Tasks from Virtual Reality Teleoperation," *ICRA*, 2018.
