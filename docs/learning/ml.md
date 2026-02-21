# 机器学习

!!! note "引言"
    机器学习 (Machine Learning) 是人工智能的核心分支，研究如何让计算机从数据中自动学习规律并做出预测或决策。本页面介绍机器学习的基本概念、主要算法类别以及在机器人领域中的应用。

机器学习 (Machine Learning) 研究的主题是如何让计算机具备与人类同等的思考和分析能力。机器学习主要基于认知学、计算机科学，统计概率学以及信息决策学。典型的机器学习应用包括照片分类、垃圾邮件识别、自然语言处理等。最近很火热的围棋人工智能AlphaGo就是采用了深度神经网络对大量棋局进行学习，从而具备了顶尖围棋选手的水平。

机器学习的应用领域有：

- 经济学模型建立
- 图像处理和机器视觉
- 生物DNA解码
- 能源负载、使用、价格预测
- 汽车、航空和制造
- 自然语言处理
- ……


## 学习范式

机器学习从其采用的学习方式来说有以下三大类：

- **监督学习 (Supervised Learning)**：用于训练的数据包含已知结果（回归与分类问题）。
- **无监督学习 (Unsupervised Learning)**：用于训练的数据不包含已知结果（聚类问题）。
- **强化学习 (Reinforcement Learning)**：用于训练的数据不包含已知结果，但是可以用奖励函数 (Reward Function) 对其进行评价。

此外，还有一些介于上述范式之间的学习方式：

- **半监督学习 (Semi-supervised Learning)**：仅部分数据带有标签，结合有标签和无标签数据共同训练
- **自监督学习 (Self-supervised Learning)**：从数据自身结构中生成监督信号，无需人工标注
- **迁移学习 (Transfer Learning)**：将在一个任务上学到的知识迁移到新任务，减少对新数据的依赖


## 监督学习 (Supervised Learning)

监督学习从带有标签的训练数据中学习映射函数 \(f: X \rightarrow Y\)，其中 \(X\) 是输入特征，\(Y\) 是输出标签。根据输出类型的不同，监督学习分为回归 (Regression) 和分类 (Classification) 两大类。

### 线性回归 (Linear Regression)

线性回归是最基础的回归算法，假设输入与输出之间存在线性关系：

$$
y = w^T x + b
$$

通过最小化均方误差 (Mean Squared Error, MSE) 来拟合参数：

$$
\min_{w, b} \frac{1}{N} \sum_{i=1}^{N} (y_i - w^T x_i - b)^2
$$

线性回归虽然简单，但在许多实际问题中仍然有效，且具有良好的可解释性。

### 逻辑回归 (Logistic Regression)

尽管名字中包含"回归"，逻辑回归实际上是一种分类算法。它使用 Sigmoid 函数将线性组合映射到 \([0, 1]\) 区间，表示属于某一类的概率：

$$
P(y=1|x) = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}
$$

### 支持向量机 (Support Vector Machine, SVM)

SVM 的核心思想是在特征空间中找到一个最大间隔超平面 (Maximum Margin Hyperplane) 来分隔不同类别的数据。对于非线性问题，SVM 通过核函数 (Kernel Function) 将数据映射到高维空间，使其线性可分。

常用的核函数有：

- 线性核 (Linear Kernel)：\(K(x_i, x_j) = x_i^T x_j\)
- 径向基核 (RBF Kernel)：\(K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2)\)
- 多项式核 (Polynomial Kernel)：\(K(x_i, x_j) = (x_i^T x_j + c)^d\)

### 决策树 (Decision Tree)

决策树通过递归地将数据按特征值进行分裂，构建一棵树形结构。每个内部节点表示一个特征上的判断条件，叶节点表示预测结果。分裂准则通常基于信息增益 (Information Gain) 或基尼不纯度 (Gini Impurity)。

### 随机森林 (Random Forest)

随机森林是一种集成学习 (Ensemble Learning) 方法，通过训练多棵决策树并取其投票结果（分类）或平均值（回归）来提高预测精度和鲁棒性。它引入了两个随机化机制：

- **样本随机化**：每棵树使用自助采样 (Bootstrap Sampling) 得到的子集训练
- **特征随机化**：每次分裂时只考虑随机选取的特征子集

### 神经网络 (Neural Network)

人工神经网络 (Artificial Neural Network, ANN) 由多层互联的神经元组成。每个神经元执行加权求和并通过激活函数 (Activation Function) 进行非线性变换：

$$
a = \sigma(W x + b)
$$

常用的激活函数有 ReLU (\(\max(0, x)\))、Sigmoid、Tanh 等。通过反向传播算法 (Backpropagation) 和梯度下降 (Gradient Descent) 更新网络参数。


## 无监督学习 (Unsupervised Learning)

无监督学习从不带标签的数据中发现隐藏的结构和模式。

### K均值聚类 (K-Means Clustering)

K-Means 将 \(N\) 个数据点划分为 \(K\) 个簇，使得每个数据点属于距其最近的簇中心所代表的簇。算法通过交替执行以下两步迭代收敛：

1. **分配步骤**：将每个数据点分配到最近的簇中心
2. **更新步骤**：重新计算每个簇的中心为其成员的均值

目标函数为最小化簇内平方和 (Within-Cluster Sum of Squares)：

$$
\min \sum_{k=1}^{K} \sum_{x_i \in C_k} \|x_i - \mu_k\|^2
$$

### 主成分分析 (Principal Component Analysis, PCA)

PCA 是一种降维 (Dimensionality Reduction) 方法，通过找到数据方差最大的方向（主成分），将高维数据投影到低维空间，同时尽可能保留原始数据的信息。PCA 在传感器数据预处理和特征压缩中广泛应用。

### 自编码器 (Autoencoder)

自编码器是一种无监督的神经网络，通过编码器 (Encoder) 将输入压缩为低维表示（潜在空间），再通过解码器 (Decoder) 重建原始输入。其变种包括变分自编码器 (Variational Autoencoder, VAE)，可用于数据生成和异常检测。


## 深度学习基础 (Deep Learning)

深度学习 (Deep Learning) 是机器学习的一个子领域，使用多层神经网络从原始数据中自动学习多层次的特征表示。

### 卷积神经网络 (Convolutional Neural Network, CNN)

CNN 专门用于处理具有网格结构的数据（如图像）。其核心操作是卷积 (Convolution)，通过可学习的卷积核提取局部特征：

- **卷积层 (Convolutional Layer)**：提取局部特征
- **池化层 (Pooling Layer)**：降低空间维度，增强平移不变性
- **全连接层 (Fully Connected Layer)**：进行最终的分类或回归

经典网络结构有 LeNet、AlexNet、VGG、ResNet、EfficientNet 等。CNN 在机器人视觉感知中应用广泛。

### 循环神经网络 (Recurrent Neural Network, RNN)

RNN 适用于处理序列数据（如时间序列、文本）。其隐藏状态能够保存历史信息：

$$
h_t = \sigma(W_h h_{t-1} + W_x x_t + b)
$$

标准 RNN 存在梯度消失/爆炸问题，改进的变种有长短期记忆网络 (Long Short-Term Memory, LSTM) 和门控循环单元 (Gated Recurrent Unit, GRU)。在机器人中，RNN 可用于轨迹预测和时间序列传感器数据处理。

### Transformer

Transformer 基于自注意力机制 (Self-Attention Mechanism)，能够并行处理序列中所有位置的信息，解决了 RNN 在长序列上的计算效率问题。其核心是缩放点积注意力 (Scaled Dot-Product Attention)：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Transformer 最初用于自然语言处理（如 BERT、GPT 系列），现已扩展到视觉领域（Vision Transformer, ViT）和机器人决策领域。


## scikit-learn 实践示例

scikit-learn 是 Python 生态中最成熟的传统机器学习库，提供统一的应用程序接口 (Application Programming Interface, API)，适合快速建立基线模型和进行特征工程实验。以下以机器人抓取成功与失败的预测为例，演示从数据准备到模型评估的完整流程。

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
import numpy as np

# 以机器人抓取成功/失败预测为例
# 特征: [gripper_force, object_width, approach_angle, surface_roughness]
X = np.random.randn(1000, 4)  # 示例数据
y = (X[:, 0] * 0.5 + X[:, 1] * 0.3 > 0).astype(int)  # 示例标签

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 特征标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 随机森林分类器
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

# 评估
print(classification_report(y_test, rf_clf.predict(X_test)))

# 特征重要性
importances = rf_clf.feature_importances_
feature_names = ['gripper_force', 'object_width', 'approach_angle', 'surface_roughness']
for name, imp in zip(feature_names, importances):
    print(f"{name}: {imp:.3f}")
```

特征重要性分析对于机器人系统调试很有价值：若 `approach_angle` 的重要性远高于其他特征，则说明抓取角度是影响成功率的关键因素，值得在控制层面重点优化。


## PyTorch 神经网络实践

PyTorch 以动态计算图著称，便于调试和科研迭代。以下示例展示如何构建一个用于机器人状态分类的多层感知机 (Multilayer Perceptron, MLP)，输入为关节角度、角速度等 12 维状态向量，输出为 5 种运动模式的分类概率。

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 定义多层感知机 (MLP)
class RobotStateClassifier(nn.Module):
    def __init__(self, input_dim=12, hidden_dims=[64, 32], num_classes=5):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# 准备示例数据
X_tensor = torch.randn(800, 12)
y_tensor = torch.randint(0, 5, (800,))
dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 训练循环
model = RobotStateClassifier()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(100):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
```

网络中使用的批归一化 (Batch Normalization) 和随机失活 (Dropout) 是缓解过拟合 (Overfitting) 的常用手段，在机器人数据集通常规模有限的情况下尤为重要。


## 迁移学习 (Transfer Learning) 实践

迁移学习的核心思想是：在大规模数据集（如 ImageNet）上预训练的模型已经学会了丰富的底层特征（边缘、纹理、形状），将其迁移到机器人视觉任务可以大幅降低对标注数据的需求。常见策略分为两种：

- **特征提取 (Feature Extraction)**：冻结骨干网络所有参数，仅训练新增分类头
- **微调 (Fine-tuning)**：解冻骨干网络的高层参数，以较小的学习率联合训练

```python
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

# 使用预训练 ResNet50
backbone = models.resnet50(pretrained=True)

# 冻结特征提取层
for param in backbone.parameters():
    param.requires_grad = False

# 替换分类头 (仅微调分类层)
num_classes = 10  # 机器人抓取物体类别数
backbone.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(backbone.fc.in_features, 256),
    nn.ReLU(),
    nn.Linear(256, num_classes)
)

# 只优化分类头参数
optimizer = optim.Adam(backbone.fc.parameters(), lr=1e-3)
```

在数据极为稀缺时（例如某种特殊工件仅有数十张图片），还可以采用少样本学习 (Few-Shot Learning) 方法，如原型网络 (Prototypical Networks) 或模型无关元学习 (Model-Agnostic Meta-Learning, MAML)。


## 生成模型 (Generative Models)

生成模型学习数据的联合概率分布 \(p(x)\) 或条件分布 \(p(x|c)\)，可用于数据增强、异常检测和机器人运动合成。

### 变分自编码器 (Variational Autoencoder, VAE)

VAE 在普通自编码器的基础上，将潜在空间建模为高斯分布，引入重参数化技巧 (Reparameterization Trick) 使得梯度可以反向传播。损失函数由重建损失和KL散度两部分组成：

$$
\mathcal{L}_{\text{VAE}} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \beta \cdot D_{\text{KL}}(q_\phi(z|x) \| p(z))
$$

在机器人领域，VAE 可以学习机器人状态（关节角度、末端执行器位姿）的紧凑潜在表示，在低维潜在空间中进行运动规划，再解码回关节空间。

### 生成对抗网络 (Generative Adversarial Network, GAN)

GAN 由生成器 (Generator) 和判别器 (Discriminator) 组成，通过对抗训练生成以假乱真的数据。在机器人仿真到真实 (Sim-to-Real) 迁移中，GAN 可以将仿真图像转换为真实风格的图像（域随机化的逆问题），减少域偏移 (Domain Gap)。CycleGAN 是无需配对数据即可实现图像风格迁移的经典方法。

### 扩散模型 (Diffusion Model)

扩散模型通过逐步去噪的过程生成高质量数据，已被用于机器人运动合成。扩散策略 (Diffusion Policy) 将机器人动作序列的生成建模为条件扩散过程，在灵巧操作任务上表现出色，能够捕捉多模态动作分布（即同一任务可以有多种合理的执行方式）。


## 主动学习 (Active Learning)

在机器人操控等任务中，采集带标签数据的代价极高：每一条演示数据都需要专家人工操控机器人完成。主动学习通过智能选择最具信息量的样本请求标注，从而以更少的标注预算达到相同的模型性能。

常用的查询策略 (Query Strategy) 有：

- **不确定性采样 (Uncertainty Sampling)**：选择模型预测概率最接近决策边界（熵最大）的样本。对于分类问题，最不确定的样本满足：

$$
x^* = \arg\max_{x} H(y | x) = \arg\max_{x} \left( -\sum_{c} p(y=c|x) \log p(y=c|x) \right)
$$

- **委员会查询 (Query by Committee)**：训练多个模型组成委员会，选择委员会意见分歧最大的样本
- **期望模型变化 (Expected Model Change)**：选择加入标签后预期使模型参数发生最大变化的样本

主动学习在机器人遥操作数据采集、医疗图像标注等高成本场景中具有重要的实际价值。


## 联邦学习 (Federated Learning)

联邦学习允许多个参与方（例如多台部署在不同场景的配送机器人）在不共享原始数据的情况下协同训练一个全局模型。其基本流程为：

1. 服务器向各客户端分发当前全局模型参数
2. 各客户端在本地数据上独立训练，得到本地梯度或更新后的参数
3. 服务器聚合各客户端上传的参数（通常采用 FedAvg 算法取加权平均）：

$$
w_{\text{global}} \leftarrow \sum_{k=1}^{K} \frac{n_k}{N} w_k
$$

其中 \(n_k\) 为第 \(k\) 个客户端的数据量，\(N\) 为总数据量。

在机器人行业中，联邦学习的典型应用场景包括：

- **配送机器人车队**：各机器人从本地路况经验中学习，无需将用户行为数据上传至中心服务器，保护用户隐私
- **工业质检**：不同工厂的生产设备共同训练缺陷检测模型，同时保护各厂的商业机密数据
- **医疗机器人**：多家医院的手术机器人联合学习组织识别模型，无需共享患者影像数据


## 不确定性估计 (Uncertainty Estimation)

在机器人系统中，模型的预测往往需要附带置信度信息，以支持安全的决策（例如：当不确定性过高时，机器人应暂停并请求人工确认）。不确定性通常分为两类：

- **认知不确定性 (Epistemic Uncertainty)**：由训练数据不足导致的模型本身的不确定性，可通过收集更多数据来降低。估计方法包括深度集成 (Deep Ensemble) 和蒙特卡洛随机失活 (Monte Carlo Dropout, MC Dropout)。

- **偶然不确定性 (Aleatoric Uncertainty)**：数据本身固有的随机噪声，无法通过增加数据消除。可让模型直接预测均值和方差来建模：

$$
\mathcal{L}_{\text{aleatoric}} = \frac{1}{N} \sum_{i=1}^{N} \left( \frac{(y_i - \hat{\mu}_i)^2}{2\hat{\sigma}_i^2} + \frac{1}{2}\log \hat{\sigma}_i^2 \right)
$$

MC Dropout 在推理阶段保持 Dropout 开启，对同一输入进行多次前向传播，将输出的方差作为不确定性估计：

```python
import torch

def mc_dropout_predict(model, x, n_samples=50):
    model.train()  # 保持 Dropout 激活
    preds = torch.stack([model(x) for _ in range(n_samples)])
    mean = preds.mean(dim=0)
    variance = preds.var(dim=0)
    return mean, variance
```

**模型校准 (Calibration)** 是另一个重要概念，要求模型预测的置信度能够反映实际的准确率——预测置信度为 80% 的样本中，实际应有约 80% 的预测是正确的。可以用可靠性图 (Reliability Diagram) 和期望校准误差 (Expected Calibration Error, ECE) 来量化校准程度：

$$
\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|
$$


## 图神经网络 (Graph Neural Networks, GNN)

许多机器人相关问题天然具有图结构，适合用图神经网络来建模：

- **多机器人协同**：将机器人群体建模为图，节点为单个机器人，边表示通信或协作关系。GNN 可以聚合邻居信息，为每个机器人生成考虑全局态势的决策。
- **场景图理解 (Scene Graph Understanding)**：将环境中的物体及其关系建模为图，支持机器人对复杂场景的语义理解（"杯子在桌子上"、"机械臂在杯子旁边"）。
- **分子属性预测**：将分子结构建模为原子图，预测材料力学、摩擦等性质，辅助机器人夹爪材料选型。

图神经网络的核心操作是消息传递 (Message Passing)，每个节点聚合邻居节点的特征来更新自身表示：

$$
h_v^{(l+1)} = \text{UPDATE}\left( h_v^{(l)},\ \text{AGGREGATE}\left( \{ h_u^{(l)} : u \in \mathcal{N}(v) \} \right) \right)
$$

常见的 GNN 变体有图卷积网络 (Graph Convolutional Network, GCN)、图注意力网络 (Graph Attention Network, GAT) 和图同构网络 (Graph Isomorphism Network, GIN)。


## 模型评估与选择

### 交叉验证 (Cross-Validation)

在数据量有限的场景（机器人领域中极为常见）下，K 折交叉验证 (K-Fold Cross-Validation) 比单次划分更能可靠地估计模型的泛化性能：

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, random_state=42)
scores = cross_val_score(clf, X, y, cv=5, scoring='f1_weighted')
print(f"F1: {scores.mean():.3f} ± {scores.std():.3f}")
```

### 超参数调优 (Hyperparameter Tuning)

超参数（如学习率、树的数量、网络层数）不能从数据中直接学习，需要借助搜索策略来选择。

**网格搜索 (Grid Search)**（sklearn）穷举所有超参数组合：

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5]
}
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, cv=5, scoring='f1_weighted', n_jobs=-1
)
grid_search.fit(X_train, y_train)
print("最优参数:", grid_search.best_params_)
```

**Optuna** 采用贝叶斯优化 (Bayesian Optimization) 策略，比网格搜索更高效：

```python
import optuna

def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 15)
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    return cross_val_score(clf, X_train, y_train, cv=3, scoring='f1_weighted').mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
print("最优参数:", study.best_params)
```

### 过拟合与欠拟合诊断

通过绘制训练误差与验证误差随模型复杂度（或训练轮数）的变化曲线，可以诊断过拟合 (Overfitting) 和欠拟合 (Underfitting)：

- **欠拟合**：训练误差与验证误差均偏高，说明模型容量不足，需要增加复杂度或特征
- **过拟合**：训练误差低但验证误差高，两者差距大，需要正则化、数据增强或减小模型容量

常用的正则化手段包括 L2 权重衰减 (Weight Decay)、Dropout、数据增强 (Data Augmentation) 和早停 (Early Stopping)。


## 机器学习在机器人中的应用

机器学习为机器人的多个子系统提供了强大的工具：

| 应用领域 | 典型方法 | 具体应用 | 推荐工具 |
|---------|---------|---------|---------|
| 感知 (Perception) | CNN、PointNet | 物体检测、语义分割、点云分类 | PyTorch + torchvision |
| 定位 (Localization) | CNN、RNN | 视觉里程计、位姿估计 | PyTorch |
| 规划 (Planning) | 强化学习、模仿学习 | 运动规划、任务规划 | Stable-Baselines3 |
| 控制 (Control) | 强化学习、神经网络 | 自适应控制、灵巧操作 | PyTorch / JAX |
| 人机交互 (HRI) | RNN、Transformer | 语音识别、手势识别、意图理解 | HuggingFace Transformers |
| 预测 (Prediction) | LSTM、Transformer | 行人轨迹预测、故障预测 | PyTorch |
| 质量检测 | CNN、SVM | 焊缝检测、表面缺陷识别 | scikit-learn / PyTorch |
| 材料选型 | GNN、随机森林 | 夹爪材料属性预测 | PyTorch Geometric |


## 常用框架

| 框架 | 语言 | 特点 |
|------|------|------|
| scikit-learn | Python | 传统机器学习算法集合，适合快速原型开发 |
| PyTorch | Python/C++ | 动态计算图，学术研究首选 |
| TensorFlow / Keras | Python/C++ | 静态图优化，工业部署友好 |
| XGBoost / LightGBM | Python/C++ | 高性能梯度提升 (Gradient Boosting) 库 |
| JAX | Python | Google 推出的高性能数值计算库，支持自动微分和即时编译 (Just-In-Time Compilation) |
| PyTorch Geometric | Python | 图神经网络专用库，提供丰富的 GNN 层和数据集 |
| HuggingFace Transformers | Python | 预训练语言模型和视觉模型的标准库 |
| Optuna | Python | 超参数自动优化框架，支持贝叶斯优化和多目标优化 |


## 参考资料

1. 戴晓天，[机器学习 | 机器学习101](https://www.yfworld.com/?p=3378)，云飞机器人实验室
2. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. [在线版本](https://www.deeplearningbook.org/)
4. Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*. MIT Press. [在线版本](https://probml.github.io/pml-book/book1.html)
5. Vaswani, A., et al. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30.
6. Settles, B. (2009). Active learning literature survey. *Computer Sciences Technical Report*, University of Wisconsin-Madison.
7. McMahan, H. B., et al. (2017). Communication-efficient learning of deep networks from decentralized data. *AISTATS*.
8. Kendall, A., & Gal, Y. (2017). What uncertainties do we need in Bayesian deep learning for computer vision? *Advances in Neural Information Processing Systems*, 30.
9. Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *ICLR*.
10. Ho, J., et al. (2020). Denoising diffusion probabilistic models. *Advances in Neural Information Processing Systems*, 33.
11. Chi, C., et al. (2023). Diffusion policy: Visuomotor policy learning via action diffusion. *Robotics: Science and Systems*.
12. scikit-learn 开发团队，[scikit-learn 官方文档](https://scikit-learn.org/stable/)
13. PyTorch 开发团队，[PyTorch 官方文档](https://pytorch.org/docs/)
