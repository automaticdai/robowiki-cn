# 深度学习

!!! note "引言"
    深度学习 (Deep Learning, DL) 是机器学习的一个子领域，通过多层神经网络从原始数据中自动学习层次化的特征表示，已成为现代机器人感知、决策与控制的核心驱动力。近年来，以大规模预训练为核心的基础模型 (Foundation Model) 正在将机器人的泛化能力推向新的高度，从简单的分类识别扩展到语言指令跟随、灵巧操作等复杂任务。本文在读者具备机器学习基础的前提下，重点讲解深度学习在机器人领域的关键方法与前沿进展。


## 神经网络基础回顾

### 多层感知机（MLP）

多层感知机 (Multilayer Perceptron, MLP) 是深度神经网络的基本构件，由输入层、若干隐藏层和输出层组成。通用近似定理 (Universal Approximation Theorem) 指出，一个具有足够多神经元的单隐藏层网络可以以任意精度逼近任意连续函数，这为深度学习的有效性提供了理论基础。

**前向传播 (Forward Propagation)**：输入信号逐层传递，第 \(l\) 层的激活值为：

$$
a^{(l)} = \sigma\!\left(W^{(l)} a^{(l-1)} + b^{(l)}\right)
$$

其中 \(\sigma\) 为激活函数，常用选择包括 ReLU (\(\max(0,x)\))、GELU 等。

**反向传播 (Backpropagation)**：通过链式法则计算损失函数对每层参数的梯度，再由优化器（如 Adam、SGD）沿负梯度方向更新权重：

$$
W^{(l)} \leftarrow W^{(l)} - \eta \frac{\partial \mathcal{L}}{\partial W^{(l)}}
$$

其中 \(\eta\) 为学习率 (Learning Rate)，\(\mathcal{L}\) 为损失函数。


### 卷积神经网络（CNN）

卷积神经网络 (Convolutional Neural Network, CNN) 专为网格结构数据（图像、点云投影）设计，通过两个核心归纳偏置实现高效学习：

- **局部感受野 (Local Receptive Field)**：每个卷积核只与输入的局部区域连接，捕获局部纹理和边缘特征。
- **权重共享 (Weight Sharing)**：同一卷积核在整幅特征图上滑动，大幅减少参数量，同时赋予网络平移不变性。

卷积层之后通常接**池化层 (Pooling Layer)**（最大池化或平均池化）进一步压缩空间维度，以及**批归一化 (Batch Normalization, BN)** 层加速训练收敛并抑制内部协变量偏移。

代表架构：

- **ResNet**：引入残差连接 (Residual Connection) \(y = \mathcal{F}(x) + x\)，解决深层网络梯度消失问题，最深可达 152 层。
- **EfficientNet**：通过复合缩放 (Compound Scaling) 同时调整网络深度、宽度和输入分辨率，在精度与计算量之间取得最优平衡。


### 循环神经网络（RNN/LSTM）

循环神经网络 (Recurrent Neural Network, RNN) 通过隐状态传递历史信息，适合处理时序数据。标准 RNN 的状态更新为：

$$
h_t = \tanh\!\left(W_h h_{t-1} + W_x x_t + b\right)
$$

在机器人领域，传感器时间序列（IMU、关节编码器）和运动轨迹均具有强时序依赖性，RNN 系列模型在此类任务上表现出色。

然而标准 RNN 存在**梯度消失/爆炸**问题，长短期记忆网络 (Long Short-Term Memory, LSTM) 通过引入输入门、遗忘门和输出门解决了这一问题，能够学习数百步以上的长程依赖。门控循环单元 (Gated Recurrent Unit, GRU) 是 LSTM 的简化变体，参数更少、训练更快。


### Transformer 与注意力机制

Transformer 基于**自注意力机制 (Self-Attention Mechanism)**，摒弃了 RNN 的序列计算限制，可以并行处理序列中任意两个位置之间的依赖关系。缩放点积注意力 (Scaled Dot-Product Attention) 的计算公式为：

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

其中查询矩阵 \(Q\)、键矩阵 \(K\)、值矩阵 \(V\) 均由输入线性投影得到，\(d_k\) 是键向量的维度，用于防止点积值过大导致梯度消失。

视觉 Transformer (Vision Transformer, ViT) 将图像切分为固定大小的图像块 (Patch)，展平后加入位置编码送入标准 Transformer 编码器，在大规模数据上的视觉识别性能已超越 CNN。在机器人领域，Transformer 被广泛用于处理多模态输入（图像 + 语言 + 关节状态）并生成动作序列。


## 端到端学习（End-to-End Learning）

### 定义与动机

端到端学习 (End-to-End Learning) 是指从原始感知输入（图像、点云、激光雷达扫描）直接映射到动作输出（转向角、速度指令、关节力矩）的学习范式，中间无需人工设计的特征提取或模块化流水线。网络自动发现对决策最有用的中间表示，从而避免了手工特征工程中的信息损失和设计偏差。

### NVIDIA PilotNet

PilotNet 是端到端自动驾驶的早期里程碑，由 NVIDIA 于 2016 年提出。其网络结构为纯 CNN：输入为车载摄像头的原始 RGB 图像（尺寸 $66 \times 200$ 像素），输出为车辆转向角的预测值，训练数据完全来自人类驾驶员的操作记录。

PilotNet 的成功说明了深度网络能够直接从像素中学习到有意义的道路感知表示，无需显式的车道线检测或语义分割模块。

### 优势与挑战

**优势**：

- 不依赖手工设计的特征，减少领域专家负担
- 整体联合优化，避免模块间误差累积
- 可以从大规模无标注数据中挖掘信息

**挑战**：

- **可解释性差**：难以理解网络的决策依据，增加安全验证难度
- **数据需求量大**：需要覆盖各种边界情况 (Corner Case) 的大量样本
- **分布迁移 (Distribution Shift)**：训练分布与部署场景不匹配时性能急剧下降
- **长尾问题**：罕见事件在数据中占比极低，模型难以充分学习


## 模仿学习（Imitation Learning）

模仿学习 (Imitation Learning, IL) 通过学习专家演示来获得技能，无需显式定义奖励函数，特别适合机器人操作任务。

### 行为克隆（Behavioral Cloning, BC）

行为克隆 (Behavioral Cloning, BC) 将模仿学习转化为标准的监督学习问题：将专家演示数据集 \(\mathcal{D} = \{(s_i, a_i)\}\) 中的状态-动作对视为输入-标签对，最小化策略输出与专家动作之间的误差：

$$
\mathcal{L} = \mathbb{E}_{(s,a) \sim \mathcal{D}}\!\left[\|a - \pi_\theta(s)\|^2\right]
$$

BC 的核心缺陷是**协变量偏移 (Covariate Shift)**：训练时策略在专家状态分布上优化，但测试时策略的微小误差会导致访问到训练集中从未出现的状态，进而引发误差的滚雪球式累积。

### DAgger（Dataset Aggregation）

DAgger (Dataset Aggregation) 通过迭代式数据收集缓解协变量偏移问题：

1. 用当前策略 \(\pi_i\) 在环境中滚动采集轨迹，记录访问到的状态
2. 邀请专家对这些**实际访问状态**标注最优动作
3. 将新数据加入数据集，重新训练策略 \(\pi_{i+1}\)

随着迭代进行，策略所访问的状态分布逐渐与训练分布对齐，理论上可以收敛到接近专家性能的策略。

### 隐式行为克隆（Implicit BC）

隐式行为克隆 (Implicit Behavioral Cloning, Implicit BC) 用**能量模型 (Energy-Based Model)** 取代显式策略网络。网络 \(E_\theta(s, a)\) 输出状态-动作对的能量值，策略通过求解最小能量动作来做决策：

$$
\pi(s) = \arg\min_a E_\theta(s, a)
$$

能量模型天然支持多模态动作分布（同一状态可有多个合理动作），克服了传统行为克隆在处理多峰分布时发生模式平均 (Mode Averaging) 的问题。

### 动作分块（Action Chunking）与 ACT

ACT (Action Chunking with Transformers) 是斯坦福大学提出的一种针对机器人操作的模仿学习方法。其核心思想是**动作分块 (Action Chunking)**：策略一次预测未来 \(k\) 步的完整动作序列，而非逐步预测单个动作，从而显著减少累积误差并支持更长视野的规划。

ACT 使用基于 Transformer 的 CVAE（条件变分自编码器）架构，编码器从专家演示中提取动作风格隐变量，解码器结合当前视觉和本体感知状态生成动作序列，在精密双臂操作任务上取得了优秀的表现。

### 演示数据采集方式

高质量演示数据是模仿学习成功的关键，常用采集方式包括：

- **遥操作 (Teleoperation)**：操作员通过手柄、力反馈设备或 ALOHA 双臂主从系统实时控制机器人
- **动作捕捉 (Motion Capture)**：在人体上佩戴标记点，将人类动作重定向到机器人
- **VR/数字孪生**：在虚拟环境中采集演示，通过仿真到现实迁移部署到实体机器人
- **视频学习**：从互联网视频中提取人类操作知识，无需机器人在场


## 基础模型与机器人（Foundation Models for Robotics）

大规模预训练基础模型 (Foundation Model) 将互联网规模的视觉、语言知识迁移到机器人控制，正在重新定义机器人的泛化能力边界。

### RT-2（Robotics Transformer 2）

RT-2 由 Google DeepMind 于 2023 年提出，将视觉-语言模型 (Vision-Language Model, VLM) 直接迁移到机器人控制。其关键创新在于用语言 Token (Token) 表示机器人动作：将连续动作值离散化为整数，并映射到语言词表中已有的 Token，从而让机器人策略可以在标准语言建模目标上共同训练。

RT-2 继承了 VLM 的语义推理能力，展现出在纯机器人数据中从未见过的语言指令上的**零样本泛化 (Zero-Shot Generalization)** 能力，例如理解"将能量最少的食物放入垃圾桶"这类需要常识推理的指令。

### OpenVLA

OpenVLA 是首个开源的视觉-语言-动作 (Vision-Language-Action, VLA) 模型，参数量为 7B。它以 Prismatic VLM 为骨干，在 Open X-Embodiment 数据集（来自 22 种不同机器人）上进行微调，支持语言条件的机器人操作。

OpenVLA 的优势在于其开放性：研究者可以在消费级 GPU（单卡 A100 或等效）上对其进行参数高效微调 (Parameter-Efficient Fine-Tuning, PEFT)，适配特定机器人和任务场景，大大降低了机器人基础模型的研究门槛。

### π₀（Pi-Zero）

π₀ (Pi-Zero) 是 Physical Intelligence（物理智能）公司推出的通用机器人策略模型，基于**流匹配 (Flow Matching)** 框架生成连续动作。其架构将预训练的视觉-语言模型与轻量级动作专家网络结合，后者负责将高级语义意图转化为精确的关节控制指令。

π₀ 在折叠衣物、组装纸箱等高度灵巧的操作任务上展现出出色性能，并支持通过少量演示快速适配新任务，体现了基础模型在机器人领域**跨机器人迁移 (Cross-Embodiment Transfer)** 的潜力。

### 扩散策略（Diffusion Policy）

扩散策略 (Diffusion Policy) 将去噪扩散概率模型 (Denoising Diffusion Probabilistic Model, DDPM) 引入机器人操作，将动作生成建模为迭代去噪过程：

$$
a_0 = \text{Denoise}_\theta\!\left(a_T, s\right) = \text{Denoise}_\theta\!\left(\epsilon, s\right), \quad \epsilon \sim \mathcal{N}(0, I)
$$

扩散模型天然支持**多模态动作分布 (Multimodal Action Distribution)**，能够表达"抓取左边或右边的物体"等具有歧义性的任务中多种合理动作，而传统回归策略只能输出所有模式的均值，导致无效的居中动作。扩散策略在多个机器人操作基准上取得了当前最优 (State-of-the-Art) 成绩。

### 基础模型的关键能力

| 能力 | 说明 |
|------|------|
| 语言指令跟随 | 理解并执行自然语言描述的操作任务 |
| 开放词汇感知 (Open-Vocabulary Perception) | 检测和定位训练时未见过的物体类别 |
| 零样本泛化 | 在全新场景和任务上无需额外训练即可运行 |
| 少样本适配 | 通过极少量演示快速学习新技能 |
| 常识推理 | 利用预训练语言知识处理需要物理或社会常识的指令 |


## 深度学习与经典控制的融合

深度学习并非要完全取代经典控制，两者的有机结合往往能取得更好的实用效果。

### 残差策略学习

残差策略学习 (Residual Policy Learning) 在经典控制器（如 PID、MPC）输出的基础上叠加学习得到的残差修正量：

$$
a = \pi_{\text{classic}}(s) + \pi_\theta(s)
$$

经典控制器提供稳定、安全的基础控制，学习策略 \(\pi_\theta\) 专注于补偿模型误差、未建模动力学和环境干扰。这种方式显著降低了学习策略的难度，并继承了经典控制器的安全保障，适合对安全性要求高的工业机器人场景。

### 学习代价函数

逆强化学习 (Inverse Reinforcement Learning, IRL) 从专家演示中反推奖励函数，而非直接克隆动作。学习到的奖励函数具有更好的可迁移性：在新环境中可以结合规划算法重新求解最优策略，而无需重新收集演示数据。生成对抗模仿学习 (Generative Adversarial Imitation Learning, GAIL) 是 IRL 的一个高效变体，用判别器隐式表示奖励信号。

### 神经网络系统辨识

精确的动力学模型是模型预测控制 (MPC) 和基于模型强化学习的核心。神经网络系统辨识 (Neural Network System Identification) 通过数据驱动的方式学习机器人动力学：

- **LSTM 动力学模型**：将历史状态-动作序列作为输入，预测下一时刻状态，适合捕捉流体、绳索等难以解析建模的动力学
- **物理信息神经网络 (Physics-Informed Neural Network, PINN)**：在损失函数中嵌入物理定律约束（如拉格朗日方程），确保学习到的模型在物理上合理，同时用数据拟合未知参数


## 模型优化与嵌入式部署

机器人对推理延迟有严格要求（控制频率通常 ≥ 50 Hz），将深度模型部署到边缘计算平台需要多维度的模型压缩与加速。

### 量化（Quantization）

量化 (Quantization) 将浮点权重和激活值转换为低精度整数表示，降低显存占用并利用整数计算单元加速推理：

- **INT8 量化**：精度损失通常 < 1%，推理速度可提升 2-4 倍
- **INT4 量化**：进一步压缩模型体积，适合极端资源受限场景，需配合量化感知训练 (Quantization-Aware Training, QAT) 弥补精度损失
- **FP16/BF16**：半精度浮点，在保留大部分精度的同时获得约 2 倍加速，是当前 GPU 推理的默认格式

### 剪枝（Pruning）

剪枝 (Pruning) 通过删除冗余权重减小模型规模：

- **非结构化剪枝 (Unstructured Pruning)**：将权重矩阵中绝对值较小的元素置零，压缩比高，但稀疏矩阵运算难以在通用硬件上高效执行
- **结构化剪枝 (Structured Pruning)**：删除整个卷积通道（通道剪枝）或注意力头，生成的小模型保持稠密结构，可直接在现有硬件上加速

### 知识蒸馏（Knowledge Distillation）

知识蒸馏 (Knowledge Distillation, KD) 由 Hinton 等人提出，通过将大型教师模型 (Teacher Model) 的"软标签"（类别概率分布）作为训练目标来指导小型学生模型 (Student Model) 的训练：

$$
\mathcal{L}_{KD} = \alpha \mathcal{L}_{CE}(y, \hat{y}_S) + (1-\alpha) \tau^2 \mathcal{L}_{KL}\!\left(\sigma\!\left(\frac{z_T}{\tau}\right),\ \sigma\!\left(\frac{z_S}{\tau}\right)\right)
$$

其中 \(\tau\) 为温度参数，较高的温度使软标签更平滑，包含类别间相似度的"暗知识 (Dark Knowledge)"；\(\alpha\) 平衡真实标签损失与蒸馏损失的权重。

### 推理引擎

| 推理引擎 | 开发方 | 支持硬件 | 特点 |
|---------|--------|---------|------|
| TensorRT | NVIDIA | NVIDIA GPU | 图优化 + 量化，延迟最低，机器人 GPU 端首选 |
| ONNX Runtime | Microsoft | CPU/GPU/NPU | 跨平台，与框架解耦，支持多种后端加速器 |
| OpenVINO | Intel | Intel CPU/iGPU/VPU | Intel 硬件深度优化，适合工控机部署 |
| TFLite | Google | ARM CPU/GPU/DSP | 极轻量，适合微控制器和移动端 |

**TensorRT 推理代码示例**（Python）：

```python
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_path: str) -> trt.ICudaEngine:
    """从 ONNX 文件构建 TensorRT 引擎（INT8 量化）。"""
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GiB
        config.set_flag(trt.BuilderFlag.INT8)  # 启用 INT8 量化

        with open(onnx_path, "rb") as f:
            parser.parse(f.read())

        return builder.build_serialized_network(network, config)


def infer(engine_bytes: bytes, input_data: np.ndarray) -> np.ndarray:
    """使用序列化的 TensorRT 引擎执行推理。"""
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    context = engine.create_execution_context()

    # 分配设备显存
    d_input = cuda.mem_alloc(input_data.nbytes)
    output = np.empty(engine.get_tensor_shape("output"), dtype=np.float32)
    d_output = cuda.mem_alloc(output.nbytes)

    stream = cuda.Stream()
    cuda.memcpy_htod_async(d_input, input_data, stream)
    context.execute_async_v2([int(d_input), int(d_output)], stream.handle)
    cuda.memcpy_dtoh_async(output, d_output, stream)
    stream.synchronize()
    return output
```

### 目标平台：NVIDIA Jetson Orin

Jetson Orin 系列是当前机器人边缘计算的主流选择，提供 AGX Orin（275 TOPS）、Orin NX（100 TOPS）和 Orin Nano（40 TOPS）等多个性能档位。其 Ampere 架构 GPU、深度学习加速器 (DLA) 和高带宽内存的组合，可在 10-60 W 功耗范围内运行实时目标检测、姿态估计和视觉-语言推理等多个并行任务。


## 常用框架与工具

| 框架 | 语言 | 特点 | 适用场景 |
|------|------|------|---------|
| PyTorch | Python/C++ | 动态计算图，调试便捷，生态丰富（torchvision、Detectron2、timm） | 学术研究、模型开发与快速原型 |
| JAX/Flax | Python | 函数式编程，XLA 即时编译，支持 TPU，`jit`/`vmap`/`grad` 组合强大 | 大规模分布式训练、RL 研究（Brax） |
| LeRobot（HuggingFace）| Python | 机器人学习专用库，集成数据集（LeRobot Dataset）、预训练模型（ACT、Diffusion Policy）和标准训练脚本 | 机器人操作、模仿学习端到端研究 |
| TensorFlow/Keras | Python/C++ | 静态图优化，TFLite/TF Serving 部署链完整 | 工业级部署、移动端推理 |
| ONNX | — | 跨框架模型交换标准，打通训练框架与推理引擎 | 模型转换与跨平台部署 |

**LeRobot 快速上手示例**：

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.act.modeling_act import ACTPolicy

# 加载 HuggingFace Hub 上的机器人演示数据集
dataset = LeRobotDataset("lerobot/aloha_sim_insertion_human")

# 加载预训练 ACT 策略并在 GPU 上推理
policy = ACTPolicy.from_pretrained("lerobot/act_aloha_sim_insertion_human")
policy.eval()

# 从数据集中取一帧进行推理
batch = dataset[0]
action = policy.select_action(batch)
print("预测动作：", action)
```


## 参考资料

1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. [在线版本](https://www.deeplearningbook.org/)
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*.
3. Bojarski, M., et al. (2016). End to End Learning for Self-Driving Cars. *arXiv:1604.07316*. NVIDIA.
4. Ross, S., Gordon, G., & Bagnell, D. (2011). A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning. *AISTATS 2011*.
5. Florence, P., et al. (2022). Implicit Behavioral Cloning. *Conference on Robot Learning (CoRL) 2021*.
6. Zhao, T. Z., et al. (2023). Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware. *RSS 2023*. (ACT)
7. Brohan, A., et al. (2023). RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control. *arXiv:2307.15818*.
8. Kim, M. J., et al. (2024). OpenVLA: An Open-Source Vision-Language-Action Model. *arXiv:2406.09246*.
9. Black, K., et al. (2024). π₀: A Vision-Language-Action Flow Model for General Robot Control. *arXiv:2410.24164*.
10. Chi, C., et al. (2023). Diffusion Policy: Visuomotor Policy Learning via Action Diffusion. *RSS 2023*.
11. Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network. *arXiv:1503.02531*.
12. Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS 2017*.
