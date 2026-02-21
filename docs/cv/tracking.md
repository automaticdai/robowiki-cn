# 目标跟踪

!!! note "引言"
    目标跟踪（Object Tracking）是计算机视觉与机器人感知的核心技术之一，旨在视频序列中持续定位特定目标的位置与状态。在机器人系统中，目标跟踪为人员跟随、抓取交互、视觉伺服等任务提供实时的空间感知能力，是连接感知与控制的重要桥梁。从工业机械臂的工件追踪到自主移动机器人的行人跟随，目标跟踪技术的鲁棒性与实时性直接决定了机器人系统的可用性与安全性。


## 跟踪任务概述

### 跟踪与检测的区别

目标检测（Object Detection）与目标跟踪虽然密切相关，但在任务定义上存在本质差异：

- **目标检测**：对单帧图像进行处理，输出场景中所有特定类别目标的边界框（Bounding Box）与类别置信度，每帧独立运算，不维护时序信息。
- **目标跟踪**：在连续视频帧中持续定位同一目标，维护目标的身份（Identity）信息，需利用时序上下文处理遮挡、快速运动、外观变化等挑战。

跟踪器的输入通常为初始帧中目标的位置（或检测结果），输出为后续每帧中该目标的位置估计。与检测不同，跟踪器通常不需要在每帧重新识别目标类别，但需要解决目标身份的时序一致性问题。

### 单目标跟踪与多目标跟踪

根据跟踪目标数量，可将跟踪任务分为两大类：

**单目标跟踪**（Single Object Tracking，SOT）：

- 初始帧给定一个目标的边界框作为模板
- 后续帧中持续定位该目标
- 不需要类别先验，泛化能力强
- 典型挑战：快速运动、遮挡、相似目标干扰、出视野再入场

**多目标跟踪**（Multi-Object Tracking，MOT）：

- 同时跟踪场景中多个同类或异类目标
- 需为每个目标分配唯一的轨迹编号（Track ID）
- 涉及轨迹初始化、维护与终止
- 典型挑战：目标间遮挡、身份切换（ID Switch）、新目标进入与目标消失

### 在机器人中的应用

| 应用场景 | 跟踪类型 | 典型算法 | 关键需求 |
|---------|---------|---------|---------|
| 人员跟随（Person Following） | SOT / MOT | KCF、DeepSORT | 实时性、鲁棒遮挡处理 |
| 目标抓取交互 | SOT | SiamRPN++、OSTrack | 精确定位、6DoF估计 |
| 视觉伺服（Visual Servoing） | SOT | CSRT、KCF | 低延迟、高帧率 |
| 多人感知与社交导航 | MOT | ByteTrack、StrongSORT | 多目标身份一致性 |
| 工业质检与装配 | SOT / MOT | SiamMask、SORT | 像素级精度 |
| 无人机目标追踪 | SOT | SiamFC、DropTrack | 视角变化鲁棒性 |

在视觉伺服中，跟踪器的输出（目标位置/速度）直接作为控制律的反馈信号，因此对跟踪的实时性与稳定性要求极高。在多机器人协作场景中，多目标跟踪则用于感知同伴机器人和障碍物的动态状态。


## 单目标跟踪：相关滤波方法

相关滤波（Correlation Filter）方法基于信号处理理论，通过学习一个滤波器使其与目标模板的互相关响应最大，从而在搜索区域中定位目标。其核心优势在于可借助快速傅里叶变换（Fast Fourier Transform，FFT）在频域高效计算，实现实时跟踪。

### MOSSE

**MOSSE**（Minimum Output Sum of Squared Error，最小输出平方误差之和）由 Bolme 等人于 2010 年提出，是相关滤波跟踪的奠基性工作。

**核心思想**：在频域学习一个滤波器 \(\hat{H}\)，使其与目标图像块 \(F_i\) 的相关响应尽量接近预期的高斯响应 \(G_i\)：

$$
\min_H \sum_i \left| F_i \odot H^* - G_i \right|^2
$$

其中 \(\odot\) 表示逐元素相乘，\({}^*\) 表示共轭。闭式解为：

$$
\hat{H}^* = \frac{\sum_i G_i \odot F_i^*}{\sum_i F_i \odot F_i^* + \varepsilon}
$$

**在线更新**：通过指数加权移动平均更新分子 \(A\) 与分母 \(B\)：

$$
A_t = (1 - \eta) A_{t-1} + \eta G_t \odot F_t^*, \quad B_t = (1 - \eta) B_{t-1} + \eta F_t \odot F_t^*
$$

其中 \(\eta\) 为学习率。MOSSE 速度极快（可达 669 FPS），但精度有限，仅使用灰度特征。

### KCF

**KCF**（Kernelized Correlation Filters，核化相关滤波）由 Henriques 等人于 2015 年提出，是相关滤波方法的重要突破。

**循环矩阵加速**：KCF 的关键洞见是：对图像块进行循环移位所得到的所有样本构成一个循环矩阵（Circulant Matrix），可在频域对角化，从而将训练复杂度从 \(O(n^3)\) 降低至 \(O(n \log n)\)。

**核技巧**：将线性相关滤波推广至核空间，引入核函数 \(k(\mathbf{x}, \mathbf{x}')\)（常用高斯核或多项式核），使模型具备非线性拟合能力。在核空间中，预测响应为：

$$
\hat{f}(\mathbf{z}) = \mathbf{w}^T \mathbf{z} = \sum_i \alpha_i k(\mathbf{z}, \mathbf{x}_i)
$$

利用循环矩阵的性质，核矩阵的对角化可在频域完成，训练与检测的计算均可通过 FFT 高效实现：

$$
\hat{\boldsymbol{\alpha}} = \frac{\hat{g}}{\hat{k}^{xx} + \lambda}
$$

$$
\hat{f}(\mathbf{z}) = \mathcal{F}^{-1}\left(\hat{k}^{xz} \odot \hat{\boldsymbol{\alpha}}\right)
$$

其中 \(\hat{k}^{xx}\) 和 \(\hat{k}^{xz}\) 分别为自相关与互相关核向量的傅里叶变换，\(\lambda\) 为正则化系数。

KCF 可使用 HOG（Histogram of Oriented Gradients，方向梯度直方图）特征，在保持实时性（~170 FPS）的同时大幅提升精度。

### CSRT

**CSRT**（Channel and Spatial Reliability Tracking，通道与空间可靠性跟踪）由 Lukezic 等人于 2017 年提出，是 OpenCV 内置的高精度相关滤波跟踪器。

**核心改进**：

1. **空间可靠性图（Spatial Reliability Map）**：对目标区域的每个像素计算可靠性权重，突出目标前景、抑制背景干扰，使滤波器的学习更集中于目标本身。

2. **通道可靠性（Channel Reliability）**：对多通道特征（HOG + Color Names）中的每个通道赋予独立权重，自适应地选择最具判别力的特征通道。

优化目标变为带空间约束的相关滤波学习：

$$
\varepsilon(h) = \left\| \sum_c w_c \cdot (h_c * f_c) - g \right\|^2 + \lambda \sum_c \| h_c \|^2
$$

其中 \(w_c\) 为通道可靠性权重，\(h_c\) 为各通道滤波器，\(f_c\) 为各通道特征图。

CSRT 速度约 25 FPS，精度在相关滤波方法中居于前列，适用于对精度要求较高但允许适当延迟的机器人应用。

### 相关滤波方法对比

| 方法 | 特征 | 速度 | 精度 | 核技巧 | 特点 |
|-----|------|------|------|--------|------|
| MOSSE | 灰度 | ~669 FPS | 低 | 否 | 速度最快，工程友好 |
| KCF | HOG | ~170 FPS | 中 | 是（高斯核） | 速度与精度均衡 |
| CSRT | HOG + CN | ~25 FPS | 高 | 否 | 空间/通道可靠性加权 |


## 单目标跟踪：孪生网络方法

孪生网络（Siamese Network）方法将跟踪问题转化为相似度度量学习问题：离线训练一个深度神经网络，使其能够度量模板图像块与候选区域的相似度，在线跟踪时无需模型更新，直接前向推理即可定位目标。

### SiamFC

**SiamFC**（Siamese Fully-Convolutional Network，孪生全卷积网络）由 Bertinetto 等人于 2016 年提出，开创了基于深度学习的孪生网络跟踪范式。

**网络结构**：采用权重共享的双分支卷积网络，分别提取模板图像块 \(\mathbf{z}\)（初始帧目标区域）与搜索区域 \(\mathbf{x}\)（当前帧）的深度特征，通过互相关（Cross-Correlation）操作计算相似度响应图：

$$
f(\mathbf{z}, \mathbf{x}) = \varphi(\mathbf{z}) \star \varphi(\mathbf{x}) + b \cdot \mathbf{1}
$$

其中 \(\varphi(\cdot)\) 为共享的特征提取网络（AlexNet），\(\star\) 表示互相关，\(b\) 为偏置项，响应图的峰值位置即为目标估计位置。

**训练策略**：使用 ImageNet VID 等大规模视频数据集离线训练，正负样本对学习度量函数，无需在线更新，推理速度可达 86 FPS。

**局限性**：

- 仅输出位置，不估计尺度变化（需多尺度搜索）
- 不进行在线模型更新，长时跟踪易漂移

### SiamRPN++

**SiamRPN++** 由 Li 等人于 2019 年提出，将区域建议网络（Region Proposal Network，RPN）引入孪生网络框架。

**核心改进**：

1. **深层骨干网络**：使用 ResNet-50 替代 AlexNet，通过调整步长和空洞卷积（Dilated Convolution）保持特征图分辨率，并通过随机裁剪技巧打破平移不变性限制，使深层网络可用于跟踪。

2. **分类与回归双分支**：在互相关特征图上增加 RPN 头，同时输出分类分数（目标/背景）与边界框回归偏移量 \((\Delta x, \Delta y, \Delta w, \Delta h)\)，直接预测精确的边界框而非仅输出响应峰值。

3. **多层特征聚合**：在 ResNet 的多个阶段分别计算互相关并加权融合，提升对尺度变化的鲁棒性。

SiamRPN++ 在 LaSOT 和 GOT-10k 基准上取得了显著提升，成为孪生网络方法的重要基准。

### SiamMask

**SiamMask** 由 Wang 等人于 2019 年提出，在孪生网络框架中增加了像素级掩码（Mask）分支，实现了同时输出边界框与目标掩码的半监督视频目标分割（Video Object Segmentation，VOS）功能。

**网络结构扩展**：在 SiamRPN 框架基础上，增加第三个分支用于预测目标的二值掩码。掩码分支以互相关特征为输入，通过上采样网络恢复到原始分辨率，输出像素级分割结果。

**应用价值**：在机器人抓取与交互场景中，像素级目标轮廓比边界框提供更精确的空间信息，有助于估计目标形状、计算抓取点。


## 单目标跟踪：Transformer 方法

随着视觉 Transformer（Vision Transformer，ViT）在图像识别任务中取得突破性进展，基于 Transformer 的目标跟踪方法迅速兴起，通过自注意力（Self-Attention）机制建模目标模板与搜索区域的全局依赖关系，克服了相关滤波方法局部性强的限制和孪生网络方法特征交互不充分的问题。

### OSTrack

**OSTrack**（One-Stream Tracking，单流跟踪）由 Ye 等人于 2022 年提出，采用单一 Transformer 流对模板图像块与搜索区域进行联合编码。

**核心思想**：将模板 Token 序列 \(\mathbf{z}\) 与搜索区域 Token 序列 \(\mathbf{x}\) 拼接后送入标准 ViT 编码器，通过多头自注意力（Multi-Head Self-Attention，MHSA）机制在两者之间建立全局交互：

$$
[\mathbf{z}', \mathbf{x}'] = \text{MHSA}\left([\mathbf{z}, \mathbf{x}]\right)
$$

相比孪生网络中的互相关操作，自注意力能够在更深层次建立模板与搜索区域的双向特征交互，无需手工设计特征融合模块。

**Early Candidate Elimination**：OSTrack 引入候选消除机制，在 ViT 编码的中间层逐步过滤掉搜索区域中与模板相关性低的背景 Token，降低计算量并聚焦于目标区域。

**性能**：OSTrack-384 在 LaSOT 上 AUC 达 71.1%，在 GOT-10k 上 AO 达 73.7%，兼顾精度与速度（~58 FPS）。

### DropTrack

**DropTrack** 在 OSTrack 框架基础上引入 Token Dropout 正则化策略，通过随机丢弃训练时的 Token 增强模型对遮挡和部分可见的鲁棒性。

**核心机制**：

- 在训练阶段随机丢弃目标模板和搜索区域的部分 Token，迫使模型从不完整的局部信息中恢复目标位置，提升抗遮挡能力。
- 推理阶段使用完整 Token 序列，无额外计算开销。

### Transformer 跟踪方法的优势

1. **全局建模能力**：自注意力机制可捕获长距离依赖，有效处理大尺度运动与目标消失再出现。
2. **端到端学习**：无需手工设计特征交互模块，特征提取与匹配统一优化。
3. **大规模预训练迁移**：可直接加载 ImageNet 预训练的 ViT 权重，充分利用大规模视觉表征。

### SOT 方法横向对比

| 方法 | 类别 | 骨干网络 | LaSOT AUC | GOT-10k AO | 速度（FPS） | 特点 |
|-----|------|---------|-----------|------------|------------|------|
| KCF | 相关滤波 | HOG | ~33% | - | ~170 | 轻量实时 |
| CSRT | 相关滤波 | HOG+CN | ~40% | - | ~25 | 精度较高 |
| SiamFC | 孪生网络 | AlexNet | ~33% | ~37% | ~86 | 范式开创 |
| SiamRPN++ | 孪生网络 | ResNet-50 | ~49.6% | ~51.7% | ~35 | 引入 RPN |
| SiamMask | 孪生网络 | ResNet-50 | - | - | ~55 | 像素级掩码 |
| OSTrack-384 | Transformer | ViT-B | ~71.1% | ~73.7% | ~58 | 联合编码 |
| DropTrack | Transformer | ViT-B | ~71.8% | ~74.1% | ~57 | Dropout 增强 |


## 多目标跟踪：检测-关联框架

多目标跟踪的主流范式是**检测后跟踪**（Tracking-by-Detection，TbD）：先用目标检测器在每帧获得检测框，再将当前帧检测结果与历史轨迹进行关联，维护每个目标的连续轨迹。

### SORT

**SORT**（Simple Online and Realtime Tracking）由 Bewley 等人于 2016 年提出，以简洁高效著称，是检测-关联范式的重要基准。

**状态表示**：SORT 使用卡尔曼滤波（Kalman Filter）对每条轨迹的状态进行预测与更新，状态向量定义为：

$$
\mathbf{x} = [u, v, s, r, \dot{u}, \dot{v}, \dot{s}]^T
$$

其中 \(u, v\) 为边界框中心坐标，\(s\) 为边界框面积，\(r\) 为宽高比（认为固定不变），\(\dot{u}, \dot{v}, \dot{s}\) 为对应的速度分量。观测量为 \([u, v, s, r]^T\)。

**预测步骤**：

$$
\hat{\mathbf{x}}_{t|t-1} = F \mathbf{x}_{t-1|t-1}
$$

$$
P_{t|t-1} = F P_{t-1|t-1} F^T + Q
$$

**更新步骤（匹配后）**：

$$
K_t = P_{t|t-1} H^T (H P_{t|t-1} H^T + R)^{-1}
$$

$$
\mathbf{x}_{t|t} = \hat{\mathbf{x}}_{t|t-1} + K_t (z_t - H \hat{\mathbf{x}}_{t|t-1})
$$

其中 \(F\) 为状态转移矩阵，\(H\) 为观测矩阵，\(Q\)、\(R\) 分别为过程噪声和观测噪声协方差矩阵。

**关联策略**：使用**匈牙利算法**（Hungarian Algorithm）求解检测框与轨迹预测框之间的最优匹配，代价矩阵基于**交并比**（Intersection over Union，IoU）：

$$
\text{cost}_{ij} = 1 - \text{IoU}(\text{det}_i, \text{pred}_j)
$$

SORT 简洁高效，速度可达 260 Hz，但纯 IoU 关联在遮挡和相似目标密集场景中身份切换率较高。

### DeepSORT

**DeepSORT**（Deep SORT）由 Wojke 等人于 2017 年提出，在 SORT 基础上引入外观特征（Appearance Feature）以减少身份切换。

**核心改进**：

1. **外观描述子**：训练一个轻量级重识别（Re-Identification，ReID）网络，对每个检测框提取 128 维外观特征向量。

2. **级联匹配**：将关联分为两阶段。首先对近期更新的轨迹进行优先匹配（按上次更新时间排序级联匹配），其次对未匹配的轨迹进行 IoU 匹配。

3. **综合代价函数**：结合马氏距离（Mahalanobis Distance）和余弦距离（Cosine Distance）：

$$
c_{ij} = \lambda \cdot d_{\text{Mah}}(i, j) + (1 - \lambda) \cdot d_{\text{cos}}(i, j)
$$

其中 \(d_{\text{Mah}}\) 利用卡尔曼滤波的不确定性估计运动一致性，\(d_{\text{cos}}\) 度量外观相似度。

DeepSORT 在 MOT16 上显著降低了身份切换次数，成为此后众多改进方法的基础。

### ByteTrack

**ByteTrack** 由 Zhang 等人于 2022 年提出，核心创新在于充分利用**低置信度检测框**（Low-Score Detections）。

**动机**：传统方法仅使用高置信度检测框（阈值通常 \(\geq 0.5\)）进行关联，而被遮挡或模糊的目标往往产生低置信度检测，直接丢弃会导致轨迹过早终止。

**两步关联策略**：

1. **第一步**：高置信度检测框（分数 \(\geq \tau_\text{high}\)）与所有轨迹进行关联，得到已匹配轨迹和未匹配轨迹。

2. **第二步**：低置信度检测框（分数在 \([\tau_\text{low}, \tau_\text{high})\) 之间）与第一步未匹配的轨迹再次关联，恢复被遮挡目标的轨迹。

低置信度检测框不用于初始化新轨迹，仅用于维持已有轨迹，有效减少遮挡导致的轨迹中断。ByteTrack 在 DanceTrack 和 MOT17 上均取得了当时 SOTA 性能。

### StrongSORT

**StrongSORT** 由 Du 等人于 2022 年提出，通过系统性整合多项改进构建了强基线：

1. **EMA（Exponential Moving Average）外观特征更新**：对轨迹的外观特征进行指数加权更新，保持对外观变化的适应性。

2. **NSA（Non-linear motion model with Speed-Adaptive）卡尔曼滤波**：根据目标运动速度自适应调整卡尔曼滤波的过程噪声，提升快速运动估计精度。

3. **AFLink（Appearance-Free Link）轨迹修复**：对短轨迹片段（Tracklet）进行离线后处理，合并被遮挡打断的轨迹，降低碎片化（Fragmentation）。

4. **GSI（Gaussian-smoothed Interpolation）插值**：对轨迹中的空缺帧进行高斯平滑插值，生成平滑连续的轨迹。

### 检测-关联方法对比

| 方法 | 运动模型 | 外观特征 | 关联算法 | 低分检测利用 | 特点 |
|-----|---------|---------|---------|------------|------|
| SORT | 卡尔曼滤波 | 无 | 匈牙利（IoU） | 否 | 极简高效 |
| DeepSORT | 卡尔曼滤波 | ReID（128维） | 级联 + IoU | 否 | 减少 ID Switch |
| ByteTrack | 卡尔曼滤波 | 无 | 两步 IoU | 是 | 遮挡处理优 |
| StrongSORT | NSA卡尔曼 | EMA ReID | 综合代价 | 否 | 系统性综合改进 |


## 多目标跟踪：端到端方法

随着 Transformer 在目标检测（DETR）领域的成功，研究者开始探索直接用 Transformer 完成检测与跟踪的端到端联合学习，无需手工设计关联模块。

### MOTR

**MOTR**（Multiple Object TRacking with Transformers）由 Zeng 等人于 2022 年提出，基于 Deformable DETR 框架，引入**轨迹查询**（Track Query）机制实现端到端多目标跟踪。

**核心机制**：

- **检测查询（Detection Query）**：用于在当前帧检测新出现的目标，生成新轨迹。
- **轨迹查询（Track Query）**：携带历史帧目标信息，跨帧传递，用于定位持续存在的目标。

轨迹查询通过 Transformer 解码器与当前帧图像特征交互，自然地建模时序上下文，无需显式的运动预测与匹配步骤。MOTR 实现了真正意义上的端到端多目标跟踪训练与推理。

### TrackFormer

**TrackFormer** 由 Meinhardt 等人于 2022 年提出，同样基于 DETR 框架，采用自回归式（Auto-regressive）跟踪设计。

**核心思路**：将上一帧的目标检测结果（以查询向量形式）作为当前帧解码器的额外输入查询，通过注意力机制在新帧中寻找对应目标，实现跨帧身份延续。新目标由标准的可学习目标查询（Object Queries）检测。

**优势与挑战**：

- 检测与跟踪统一优化，无手工设计关联规则
- 对大规模带标注跟踪数据需求更高
- 当前在速度与精度上相比 ByteTrack 等优化方案仍有差距，但代表未来技术趋势


## 评测指标

### 多目标跟踪精度（MOTA）

**MOTA**（Multiple Object Tracking Accuracy，多目标跟踪精度）综合衡量漏检（False Negative，FN）、误检（False Positive，FP）和身份切换（ID Switch，IDSW）：

$$
\text{MOTA} = 1 - \frac{\sum_t (\text{FN}_t + \text{FP}_t + \text{IDSW}_t)}{\sum_t \text{GT}_t}
$$

其中 \(\text{GT}_t\) 为第 \(t\) 帧的真实目标数量。MOTA 越高越好，最优值为 1。MOTA 对漏检和误检敏感，但不直接反映定位精度。

### 多目标跟踪精准度（MOTP）

**MOTP**（Multiple Object Tracking Precision，多目标跟踪精准度）衡量已匹配目标对之间的平均定位误差：

$$
\text{MOTP} = \frac{\sum_{t,i} d_{t,i}}{\sum_t c_t}
$$

其中 \(d_{t,i}\) 为第 \(t\) 帧第 \(i\) 个匹配对的 IoU（或距离），\(c_t\) 为第 \(t\) 帧匹配对总数。MOTP 反映跟踪精度，与 MOTA 互补。

### IDF1

**IDF1** 基于身份的 F1 分数，重点评估跟踪器维护目标身份的能力：

$$
\text{IDF1} = \frac{2 \cdot \text{IDTP}}{2 \cdot \text{IDTP} + \text{IDFP} + \text{IDFN}}
$$

其中 IDTP（Identity True Positive）为轨迹正确匹配的帧数，IDFP 和 IDFN 分别为身份误检和身份漏检。IDF1 对身份一致性更敏感，是评估长时跟踪质量的重要指标。

### HOTA

**HOTA**（Higher Order Tracking Accuracy）由 Luiten 等人于 2021 年提出，旨在同时平衡检测精度与关联精度：

$$
\text{HOTA}(\alpha) = \sqrt{\text{DetA}(\alpha) \cdot \text{AssA}(\alpha)}
$$

$$
\text{HOTA} = \frac{1}{|\mathcal{A}|} \sum_{\alpha \in \mathcal{A}} \text{HOTA}(\alpha)
$$

其中 \(\alpha\) 为 IoU 阈值（从 0.05 到 0.95 均匀采样），\(\text{DetA}\)（Detection Accuracy）衡量检测准确率，\(\text{AssA}\)（Association Accuracy）衡量关联准确率。HOTA 已成为 MOTChallenge 官方主要指标。

### 单目标跟踪指标

| 指标 | 全称 | 定义 |
|-----|------|------|
| Prec（精度） | Precision | 预测中心点与真实中心点欧氏距离 \(\leq 20\) 像素的帧比例 |
| AUC | Area Under Curve | Success Plot（IoU 阈值 0~1）曲线下面积，综合衡量定位精度 |
| AO | Average Overlap | 预测框与真实框 IoU 的平均值（GOT-10k 主要指标） |
| SR0.5 | Success Rate at 0.5 | IoU \(\geq 0.5\) 的帧比例 |


## 机器人应用场景

### 人员跟随（Person Following）

人员跟随是服务机器人、物流机器人和社交机器人的核心功能。典型系统架构：

1. **检测**：使用 YOLOv8 等实时检测器定位画面中所有行人。
2. **跟踪**：使用 DeepSORT 或 ByteTrack 维护多人轨迹，通过 ReID 特征锁定目标用户。
3. **控制**：将目标边界框中心与图像中心的像差转化为线速度和角速度指令，驱动机器人跟随。

**挑战与对策**：

- **目标遮挡**：利用卡尔曼预测在遮挡期间维持轨迹，结合 ByteTrack 低分检测恢复策略。
- **人群密集**：使用深度相机提供 3D 位置，在三维空间进行跟踪关联。
- **目标丢失**：结合语义先验（如衣物颜色）和运动预测进行搜索恢复。

### 多目标感知与社交导航

在多人环境中，机器人需实时感知所有行人的位置、速度和预测轨迹，以规划安全路径：

- 使用 MOT 系统输出每个行人的轨迹历史
- 基于恒速或社交力模型（Social Force Model）预测未来 1~3 秒的运动轨迹
- 将预测轨迹作为动态障碍物输入局部规划器（如 DWA、TEB 算法）

### 视觉伺服（Visual Servoing）

视觉伺服利用视觉反馈信息直接驱动机械臂或移动平台，无需精确的三维建模：

**基于图像的视觉伺服（Image-Based Visual Servoing，IBVS）**：

$$
\dot{\mathbf{q}} = \lambda \mathbf{J}_e^+ \mathbf{e}
$$

其中 \(\mathbf{e} = \mathbf{s} - \mathbf{s}^*\) 为当前图像特征 \(\mathbf{s}\)（由跟踪器提供）与目标特征 \(\mathbf{s}^*\) 的误差，\(\mathbf{J}_e^+\) 为图像雅可比矩阵（Jacobian Matrix）的伪逆，\(\lambda\) 为控制增益，\(\dot{\mathbf{q}}\) 为关节速度指令。

跟踪器为视觉伺服提供实时的目标特征（中心点、轮廓等），跟踪精度和延迟直接影响伺服控制性能。KCF 和 CSRT 因其低延迟特性常被用于对响应速度要求严苛的视觉伺服场景。


## 常用工具与数据集

### 开发工具与框架

**OpenCV Tracking API**：

OpenCV 内置了多种经典跟踪器，提供统一的接口：

```python
import cv2

# 创建 CSRT 跟踪器
tracker = cv2.TrackerCSRT_create()

# 在第一帧中初始化（bbox 格式：x, y, w, h）
ret, frame = cap.read()
bbox = cv2.selectROI(frame)
tracker.init(frame, bbox)

# 后续帧更新
while True:
    ret, frame = cap.read()
    success, bbox = tracker.update(frame)
    if success:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
```

支持的跟踪器包括：`TrackerKCF`、`TrackerCSRT`、`TrackerMOSSE`、`TrackerGOTURN`、`TrackerMIL` 等。

**其他常用框架**：

| 工具 | 类型 | 特点 |
|-----|------|------|
| BoxMOT | MOT 库 | 集成 SORT、DeepSORT、ByteTrack、StrongSORT 等，统一接口 |
| MMTracking | MOT/SOT 框架 | OpenMMLab 生态，支持大量算法和数据集 |
| PyTracking | SOT 研究框架 | 支持 OSTrack、DiMP 等，适合学术研究 |
| pysot | SOT 框架 | 支持 SiamRPN++、SiamMask 等孪生网络方法 |
| VisDrone SDK | 无人机跟踪 | 专为无人机视角跟踪设计 |

### 单目标跟踪数据集

| 数据集 | 视频数 | 平均帧数 | 挑战属性 | 主要用途 |
|-------|-------|---------|---------|---------|
| OTB-100 | 100 | ~590 | 遮挡、快速运动等 11 类 | 早期标准基准 |
| GOT-10k | 10,000（训练） | ~150 | 480 类目标，零样本泛化测试 | 泛化性评测 |
| LaSOT | 1,400 | ~2,500 | 长时跟踪，14 类目标 | 长时跟踪基准 |
| TrackingNet | 30,000+ | - | 自然视频，大规模 | 大规模训练 |
| NFS（Need for Speed） | 100 | ~3,830 | 240 FPS 高速视频 | 快速运动 |

### 多目标跟踪数据集

| 数据集 | 场景 | 目标类型 | 帧率 | 特点 |
|-------|------|---------|------|------|
| MOT16/17 | 城市行人 | 行人 | 25~30 FPS | MOT 基准，标注精细 |
| MOT20 | 密集人群 | 行人 | 25 FPS | 超高密度场景 |
| DanceTrack | 舞蹈视频 | 人体 | 20 FPS | 外观相似、大范围运动 |
| BDD100K-MOT | 自动驾驶 | 多类别 | 5 FPS | 驾驶场景，8 类目标 |
| KITTI Tracking | 自动驾驶 | 车辆、行人 | 10 FPS | 激光雷达辅助标注 |
| VisDrone-MOT | 无人机视角 | 多类别 | 25 FPS | 小目标、密集场景 |


## 参考资料

1. Bewley, A., Ge, Z., Ott, L., Ramos, F., & Upcroft, B. (2016). Simple online and realtime tracking. *ICIP 2016*. https://doi.org/10.1109/ICIP.2016.7533003

2. Bertinetto, L., Valmadre, J., Henriques, J. F., Vedaldi, A., & Torr, P. H. S. (2016). Fully-convolutional siamese networks for object tracking. *ECCV 2016 Workshops*. https://doi.org/10.1007/978-3-319-48881-3_56

3. Bolme, D. S., Beveridge, J. R., Draper, B. A., & Lui, Y. M. (2010). Visual object tracking using adaptive correlation filters. *CVPR 2010*. https://doi.org/10.1109/CVPR.2010.5539960

4. Du, Y., Zhao, Z., Song, Y., Zhao, Y., Su, F., Gong, T., & Meng, H. (2023). StrongSORT: Make DeepSORT great again. *IEEE Transactions on Multimedia*. https://doi.org/10.1109/TMM.2023.3240881

5. Henriques, J. F., Caseiro, R., Martins, P., & Batista, J. (2015). High-speed tracking with kernelized correlation filters. *IEEE TPAMI*, 37(3), 583–596. https://doi.org/10.1109/TPAMI.2014.2345390

6. Li, B., Wu, W., Wang, Q., Zhang, F., Xing, J., & Yan, J. (2019). SiamRPN++: Evolution of Siamese visual tracking with very deep networks. *CVPR 2019*. https://doi.org/10.1109/CVPR.2019.00441

7. Luiten, J., Osep, A., Dendorfer, P., Torr, P., Geiger, A., Leal-Taixé, L., & Leibe, B. (2021). HOTA: A higher order metric for evaluating multi-object tracking. *IJCV*, 129, 548–578. https://doi.org/10.1007/s11263-020-01375-2

8. Lukežič, A., Vojíř, T., Čehovin Zajc, L., Matas, J., & Kristan, M. (2018). Discriminative correlation filter tracker with channel and spatial reliability. *IJCV*, 126(7), 671–688. https://doi.org/10.1007/s11263-017-1061-3

9. Meinhardt, T., Kirillov, A., Leal-Taixe, L., & Feichtenhofer, C. (2022). TrackFormer: Multi-object tracking with transformers. *CVPR 2022*. https://doi.org/10.1109/CVPR52688.2022.00864

10. Wang, Q., Zhang, L., Bertinetto, L., Hu, W., & Torr, P. H. S. (2019). Fast online object tracking and segmentation: A unifying approach. *CVPR 2019*. https://doi.org/10.1109/CVPR.2019.00142

11. Wojke, N., Bewley, A., & Paulus, D. (2017). Simple online and realtime tracking with a deep association metric. *ICIP 2017*. https://doi.org/10.1109/ICIP.2017.8296962

12. Ye, B., Chang, H., Ma, B., Shan, S., & Chen, X. (2022). Joint feature learning and relation modeling for tracking: A one-stream framework. *ECCV 2022*. https://doi.org/10.1007/978-3-031-20047-2_22

13. Zeng, F., Dong, B., Zhang, Y., Wang, T., Zhang, X., & Wei, Y. (2022). MOTR: End-to-end multiple-object tracking with transformer. *ECCV 2022*. https://doi.org/10.1007/978-3-031-19812-0_39

14. Zhang, Y., Sun, P., Jiang, Y., Yu, D., Weng, F., Yuan, Z., Luo, P., Liu, W., & Wang, X. (2022). ByteTrack: Multi-object tracking by associating every detection box. *ECCV 2022*. https://doi.org/10.1007/978-3-031-20047-2_1

