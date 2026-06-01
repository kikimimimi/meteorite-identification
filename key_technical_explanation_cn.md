# 关键技术讲解

本文档与当前仓库代码保持一致：当前最高分主线是 `train_dinov2_lora_v2.py`，对应 public F1 为 **0.80628**。最终代码没有使用 INR；INR、SAM、Label Propagation、kNN、Meta-INR 等方法全部归入“尝试过但未作为最终方案”的技术。

当前最佳运行命令：

```bash
python train_dinov2_lora_v2.py --data-root .. --batch-size 4 --amp
```

## 一、最终使用的技术

### 1. 原图输入

最终主线使用原始图像，而不是 SAM/CLIP 裁剪图像。

一开始我们认为分割出石头主体可以减少背景干扰，但实验发现 SAM 裁剪会移除部分有用信息，例如整体尺度、边缘、阴影、拍摄环境和物体形状。对于陨石识别来说，这些信息不一定是噪声，反而可能帮助模型区分陨石和普通石头。

实验现象：

```text
SAM crop + DINOv2 ViT-B/14: 约 0.70
原图 + frozen DINOv2 ViT-B/14 多中心视角: 约 0.74
```

因此最终主线选择原图输入。

### 2. DINOv2 ViT-B/14 视觉基础模型

项目最核心的视觉表征来自 DINOv2 ViT-B/14。DINOv2 是自监督视觉基础模型，已经在大规模图像上学习到强通用表征。

早期路线是冻结 DINOv2，只提取特征，再训练 Logistic Regression 等轻量分类器。这个路线已经能达到较强基线：

```text
原图 + frozen DINOv2 ViT-B/14 多中心视角 top105: 约 0.74
```

但是 frozen feature 只能利用通用视觉表征，不能主动适应当前陨石识别任务。因此最终改为 LoRA/PEFT 微调。

### 3. DINOv2 LoRA / PEFT 微调

最终突破 0.8 的关键技术是 DINOv2 LoRA/PEFT 微调。

具体做法：

```text
冻结 DINOv2 主干参数
在最后若干个 ViT block 的 Linear 层加入 LoRA adapter
只训练 LoRA 参数和分类头
```

这样既保留 DINOv2 的通用视觉知识，又让模型用少量参数适应陨石任务中的纹理、形状和表面结构。

### 4. LoRA v2：更强的分类头与正则化

当前最高分来自 `train_dinov2_lora_v2.py`。相比 v1，v2 的主要改进是减少过拟合、提升小样本稳定性：

```text
更深的 MLP 分类头
BatchNorm
Dropout
Label Smoothing BCE
更低学习率 lr = 1e-4
更强 weight_decay = 1e-3
默认 5-fold 预测平均
```

v2 默认配置包括：

```text
backbone = dinov2_vitb14
image_size = 518
pooling = CLS token + patch-token mean
LoRA rank = 8
LoRA alpha = 16
trainable blocks = last 4 ViT blocks
head_hidden_dim = 256
head_dropout = 0.3
label_smoothing = 0.05
inference = flip TTA
```

记录到的结果：

```text
LoRA v1 top105: 0.79581
LoRA v1 top106: 0.80208
LoRA v2: 0.80628
```

这说明在当前小数据任务中，单纯增加复杂后处理不如对 DINOv2 做轻量任务适配，并通过正则化控制过拟合。

### 5. CLS + Patch Mean Pooling

最终模型不仅使用 DINOv2 的 CLS token，还加入 patch tokens 的平均池化结果：

```text
feature = concat(CLS token, mean(patch tokens))
```

CLS token 更偏向整图语义，patch mean 更偏向局部区域信息。陨石识别往往依赖表面纹理、熔壳、孔洞、裂纹、颗粒和局部反光，因此 patch-level 信息很重要。

### 6. 数据增强与 TTA

训练阶段使用：

```text
RandomResizedCrop
HorizontalFlip / VerticalFlip
RandomRotation
ColorJitter
```

推理阶段使用 flip TTA：

```text
原图预测
水平翻转预测
垂直翻转预测
平均概率
```

这些策略可以减轻小数据集过拟合，使边界样本排序更稳定。

### 7. Top-k 提交

本任务提交的是 0/1 标签，评估指标是 F1，所以除了模型概率排序，还需要决定预测多少张为正类。

public score 反推显示，测试集真实正类数量大约为 86，但直接提交 top86 并不是最好。原因是 F1 同时考虑 precision 和 recall，适当多预测一些正类可以换取更高召回率。

v2 命令默认生成：

```text
top85, top90, top95, top100, top105
```

当前记录到的 v2 最好分数为：

```text
0.80628
```

## 二、尝试过但未作为最终方案的技术

### 1. INR / SIREN 描述符

INR 的思路是对每张图像单独拟合一个小型 SIREN 网络，让网络从坐标 `(x, y)` 重建 RGB 像素值，然后提取拟合过程中的统计特征，例如：

```text
初始重建误差
最终重建误差 / PSNR
loss 曲线
参数变化统计
RGB 残差统计
hidden activation 统计
```

这些特征可以反映图像纹理复杂度、重建难度和表面结构。INR 在 frozen DINOv2 阶段确实带来过提升：

```text
原图 + frozen DINOv2 ViT-B/14: 约 0.74
加入 strong INR re-ranking: 0.76439
```

但是当前最终代码没有使用 INR。并且在 LoRA 路线已经适应任务之后，INR 没有继续带来提升：

```text
LoRA v1 top106: 0.80208
LoRA + INR blend top106: 0.79166
LoRA v2: 0.80628
```

因此 INR 是有价值的探索路线，但不是最终方案。

### 2. SAM / CLIP 裁剪预处理

SAM 裁剪的直觉是去掉背景、保留石头主体。但实验发现它没有提升，甚至会降低效果。

可能原因：

```text
裁剪移除了有用上下文和尺度信息
SAM 处理改变了训练/测试图像分布
陨石识别不仅依赖主体纹理，也依赖整体形态和拍摄环境
```

因此最终不使用 SAM crop。

### 3. Label Propagation

Label Propagation 试图利用训练集和测试集在特征空间中的邻近关系，把标签传播到测试样本。这个方法理论上适合半监督学习，但在当前任务中不稳定：

```text
原图 DINOv2 baseline: 约 0.74
加入 Label Propagation: 约 0.69
Label Propagation + INR: 约 0.71
```

原因可能是：外观相似不一定类别相同，普通石头和陨石在视觉特征空间里容易混在一起。

### 4. kNN / Nearest Neighbor Voting

kNN 根据测试图像在 DINOv2/CLIP 特征空间中最近的训练样本进行投票。它适合做样本分析，但作为最终模型没有带来明显提升：

```text
vitb14 multicenter + kNN blend: 约 0.70
```

因此 kNN 最终作为分析工具，而不是提交主线。

### 5. Meta-INR

Meta-INR 的想法是对 INR descriptor 再训练一个小型 meta encoder：

```text
INR descriptor -> MLP encoder -> meta embedding + meta probability
```

但由于样本数量较少，meta encoder 容易学习训练集中的偶然模式，导致测试集边界排序被扰乱。最终没有进入方案。

### 6. K-fold 多 seed INR Rank Average

我们尝试过更稳定的 K-fold + 多 seed INR 集成：

```text
多个 INR seed
每个 seed 做 K-fold
测试集概率按 fold 平均
多个 seed 做 rank average
```

结果低于 single strong INR：

```text
strong INR single split top105: 0.76439
K-fold multi-seed INR top104: 0.75789
K-fold multi-seed INR top105: 0.75392
```

### 7. LoRA 坏 seed

LoRA 对随机种子有一定敏感性。例如 seed4 的 top105 结果只有：

```text
0.74345
```

这说明不能无脑做 multi-seed 平均。后续如果做 ensemble，需要先筛掉明显跑偏的 seed。

## 三、总结

最终的关键发现是：

```text
最有效的提升不是 INR、SAM 或标签传播，而是让 DINOv2 通过 LoRA 进行轻量任务适配，并用 v2 的分类头和正则化提高泛化能力。
```

当前最终技术路线：

```text
Original Images
+ DINOv2 ViT-B/14
+ LoRA / PEFT fine-tuning
+ CLS + patch-token mean pooling
+ MLP head + BatchNorm + Dropout
+ label smoothing
+ data augmentation
+ flip TTA
= 0.80628 public F1
```
