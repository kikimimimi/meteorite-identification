# 关键技术讲解

## 一、最终使用的技术

### 1. 原图输入而不是 SAM 裁剪图

最终最有效的输入路线是使用原始测试图像，而不是使用 SAM/CLIP 处理后的裁剪图像。

在实验中，SAM 裁剪图像虽然能突出石头主体，但也会移除背景、尺度、边缘、阴影和拍摄环境等信息。对于本任务来说，这些信息并不一定是噪声，反而可能帮助模型区分陨石和普通石头。因此最终主线选择原图输入。

实验结果也支持这一点：

```text
SAM crop + DINOv2 ViT-B/14 多中心视角: 约 0.70
原图 + DINOv2 ViT-B/14 多中心视角: 约 0.74
```

这说明原图保留了更完整的判别信息。

### 2. DINOv2 ViT-B/14 作为视觉基础模型

项目最核心的视觉表征来自 DINOv2 ViT-B/14。DINOv2 是一个自监督视觉基础模型，能够在没有针对本任务训练的情况下提取较强的通用视觉特征。

最初我们使用 frozen DINOv2 提取特征，再训练轻量分类器。这个路线已经能达到较强的基线效果：

```text
原图 + DINOv2 ViT-B/14 多中心视角 top105: 约 0.74
```

多中心视角包括：

```text
full, center90, center80, center75, center70, center60
```

这样做的目的是让模型同时看到完整图像和不同缩放程度的中心区域，从而兼顾全局形状、背景上下文和局部纹理。

### 3. DINOv2 LoRA / PEFT 微调

最终突破 0.8 的关键技术是 DINOv2 LoRA/PEFT 微调。

相比 frozen DINOv2 只提取通用特征，LoRA 允许模型用很少的可训练参数适应当前陨石识别任务。具体做法是：

```text
冻结 DINOv2 主干参数
只在最后若干个 ViT block 的 Linear 层加入 LoRA adapter
训练 LoRA 参数和一个轻量二分类 head
```

这样既保留了 DINOv2 的强通用视觉表征，又让模型学习到陨石任务中特有的纹理、形状和表面结构。

当前最佳 LoRA 设置包括：

```text
backbone = dinov2_vitb14
image_size = 518
pooling = CLS token + patch-token mean
LoRA rank = 8
LoRA alpha = 16
trainable blocks = last 4 ViT blocks
augmentation = random resized crop, flip, rotation, color jitter
inference = flip TTA
```

最终结果：

```text
DINOv2 LoRA quick top105: 0.79581
DINOv2 LoRA quick top106: 0.80208
```

这是目前最强的算法路线。

### 4. CLS + Patch Mean Pooling

最终 LoRA 模型不仅使用 DINOv2 的 CLS token，还加入了 patch tokens 的平均池化结果。

CLS token 更偏向整体图像语义，patch mean 更偏向局部区域信息。陨石识别往往依赖表面纹理、熔壳、孔洞、裂纹、颗粒和局部反光等细节，因此 patch-level 信息很重要。

最终使用：

```text
feature = concat(CLS token, mean(patch tokens))
```

这使模型既能看整体形状，也能关注局部纹理。

### 5. Top-k 校准

本任务最终提交的是 0/1 标签，且评估指标是 F1。因此除了模型概率排序外，还需要选择预测为正类的数量。

通过 public score 反推，我们发现真实正类数量大约为 86，但最优提交并不一定是 top86。因为 F1 同时考虑 precision 和 recall，多预测一些正类可能带来更高 recall，从而提升 F1。

LoRA 最佳结果出现在：

```text
top106 = 0.80208
```

根据 F1 反推，该提交大约对应：

```text
预测正类数 = 106
命中正类数 TP ≈ 77
FP ≈ 29
FN ≈ 9
```

说明 LoRA 模型成功找到了更多真实陨石样本。

### 6. INR 作为有效的辅助技术

虽然最终最佳结果来自 DINOv2 LoRA，但 INR 仍然是一个有效的辅助路线。

INR 的方法是：对每张图片单独拟合一个小型 SIREN 网络，让网络从坐标 `(x, y)` 重建 RGB 像素值。然后提取拟合过程中的特征，例如：

```text
初始重建误差
最终重建误差
PSNR
loss 曲线
参数变化统计
RGB 残差统计
hidden activation 统计
```

这些特征反映图像的纹理复杂度、重建难度和局部表面结构。

INR 的最好结果：

```text
原图 + DINOv2 ViT-B/14 多中心视角: 约 0.74
加入 strong INR re-ranking: 0.76439
```

这说明 INR 确实能帮助边界样本排序，但它不是最终最强主线。最终 LoRA 进一步提升到了 0.80208。

## 二、尝试过但未使用的技术

### 1. SAM / CLIP 裁剪预处理

SAM 裁剪的直觉是：去掉背景，只保留石头主体，减少噪声。

但实验发现这个方法没有提升，甚至会降低效果：

```text
SAM crop + DINOv2 ViT-B/14: 约 0.70
原图 + DINOv2 ViT-B/14: 约 0.74
SAM crop + strong INR: 约 0.72
```

主要原因可能是：

```text
裁剪移除了有用的上下文和尺度信息
SAM 处理改变了训练/测试图像分布
陨石识别不仅依赖主体纹理，也依赖整体形态和拍摄环境
```

因此最终没有把 SAM crop 作为主线。

### 2. Label Propagation

Label Propagation 的思路是利用无标签测试图像和训练图像在特征空间中的邻近关系，把标签传播到测试集。

这个方法理论上适合半监督学习，但在当前任务中效果不稳定：

```text
原图 DINOv2 ViT-B/14 baseline: 约 0.74
加入 Label Propagation: 约 0.69
Label Propagation + INR: 约 0.71
```

主要原因可能是训练集和测试集在特征空间中并不是完美同分布。相似的石头外观不一定代表相同标签，导致标签传播容易把普通石头和陨石混淆。

因此最终未使用 LP。

### 3. kNN / Nearest Neighbor Voting

kNN 方法根据测试图像在 DINOv2/CLIP 特征空间中最近的训练样本进行投票。

它适合做样本分析，但作为最终模型没有带来明显提升：

```text
vitb14 multicenter + kNN blend: 约 0.70
```

主要问题是：最近邻相似并不等于类别相同。尤其在陨石和普通石头外观接近时，kNN 很容易受到局部相似性的误导。

最终我们保留 kNN 作为分析工具，而不是提交主线。

### 4. 全局 Meta-INR

Meta-INR 的思路是对 INR descriptor 再训练一个小型 meta encoder，让模型自动学习 INR 特征的非线性表示。

我们实现了：

```text
INR descriptor -> MLP encoder -> meta embedding + meta probability
```

并将其拼接到原有 DINOv2 + INR 特征中。

但实验结果没有提升，甚至出现排序大幅变化后分数不变或下降的情况。主要原因是样本数量较少，meta encoder 容易学习训练集中的偶然模式，导致测试集边界排序被扰乱。

因此全局 Meta-INR 没有进入最终方案。

### 5. K-fold 多 seed INR Rank Average

我们也尝试过更稳定的 K-fold + 多 seed INR 集成：

```text
多个 INR seed
每个 seed 做 K-fold
测试集概率按 fold 平均
多个 seed 用 rank average 融合
```

这个方法理论上可以降低随机性，但实际结果低于 single strong INR：

```text
strong INR single split top105: 0.76439
K-fold multi-seed INR top104: 0.75789
K-fold multi-seed INR top105: 0.75392
```

原因可能是：平均化虽然提高稳定性，但也抹平了 single strong INR 对关键边界样本的有用偏置。

因此该方法没有作为最终提交。

### 6. 更大的 INR 网络

我们尝试过更强的 INR 设置，例如更大的图像尺寸、更多训练步数、更宽更深的 SIREN 网络。

但结果并没有继续提升，反而可能降低。这说明对于 INR 来说，更好的像素重建不一定等于更好的分类特征。

这和神经场表示的经验一致：下游分类效果不仅取决于重建质量，也取决于拟合动态和特征是否具有判别性。

因此最终保留 strong INR 作为有效辅助，但没有继续扩大 INR 作为主线。

### 7. Linear SVM 作为 INR 分类器

我们也尝试用 calibrated Linear SVM 替代 Logistic Regression 处理 DINOv2 + INR 特征。

结果：

```text
strong INR LogisticRegression: 约 0.764
strong INR LinearSVM: 约 0.75
```

Linear SVM 没有超过 Logistic Regression，因此最终 INR 分支保留 Logistic Regression。

## 三、总结

最终的关键发现是：

```text
最有效的提升不是更复杂的后处理，而是让 DINOv2 通过 LoRA 进行轻量任务适配。
```

整体实验路线可以总结为：

```text
1. 原图比 SAM crop 更有效
2. DINOv2 ViT-B/14 是最强视觉基础模型
3. frozen DINOv2 已经很强，但仍受限于通用特征
4. INR 能补充纹理和重建动态信息，使分数从约 0.74 到 0.76439
5. LoRA/PEFT 微调让 DINOv2 学到任务相关边界，使分数进一步到 0.80208
```

最终使用的技术路线是：

```text
Original Images
+ DINOv2 ViT-B/14
+ LoRA / PEFT fine-tuning
+ CLS + patch-token mean pooling
+ data augmentation
+ flip TTA
+ top106 calibration
= 0.80208 public F1
```
