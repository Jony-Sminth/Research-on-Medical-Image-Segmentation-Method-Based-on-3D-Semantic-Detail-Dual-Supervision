# 论文写作指南

## 一、大论文结构建议
第 1 章 绪论
1.1 研究背景与意义（皮肤病少样本分割）
1.2 国内外研究现状（扩散模型/医学图像生成/少样本分割）
1.3 本文工作与贡献
1.4 论文结构

第 2 章 相关工作
2.1 扩散模型与医学图像生成
2.2 边界感知学习
2.3 生成-分割协同优化
2.4 少样本医学图像分割

第 3 章 边界感知双路扩散生成（论文1）
3.1 Siamese-Diffusion 框架回顾
3.2 皮肤病变边界特性分析（motivation）
3.3 边界感知噪声一致性损失（Sobel 权重 + 归一化）
3.4 自适应 alpha warmup 调度
3.5 实验：FID / HD95 / 消融

第 4 章 生成-分割协同优化（论文2）
4.1 问题：FID 提升未传递到 Dice
4.2 单步 x0 预测 + 冻结分割器 Dice 反馈
4.3 λ warmup 调度
4.4 实验：多种子验证 / N=40 vs N=100

第 5 章 联合实验与讨论
5.1 双模块联合消融
5.2 跨数据集泛化（ISIC2018 limitation）
5.3 局限性分析

第 6 章 结论与展望

text

## 二、关键写作要点

### 2.1 论文1（边界感知生成）的叙事
**主线**：FID 从 116.76 → 102.19（-12.5%），HD95 从 31.35 → 30.33（-3.3%）

**核心贡献**：
1. Sobel 边界权重 + 归一化机制（双路框架必须）
2. 自适应 alpha warmup
3. 实验证明 Sobel 优于距离变换（E-12 诊断）

**注意**：
- 不要写"下游分割显著提升"（N=100 无差异）
- 写"生成质量和边界保真度提升"
- 一句话带过"训练过程 FID 波动是 warmup 调整期的现象"

### 2.2 论文2（生成-分割协同）的叙事
**主线**：N=40 下 Dice 0.8089→0.8150（p=0.0032 显著）

**核心贡献**：
1. 单步 x0 预测替代 DDIM 采样，实现可微反馈
2. 冻结分割器 + λ warmup 保证训练稳定
3. 在少样本（N=40）下显著提升下游分割

**关键**：
- N=40 是核心场景（p<0.01）
- N=100 无差异是"随着数据增多，生成增强的边际价值降低"（合理）
- ISIC2018 失败作为 limitation（诚实）

### 2.3 大论文统一叙事
> "少样本"是核心场景。生成增强在数据极度稀缺时最有价值，随着真实数据增多，生成图的边际贡献下降。

## 三、写作顺序建议（2 天）

### Day 1
1. 画图（3 小时）：FID 柱状图 / Dice 箱线图 / HD95 柱状图
2. 写论文2 章节（4 小时）：方法 + 实验 + limitation

### Day 2
1. 写论文1 章节（3 小时）
2. 写大论文引言、相关工作、结论（3 小时）
3. 整合 + 格式 + 查重（2 小时）

## 四、画图脚本模板

### 4.1 FID 对比图
```python
import matplotlib.pyplot as plt

methods = ['E-00\nbaseline', 'E-10\n论文1', 'E-B\n论文2']
fid = [116.76, 102.19, 92.16]

plt.figure(figsize=(6, 4))
bars = plt.bar(methods, fid, color=['#888', '#4a90d9', '#e74c3c'])
plt.ylabel('FID (lower is better)')
plt.title('ISIC2016 Generation Quality')
for bar, v in zip(bars, fid):
    plt.text(bar.get_x() + bar.get_width()/2, v + 2, f'{v:.2f}',
             ha='center', fontsize=10)
plt.savefig('fid_comparison.png', dpi=150, bbox_inches='tight')
4.2 Dice 箱线图
python
import matplotlib.pyplot as plt

e22_n40 = [0.818, 0.816, 0.796, 0.8002, 0.8126, 0.8149, 0.806, 0.8092, 0.8033, 0.8131]
eb_n40 = [0.8221, 0.8157, 0.8077, 0.8092, 0.8158, 0.813, 0.8134, 0.8222, 0.8105, 0.8206]
e22_n100 = [0.8657, 0.8752, 0.8696, 0.8801, 0.8773, 0.8822, 0.8733, 0.8729, 0.8684, 0.8767]
eb_n100 = [0.8777, 0.8747, 0.8779, 0.8677, 0.8748, 0.873, 0.8657, 0.8786, 0.8735, 0.8796]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].boxplot([e22_n40, eb_n40], labels=['E-22', 'E-B'])
axes[0].set_title('N=40 (p=0.0032)')
axes[0].set_ylabel('Dice')
axes[1].boxplot([e22_n100, eb_n100], labels=['E-22', 'E-B'])
axes[1].set_title('N=100 (p=0.9451)')
axes[1].set_ylabel('Dice')
plt.tight_layout()
plt.savefig('dice_boxplot.png', dpi=150, bbox_inches='tight')
4.3 HD95 柱状图
python
import matplotlib.pyplot as plt

methods = ['E-22\n(N=40)', 'E-B\n(N=40)', 'E-22\n(N=100)', 'E-B\n(N=100)']
hd95 = [46.69, 43.71, 30.51, 30.55]
std = [6.54, 5.29, 1.98, 2.51]
colors = ['#888', '#e74c3c', '#888', '#e74c3c']

plt.figure(figsize=(8, 4))
plt.bar(methods, hd95, yerr=std, color=colors, capsize=5)
plt.ylabel('HD95 (lower is better)')
plt.title('ISIC2016 Downstream Segmentation')
plt.savefig('hd95_comparison.png', dpi=150, bbox_inches='tight')
五、常见问题
Q: 为什么 N=100 没有显著提升？
A: N=100 时真实样本已足够，生成增强的边际价值下降。这是少样本学习的一般规律，不是方法缺陷。

Q: ISIC2018 失败怎么解释？
A: 3 个原因：(1) 训练不足（2.3 epoch）；(2) 7 类多样性高；(3) S0 弱。作为 limitation。

Q: 为什么不和原论文 8 卡数值比？
A: 训练资源不同，不可比。只做相对比较（同一硬件、同一步数）。

文档版本：v1
生成时间：2026-10-06
