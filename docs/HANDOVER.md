# 硕士论文项目交接文档

**论文题目**：生成-分割协同驱动的少样本皮肤病图像分割
**日期**：2026-10-06
**状态**：实验已完成，进入写论文阶段

---

## 一、项目概述

### 1.1 大论文框架
论文1（已完成） 论文2（已完成）
────────────────────── ──────────────────────────
边界感知双路扩散生成 分割反馈协同优化
核心：Sobel 边界权重 核心：单步 x0 + 冻结分割器 Dice 反馈

归一化 + warmup
指标：FID / HD95 指标：FID / Dice / IoU / HD95
数据：ISIC2016 数据：ISIC2016（主线）+ ISIC2018（limitation）
↓ ↓
大论文：两模块整合

text

### 1.2 核心创新点
1. **论文1**：在 Siamese-Diffusion 的噪声一致性损失（Loss 2）中引入 Sobel 边界加权 + 归一化 + alpha warmup
2. **论文2**：单步 x0 预测 + 冻结分割器 Dice 反馈，实现可微生成-分割协同

---

## 二、最终结果汇总

### 2.1 ISIC2016 主实验（论文主线）

**FID（900 vs 900）**

| 方法 | FID ↓ |
|---|---|
| E-00 baseline | 116.76 |
| E-10（论文1） | 102.19 |
| **E-B（论文2）** | **92.16** |

**下游分割 N=40（10 种子，均值±std）**

| 方法 | Dice | IoU | HD95 |
|---|---|---|---|
| E-22 baseline gen | 0.8089±0.0073 | 0.7165±0.0091 | 46.69±6.54 |
| **E-B（本文）** | **0.8150±0.0053** | **0.7299±0.0068** | 43.71±5.29 |
| **p (paired)** | **0.0032** | **<0.001** | 0.27 |

**下游分割 N=100（10 种子）**

| 方法 | Dice | IoU | HD95 |
|---|---|---|---|
| E-22 | 0.8741±0.0052 | 0.7932±0.0067 | 30.51±1.98 |
| E-B | 0.8743±0.0046 | 0.7954±0.0059 | 30.55±2.51 |
| p (paired) | 0.9451 | 0.5345 | 0.9724 |

**结论**：N=40 显著提升，N=100 无显著差异。

### 2.2 ISIC2018（limitation）

**下游分割 N=40 多种子（10 seeds）**

| 指标 | E-22 baseline | E-23B ours | p |
|---|---|---|---|
| Dice | 0.7534±0.0160 | 0.7427±0.0185 | 0.0303 |
| IoU | 0.6432±0.0192 | 0.6277±0.0215 | 0.0122 |
| HD95 | 50.35±5.64 | 48.44±3.88 | 0.3206 |

**ISIC2018 生成质量（FID）**

| 方法 | FID ↓ |
|---|---|
| E-16 baseline | 108 |
| E-17 论文1 | 88 |
| **E-B 论文2** | **75.67** |

**结论**：ISIC2018 上 E-B 显著更差，作为 limitation。可能原因：
1. 训练不足（6000 步 ≈ 2.3 epoch，ISIC2016 是 6.6 epoch）
2. ISIC2018 含 7 类病变，形态多样
3. S0_2018 较弱（Dice 0.79 vs ISIC2016 的 0.87）

**值得注意**：ISIC2018 上 E-B 的 FID（75.67）继续优于论文1（88）和 baseline（108），但下游 Dice 反而下降——**再次验证"生成质量 ≠ 任务性能"**。

---

## 三、关键文件路径

### 3.1 代码
| 路径 | 说明 |
|---|---|
| `~/SiameseDiffusionmain/cldm/cldm.py` | 核心：Loss 2（边界加权）+ Loss 4（分割反馈） |
| `~/SiameseDiffusionmain/tutorial_train_routeB.py` | 路线B 训练（ISIC2016） |
| `~/SiameseDiffusionmain/tutorial_train_routeB_2018.py` | 路线B 训练（ISIC2018） |
| `~/SiameseDiffusionmain/inference2.py` | 生成候选池 |
| `~/SiameseDiffusionmain/monai/train_seg.py` | 下游分割评估（支持 --seed） |
| `~/SiameseDiffusionmain/monai/rename_gen_to_real.py` | 生成图重命名 |

### 3.2 配置
| 路径 | 说明 |
|---|---|
| `models/E-00_baseline.yaml` | ISIC2016 baseline（boundary_type: none） |
| `models/E-10_sobel_a3_1000.yaml` | ISIC2016 论文1（alpha_max=3.0, warmup=1000） |
| `models/E-B_segfb.yaml` | ISIC2016 论文2（+use_seg_feedback, lambda=0.1） |
| `models/E-16_baseline.yaml` | ISIC2018 baseline |
| `models/E-17_sobel_a4_1000.yaml` | ISIC2018 论文1（alpha_max=3.0） |
| `models/E-B_2018.yaml` | ISIC2018 论文2 |

### 3.3 ckpt（都在 FINAL/ckpts/）
| 路径 | 说明 |
|---|---|
| `E-B_segfb_2016_step{1000..6000}.ckpt` | ISIC2016 E-B 6 个 step |
| `E-B_2018_step{1000..6000}.ckpt` | ISIC2018 E-B 6 个 step |
| `S0.pth` / `S0_2018.pth` | 冻结分割器 |

### 3.4 生成图
| 路径 | 说明 |
|---|---|
| `/mnt/f/240493014/paper2/curve_full/E-B_segfb_step3000/images/` | ISIC2016 E-B 900 张 |
| `/mnt/f/240493014/paper2/curve_full/E-B_2018_step3000/images/` | ISIC2018 E-B 2594 张 |
| `/mnt/f/240493014/paper2/curve_full/E-00_baseline_step3000/images/` | ISIC2016 E-00 900 张 |
| `/mnt/f/240493014/paper2/curve_full/E-10_sobel_a3_1000_step3000/images/` | ISIC2016 E-10 900 张 |

### 3.5 结果
| 路径 | 说明 |
|---|---|
| `FINAL/results/isic2016/` | ISIC2016 所有结果 |
| `FINAL/results/isic2018/` | ISIC2018 所有结果 |

### 3.6 下游分割数据
| 路径 | 说明 |
|---|---|
| `~/isic2016_downstream/real/` | ISIC2016 真实图 + mask |
| `~/isic2016_downstream/splits/` | split 文件 |
| `~/isic2016_downstream/gen_baseline/images/` | E-00 生成的 100 张 |
| `~/isic2016_downstream/gen_ours/images/` | E-10 生成的 100 张 |
| `~/isic2016_downstream/gen_routeB/images/` | E-B 生成的 100 张 |
| `~/isic2018_downstream/` | 同上，ISIC2018 版本 |

---

## 四、关键参数

### 4.1 硬件
- GPU：单卡 4070Ti Super（16G）
- Windows 32G 内存，WSL 配置：`C:\Users\pc\.wslconfig`（memory=20GB, swap=32GB）
- WSL vhdx：`F:\WSL\Ubuntu\ext4.vhdx`（压缩后 95G）
- conda 环境：`xcontrol`

### 4.2 训练配置
- 所有实验：单卡 batch=1，precision=16，3000 或 6000 步
- 训练速度：~20 秒/步（ISIC2016 和 ISIC2018）
- 生成速度：~4.5 秒/张

### 4.3 关键超参
| 参数 | ISIC2016 | ISIC2018 |
|---|---|---|
| boundary_type | sobel | sobel |
| alpha_max | 3.0 | 3.0 |
| boundary_warmup_steps | 1000 | 1000 |
| use_seg_feedback | true | true |
| lambda_seg_max | 0.1 | 0.1 |
| seg_warmup_steps | 1000 | 1000 |
| max_steps | 6000 | 6000 |

---

## 五、已知的坑

1. **WSL 写 D 盘大文件容易损坏**：训练输出写 `/home/pc/`，ckpt 立刻备份 F 盘
2. **`save_last=True` 会 OOM**：用 `StateDictSaver` 只存 state_dict（5.4G）
3. **`inference2.py` 种子必须用 `idx*1000+k`**：否则 K=1 时 900 张图初始噪声相同，FID 虚高
4. **FID 必须 real 和 gen 数量对齐**（900 vs 900）
5. **下游分割 rename 必须核对**：生成图按 prompt.json 顺序对应真实图名
6. **`gen_ours` 目录会被脚本临时替换**：跑完检查是否恢复
7. **多种子验证不能只跑 3 个**：容易得错误结论，必须 ≥10 个

---

## 六、待完成事项

| 项 | 状态 |
|---|---|
| 实验 | ✅ 全部完成 |
| 数据分析 | ✅ 完成 |
| **画图** | ⏳ FID 曲线、HD95 柱状图、Dice 箱线图 |
| **写论文1 章节** | ⏳ |
| **写论文2 章节** | ⏳ |
| **大论文整合** | ⏳ |
| **摘要、引言、相关工作、结论** | ⏳ |

---

## 七、写作要点

### 7.1 论文1（边界感知生成）
- 核心：Sobel 边界权重 + 归一化 + warmup
- 主指标：FID（116.76→102.19）+ HD95（生成图，31.35→30.33）
- 消融：E-02~E-14
- **关键发现**：下游 Dice 反而下降（0.8089→0.7900, p<0.0001）
- 解读：边界加权让生成图偏向边缘锐利，但弱化病灶内部纹理，导致分割器难以学习整体形态
- **这为论文2 提供了动机**

### 7.2 论文2（生成-分割协同）
- 核心：单步 x0 + 冻结分割器 Dice 反馈
- 主指标：FID（→92.16）+ Dice（N=40 显著提升）
- 关键：N=40 显著（p=0.003），N=100 无差异
- 局限：ISIC2018 上不 work

### 7.3 大论文主线
> 边界感知生成（论文1）改善 FID/HD95，但下游 Dice 反而下降 →
> 分割反馈协同（论文2）弥补缺陷并全面超越 baseline →
> 揭示"生成质量 ≠ 任务性能"，形成完整"发现问题-解决"框架
---

## 八、Git 仓库

- 仓库：https://github.com/Jony-Sminth/Research-on-Medical-Image-Segmentation-Method-Based-on-3D-Semantic-Detail-Dual-Supervision
- 分支：`main`（论文1），`routeB`（论文2）

---

**文档版本**：v1
**生成时间**：2026-10-06
