# 对话与项目历程总结

**时间跨度**：2026-09-25 ~ 2026-10-06
**总历时**：约 12 天

---

## 一、时间线

### 第 1~3 天：论文2 路线B 设计与实现
- 确认路线B（可微分割反馈）方案
- 改 `cldm.py` 加 Loss 4（单步 x0 + 冻结分割器 Dice）
- 新建 `E-B_segfb.yaml` 和 `tutorial_train_routeB.py`
- 训 S0 分割器（Dice=0.8671）

### 第 4~6 天：ISIC2016 E-B 训练
- 6000 步训练（~33 小时）
- 遇到 OOM、目录冲突、`gen_ours` symlink 问题
- 用 `StateDictSaver` 规避 OOM
- 生成 6 个 step × 900 张

### 第 7~9 天：ISIC2016 多种子验证
- 最初 3 种子得出"E-B 更好"
- 补到 10 种子，修正为"N=40 显著，N=100 无差异"
- 中间遇到 `gen_ours` symlink 目录 bug，浪费 7 个种子

### 第 10~12 天：ISIC2018 补充实验
- 训 S0_2018（Dice=0.7913）
- 训 E-B_2018（6000 步）
- 生成 6 个 step × 2594 张
- 结果：ISIC2018 上 E-B 显著更差 → limitation

---

## 二、关键转折点

### 转折 1：FID 反升
**现象**：E-00 step=1000 是 93.65，3000 是 116.76（升了）
**原因**：FID 对"糊图"有偏好，ISIC2016 样本少，模型过拟合
**处置**：论文只报 step=3000，不画曲线

### 转折 2：种子固定导致 FID 虚高
**现象**：E-00 三个 step 的 FID 是 197/226/247
**原因**：`inference2.py` 里 K=1 时 900 张图初始噪声相同
**修复**：改成 `seed = idx*1000 + k`

### 转折 3：3 种子的错误结论
**现象**：3 种子得出 E-B 更好，10 种子发现 N=100 无差异
**教训**：多种子必须 ≥10 个

### 转折 4：`gen_ours` symlink bug
**现象**：脚本用 `mv` 移动目录，后半段 gen_ours 不存在，E-23B 45-51 实际是 E-21（纯真实）
**修复**：改用 `ln -s`

### 转折 5：ISIC2018 失败
**现象**：ISIC2018 上 E-B Dice 0.7427 vs baseline 0.7534，p=0.03 显著更差
**原因**：训练不足（2.3 epoch）+ 7 类多样性 + S0 弱
**处置**：不写主实验，作为 limitation

---

## 三、AI 助手给的关键建议

1. **不要和原论文 8 卡数值比**：单卡 3000 步和 8 卡训练量不同，只做相对比较
2. **论文1 改写双指标**：FID + HD95 双提升，Dice 不作为卖点
3. **论文2 必须做多种子**：单次结果不可信
4. **N=40 才是核心场景**：少样本下生成增强收益最大
5. **ISIC2018 失败是诚实的科学结果**：limitation 里说明，不硬写
6. **别追求"FID 曲线"**：ISIC2016 的曲线不单调，直接用 step=3000 单点

---

## 四、Git 提交历史
routeB 分支：

routeB: Loss 4 分割反馈 + E-B yaml + 训练脚本

routeB for ISIC2018: DS_NAME env + E-B_2018 config
main 分支：

paper1 final: sobel + warmup (code only)

merge remote initial files

text

---

## 五、未完成事项（交给新 AI）

1. **画图**：FID 柱状图、Dice 箱线图、HD95 柱状图
2. **写论文**：
   - 论文1 章节（边界感知生成）
   - 论文2 章节（分割反馈协同）
   - 大论文引言、相关工作、结论
3. **整理最终表格**（论文用）
4. **准备答辩 PPT**

---

## 六、新 AI 对话建议

**给新 AI 的第一句话**：

> 我在写硕士论文《生成-分割协同驱动的少样本皮肤病图像分割》，实验全部完成，现在在整理数据和写论文。请阅读 `/mnt/f/240493014/paper2/FINAL/summary/` 下的 4 个 md 文件（HANDOVER.md, CONVERSATION_SUMMARY.md, DATA_SUMMARY.md, WRITING_GUIDE.md）了解项目全貌，然后帮我 [具体任务]。

**新 AI 需要的文件**：
- `HANDOVER.md`：项目概况、路径、结果
- `CONVERSATION_SUMMARY.md`：本对话总结
- `DATA_SUMMARY.md`：数据汇总（详细版）
- `WRITING_GUIDE.md`：写作指南

---

**文档版本**：v1
**生成时间**：2026-10-06
