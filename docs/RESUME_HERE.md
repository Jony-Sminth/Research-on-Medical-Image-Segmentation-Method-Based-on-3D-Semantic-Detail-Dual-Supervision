# 从哪里开始 / Resume Here

## 给新 AI 助手

请先读 `/mnt/f/240493014/paper2/FINAL/summary/` 下的 4 个文件：

1. **HANDOVER.md** ← 最重要，项目概况 + 结果 + 路径
2. **CONVERSATION_SUMMARY.md** ← 对话历程 + 关键转折
3. **DATA_SUMMARY.md** ← 所有数据详细版
4. **WRITING_GUIDE.md** ← 写作指南 + 画图脚本

## 项目当前状态

- 实验全部完成（ISIC2016 + ISIC2018）
- 数据已汇总到 FINAL/
- 待做：画图 + 写论文
- 剩余时间：约 2 天

## 下一步任务

1. 画图（FID / Dice / HD95）
2. 写论文2 章节（数据最全）
3. 写论文1 章节
4. 整合大论文

## 关键提醒

- ISIC2016 是主线，ISIC2018 只作 limitation
- N=40 是核心场景（p=0.003 显著）
- N=100 无差异（不要写成"提升"）
- 所有数据已备份，不要重新跑实验

## 目录结构
/mnt/f/240493014/paper2/FINAL/
├── RESUME_HERE.md ← 本文件
├── summary/ ← 4 个交接文档
│ ├── HANDOVER.md
│ ├── CONVERSATION_SUMMARY.md
│ ├── DATA_SUMMARY.md
│ └── WRITING_GUIDE.md
├── ckpts/ ← 14 个 ckpt
├── configs/ ← 18 个配置 + 代码 + 脚本
├── results/ ← 67 个 result.txt
│ ├── isic2016/
│ └── isic2018/
└── samples/ ← 40 张示例生成图

text

## 给新 AI 的第一句话模板

> 我在写硕士论文《生成-分割协同驱动的少样本皮肤病图像分割》，实验全部完成，现在在整理数据和写论文。请阅读 `/mnt/f/240493014/paper2/FINAL/summary/` 下的 4 个 md 文件（HANDOVER.md, CONVERSATION_SUMMARY.md, DATA_SUMMARY.md, WRITING_GUIDE.md）了解项目全貌，然后帮我 [具体任务]。

---

**文档版本**：v1
**生成时间**：2026-10-06
