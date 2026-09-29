#!/bin/bash
# ═══════════════════════════════════════════════════════════
#  setup_isic2018.sh
#  一次性建立 ISIC2018 下游目录结构（用 symlink，不复制原始数据）
#  执行一次即可，之后直接跑 generate_splits_2018.py
# ═══════════════════════════════════════════════════════════

set -e

# ── 根据你的实际路径修改这两行 ──────────────────────────────
SRC_IMAGES=~/SiameseDiffusionmain/data/ISIC2018/images
SRC_MASKS=~/SiameseDiffusionmain/data/ISIC2018/masks   # ← 如果masks和images同目录就改这里
# ──────────────────────────────────────────────────────────

DOWNSTREAM=~/isic2018_downstream

echo "建立目录结构..."
mkdir -p $DOWNSTREAM/splits
mkdir -p $DOWNSTREAM/gen_baseline/{images,masks}
mkdir -p $DOWNSTREAM/gen_ours/{images,masks}
mkdir -p $DOWNSTREAM/real

# 用 symlink 指向原始数据，避免复制大文件
ln -sfn $(realpath $SRC_IMAGES) $DOWNSTREAM/real/images
ln -sfn $(realpath $SRC_MASKS)  $DOWNSTREAM/real/masks

echo ""
echo "✓ 目录结构："
ls -la $DOWNSTREAM/
echo ""
ls -la $DOWNSTREAM/real/
echo ""
echo "⚠️  gen_baseline 和 gen_ours 目录已建好，等 E-16/E-17 生成图就绪后手动填入。"
echo "✓ 完成！下一步运行: python generate_splits_2018.py"