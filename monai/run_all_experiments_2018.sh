#!/bin/bash
# ═══════════════════════════════════════════════════════════
#  run_all_experiments_2018.sh
#  运行 E-24 ～ E-29 分割实验（ISIC2018）
#  依赖：train_seg.py（需已加 --mask_suffix 参数，见 README）
# ═══════════════════════════════════════════════════════════

set -e

DATA2018=~/isic2018_downstream

RESULTS_DIR=/mnt/f/seg_experiments/results
CKPT_DIR=/mnt/f/seg_experiments/checkpoints
LOG_DIR=/mnt/f/seg_experiments/logs

# ISIC2018 Task1：二分类皮损分割，out_channels=1，mask后缀小写
COMMON="--data_root $DATA2018 --out_channels 1 --mask_suffix _segmentation.png \
        --result_dir $RESULTS_DIR --checkpoint_dir $CKPT_DIR --log_dir $LOG_DIR"

echo "=========================================="
echo " 后端分割实验 E-24 ~ E-29 启动（ISIC2018）"
echo " $(date)"
echo "=========================================="

echo ""
echo "===== ISIC2018 · N=40 ====="

echo "[E-24] 纯真实 N=40"
python train_seg.py --exp_id E-24 --n_real 40 $COMMON

echo "[E-25] +baseline生成（E-16）N=40"
python train_seg.py --exp_id E-25 --n_real 40 --use_baseline_gen $COMMON

echo "[E-26] +本文生成（E-17）N=40"
python train_seg.py --exp_id E-26 --n_real 40 --use_ours_gen $COMMON

echo ""
echo "===== ISIC2018 · N=100 ====="

echo "[E-27] 纯真实 N=100"
python train_seg.py --exp_id E-27 --n_real 100 $COMMON

echo "[E-28] +baseline生成（E-16）N=100"
python train_seg.py --exp_id E-28 --n_real 100 --use_baseline_gen $COMMON

echo "[E-29] +本文生成（E-17）N=100"
python train_seg.py --exp_id E-29 --n_real 100 --use_ours_gen $COMMON

echo ""
echo "=========================================="
echo " 全部实验完成！$(date)"
echo " 结果目录：$RESULTS_DIR"
echo " 模型目录：$CKPT_DIR"
echo " 日志目录：$LOG_DIR"
echo "=========================================="