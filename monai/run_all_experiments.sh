#!/bin/bash
# ═══════════════════════════════════════════════════════════
#  run_all_experiments.sh
#  运行 E-18 ～ E-23 分割实验（仅 ISIC2016）
#  依赖：train_seg.py
# ═══════════════════════════════════════════════════════════

set -e

DATA2016=~/isic2016_downstream

RESULTS_DIR=/mnt/f/seg_experiments/results
CKPT_DIR=/mnt/f/seg_experiments/checkpoints
LOG_DIR=/mnt/f/seg_experiments/logs

COMMON="--result_dir $RESULTS_DIR --checkpoint_dir $CKPT_DIR --log_dir $LOG_DIR"

echo "=========================================="
echo " 后端分割实验 E-18 ~ E-23 启动"
echo " $(date)"
echo "=========================================="

echo ""
echo "===== ISIC2016 · N=40 ====="

echo "[E-18] 纯真实 N=40"
python train_seg.py --exp_id E-18 --data_root $DATA2016 \
    --n_real 40 --out_channels 1 $COMMON

echo "[E-19] +baseline生成 N=40"
python train_seg.py --exp_id E-19 --data_root $DATA2016 \
    --n_real 40 --out_channels 1 --use_baseline_gen $COMMON

echo "[E-20] +本文生成 N=40"
python train_seg.py --exp_id E-20 --data_root $DATA2016 \
    --n_real 40 --out_channels 1 --use_ours_gen $COMMON

echo ""
echo "===== ISIC2016 · N=100 ====="

echo "[E-21] 纯真实 N=100"
python train_seg.py --exp_id E-21 --data_root $DATA2016 \
    --n_real 100 --out_channels 1 $COMMON

echo "[E-22] +baseline生成 N=100"
python train_seg.py --exp_id E-22 --data_root $DATA2016 \
    --n_real 100 --out_channels 1 --use_baseline_gen $COMMON

echo "[E-23] +本文生成 N=100"
python train_seg.py --exp_id E-23 --data_root $DATA2016 \
    --n_real 100 --out_channels 1 --use_ours_gen $COMMON

echo ""
echo "=========================================="
echo " 全部实验完成！$(date)"
echo " 结果目录：$RESULTS_DIR"
echo " 模型目录：$CKPT_DIR"
echo " 日志目录：$LOG_DIR"
echo "=========================================="