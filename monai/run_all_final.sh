#!/bin/bash
set -e

LOG_ALL=/mnt/f/seg_experiments/run_all_final.log
exec > >(tee -a $LOG_ALL) 2>&1

echo "=========================================="
echo " 全量分割实验启动"
echo " $(date)"
echo "=========================================="

DATA2016=~/isic2016_downstream
DATA2018=~/isic2018_downstream
RESULTS=/mnt/f/seg_experiments/results
CKPT=/mnt/f/seg_experiments/checkpoints
LOGS=/mnt/f/seg_experiments/logs

C16="--data_root $DATA2016 --out_channels 1 \
     --result_dir $RESULTS --checkpoint_dir $CKPT --log_dir $LOGS"

C18="--data_root $DATA2018 --out_channels 1 --mask_suffix _segmentation.png \
     --result_dir $RESULTS --checkpoint_dir $CKPT --log_dir $LOGS"

echo "===== ISIC2016 · N=40 ====="
python train_seg.py --exp_id E-18 --n_real 40 $C16
python train_seg.py --exp_id E-19 --n_real 40 --use_baseline_gen $C16
python train_seg.py --exp_id E-20 --n_real 40 --use_ours_gen $C16
python train_seg.py --exp_id E-30 --n_real 40 --use_controlnet_gen $C16

echo "===== ISIC2016 · N=100 ====="
python train_seg.py --exp_id E-21 --n_real 100 $C16
python train_seg.py --exp_id E-22 --n_real 100 --use_baseline_gen $C16
python train_seg.py --exp_id E-23 --n_real 100 --use_ours_gen $C16
python train_seg.py --exp_id E-31 --n_real 100 --use_controlnet_gen $C16

echo "===== ISIC2018 · N=40 ====="
python train_seg.py --exp_id E-24 --n_real 40 $C18
python train_seg.py --exp_id E-25 --n_real 40 --use_baseline_gen $C18
python train_seg.py --exp_id E-26 --n_real 40 --use_ours_gen $C18
python train_seg.py --exp_id E-32 --n_real 40 --use_controlnet_gen $C18

echo "===== ISIC2018 · N=100 ====="
python train_seg.py --exp_id E-27 --n_real 100 $C18
python train_seg.py --exp_id E-28 --n_real 100 --use_baseline_gen $C18
python train_seg.py --exp_id E-29 --n_real 100 --use_ours_gen $C18
python train_seg.py --exp_id E-33 --n_real 100 --use_controlnet_gen $C18

echo ""
echo "=========================================="
echo " 全部完成！$(date)"
echo " 汇总日志：$LOG_ALL"
echo "=========================================="
