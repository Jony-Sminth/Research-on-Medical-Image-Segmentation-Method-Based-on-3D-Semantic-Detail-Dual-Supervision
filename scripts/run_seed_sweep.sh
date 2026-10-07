#!/bin/bash
set +e

RESULT_DIR=/mnt/f/seg_experiments/results
CKPT_DIR=/mnt/f/seg_experiments/checkpoints
LOG_DIR=/mnt/f/seg_experiments/logs
DATA=/home/pc/isic2016_downstream

cd /home/pc/SiameseDiffusionmain/monai

echo "================================"
echo "种子扩展实验开始：$(date)"
echo "================================"

# ---- E-22 (baseline 生成图) ----
echo ""
echo "===== E-22 (baseline gen) ====="
mv $DATA/gen_ours     $DATA/gen_ours_E10
mv $DATA/gen_baseline $DATA/gen_ours

for s in 45 46 47 48 49 50 51; do
  if [ -f "$RESULT_DIR/E-22_s${s}_result.txt" ]; then
    echo "[skip] E-22_s${s} 已存在"
    continue
  fi
  echo "=== E-22 seed=$s  START: $(date) ==="
  python train_seg.py --exp_id E-22_s${s} \
    --data_root $DATA \
    --n_real 100 --out_channels 1 --use_ours_gen --seed ${s} \
    --result_dir $RESULT_DIR \
    --checkpoint_dir $CKPT_DIR \
    --log_dir $LOG_DIR
  echo "=== E-22 seed=$s  DONE: $(date) ==="
done

mv $DATA/gen_ours     $DATA/gen_baseline
mv $DATA/gen_ours_E10 $DATA/gen_ours

# ---- E-23B (E-B 生成图) ----
echo ""
echo "===== E-23B (E-B gen) ====="
mv $DATA/gen_ours   $DATA/gen_ours_E10
mv $DATA/gen_routeB $DATA/gen_ours

for s in 45 46 47 48 49 50 51; do
  if [ -f "$RESULT_DIR/E-23B_s${s}_result.txt" ]; then
    echo "[skip] E-23B_s${s} 已存在"
    continue
  fi
  echo "=== E-23B seed=$s  START: $(date) ==="
  python train_seg.py --exp_id E-23B_s${s} \
    --data_root $DATA \
    --n_real 100 --out_channels 1 --use_ours_gen --seed ${s} \
    --result_dir $RESULT_DIR \
    --checkpoint_dir $CKPT_DIR \
    --log_dir $LOG_DIR
  echo "=== E-23B seed=$s  DONE: $(date) ==="
done

mv $DATA/gen_ours     $DATA/gen_routeB
mv $DATA/gen_ours_E10 $DATA/gen_ours

echo ""
echo "================================"
echo "全部完成：$(date)"
echo "================================"
