#!/bin/bash
DATA=/home/pc/isic2016_downstream
RESULT_DIR=/mnt/f/seg_experiments/results
CKPT_DIR=/mnt/f/seg_experiments/checkpoints
LOG_DIR=/mnt/f/seg_experiments/logs

cd /home/pc/SiameseDiffusionmain/monai

# 清掉任何残留的软链接
[ -L $DATA/gen_ours ] && rm -f $DATA/gen_ours

# ---- E-22 (baseline) ----
if [ -d "$DATA/gen_ours" ]; then
    mv $DATA/gen_ours $DATA/gen_ours_stash_$$
fi
ln -s $DATA/gen_baseline $DATA/gen_ours
for s in 42 43 44 45 46 47 48 49 50 51; do
    python train_seg.py --exp_id E-22_N40_s${s} \
        --data_root $DATA --n_real 40 --out_channels 1 --use_ours_gen --seed ${s} \
        --result_dir $RESULT_DIR --checkpoint_dir $CKPT_DIR --log_dir $LOG_DIR
done
rm -f $DATA/gen_ours
[ -d "$DATA/gen_ours_stash_$$" ] && mv $DATA/gen_ours_stash_$$ $DATA/gen_ours

# ---- E-23B (E-B) ----
if [ -d "$DATA/gen_ours" ]; then
    mv $DATA/gen_ours $DATA/gen_ours_stash2_$$
fi
ln -s $DATA/gen_routeB $DATA/gen_ours
for s in 42 43 44 45 46 47 48 49 50 51; do
    python train_seg.py --exp_id E-23B_N40_s${s} \
        --data_root $DATA --n_real 40 --out_channels 1 --use_ours_gen --seed ${s} \
        --result_dir $RESULT_DIR --checkpoint_dir $CKPT_DIR --log_dir $LOG_DIR
done
rm -f $DATA/gen_ours
[ -d "$DATA/gen_ours_stash2_$$" ] && mv $DATA/gen_ours_stash2_$$ $DATA/gen_ours

echo "===== N40 全部完成: $(date) ====="
