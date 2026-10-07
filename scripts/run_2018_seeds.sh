#!/bin/bash
cd /home/pc/SiameseDiffusionmain/monai

DATA=/home/pc/isic2018_downstream
RESULT_DIR=/mnt/f/seg_experiments/results
CKPT_DIR=/mnt/f/seg_experiments/checkpoints
LOG_DIR=/mnt/f/seg_experiments/logs

run_seed () {
    local exp_id=$1
    local seed=$2
    local gen_dir=$3

    [ -L $DATA/gen_ours ] && rm -f $DATA/gen_ours

    if [ "$gen_dir" = "ours" ]; then
        python train_seg.py --exp_id $exp_id \
            --data_root $DATA --n_real 40 --out_channels 1 --use_ours_gen \
            --seed $seed --mask_suffix _segmentation.png \
            --result_dir $RESULT_DIR --checkpoint_dir $CKPT_DIR --log_dir $LOG_DIR
    else
        if [ -d "$DATA/gen_ours" ]; then
            mv $DATA/gen_ours $DATA/gen_ours_E17_stash
        fi
        ln -s $DATA/gen_${gen_dir} $DATA/gen_ours
        python train_seg.py --exp_id $exp_id \
            --data_root $DATA --n_real 40 --out_channels 1 --use_ours_gen \
            --seed $seed --mask_suffix _segmentation.png \
            --result_dir $RESULT_DIR --checkpoint_dir $CKPT_DIR --log_dir $LOG_DIR
        rm -f $DATA/gen_ours
        if [ -d "$DATA/gen_ours_E17_stash" ]; then
            mv $DATA/gen_ours_E17_stash $DATA/gen_ours
        fi
    fi
}

echo "===== ISIC2018 N=40 多种子开始: $(date) ====="

for s in 42 43 44 45 46 47 48 49 50 51; do
    echo "=== E-22_2018_N40 seed=$s START: $(date) ==="
    run_seed E-22_2018_N40_s${s} $s baseline
    echo "=== E-22_2018_N40 seed=$s DONE: $(date) ==="
done

for s in 42 43 44 45 46 47 48 49 50 51; do
    echo "=== E-23B_2018_N40 seed=$s START: $(date) ==="
    run_seed E-23B_2018_N40_s${s} $s routeB
    echo "=== E-23B_2018_N40 seed=$s DONE: $(date) ==="
done

echo "===== 全部完成: $(date) ====="
