#!/bin/bash
cd /home/pc/SiameseDiffusionmain/monai

DATA=/home/pc/isic2018_downstream
RESULT_DIR=/mnt/f/seg_experiments/results
CKPT_DIR=/mnt/f/seg_experiments/checkpoints
LOG_DIR=/mnt/f/seg_experiments/logs

run_one () {
    local exp_id=$1
    local n_real=$2
    local gen_dir=$3   # baseline / ours / routeB

    # 清掉残留软链接
    [ -L $DATA/gen_ours ] && rm -f $DATA/gen_ours

    # 根据 gen_dir 决定是否临时替换 gen_ours
    if [ "$gen_dir" = "ours" ]; then
        # 直接用现有的 gen_ours（E-17 生成图）
        python train_seg.py --exp_id $exp_id \
            --data_root $DATA --n_real $n_real --out_channels 1 --use_ours_gen \
            --mask_suffix _segmentation.png \
            --result_dir $RESULT_DIR --checkpoint_dir $CKPT_DIR --log_dir $LOG_DIR
    else
        # 临时用 gen_baseline 或 gen_routeB 冒充 gen_ours
        if [ -d "$DATA/gen_ours" ]; then
            mv $DATA/gen_ours $DATA/gen_ours_E17_stash
        fi
        ln -s $DATA/gen_${gen_dir} $DATA/gen_ours
        python train_seg.py --exp_id $exp_id \
            --data_root $DATA --n_real $n_real --out_channels 1 --use_ours_gen \
            --mask_suffix _segmentation.png \
            --result_dir $RESULT_DIR --checkpoint_dir $CKPT_DIR --log_dir $LOG_DIR
        rm -f $DATA/gen_ours
        if [ -d "$DATA/gen_ours_E17_stash" ]; then
            mv $DATA/gen_ours_E17_stash $DATA/gen_ours
        fi
    fi
}

echo "===== ISIC2018 下游分割开始: $(date) ====="

# ===== N=40 =====
echo ""
echo "######## N=40 ########"
run_one E-22_2018_N40  40 baseline
run_one E-23_2018_N40  40 ours
run_one E-23B_2018_N40 40 routeB

# ===== N=100 =====
echo ""
echo "######## N=100 ########"
run_one E-22_2018  100 baseline
run_one E-23_2018  100 ours
run_one E-23B_2018 100 routeB

echo ""
echo "===== 全部完成: $(date) ====="
