#!/bin/bash
DATA=/home/pc/isic2016_downstream
RESULT_DIR=/mnt/f/seg_experiments/results
CKPT_DIR=/mnt/f/seg_experiments/checkpoints
LOG_DIR=/mnt/f/seg_experiments/logs

cd /home/pc/SiameseDiffusionmain/monai

echo "===== E-23B 重跑：$(date) ====="

# 安全断言：gen_routeB 必须 100 张
if [ ! -d "$DATA/gen_routeB/images" ]; then
    echo "ERROR: $DATA/gen_routeB/images 不存在"; exit 1
fi
N=$(ls $DATA/gen_routeB/images/ | wc -l)
if [ "$N" -ne 100 ]; then
    echo "ERROR: gen_routeB 应有 100 张，实际 $N"; exit 1
fi
echo "gen_routeB OK: 100 张"

# 如果 gen_ours 已存在，先挪走
if [ -e "$DATA/gen_ours" ]; then
    mv $DATA/gen_ours $DATA/gen_ours_stash_$$
fi

# 建软链接 gen_ours -> gen_routeB
ln -s $DATA/gen_routeB $DATA/gen_ours

# 逐个跑
for s in 45 46 47 48 49 50 51; do
    echo "=== E-23B seed=$s START: $(date) ==="
    python train_seg.py --exp_id E-23B_s${s} \
        --data_root $DATA \
        --n_real 100 --out_channels 1 --use_ours_gen --seed ${s} \
        --result_dir $RESULT_DIR \
        --checkpoint_dir $CKPT_DIR \
        --log_dir $LOG_DIR
    echo "=== E-23B seed=$s DONE: $(date) ==="
done

# 清理软链接
rm -f $DATA/gen_ours
if [ -e "$DATA/gen_ours_stash_$$" ]; then
    mv $DATA/gen_ours_stash_$$ $DATA/gen_ours
fi

echo "===== 全部完成: $(date) ====="
