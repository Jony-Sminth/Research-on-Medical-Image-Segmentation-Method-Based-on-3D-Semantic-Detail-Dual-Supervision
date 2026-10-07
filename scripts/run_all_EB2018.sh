#!/bin/bash
set -e
cd /home/pc/SiameseDiffusionmain
OUT_ROOT=/mnt/f/240493014/paper2/curve_full
CKPT_DIR=/home/pc/SiameseDiffusion_outputs/E-B_2018/checkpoints

echo "===== E-B_2018 全 step 生成开始: $(date) ====="

for s in 1000 2000 3000 4000 5000 6000; do
    OUT_DIR=$OUT_ROOT/E-B_2018_step${s}
    N=$(ls $OUT_DIR/images/ 2>/dev/null | wc -l)
    if [ "$N" -ge 2500 ]; then
        echo "[skip] step=$s 已完成 ($N 张)"
        continue
    fi
    echo "=== step=$s START: $(date) ==="
    DS_NAME=ISIC2018 python inference2.py \
        --exp_name E-B_2018 \
        --K 1 \
        --ckpt $CKPT_DIR/ckpt-${s}.ckpt \
        --out_dir $OUT_DIR
    echo "=== step=$s DONE:  $(date) ==="
done

echo "===== 全部完成: $(date) ====="
