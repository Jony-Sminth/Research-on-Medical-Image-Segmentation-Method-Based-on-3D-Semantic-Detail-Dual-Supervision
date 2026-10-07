#!/bin/bash
echo "===== $(date) ====="
echo ""
echo "进程:"
pgrep -f "train_seg.py" > /dev/null && echo "  ✅ 训练中" || echo "  ⏹  未运行"
echo ""
echo "进度:"
for exp in E-22_2018_N40 E-23_2018_N40 E-23B_2018_N40 E-22_2018 E-23_2018 E-23B_2018; do
    if [ -f /mnt/f/seg_experiments/results/${exp}_result.txt ]; then
        echo "  ✅ $exp"
    else
        echo "  ⏳ $exp"
    fi
done
echo ""
echo "日志尾部:"
tail -2 /tmp/down_2018.log
