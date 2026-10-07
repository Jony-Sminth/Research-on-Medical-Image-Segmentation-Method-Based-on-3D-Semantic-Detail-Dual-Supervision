#!/bin/bash
echo "=========================================="
echo " 时间: $(date)"
echo "=========================================="
echo ""
echo "【进程状态】"
if pgrep -f "train_seg.py" > /dev/null; then
    echo "  ✅ 训练中 (PID: $(pgrep -f train_seg.py | head -1))"
else
    echo "  ⏹  未运行"
fi
if pgrep -f "run_2018_seeds" > /dev/null; then
    echo "  ✅ 脚本在跑"
else
    echo "  ⏹  脚本结束"
fi
echo ""
echo "【当前进度】"
tail -3 /tmp/seeds_2018.log | grep -oE "seed=[0-9]+|START|DONE" | tail -3
echo ""
echo "【完成计数】"
echo -n "  E-22_2018_N40:  "
ls /mnt/f/seg_experiments/results/E-22_2018_N40_s*.txt 2>/dev/null | wc -l
echo -n "  E-23B_2018_N40: "
ls /mnt/f/seg_experiments/results/E-23B_2018_N40_s*.txt 2>/dev/null | wc -l
echo "  （目标: 各 10）"
echo ""
echo "【最新结果】"
for f in $(ls -t /mnt/f/seg_experiments/results/*_2018_N40_s*_result.txt 2>/dev/null | head -3); do
    echo "--- $(basename $f) ---"
    cat $f
    echo ""
done
echo "=========================================="
