# generate_splits_2018.py
import os, random

random.seed(42)

real_dir = os.path.expanduser("~/isic2018_downstream/real/images")
all_files = sorted([f for f in os.listdir(real_dir) if f.lower().endswith('.jpg')])
print(f"找到图片: {len(all_files)} 张")

# 不做硬断言，仅提示（ISIC2018 不同版本图片数不同）
if len(all_files) < 300:
    raise RuntimeError(f"图片数量异常（{len(all_files)}），请检查路径: {real_dir}")

random.shuffle(all_files)

test_files  = all_files[:200]
val_files   = all_files[200:250]
train_files = all_files[250:]    # 剩余全部作为训练候选池

# N=40 是 N=100 的子集（保证趋势可比性）
train_40  = train_files[:40]
train_100 = train_files[:100]

splits_dir = os.path.expanduser("~/isic2018_downstream/splits")
os.makedirs(splits_dir, exist_ok=True)

for name, lst in [
    ("test_200_seed42.txt",  test_files),
    ("val_50_seed42.txt",    val_files),
    ("train_40_seed42.txt",  train_40),
    ("train_100_seed42.txt", train_100),
]:
    with open(os.path.join(splits_dir, name), "w") as f:
        f.write("\n".join(lst))
    print(f"  {name}: {len(lst)} 张")

print(f"\n✓ splits 已保存到: {splits_dir}")
print(f"  训练候选池: {len(train_files)} 张（N=40/100 均从此池抽取）")