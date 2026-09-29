# generate_splits.py
import os, random

random.seed(42)

# 读取所有真实图文件名
real_dir = os.path.expanduser("~/isic2018_downstream/real/images")
all_files = sorted(os.listdir(real_dir))  # 排序保证可复现
assert len(all_files) == 2597, f"期望2354张，实际{len(all_files)}张"

random.shuffle(all_files)

test_files  = all_files[:200]
val_files   = all_files[200:250]
train_files = all_files[250:]   # 剩余650张可用于抽取N=40/100

# 从train_files中抽取
train_40  = train_files[:40]
train_100 = train_files[:100]   # 40是100的子集

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
    print(f"{name}: {len(lst)}张")