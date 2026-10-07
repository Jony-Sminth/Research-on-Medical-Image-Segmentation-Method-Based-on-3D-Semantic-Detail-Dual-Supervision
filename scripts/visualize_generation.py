# visualize_generation.py
# 生成论文用的生成质量对比图
import os, json
import matplotlib.pyplot as plt
from PIL import Image

# ===== 配置 =====
BASE = '/mnt/f/240493014/paper2/curve_full'
REAL_DIR = os.path.expanduser('~/SiameseDiffusionmain/data/ISIC2016/images')
PROMPT_JSON = os.path.expanduser('~/SiameseDiffusionmain/data/ISIC2016/prompt.json')
OUT_DIR = '/mnt/f/240493014/paper2/FINAL/figures'
os.makedirs(OUT_DIR, exist_ok=True)

# 选 5 个样本（idx 是 prompt.json 里的行号，0~899）
SAMPLE_IDX = [0, 150, 350, 600, 800]

METHODS = [
    ('E-00_baseline_step3000',       'E-00 Baseline'),
    ('E-10_sobel_a3_1000_step3000',  'E-10 (Paper1)'),
    ('E-B_segfb_step3000',           'E-B (Paper2)'),
]

# ===== 读 prompt.json =====
idx2name = []
with open(PROMPT_JSON) as f:
    for line in f:
        item = json.loads(line)
        idx2name.append(os.path.basename(item['target']))

# ===== 画图 A：mask + 3 方法生成 =====
n_row = len(SAMPLE_IDX)
n_col = 1 + len(METHODS)   # mask + 3

fig, axes = plt.subplots(n_row, n_col, figsize=(4 * n_col, 4 * n_row))

for row, idx in enumerate(SAMPLE_IDX):
    # mask
    mask_path = f'{BASE}/E-B_segfb_step3000/masks/{idx:06d}.png'
    mask_img = Image.open(mask_path).convert('L').resize((256, 256), Image.NEAREST)
    axes[row, 0].imshow(mask_img, cmap='gray')

    # 各方法生成图
    for col, (method, _) in enumerate(METHODS, start=1):
        gen_path = f'{BASE}/{method}/images/{idx:06d}_k0.png'
        gen_img = Image.open(gen_path).convert('RGB').resize((256, 256))
        axes[row, col].imshow(gen_img)

# 列标题
titles = ['Mask'] + [m[1] for m in METHODS]
for col, title in enumerate(titles):
    axes[0, col].set_title(title, fontsize=14, fontweight='bold')

for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
out_a = f'{OUT_DIR}/fig_A_generation_quality.png'
plt.savefig(out_a, dpi=200, bbox_inches='tight')
plt.close()
print(f'saved: {out_a}')

# ===== 画图 B：真实图 + 3 方法生成 =====
n_col_b = 1 + len(METHODS)   # real + 3
fig, axes = plt.subplots(n_row, n_col_b, figsize=(4 * n_col_b, 4 * n_row))

for row, idx in enumerate(SAMPLE_IDX):
    # 真实图
    real_name = idx2name[idx]
    real_img = Image.open(os.path.join(REAL_DIR, real_name)).convert('RGB').resize((256, 256))
    axes[row, 0].imshow(real_img)

    # 各方法生成图
    for col, (method, _) in enumerate(METHODS, start=1):
        gen_path = f'{BASE}/{method}/images/{idx:06d}_k0.png'
        gen_img = Image.open(gen_path).convert('RGB').resize((256, 256))
        axes[row, col].imshow(gen_img)

titles_b = ['Real'] + [m[1] for m in METHODS]
for col, title in enumerate(titles_b):
    axes[0, col].set_title(title, fontsize=14, fontweight='bold')

for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
out_b = f'{OUT_DIR}/fig_B_real_vs_generated.png'
plt.savefig(out_b, dpi=200, bbox_inches='tight')
plt.close()
print(f'saved: {out_b}')

print('done')
