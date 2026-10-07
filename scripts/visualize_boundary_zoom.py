import os, json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion

BASE = '/mnt/f/240493014/paper2/curve_full'
OUT_DIR = '/mnt/f/240493014/paper2/FINAL/figures'
os.makedirs(OUT_DIR, exist_ok=True)

SAMPLE_IDX = [150, 350, 600]
METHODS = [
    ('E-00_baseline_step3000',      'E-00 Baseline'),
    ('E-10_sobel_a3_1000_step3000', 'E-10 (Paper1)'),
    ('E-B_segfb_step3000',          'E-B (Paper2)'),
]
SIZE = 384           # 统一分辨率
CROP = 120           # 裁剪尺寸（比之前大）
SCALE = 3

def find_boundary_point(mask_arr):
    """找 mask 边界上、最靠近中心的一个点"""
    m = mask_arr > 127
    boundary = m & ~binary_erosion(m)
    ys, xs = np.where(boundary)
    if len(ys) == 0:
        return SIZE // 2, SIZE // 2
    cy, cx = SIZE // 2, SIZE // 2
    # 选离中心最近的边界点
    dist = (ys - cy) ** 2 + (xs - cx) ** 2
    i = dist.argmin()
    return int(ys[i]), int(xs[i])

n_row = len(SAMPLE_IDX)
n_col = 1 + len(METHODS)

fig, axes = plt.subplots(n_row, n_col, figsize=(3.2 * n_col, 3.2 * n_row))

for row, idx in enumerate(SAMPLE_IDX):
    # ★ 关键：mask resize 到 384×384
    mask_path = f'{BASE}/E-B_segfb_step3000/masks/{idx:06d}.png'
    mask_img = Image.open(mask_path).convert('L').resize((SIZE, SIZE), Image.NEAREST)
    mask_arr = np.array(mask_img)

    # 在边界上定位
    cy, cx = find_boundary_point(mask_arr)
    y1 = max(0, cy - CROP // 2); y2 = y1 + CROP
    x1 = max(0, cx - CROP // 2); x2 = x1 + CROP
    # 边界越界修正
    if y2 > SIZE: y2 = SIZE; y1 = SIZE - CROP
    if x2 > SIZE: x2 = SIZE; x1 = SIZE - CROP

    # Mask crop
    mask_crop = mask_img.crop((x1, y1, x2, y2)).resize((CROP * SCALE, CROP * SCALE), Image.NEAREST)
    axes[row, 0].imshow(mask_crop, cmap='gray', vmin=0, vmax=255)

    # 生成图 crop（生成图本身是 384×384）
    for col, (method, _) in enumerate(METHODS, start=1):
        gen_path = f'{BASE}/{method}/images/{idx:06d}_k0.png'
        gen_img = Image.open(gen_path).convert('RGB').resize((SIZE, SIZE), Image.LANCZOS)
        gen_crop = gen_img.crop((x1, y1, x2, y2)).resize((CROP * SCALE, CROP * SCALE), Image.LANCZOS)
        axes[row, col].imshow(gen_crop)

titles = ['Mask (GT)'] + [m[1] for m in METHODS]
for col, t in enumerate(titles):
    axes[0, col].set_title(t, fontsize=13, fontweight='bold')

for ax in axes.flat:
    ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
out = f'{OUT_DIR}/fig_B_boundary_zoom.png'
plt.savefig(out, dpi=200, bbox_inches='tight')
plt.close()
print(f'saved: {out}')
