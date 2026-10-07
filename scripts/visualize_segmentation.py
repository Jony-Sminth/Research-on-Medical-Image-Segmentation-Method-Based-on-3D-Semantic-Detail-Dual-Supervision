# 分割预测可视化：Real + GT + E-22 预测 + E-B 预测
import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from monai.networks.nets import UNet

# ===== 配置 =====
DATA = os.path.expanduser('~/isic2016_downstream')
CKPT_DIR = '/mnt/f/seg_experiments/checkpoints'
OUT_DIR = '/mnt/f/240493014/paper2/FINAL/figures'
os.makedirs(OUT_DIR, exist_ok=True)

# 用多种子里的 seed=42 版本
CKPT_22 = f'{CKPT_DIR}/E-22_s42_best.pth'      # baseline 生成
CKPT_EB = f'{CKPT_DIR}/E-23B_s42_best.pth'     # E-B 生成

SPLIT = f'{DATA}/splits/test_200_seed42.txt'
MASK_SUFFIX = '_Segmentation.png'
SIZE = 256
SAMPLE_IDX =  [33, 50, 96, 82, 72]   # 按 diff 从大到小

device = torch.device('cuda')

def load_model(ckpt):
    m = UNet(spatial_dims=2, in_channels=3, out_channels=1,
             channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2)).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device))
    m.eval()
    return m

def predict(model, img_arr):
    # img_arr: [H,W,3] 0~255
    x = torch.tensor(img_arr).permute(2,0,1).float() / 255.0
    x = x.unsqueeze(0).to(device)
    with torch.no_grad():
        logit = model(x)
        prob = torch.sigmoid(logit)[0, 0].cpu().numpy()
    return prob

# 读测试列表
with open(SPLIT) as f:
    test_names = [l.strip() for l in f if l.strip()]

# 加载模型
print('loading E-22...')
m22 = load_model(CKPT_22)
print('loading E-B...')
mEB = load_model(CKPT_EB)

# 画图
n_row = len(SAMPLE_IDX)
n_col = 4
fig, axes = plt.subplots(n_row, n_col, figsize=(3.5 * n_col, 3.5 * n_row))

for row, idx in enumerate(SAMPLE_IDX):
    name = test_names[idx]
    stem = os.path.splitext(name)[0]

    # 读数据
    real = np.array(Image.open(f'{DATA}/real/images/{name}').convert('RGB').resize((SIZE, SIZE)))
    gt   = np.array(Image.open(f'{DATA}/real/masks/{stem}{MASK_SUFFIX}').convert('L').resize((SIZE, SIZE), Image.NEAREST))
    gt_bin = (gt > 127).astype(np.uint8)

    # 预测
    p22 = predict(m22, real)
    pEB = predict(mEB, real)
    p22_bin = (p22 > 0.5).astype(np.uint8)
    pEB_bin = (pEB > 0.5).astype(np.uint8)

    # --- 第 1 列：Real + GT 轮廓 ---
    axes[row, 0].imshow(real)
    axes[row, 0].contour(gt_bin, colors='lime', linewidths=1.5)

    # --- 第 2 列：Real + E-22 预测轮廓 ---
    axes[row, 1].imshow(real)
    axes[row, 1].contour(p22_bin, colors='blue', linewidths=1.5)

    # --- 第 3 列：Real + E-B 预测轮廓 ---
    axes[row, 2].imshow(real)
    axes[row, 2].contour(pEB_bin, colors='red', linewidths=1.5)

    # --- 第 4 列：叠加对比 GT vs E-22 vs E-B ---
    axes[row, 3].imshow(real)
    axes[row, 3].contour(gt_bin, colors='lime', linewidths=1.5)
    axes[row, 3].contour(p22_bin, colors='blue', linewidths=1.0, linestyles='--')
    axes[row, 3].contour(pEB_bin, colors='red', linewidths=1.0, linestyles='--')

titles = ['Real + GT', 'Real + E-22 pred', 'Real + E-B pred', 'GT (green)\nE-22 (blue, dash)\nE-B (red, dash)']
for col, t in enumerate(titles):
    axes[0, col].set_title(t, fontsize=12, fontweight='bold')

for ax in axes.flat:
    ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
out = f'{OUT_DIR}/fig_C_segmentation_prediction.png'
plt.savefig(out, dpi=180, bbox_inches='tight')
plt.close()
print(f'saved: {out}')
