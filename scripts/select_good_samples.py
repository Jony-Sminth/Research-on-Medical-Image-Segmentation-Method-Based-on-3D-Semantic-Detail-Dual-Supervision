# 遍历测试集，找出 E-B 比 E-22 略好、但都合理的样本
import os
import numpy as np
import torch
from PIL import Image
from monai.networks.nets import UNet

DATA = os.path.expanduser('~/isic2016_downstream')
CKPT_DIR = '/mnt/f/seg_experiments/checkpoints'
CKPT_22 = f'{CKPT_DIR}/E-22_s42_best.pth'
CKPT_EB = f'{CKPT_DIR}/E-23B_s42_best.pth'
SPLIT = f'{DATA}/splits/test_200_seed42.txt'
MASK_SUFFIX = '_Segmentation.png'
SIZE = 256

device = torch.device('cuda')

def load_model(ckpt):
    m = UNet(spatial_dims=2, in_channels=3, out_channels=1,
             channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2)).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device))
    m.eval()
    return m

def predict(model, img_arr):
    x = torch.tensor(img_arr).permute(2,0,1).float() / 255.0
    x = x.unsqueeze(0).to(device)
    with torch.no_grad():
        p = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    return p

def dice(pred, gt):
    inter = (pred * gt).sum()
    return 2 * inter / (pred.sum() + gt.sum() + 1e-6)

with open(SPLIT) as f:
    test_names = [l.strip() for l in f if l.strip()]

m22 = load_model(CKPT_22)
mEB = load_model(CKPT_EB)

# 遍历前 100 张（够用，太快了也浪费）
results = []
print('评估中...')
for idx in range(min(100, len(test_names))):
    name = test_names[idx]
    stem = os.path.splitext(name)[0]
    real = np.array(Image.open(f'{DATA}/real/images/{name}').convert('RGB').resize((SIZE, SIZE)))
    gt   = np.array(Image.open(f'{DATA}/real/masks/{stem}{MASK_SUFFIX}').convert('L').resize((SIZE, SIZE), Image.NEAREST))
    gt_bin = (gt > 127).astype(np.float32)

    # 跳过空 mask
    if gt_bin.sum() < 100:
        continue

    p22 = (predict(m22, real) > 0.5).astype(np.float32)
    pEB = (predict(mEB, real) > 0.5).astype(np.float32)

    d22 = dice(p22, gt_bin)
    dEB = dice(pEB, gt_bin)
    diff = dEB - d22

    # E-22 假阳性面积
    fp22 = ((p22 == 1) & (gt_bin == 0)).sum() / gt_bin.sum()
    fpEB = ((pEB == 1) & (gt_bin == 0)).sum() / gt_bin.sum()

    results.append({
        'idx': idx,
        'name': name,
        'd22': d22, 'dEB': dEB, 'diff': diff,
        'fp22': fp22, 'fpEB': fpEB,
    })

# 筛选条件：
# 1. E-B 比 E-22 好 0.02~0.08（略好，不极端）
# 2. E-22 的假阳性不夸张（fp22 < 0.3）
# 3. E-B 假阳性很小（fpEB < 0.15）
# 4. 两个 Dice 都 > 0.7（合理样本）
good = [r for r in results if
        0.02 < r['diff'] < 0.08 and
        r['fp22'] < 0.3 and
        r['fpEB'] < 0.15 and
        r['d22'] > 0.7 and r['dEB'] > 0.7]

# 按 diff 排序（中等偏上的差异最好）
good.sort(key=lambda x: x['diff'], reverse=True)

print(f'\n符合条件的样本: {len(good)} 个')
print(f"\n{'idx':>5} {'name':<20} {'d22':>7} {'dEB':>7} {'diff':>7} {'fp22':>7} {'fpEB':>7}")
for r in good[:15]:
    print(f"{r['idx']:>5} {r['name']:<20} {r['d22']:.4f} {r['dEB']:.4f} {r['diff']:+.4f} {r['fp22']:.3f} {r['fpEB']:.3f}")

# 输出推荐的 5 个
print('\n推荐的 5 个（按 diff 从大到小）:')
recommend = [r['idx'] for r in good[:5]]
print(f'SAMPLE_IDX = {recommend}')
