# train_seg.py
import os, random, argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from monai.networks.nets import UNet
from monai.losses import DiceLoss, DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def img_name_to_mask_name(name, mask_suffix="_Segmentation.png"):
    """ISIC_0000000.jpg  →  ISIC_0000000{mask_suffix}"""
    stem = os.path.splitext(name)[0]
    return stem + mask_suffix

# ── Dataset ─────────────────────────────────────────────────
class ISICDataset(Dataset):
    def __init__(self, real_dir, split_file, gen_dirs=None, augment=False, image_size=256, mask_suffix="_Segmentation.png"):
        """
        real_dir:    {"images": ..., "masks": ...}
        gen_dirs:    [{"images": ..., "masks": ...}, ...]
        split_file:  真实图文件名列表（如 ISIC_0000000.jpg）
        mask_suffix: mask文件名后缀，如 _segmentation.png 或 _Segmentation.png
        """
        self.mask_suffix = mask_suffix

        with open(split_file) as f:
            real_names = [l.strip() for l in f if l.strip()]

        self.samples = []

        # 真实图
        for name in real_names:
            self.samples.append((
                os.path.join(real_dir["images"], name),
                os.path.join(real_dir["masks"],  img_name_to_mask_name(name, self.mask_suffix)),
            ))

        # 生成图（images用真实图文件名，masks复用GT mask）
        if gen_dirs:
            for gd in gen_dirs:
                for name in real_names:
                    img_path  = os.path.join(gd["images"], name)
                    mask_path = os.path.join(gd["masks"],  img_name_to_mask_name(name, self.mask_suffix))
                    if os.path.exists(img_path):
                        self.samples.append((img_path, mask_path))

        self.augment    = augment
        self.image_size = image_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        img  = np.array(Image.open(img_path).convert("RGB").resize(
                    (self.image_size, self.image_size))) / 255.0
        mask = np.array(Image.open(mask_path).convert("L").resize(
                    (self.image_size, self.image_size), Image.NEAREST))

        img  = img.transpose(2, 0, 1).astype(np.float32)   # [3,H,W]
        mask = (mask > 127).astype(np.float32)[None]        # [1,H,W] 二分类

        if self.augment:
            if random.random() > 0.5:
                img, mask = img[:, ::-1, :].copy(), mask[:, ::-1, :].copy()
            if random.random() > 0.5:
                img, mask = img[:, :, ::-1].copy(), mask[:, :, ::-1].copy()
            k = random.randint(0, 3)
            img  = np.rot90(img,  k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k, axes=(1, 2)).copy()

        return torch.tensor(img), torch.tensor(mask)


# ── 训练主函数 ───────────────────────────────────────────────
def train(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 路径配置
    base     = os.path.expanduser(args.data_root)
    real_dir = {"images": f"{base}/real/images", "masks": f"{base}/real/masks"}
    gen_dirs = []
    if args.use_baseline_gen:
        gen_dirs.append({"images": f"{base}/gen_baseline/images",
                         "masks":  f"{base}/gen_baseline/masks"})
    if args.use_ours_gen:
        gen_dirs.append({"images": f"{base}/gen_ours/images",
                         "masks":  f"{base}/gen_ours/masks"})
    if args.use_controlnet_gen:
        gen_dirs.append({"images": f"{base}/gen_controlnet/images",
                         "masks":  f"{base}/gen_controlnet/masks"})

    split_prefix = f"{base}/splits"
    train_split  = f"{split_prefix}/train_{args.n_real}_seed42.txt"
    val_split    = f"{split_prefix}/val_50_seed42.txt"
    test_split   = f"{split_prefix}/test_200_seed42.txt"

    train_ds = ISICDataset(real_dir, train_split, gen_dirs=gen_dirs, augment=True,  mask_suffix=args.mask_suffix)
    val_ds   = ISICDataset(real_dir, val_split,   augment=False,                    mask_suffix=args.mask_suffix)
    test_ds  = ISICDataset(real_dir, test_split,  augment=False,                    mask_suffix=args.mask_suffix)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=4, shuffle=False, num_workers=4)

    # ── 日志 ────────────────────────────────────────────────
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f"{args.exp_id}_train.log")
    log_f = open(log_path, "w")

    def log(msg):
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()

    log(f"实验ID: {args.exp_id}")
    log(f"设备:   {device}")
    log(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # 模型
    model = UNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=args.out_channels,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
    ).to(device)
    torch.manual_seed(42)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = (DiceLoss(sigmoid=True, reduction="mean")
                 if args.out_channels == 1
                 else DiceCELoss(softmax=True, reduction="mean"))

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hd95_metric = HausdorffDistanceMetric(include_background=False,
                                          distance_metric="euclidean",
                                          percentile=95, reduction="mean")

    # 训练循环
    best_val_dice = 0
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        total_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss  = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        if epoch % 10 == 0:
            log(f"Epoch {epoch}/{args.max_epochs} | Loss: {avg_loss:.4f}")

        # 验证集评估，保存最优checkpoint
        if epoch % 10 == 0:
            model.eval()
            dice_metric.reset()
            with torch.no_grad():
                for imgs, masks in val_loader:
                    imgs, masks = imgs.to(device), masks.to(device)
                    preds = model(imgs)
                    if args.out_channels == 1:
                        preds_bin = (torch.sigmoid(preds) > 0.5).float()
                    else:
                        preds_bin = torch.argmax(preds, dim=1, keepdim=True).float()
                    dice_metric(y_pred=preds_bin, y=masks)
            val_dice = dice_metric.aggregate().item()
            log(f"  Val Dice: {val_dice:.4f}")

            if val_dice > best_val_dice:
                best_val_dice = val_dice
                os.makedirs(args.checkpoint_dir, exist_ok=True)
                ckpt_path = os.path.join(args.checkpoint_dir, f"{args.exp_id}_best.pth")
                torch.save(model.state_dict(), ckpt_path)
                log(f"  ✓ 保存最优模型 -> {ckpt_path}  (val_dice={best_val_dice:.4f})")

    # 加载最优模型进行测试集评估
    best_ckpt = os.path.join(args.checkpoint_dir, f"{args.exp_id}_best.pth")
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
        log(f"\n加载最优模型: {best_ckpt}")
    else:
        log("\n未找到checkpoint，使用最终epoch模型评估")

    model.eval()
    dice_metric.reset()
    hd95_metric.reset()
    iou_list = []

    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            if args.out_channels == 1:
                preds_bin = (torch.sigmoid(preds) > 0.5).float()
            else:
                preds_bin = torch.argmax(preds, dim=1, keepdim=True).float()

            dice_metric(y_pred=preds_bin, y=masks)
            hd95_metric(y_pred=preds_bin, y=masks)

            inter = (preds_bin * masks).sum(dim=(2, 3))
            union = ((preds_bin + masks) > 0).float().sum(dim=(2, 3))
            iou_list.append((inter / (union + 1e-8)).mean().item())

    dice = dice_metric.aggregate().item()
    hd95 = hd95_metric.aggregate().item()
    iou  = float(np.mean(iou_list))

    log(f"\n{'='*40}")
    log(f"实验ID: {args.exp_id}")
    log(f"Dice:  {dice:.4f}")
    log(f"IoU:   {iou:.4f}")
    log(f"HD95:  {hd95:.2f}")
    log(f"{'='*40}")

    # 保存结果到 F 盘
    os.makedirs(args.result_dir, exist_ok=True)
    result_file = os.path.join(args.result_dir, f"{args.exp_id}_result.txt")
    with open(result_file, "w") as f:
        f.write(f"exp_id={args.exp_id}\n")
        f.write(f"Dice={dice:.4f}\n")
        f.write(f"IoU={iou:.4f}\n")
        f.write(f"HD95={hd95:.2f}\n")
    log(f"结果已保存: {result_file}")
    log(f"日志已保存: {log_path}")

    log_f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id",           type=str, required=True)
    parser.add_argument("--data_root",        type=str, default="~/isic2016_downstream")
    parser.add_argument("--n_real",           type=int, choices=[40, 100], required=True)
    parser.add_argument("--out_channels",     type=int, default=1)
    parser.add_argument("--max_epochs",       type=int, default=150)
    parser.add_argument("--use_baseline_gen",   action="store_true")
    parser.add_argument("--use_ours_gen",       action="store_true")
    parser.add_argument("--use_controlnet_gen", action="store_true")
    parser.add_argument("--result_dir",       type=str, default="/mnt/f/seg_experiments/results")
    parser.add_argument("--mask_suffix",      type=str, default="_Segmentation.png")
    parser.add_argument("--checkpoint_dir",   type=str, default="/mnt/f/seg_experiments/checkpoints")
    parser.add_argument("--log_dir",          type=str, default="/mnt/f/seg_experiments/logs")
    args = parser.parse_args()
    train(args)