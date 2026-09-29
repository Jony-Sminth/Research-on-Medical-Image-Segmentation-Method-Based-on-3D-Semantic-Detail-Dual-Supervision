# inference2.py —— 论文二路线A：多种子候选池生成
# 关键修正：每个 mask 用独立种子，避免 K=1 时 900 张图初始噪声相同导致 FID 虚高
import os
import argparse
import torch
from share import *
import numpy as np
from PIL import Image
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset_sample import MyDataset
from cldm.model import create_model, load_state_dict

pl.seed_everything(0, workers=True)

BATCH_SIZE = 1
DDIM_STEPS = 50
DDIM_ETA = 0.0
LEARNING_RATE = 1e-5
SD_LOCKED = False
ONLY_MID_CONTROL = False
CFG_SCALE = 9.0

DEFAULT_OUT_ROOT = '/mnt/f/240493014/paper2/gen_pools'
DEFAULT_CKPT_ROOT = '/mnt/d/SiameseDiffusion_outputs'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--exp_name', default='E-10_sobel_a3_1000')
    p.add_argument('--ckpt', default=None)
    p.add_argument('--K', type=int, default=4)
    p.add_argument('--seed_base', type=int, default=1000)
    p.add_argument('--out_dir', default=None)
    p.add_argument('--out_root', default=DEFAULT_OUT_ROOT)
    p.add_argument('--ckpt_root', default=DEFAULT_CKPT_ROOT)
    p.add_argument('--limit', type=int, default=0, help='只生成前 N 张，0=全部')
    return p.parse_args()


def get_model(model_config, ckpt_path):
    model = create_model(model_config).cpu()
    model.load_state_dict(load_state_dict(ckpt_path, location='cpu'), strict=False)
    model.learning_rate = LEARNING_RATE
    model.sd_locked = SD_LOCKED
    model.only_mid_control = ONLY_MID_CONTROL
    model.to("cuda:0")
    model.eval()
    return model


def save_image(tensor, path):
    img = (tensor + 1.0) / 2.0
    img = img.permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(img).save(path)


def save_mask(tensor, path):
    m = tensor.permute(1, 2, 0).squeeze(-1).numpy()
    m = (m * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(m).convert('1').save(path)


def main():
    args = parse_args()

    model_config = f'./models/{args.exp_name}.yaml'
    ckpt_path = args.ckpt or (
        f'{args.ckpt_root}/{args.exp_name}/lightning_logs/version_0/checkpoints/'
        f'epoch=3-step=3000.ckpt'
    )
    result_dir = args.out_dir or os.path.join(args.out_root, args.exp_name)
    img_root = os.path.join(result_dir, 'images')
    mask_root = os.path.join(result_dir, 'masks')
    os.makedirs(img_root, exist_ok=True)
    os.makedirs(mask_root, exist_ok=True)

    assert os.path.exists(model_config), f'yaml 不存在: {model_config}'
    assert os.path.exists(ckpt_path),   f'ckpt 不存在: {ckpt_path}'

    img_key = f'samples_cfg_scale_{CFG_SCALE:.2f}_mask'
    print(f'[inference2] exp={args.exp_name}')
    print(f'[inference2] cfg={model_config}')
    print(f'[inference2] ckpt={ckpt_path}')
    print(f'[inference2] out ={result_dir}')
    print(f'[inference2] K   ={args.K}')
    print(f'[inference2] limit={args.limit}')

    with torch.cuda.device(0):
        model = get_model(model_config, ckpt_path)

        dataset = MyDataset()
        dataloader = DataLoader(dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=False)

        total = len(dataloader)
        with torch.no_grad():
            for idx, batch in enumerate(dataloader):
                if args.limit > 0 and idx >= args.limit:
                    break

                for k in range(args.K):
                    # ★ 关键修正：用 idx 参与种子，保证每张图初始噪声不同
                    #   idx*1000 给每个 mask 一个独立区间，+k 区分 K 张候选
                    pl.seed_everything(args.seed_base + idx * 1000 + k, workers=True)

                    with model.ema_scope():
                        images = model.log_images(
                            batch,
                            N=BATCH_SIZE,
                            ddim_steps=DDIM_STEPS,
                            ddim_eta=DDIM_ETA,
                        )

                    for key in images:
                        if isinstance(images[key], torch.Tensor):
                            images[key] = torch.clamp(
                                images[key].detach().cpu(), -1.0, 1.0
                            )

                    save_image(images[img_key][0],
                               os.path.join(img_root, f'{idx:06}_k{k}.png'))

                    if k == 0:
                        save_mask(images['control_mask'][0],
                                  os.path.join(mask_root, f'{idx:06}.png'))

                if idx % 20 == 0:
                    print(f'[{idx}/{total}] done  (K={args.K})')

    print(f'[inference2] finished. images -> {img_root}')
    print(f'[inference2] finished. masks  -> {mask_root}')


if __name__ == "__main__":
    main()