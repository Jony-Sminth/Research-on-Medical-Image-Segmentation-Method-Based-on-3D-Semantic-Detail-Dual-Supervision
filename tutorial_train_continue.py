import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from share import *
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

torch.backends.cudnn.benchmark = False
pl.seed_everything(42, workers=True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# ========== 配置区 ==========
exp_name         = 'E-10_sobel_a3_1000'
model_config     = f'./models/{exp_name}.yaml'
output_dir       = f'/home/pc/SiameseDiffusion_outputs/{exp_name}'

resume_path      = './stable-diffusion-v1-5/control_sd15.ckpt'
batch_size       = 1
logger_freq      = 400
learning_rate    = 1e-5
sd_locked        = False
only_mid_control = False
max_steps        = 3000

# resume 路径：填你实际要接着跑的 ckpt
resume_ckpt      = '/home/pc/SiameseDiffusion_outputs/E-10_sobel_a3_1000/checkpoints/step=step=1000.ckpt'
# ============================

model = create_model(model_config).cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=True)
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

dataset    = MyDataset()
dataloader = DataLoader(
    dataset,
    num_workers=0,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True
)

logger = ImageLogger(batch_frequency=logger_freq)

ckpt_cb = ModelCheckpoint(
    dirpath=f'{output_dir}/checkpoints',
    filename='ckpt-{step}',         # 避免 step=step= 重复
    every_n_train_steps=1000,
    save_top_k=-1,
    save_last=True,
)

trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,
    precision=16,
    callbacks=[logger, ckpt_cb],
    deterministic=True,
    max_steps=max_steps,
    default_root_dir=output_dir,
)

trainer.fit(
    model,
    dataloader,
    ckpt_path=resume_ckpt,          # ← 关键
)