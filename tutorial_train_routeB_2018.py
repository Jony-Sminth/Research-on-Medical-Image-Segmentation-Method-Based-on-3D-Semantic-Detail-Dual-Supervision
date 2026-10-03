# tutorial_train_routeB.py —— 论文2路线B：分割反馈协同
import os
os.environ['DS_NAME'] = 'ISIC2018'  
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from share import *
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import Callback
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from monai.networks.nets import UNet

torch.backends.cudnn.benchmark = False
pl.seed_everything(42, workers=True)
torch.backends.cudnn.deterministic = True


class StateDictSaver(Callback):
    def __init__(self, dirpath, every_n_steps=1000):
        self.dirpath = dirpath
        self.every_n_steps = every_n_steps
        os.makedirs(dirpath, exist_ok=True)
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step > 0 and step % self.every_n_steps == 0:
            sd = {k: v.detach().cpu() for k, v in pl_module.state_dict().items()}
            torch.save(sd, f'{self.dirpath}/ckpt-{step}.ckpt')
            del sd
            torch.cuda.empty_cache()
            print(f'[saver] saved state_dict at step {step}')


def load_frozen_seg(ckpt_path, device='cuda:0'):
    """加载冻结的 S0 分割器"""
    S = UNet(spatial_dims=2, in_channels=3, out_channels=1,
             channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2))
    S.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    S.to(device).eval()
    for p in S.parameters():
        p.requires_grad = False
    return S


# ========== 配置区 ==========
exp_name     = 'E-B_2018'      
model_config = f'./models/{exp_name}.yaml'
output_dir   = f'/home/pc/SiameseDiffusion_outputs/{exp_name}'

resume_path      = './stable-diffusion-v1-5/control_sd15.ckpt'
batch_size       = 1
logger_freq      = 1000
learning_rate    = 1e-5
sd_locked        = False
only_mid_control = False
max_steps        = 6000

S0_CKPT      = '/home/pc/ckpts/S0_2018.pth'  # ← 改
# ============================

model = create_model(model_config).cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=True)
model.learning_rate    = learning_rate
model.sd_locked        = sd_locked
model.only_mid_control = only_mid_control

# 打开梯度检查点（省显存，路线B 必须）
# model.enable_gradient_checkpointing()

# 注入冻结分割器
model.seg_model = load_frozen_seg(S0_CKPT)
print(f'[routeB] frozen seg loaded from {S0_CKPT}')

dataset    = MyDataset()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size,
                        shuffle=True, drop_last=True)

logger = ImageLogger(batch_frequency=logger_freq)

trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,
    precision=16,
    callbacks=[logger, StateDictSaver(f'{output_dir}/checkpoints', every_n_steps=1000)],
    deterministic=True,
    max_steps=max_steps,
    default_root_dir=output_dir,
    enable_checkpointing=False,
)

trainer.fit(model, dataloader)