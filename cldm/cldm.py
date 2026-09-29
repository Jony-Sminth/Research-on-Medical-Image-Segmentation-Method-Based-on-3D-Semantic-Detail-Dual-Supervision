import einops
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config, default
from ldm.models.diffusion.ddim import DDIMSampler

import copy
from cldm.dhi import FeatureExtractor

import numpy as np
from scipy.ndimage import distance_transform_edt

class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        if control is not None:
            h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)


class ControlNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,
            transformer_depth=1,
            context_dim=None,
            n_embed=None,
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self.input_hint_block = TimestepEmbedSequential(
            FeatureExtractor(hint_channels),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        guided_hint = self.input_hint_block(hint, emb, context)

        outs = []

        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs


class ControlLDM(LatentDiffusion):

    def __init__(self, control_stage_config, control_key, only_mid_control, *args, **kwargs):
        # ===== 消融实验超参数：在调用super().__init__之前先pop，避免传递给DDPM =====
        # boundary_type: 'none'(原版均匀) | 'sobel'(本文) | 'distance'(距离变换对比)
        self.boundary_type = kwargs.pop('boundary_type', 'sobel')
        self.alpha_max = kwargs.pop('alpha_max', 4.0)
        self.boundary_warmup_steps = kwargs.pop('boundary_warmup_steps', 0)  # 0=固定alpha
        self.warmup_mode = kwargs.pop('warmup_mode', 'step')  # 
        self.mode = kwargs.pop('mode', 'siamese')  # 默认siamese，现有yaml不受影响
        self.normalize_boundary = kwargs.pop('normalize_boundary', True)
        # ===========================================================================
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        control_mask = batch[self.control_key]
        if bs is not None:
            control_mask = control_mask[:bs]
        control_mask = control_mask.to(self.device)
        control_mask = einops.rearrange(control_mask, 'b h w c -> b c h w')
        control_mask = control_mask.to(memory_format=torch.contiguous_format).float()

        control_image = (batch["jpg"] + 1.0) / 2.0
        if bs is not None:
            control_image = control_image[:bs]
        control_image = control_image.to(self.device)
        control_image = einops.rearrange(control_image, 'b h w c -> b c h w')
        control_image = control_image.to(memory_format=torch.contiguous_format).float()

        return x, dict(c_crossattn=[c], c_concat_mask=[control_mask], c_concat_image=[control_image])

    # =========================================================
    # apply_model：与原版 cldm_linear_backup.py 完全一致，未做任何修改
    # =========================================================
    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1)

        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            if 'c_concat_image' in cond:
                control_model_mask = copy.deepcopy(self.control_model).requires_grad_(False)
                diffusion_model_image = copy.deepcopy(diffusion_model)
                control_weights_mask = 1.0
                control_weights_image = 1.0 * self.global_step / self.trainer.max_steps
                control_image = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat_image'], 1), timesteps=t, context=cond_txt)
                control_image = [c * scale for c, scale in zip(control_image, self.control_scales)]
                with torch.no_grad():
                    control_mask = control_model_mask(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt)
                    control_mask = [c * scale for c, scale in zip(control_mask, self.control_scales)]
                control = [control_weights_mask * c_mask.detach() + control_weights_image * c_image for c_mask, c_image in zip(control_mask, control_image)]
                eps = diffusion_model_image(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)
            else:
                control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt)
                control = [c * scale for c, scale in zip(control, self.control_scales)]
                eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)

        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        c_cat_mask, c_cat_image, c = c["c_concat_mask"][0][:N], c["c_concat_image"][0][:N], c["c_crossattn"][0][:N]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["control_mask"] = c_cat_mask * 2.0 - 1.0
        log["control_image"] = c_cat_image * 2.0 - 1.0
        log["conditioning"] = log_txt_as_img((384, 384), batch[self.cond_stage_key], size=16)

        if plot_diffusion_rows:
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat_mask], "c_crossattn": [c]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat_mask
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat_mask], "c_crossattn": [c]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}_mask"] = x_samples_cfg

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates
    
    def get_boundary_weight_distance(self, mask_control, target_shape, alpha=4.0):
        """
        归一化由 p_losses 调用处统一处理，此处不做。
        """
        mask_gray = mask_control.mean(dim=1, keepdim=True).float()
        mask_binary = (mask_gray > 0.5).squeeze(1).cpu().numpy()  # [B,H,W]

        B = mask_binary.shape[0]
        weight_maps = []

        for b in range(B):
            m = mask_binary[b].astype(np.float32)
            dist_fg = distance_transform_edt(m)
            dist_bg = distance_transform_edt(1.0 - m)
            dist_combined = dist_fg + dist_bg          # ← sum，不是minimum

            max_val = dist_combined.max()
            if max_val > 1e-8:
                boundary = 1.0 - dist_combined / max_val   # ← 线性反转
            else:
                boundary = np.zeros_like(dist_combined)

            boundary = np.power(boundary.clip(0, 1), 0.5)  # ← gamma=0.5，不能省

            weight_map = torch.from_numpy(boundary).float().to(mask_control.device)
            weight_maps.append(weight_map)

        weight_map = torch.stack(weight_maps, dim=0).unsqueeze(1)  # [B,1,H,W]
        weight_map = 1.0 + alpha * weight_map
        weight_map = F.interpolate(weight_map, size=target_shape,
                                mode='bilinear', align_corners=False)
        return weight_map
    def get_boundary_weight(self, mask_control, target_shape, alpha=4.0):
        """
        Sobel 边界权重图（纯 GPU，无 CPU 同步）
        Args:
            mask_control : [B, C, H, W]  掩码控制输入
            target_shape : (H_lat, W_lat) 潜在空间目标尺寸
            alpha        : 边界处额外权重倍数
        Returns:
            weight_map   : [B, 1, H_lat, W_lat]，内部≈1，边界最高≈1+alpha
        """
        mask_gray = mask_control.mean(dim=1, keepdim=True).float()  # [B,1,H,W]

        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32, device=mask_control.device
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32, device=mask_control.device
        ).view(1, 1, 3, 3)

        gx = F.conv2d(mask_gray, sobel_x, padding=1)
        gy = F.conv2d(mask_gray, sobel_y, padding=1)
        grad_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

        # 逐样本归一化到 [0,1]
        grad_max = grad_mag.amax(dim=[2, 3], keepdim=True).clamp(min=1e-8)
        boundary = grad_mag / grad_max

        weight_map = 1.0 + alpha * boundary  # [B,1,H,W]
        weight_map = F.interpolate(
            weight_map, size=target_shape, mode='bilinear', align_corners=False
        )
        return weight_map  # [B,1,H_lat,W_lat]

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()

    def p_losses(self, x_start, cond, t, noise=None):

        cond_mask = {}
        cond_mask["c_crossattn"] = [cond["c_crossattn"][0]]
        cond_mask["c_concat"] = [cond["c_concat_mask"][0]]

        cond_image = {}
        cond_image["c_crossattn"] = [cond["c_crossattn"][0]]
        cond_image["c_concat"] = [cond["c_concat_mask"][0]]
        cond_image["c_concat_image"] = [cond["c_concat_image"][0]]

        weights_ones = torch.ones_like(t).to(x_start.device)
        weights_thre = torch.where(t <= 200, torch.tensor(1), torch.tensor(0))

        weights_mask = 1.0 * weights_ones               # Loss 0
        weights_image = 1.0 * weights_ones              # Loss 1
        weights_mask_2_image = 1.0 * weights_ones       # Loss 2
        weights_mask_regularization = 1.0 * weights_thre  # Loss 3

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output_mask = self.apply_model(x_noisy, t, cond_mask)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        # Loss 0
        loss_simple = weights_mask * self.get_loss(model_output_mask, target, mean=False).mean([1, 2, 3])
        print(f"loss_simple_mask: {loss_simple.mean():.6f}")
        
        if self.mode == 'siamese':
            # Loss 1
            if weights_image.all():
                model_output_image = self.apply_model(x_noisy, t, cond_image)
                loss_simple_image = self.get_loss(model_output_image, target, mean=False).mean([1, 2, 3])
                print(f"loss_simple_image: {loss_simple_image.mean():.6f}")
                loss_simple = loss_simple + weights_image * loss_simple_image

            # =========================================================
            # Loss 2：边界感知噪声一致性损失（本文唯一改动）
            # 原版：均匀加权 MSE(mask_output, image_output.detach())
            # 改进：对皮肤病变边界区域施加更强权重
            # 注意：model_output_image.detach() 与原版一致，
            #       boundary_weight 在 no_grad 下计算，不参与反传，
            #       梯度路径与原版完全相同，只是边界处 loss 更大。
            # =========================================================
            # if weights_mask_2_image.all():
            #     # 逐像素误差，保留空间维度 [B, C, H_lat, W_lat]
            #     pixel_loss = self.get_loss(
            #         model_output_mask,
            #         model_output_image.detach(),  # 与原版一致：detach image 路
            #         mean=False
            #     )

            #     # 边界权重图，全程在 GPU 上，no_grad
            #     alpha_max = 3.0         # 由消融实验一确定（你当前用的是4.0，不是文档里的2.0，注意对齐）
            #     warmup_steps = 1000     # 由消融实验二确定（候选 500/1000/1500）
            #     alpha = alpha_max * min(self.global_step / warmup_steps, 1.0)
            #     with torch.no_grad():
            #         boundary_weight = self.get_boundary_weight(
            #             cond["c_concat_mask"][0],
            #             pixel_loss.shape[-2:],
            #             alpha=alpha
            #         )
            #         # 归一化，保持总损失量级与原版一致
            #         boundary_weight = boundary_weight / boundary_weight.mean(dim=[2, 3], keepdim=True)
            #     # with torch.no_grad():
            #     #     boundary_weight = self.get_boundary_weight(
            #     #         cond["c_concat_mask"][0],  # [B, C, H, W]
            #     #         pixel_loss.shape[-2:],     # (H_lat, W_lat)
            #     #         alpha=4.0
            #     #     )  # [B, 1, H_lat, W_lat]，广播到 [B, C, H_lat, W_lat]

            #     loss_simple_mask_2_image = (boundary_weight * pixel_loss).mean([1, 2, 3])
            #     print(f"loss_simple_mask_2_image(boundary-aware): {loss_simple_mask_2_image.mean():.6f}")
            #     loss_simple = loss_simple + weights_mask_2_image * loss_simple_mask_2_image
            # =========================================================
            # Loss 2：边界感知噪声一致性损失（距离变换版，固定alpha，含诊断打印）
            # =========================================================
            # =========================================================
            # Loss 2：边界感知噪声一致性损失
            # =========================================================
            if weights_mask_2_image.all():
                pixel_loss = self.get_loss(
                    model_output_mask,
                    model_output_image.detach(),
                    mean=False
                )  # [B, C, H_lat, W_lat]

                # alpha 调度
                if self.boundary_type in ('sobel', 'distance'):
                    if self.warmup_mode == 't':
                        alpha = self.alpha_max * (1.0 - t.float().mean().item() / 1000.0)
                    else:
                        if self.boundary_warmup_steps > 0:
                            alpha = self.alpha_max * min(self.global_step / self.boundary_warmup_steps, 1.0)
                        else:
                            alpha = self.alpha_max

                    with torch.no_grad():
                        if self.boundary_type == 'sobel':
                            boundary_weight = self.get_boundary_weight(
                                cond["c_concat_mask"][0], pixel_loss.shape[-2:], alpha=alpha
                            )
                        else:  # distance
                            boundary_weight = self.get_boundary_weight_distance(
                                cond["c_concat_mask"][0], pixel_loss.shape[-2:], alpha=alpha
                            )
                        # ===== E-12 诊断打印（只打前5步，拿到数据后删掉） =====
                        # if self.global_step < 5:
                        #     peak = boundary_weight.amax(dim=[2, 3]).mean().item()
                        #     mean = boundary_weight.mean(dim=[2, 3]).mean().item()
                        #     print(f"[DIAG|{self.boundary_type}] step={self.global_step} | peak_before_norm={peak:.4f} | mean_before_norm={mean:.4f} | ratio={peak/mean:.4f}")
                        
                        
                        # 归一化，保持总损失量级与原版一致（在 no_grad 里）
                        if self.normalize_boundary:
                            boundary_weight = boundary_weight / boundary_weight.mean(dim=[2, 3], keepdim=True)

                    loss_simple_mask_2_image = (boundary_weight * pixel_loss).mean([1, 2, 3])

                else:  # boundary_type == 'none'，原版均匀加权
                    loss_simple_mask_2_image = pixel_loss.mean([1, 2, 3])

                loss_simple = loss_simple + weights_mask_2_image * loss_simple_mask_2_image
            # Loss 3（与原版完全一致）
            if (self.global_step > (self.trainer.max_steps * 1 / 3)) and weights_mask_regularization.any():
                recon_output_image = self.predict_start_from_noise(x_noisy, t=t, noise=model_output_image)
                noise_image_2_mask = default(noise, lambda: torch.randn_like(recon_output_image))
                x_noisy_mask_recon = self.q_sample(x_start=recon_output_image, t=t, noise=noise_image_2_mask)

                model_output_mask_xt = self.apply_model(x_noisy_mask_recon.detach(), t, cond_mask)
                loss_simple_mask_regularization = self.get_loss(model_output_mask_xt, noise_image_2_mask, mean=False).mean([1, 2, 3])
                print(f"loss_simple_mask_regularization: {loss_simple_mask_regularization.mean():.6f}")
                loss_simple = loss_simple + weights_mask_regularization * loss_simple_mask_regularization

        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = loss_simple
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict