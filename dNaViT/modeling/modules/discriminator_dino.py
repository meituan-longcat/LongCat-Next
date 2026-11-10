# Copyright (c) 2026 LongCat dNaViT Team.
# SPDX-License-Identifier: Apache-2.0

# ref: https://github.com/CVMI-Lab/Hita/blob/e67a55ea80101f5c1855d4fea39553a248ee1c16/hita/tokenizer/tokenizer_image/discriminator_dino.py
import numpy as np
import os.path as osp
import torch.nn as nn
from typing import List, Tuple
import torch.nn.functional as F
import math, random, torch, pdb
from torchvision.transforms import RandomCrop
from torch.nn.utils.spectral_norm import SpectralNorm

# import dist

try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
    from flash_attn.ops.fused_dense import fused_mlp_func
except:
    dropout_add_layer_norm = fused_mlp_func = None

try:
    from flash_attn import flash_attn_qkvpacked_func  # qkv: BL3Hc, ret: BLHcq
except:
    flash_attn_qkvpacked_func = None

try:
    assert torch.cuda.is_available()
    from torch.nn.functional import scaled_dot_product_attention as slow_attn  # q, k, v: BHLc
except:
    def slow_attn(query, key, value, scale: float, attn_mask=None, dropout_p=0.0):
        attn = query.mul(scale) @ key.transpose(-2, -1)  # BHLc @ BHcL => BHLL
        if attn_mask is not None: attn.add_(attn_mask)
        return (F.dropout(attn.softmax(dim=-1), p=dropout_p, inplace=True) if dropout_p > 0 else attn.softmax(
            dim=-1)) @ value


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

class MLPNoDrop(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, fused_if_available=True):
        super().__init__()
        self.fused_mlp_func = fused_mlp_func if (torch.cuda.is_available() and fused_if_available) else None
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        if self.fused_mlp_func is not None:
            return self.fused_mlp_func(
                x=x,
                weight1=self.fc1.weight,
                weight2=self.fc2.weight,
                bias1=self.fc1.bias,
                bias2=self.fc2.bias,
                activation='gelu_approx',
                save_pre_act=self.training,
                return_residual=False,
                checkpoint_lvl=0,
                heuristic=0,
                process_group=None,
            )
        else:
            return self.fc2(self.act(self.fc1(x)))

    def extra_repr(self) -> str:
        return f'fused_mlp_func={self.fused_mlp_func is not None}'


class SelfAttentionNoDrop(nn.Module):
    def __init__(
            self, block_idx, embed_dim=768, num_heads=12, flash_if_available=True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.block_idx, self.num_heads, self.head_dim = block_idx, num_heads, embed_dim // num_heads  # =64
        self.scale = 1 / math.sqrt(self.head_dim)
        self.qkv, self.proj = nn.Linear(embed_dim, embed_dim * 3, bias=True), nn.Linear(embed_dim, embed_dim, bias=True)
        self.using_flash_attn = torch.cuda.is_available() and flash_if_available and flash_attn_qkvpacked_func is not None

    def forward(self, x):
        B, L, C = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.num_heads, self.head_dim)
        if self.using_flash_attn and qkv.dtype != torch.float32:
            oup = flash_attn_qkvpacked_func(qkv, softmax_scale=self.scale).view(B, L, C)
        else:
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)  # BHLc
            oup = slow_attn(query=q, key=k, value=v, scale=self.scale).transpose(1, 2).reshape(B, L, C)
        return self.proj(oup)

    def extra_repr(self) -> str:
        return f'using_flash_attn={self.using_flash_attn}'


class SABlockNoDrop(nn.Module):
    def __init__(self, block_idx, embed_dim, num_heads, mlp_ratio, norm_eps):
        super(SABlockNoDrop, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=norm_eps)
        self.attn = SelfAttentionNoDrop(block_idx=block_idx, embed_dim=embed_dim, num_heads=num_heads,
                                        flash_if_available=True)
        self.norm2 = nn.LayerNorm(embed_dim, eps=norm_eps)
        self.mlp = MLPNoDrop(in_features=embed_dim, hidden_features=round(embed_dim * mlp_ratio),
                             fused_if_available=True)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ResidualBlock(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.ratio = 1 / np.sqrt(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = x.float()
        return (self.fn(x).add(x)).mul_(self.ratio)


class SpectralConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        SpectralNorm.apply(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12)


class BatchNormLocal(nn.Module):
    def __init__(self, num_features: int, affine: bool = True, virtual_bs: int = 8, eps: float = 1e-6):
        super().__init__()
        self.virtual_bs = virtual_bs
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()                         # (B, C, L)
        x = x.float()
        B = shape[0]

        # Group the batch into chunks of ``virtual_bs`` for "virtual batch norm".
        # The original code used ``view(ceil(B/virtual_bs), -1, C, L)``, which
        # silently assumes ``B`` is a multiple of the group count. Under
        # token-budget (anyres) batching ``B`` is arbitrary (e.g. 19, 22, 23),
        # so that view raises ``shape '[3, -1, C, L]' is invalid``. Pad the batch
        # up to a multiple of ``virtual_bs`` so the grouping is always exact, then
        # drop the padding afterwards. When ``B`` already divides ``virtual_bs``
        # this is a no-op and reproduces the original behaviour bit-for-bit.
        vbs = min(self.virtual_bs, B)
        pad = (-B) % vbs
        if pad:
            x = torch.cat([x, x[:pad]], dim=0)
        Bp = x.size(0)
        G = Bp // vbs
        x = x.view(G, vbs, shape[-2], shape[-1])

        # Calculate stats.
        mean = x.mean([1, 3], keepdim=True)
        var = x.var([1, 3], keepdim=True, unbiased=False)
        x = (x - mean) / (torch.sqrt(var + self.eps))

        # Back to (Bp, C, L) and drop the padding rows.
        x = x.view(Bp, shape[-2], shape[-1])[:B]

        if self.affine:
            x = x * self.weight[None, :, None] + self.bias[None, :, None]

        return x.view(shape)


def make_block(channels: int, kernel_size: int, norm_type: str, norm_eps: float, using_spec_norm: bool) -> nn.Module:
    if norm_type == 'bn':
        norm = BatchNormLocal(channels, eps=norm_eps)
    elif norm_type == 'sbn':
        norm = nn.SyncBatchNorm(channels, eps=norm_eps, process_group=None)
    # elif norm_type in {'lbn', 'hbn'}:
    #     norm = nn.SyncBatchNorm(channels, eps=norm_eps, process_group=dist.new_local_machine_group())
    elif norm_type == 'gn':
        norm = nn.GroupNorm(num_groups=32, num_channels=channels, eps=norm_eps, affine=True)
    else:
        raise NotImplementedError

    return nn.Sequential(
        (SpectralConv1d if using_spec_norm else nn.Conv1d)(channels, channels, kernel_size=kernel_size,
                                                           padding=kernel_size // 2, padding_mode='circular'),
        norm,
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
    )


class DinoDisc(nn.Module):
    def __init__(self, dino_ckpt_path='https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth',
                 device='cuda', ks=9, depth=12, key_depths=(2, 5, 8, 11), norm_type='bn',
                 using_spec_norm=True, norm_eps=1e-6):
        super().__init__()
        # load state (accepts a local .pth file or an http(s) URL)
        if str(dino_ckpt_path).startswith(("http://", "https://")):
            state = torch.hub.load_state_dict_from_url(dino_ckpt_path, map_location="cpu")
        else:
            state = torch.load(dino_ckpt_path, map_location="cpu")
        for k in sorted(state.keys()):
            if '.attn.qkv.bias' in k:
                bias = state[k]
                C = bias.numel() // 3
                bias[C:2 * C].zero_()  # zero out k_bias
        # build DINO
        key_depths = tuple(d for d in key_depths if d < depth)
        d = FrozenDINOSmallNoDrop(
            patch_size=8 if "patch8" in dino_ckpt_path else 16, depth=depth, key_depths=key_depths, norm_eps=norm_eps)
        missing, unexpected = d.load_state_dict(state, strict=False)
        missing = [m for m in missing if all(x not in m for x in {
            'x_scale', 'x_shift',
        })]
        if torch.cuda.is_available():
            assert len(missing) == 0, f'missing keys: {missing}'
            assert len(unexpected) == 0, f'unexpected keys: {unexpected}'

        requires_grad(d, False)
        # todo: don't compile! reduce-overhead would raise CudaERR
        self.dino_proxy: Tuple[FrozenDINOSmallNoDrop] = (d.to(device=device),)
        dino_C = self.dino_proxy[0].embed_dim
        # if 'KEVIN_LOCAL' in os.environ:
        #     torch.manual_seed(0)
        #     np.random.seed(0)
        #     random.seed(0)
        self.heads = nn.ModuleList([
            nn.Sequential(
                make_block(dino_C, kernel_size=1, norm_type=norm_type, norm_eps=norm_eps,
                           using_spec_norm=using_spec_norm),
                ResidualBlock(make_block(dino_C, kernel_size=ks, norm_type=norm_type, norm_eps=norm_eps,
                                         using_spec_norm=using_spec_norm)),
                (SpectralConv1d if using_spec_norm else nn.Conv1d)(dino_C, 1, kernel_size=1, padding=0)
            )
            for _ in range(len(key_depths) + 1)  # +1: before all attention blocks
        ])

    def reinit(self, dino_ckpt_path='https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth',
                 device='cuda', ks=9, depth=12, key_depths=(2, 5, 8, 11), norm_type='bn',
                 using_spec_norm=True, norm_eps=1e-6):
        dino_C = self.dino_proxy[0].embed_dim
        heads = nn.ModuleList([
            nn.Sequential(
                make_block(dino_C, kernel_size=1, norm_type=norm_type, norm_eps=norm_eps,
                           using_spec_norm=using_spec_norm),
                ResidualBlock(make_block(dino_C, kernel_size=ks, norm_type=norm_type, norm_eps=norm_eps,
                                         using_spec_norm=using_spec_norm)),
                (SpectralConv1d if using_spec_norm else nn.Conv1d)(dino_C, 1, kernel_size=1, padding=0)
            )
            for _ in range(len(key_depths) + 1)
        ])

        self.heads.load_state_dict(heads.state_dict())


    def main(self, x_in_pm1, grad_ckpt=False):  # x_in_pm1: image tensor normalized to [-1, 1]
        dino_grad_ckpt = grad_ckpt and x_in_pm1.requires_grad
        FrozenDINOSmallNoDrop.forward
        activations: List[torch.Tensor] = self.dino_proxy[0](x_in_pm1.float(), grad_ckpt=dino_grad_ckpt)
        B = x_in_pm1.shape[0]
        with torch.autocast('cuda', dtype=torch.float32, enabled=True):
            x = torch.cat([
                (
                    h(act) if not grad_ckpt
                    else torch.utils.checkpoint.checkpoint(h, act, use_reentrant=False)
                ).view(B, -1)
                for h, act in zip(self.heads, activations)
            ], dim=1)  # cat 5 BL => B, 5L

        return x

    def forward(self, input):
        """Standard forward."""
        return self.main(input)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)  # BCHW => BCL => BLC
        return self.norm(x)


class FrozenDINOSmallNoDrop(nn.Module):
    """
    Frozen DINO ViT without any dropout or droppath layers (eval node only), based on timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=0)

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(
            self, depth=12, key_depths=(2, 5, 8, 11), norm_eps=1e-6,  # 4 stages: 012, 345, 678, 9 10 11
            patch_size=16, in_chans=3, num_classes=0,
            embed_dim=384, num_heads=6, mlp_ratio=4.,
            # drop_rate=0., attn_drop_rate=0., drop_path_rate=0.    # no drop for frozen model
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.img_size = 224
        self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=patch_size, in_chans=in_chans,
                                      embed_dim=embed_dim)
        self.patch_size = patch_size
        self.patch_nums = self.img_size // patch_size

        # x \in [-1, 1]
        # x = ((x+1)/2 - m) / s = 0.5x/s + 0.5/s - m/s = (0.5/s) x + (0.5-m)/s
        m, s = torch.tensor((0.485, 0.456, 0.406)), torch.tensor((0.229, 0.224, 0.225))
        self.register_buffer('x_scale', (0.5 / s).reshape(1, 3, 1, 1))
        self.register_buffer('x_shift', ((0.5 - m) / s).reshape(1, 3, 1, 1))
        self.crop = RandomCrop(self.img_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = None
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_nums * self.patch_nums + 1, embed_dim))  # +1: for cls
        # self.pos_drop = nn.Dropout(p=drop_rate)
        # self.pos_pool = dict()

        self.key_depths = set(d for d in key_depths if d < depth)
        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # no drop for frozen model
        self.blocks = nn.Sequential(*[
            SABlockNoDrop(block_idx=i, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, norm_eps=norm_eps)
            for i in range(max(depth, 1 + max(self.key_depths)))
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=norm_eps)

        # eval mode only
        self.eval()
        [p.requires_grad_(False) for p in self.parameters()]

    def inter_pos_embed(self, patch_nums=(14, 14)):
        if patch_nums[0] == self.patch_nums and patch_nums[1] == self.patch_nums:
            return self.pos_embed
        pe_cls, pe_grid = self.pos_embed[:, :1], self.pos_embed[0, 1:]
        pe_grid = pe_grid.reshape(1, self.patch_nums, self.patch_nums, -1).permute(0, 3, 1, 2)
        pe_grid = F.interpolate(pe_grid, size=(patch_nums[0], patch_nums[1]), mode='bilinear', align_corners=False)
        pe_grid = pe_grid.permute(0, 2, 3, 1).reshape(1, patch_nums[0] * patch_nums[1], -1)
        return torch.cat([pe_cls, pe_grid], dim=1)

    def forward(self, x, grad_ckpt=False):
        # in RAE, x is directly resized to 224x224
        with torch.cuda.amp.autocast(enabled=False):
            x = (self.x_scale * x.float()).add_(self.x_shift)
            H, W = x.shape[-2], x.shape[-1]
            if H > self.img_size and W > self.img_size and random.random() <= 0.5:
                x = self.crop(x)
            else:
                x = F.interpolate(x, size=(self.img_size, self.img_size),
                                  mode='area' if H > self.img_size else 'bicubic')
            # x now must be self.img_size x self.img_size

        # patch_nums = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        # x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), self.patch_embed(x)), dim=1)
        # if patch_nums in self.pos_pool:
        #     x += self.pos_pool[patch_nums]
        # else:
        #     self.pos_pool[patch_nums] = pe = self.inter_pos_embed(patch_nums)
        #     x += pe
        # x = self.pos_drop(x)

        x = self.patch_embed(x)

        with torch.cuda.amp.autocast(enabled=False):
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x.float()), dim=1)
            x = x + self.pos_embed
            activations = [(x[:, 1:] + x[:, :1]).transpose_(1, 2)]  # readout
        for i, b in enumerate(self.blocks):
            if not grad_ckpt:
                x = b(x)
            else:
                x = torch.utils.checkpoint.checkpoint(b, x, use_reentrant=False)
            if i in self.key_depths:
                activations.append((x[:, 1:].float() + x[:, :1].float()).transpose_(1, 2))  # readout
        # x = self.norm(x)
        return activations


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    ks = 9
    norm_type = 'sbn'
    norm_eps = 1e-6
    dino_C = 384
    key_layers = (2, 5, 8, 11)
    using_spec_norm = True

    heads = nn.ModuleList([
        nn.Sequential(
            make_block(dino_C, kernel_size=1, norm_type=norm_type, norm_eps=norm_eps, using_spec_norm=using_spec_norm),
            ResidualBlock(make_block(dino_C, kernel_size=ks, norm_type=norm_type, norm_eps=norm_eps,
                                     using_spec_norm=using_spec_norm)),
            (SpectralConv1d if using_spec_norm else nn.Conv1d)(dino_C, 1, kernel_size=1, padding=0)
        )
        for _ in range(len(key_layers) + 1)
    ])

    ckpt = 'https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth'
    DinoDisc.forward
    dd = DinoDisc(dino_ckpt_path=ckpt, device='cpu', ks=ks, norm_type=norm_type, norm_eps=norm_eps, key_depths=key_layers)
    dd.eval()
    dd.heads.load_state_dict(heads.state_dict())
    print(f'{sum(p.numel() for p in dd.parameters() if p.requires_grad) / 1e6:.2f}M')
    inp = torch.linspace(-2, 2, 2 * 3 * 224 * 224).reshape(2, 3, 224, 224)
    inp.requires_grad = True
    cond = torch.rand(2, 64)
    mid_ls = dd.dino_proxy[0](inp)
    means = [round(m.mean().item(), 3) for m in mid_ls]
    stds = [round(m.std().item(), 3) for m in mid_ls]
    print(f'mean: {means}')
    print(f'std: {stds}')

    o = dd(inp, grad_ckpt=True)
    print(f'o: {o.abs().mean().item():.9f}, {o.abs().std().item():.9f}')
    o.abs().mean().backward()

    # for n, p in dd.named_parameters():
    #     tag = n.split('heads.')[-1][0]
    #     if p.ndim == 3: tag += '.conv1d'
    #     print(f'[{tag}] {n}: {p.shape}')