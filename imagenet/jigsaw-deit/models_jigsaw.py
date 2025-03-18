# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial
import numpy as np
import torch.nn.functional as F

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

import einops
from munch import Munch


class FlexiblePatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.default_patch_size = patch_size  # Default patch size (e.g., 16)

        # Create a max-size convolution kernel and resize dynamically
        self.max_patch_size = 48  # Maximum patch size allowed
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.max_patch_size, stride=self.max_patch_size)

    def forward(self, x, patch_size):
        """
        Args:
            x (Tensor): Input image tensor of shape (B, C, H, W)
            patch_size (int): The flexible patch size to use for this forward pass
        Returns:
            Tensor: Patch embeddings of shape (B, N, D)
        """
        # Resize the kernel dynamically to match requested patch_size
        resized_weight = F.interpolate(self.proj.weight.to(dtype=x.dtype), size=(patch_size, patch_size), mode='bilinear',
                                       align_corners=False)
        resized_proj = nn.Conv2d(self.in_chans, self.embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)

        # # 2. Set device/dtype AFTER creation
        # resized_proj = resized_proj.to(device=x.device, dtype=x.dtype)


        resized_proj.weight = nn.Parameter(resized_weight)

        # 3. Handle bias separately
        if self.proj.bias is not None:
            resized_proj.bias = nn.Parameter(self.proj.bias.clone())

        # 4. Ensure it’s on the right device and dtype
        resized_proj = resized_proj.to(device=x.device, dtype=x.dtype)

        x = resized_proj(x)  # Apply convolution
        x = x.flatten(2).transpose(1, 2)  # Convert to sequence form
        return x

# def resample_patch_embed(
#         patch_embed,
#         new_size: List[int],
#         interpolation: str = 'bicubic',
#         antialias: bool = True,
#         verbose: bool = False,
# ):
#     """Resample the weights of the patch embedding kernel to target resolution.
#     We resample the patch embedding kernel by approximately inverting the effect
#     of patch resizing.
#
#     Code based on:
#       https://github.com/google-research/big_vision/blob/b00544b81f8694488d5f36295aeb7972f3755ffe/big_vision/models/proj/flexi/vit.py
#
#     With this resizing, we can for example load a B/8 filter into a B/16 model
#     and, on 2x larger input image, the result will match.
#
#     Args:
#         patch_embed: original parameter to be resized.
#         new_size (tuple(int, int): target shape (height, width)-only.
#         interpolation (str): interpolation for resize
#         antialias (bool): use anti-aliasing filter in resize
#         verbose (bool): log operation
#     Returns:
#         Resized patch embedding kernel.
#     """
#     import numpy as np
#     try:
#         from torch import vmap
#     except ImportError:
#         from functorch import vmap
#
#     assert len(patch_embed.shape) == 4, "Four dimensions expected"
#     assert len(new_size) == 2, "New shape should only be hw"
#     old_size = patch_embed.shape[-2:]
#     if tuple(old_size) == tuple(new_size):
#         return patch_embed
#
#     if verbose:
#         _logger.info(f"Resize patch embedding {patch_embed.shape} to {new_size}, w/ {interpolation} interpolation.")
#
#     def resize(x_np, _new_size):
#         x_tf = torch.Tensor(x_np)[None, None, ...]
#         x_upsampled = F.interpolate(
#             x_tf, size=_new_size, mode=interpolation, antialias=antialias)[0, 0, ...].numpy()
#         return x_upsampled
#
#     def get_resize_mat(_old_size, _new_size):
#         mat = []
#         for i in range(np.prod(_old_size)):
#             basis_vec = np.zeros(_old_size)
#             basis_vec[np.unravel_index(i, _old_size)] = 1.
#             mat.append(resize(basis_vec, _new_size).reshape(-1))
#         return np.stack(mat).T
#
#     resize_mat = get_resize_mat(old_size, new_size)
#     resize_mat_pinv = torch.tensor(np.linalg.pinv(resize_mat.T), device=patch_embed.device)
#
#     def resample_kernel(kernel):
#         resampled_kernel = resize_mat_pinv @ kernel.reshape(-1)
#         return resampled_kernel.reshape(new_size)
#
#     v_resample_kernel = vmap(vmap(resample_kernel, 0, 0), 1, 1)
#     orig_dtype = patch_embed.dtype
#     patch_embed = patch_embed.float()
#     patch_embed = v_resample_kernel(patch_embed)
#     patch_embed = patch_embed.to(orig_dtype)
#     return patch_embed


# def divs(n, m=None):
#     m = m or n // 2
#     if m == 1:
#         return [1]
#     if n % m == 0:
#         return [m] + divs(n, m - 1)
#     return divs(n, m - 1)
#
#
# class FlexiPatchEmbed(nn.Module):
#     """ 2D Image to Patch Embedding w/ Flexible Patch sizes (FlexiViT)
#     FIXME WIP
#     """
#     def __init__(
#             self,
#             img_size=240,
#             patch_size=16,
#             in_chans=3,
#             embed_dim=768,
#             base_img_size=240,
#             base_patch_size=32,
#             norm_layer=None,
#             flatten=True,
#             bias=True,
#     ):
#         super().__init__()
#         self.img_size = to_2tuple(img_size)
#         self.patch_size = to_2tuple(patch_size)
#         self.num_patches = 0
#
#         # full range for 240 = (5, 6, 8, 10, 12, 14, 15, 16, 20, 24, 30, 40, 48)
#         self.seqhw = (6, 8, 10, 12, 14, 15, 16, 20, 24, 30)
#
#         self.base_img_size = to_2tuple(base_img_size)
#         self.base_patch_size = to_2tuple(base_patch_size)
#         self.base_grid_size = tuple([i // p for i, p in zip(self.base_img_size, self.base_patch_size)])
#         self.base_num_patches = self.base_grid_size[0] * self.base_grid_size[1]
#
#         self.flatten = flatten
#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=bias)
#         self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
#
#     def forward(self, x):
#         B, C, H, W = x.shape
#
#         if self.patch_size == self.base_patch_size:
#             weight = self.proj.weight
#         else:
#             weight = resample_patch_embed(self.proj.weight, self.patch_size)
#         patch_size = self.patch_size
#         x = F.conv2d(x, weight, bias=self.proj.bias, stride=patch_size)
#         if self.flatten:
#             x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
#         x = self.norm(x)
#         return x

class JigsawVisionTransformer(VisionTransformer):
    def __init__(self, mask_ratio, use_jigsaw, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_ratio = mask_ratio
        self.use_jigsaw = use_jigsaw

        self.flexi_patch_embed = FlexiblePatchEmbed(embed_dim=kwargs.get('embed_dim', 768))

        self.num_patches = self.patch_embed.num_patches

        if self.use_jigsaw:
            self.jigsaw = torch.nn.Sequential(*[torch.nn.Linear(self.embed_dim, self.embed_dim),
                                              torch.nn.ReLU(),
                                              torch.nn.Linear(self.embed_dim, self.embed_dim),
                                              torch.nn.ReLU(),
                                              torch.nn.Linear(self.embed_dim, self.num_patches)])
            self.target = torch.arange(self.num_patches)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        # target = einops.repeat(self.target, 'L -> N L', N=N) 
        # target = target.to(x.device)
        
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep] # N, len_keep
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        target_masked = ids_keep
        
        return x_masked, target_masked

    def forward_jigsaw(self, x):
        # masking: length -> length * mask_ratio
        x, target = self.random_masking(x, self.mask_ratio)

        # append cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        x = self.blocks(x)
        x = self.norm(x)
        x = self.jigsaw(x[:, 1:])
        return x.reshape(-1, self.num_patches), target.reshape(-1)

    def forward_cls(self, x): 
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.blocks(x)
        x = self.norm(x)
        x = self.head(x[:, 0])
        return x

    def forward(self, x):

        x_cls = self.flexi_patch_embed(x, 16)
        x_jigsaw = self.flexi_patch_embed(x, 32)
        pred_cls = self.forward_cls(x_cls)
        outs = Munch(sup=pred_cls)
        if self.use_jigsaw:
            pred_jigsaw, targets_jigsaw = self.forward_jigsaw(x_jigsaw)
            outs.pred_jigsaw = pred_jigsaw
            outs.gt_jigsaw = targets_jigsaw
        return outs


@register_model
def jigsaw_tiny_patch16_224(mask_ratio=0.5, use_jigsaw=True, pretrained=False, **kwargs):
    model = JigsawVisionTransformer(
        mask_ratio=mask_ratio, use_jigsaw=use_jigsaw,
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def jigsaw_small_patch16_224(mask_ratio=0.5, use_jigsaw=True, pretrained=False, **kwargs):
    model = JigsawVisionTransformer(
        mask_ratio=mask_ratio, use_jigsaw=use_jigsaw,
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def jigsaw_base_patch16_224(mask_ratio=0.5, use_jigsaw=True, pretrained=False, **kwargs):
    model = JigsawVisionTransformer(
        mask_ratio=mask_ratio, use_jigsaw=use_jigsaw,
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model

if __name__ == '__main__':
    net = jigsaw_base_patch16_224(mask_ratio=0.5, use_jigsaw=True, pretrained=False)
    net = net.cuda()
    img = torch.cuda.FloatTensor(6, 3, 224, 224)
    with torch.no_grad():
        outs = net(img)