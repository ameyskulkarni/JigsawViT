# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

import einops
from munch import Munch
import random
import torch.nn.functional as F


class JigsawVisionTransformer(VisionTransformer):
    def __init__(self, mask_ratio, use_jigsaw, jigsaw_patch_sizes=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_ratio = mask_ratio
        self.use_jigsaw = use_jigsaw
        self.num_patches = self.patch_embed.num_patches

        # Default patch size for the main classifier branch (typically 16x16)
        self.default_patch_size = self.patch_embed.patch_size

        # List of different patch sizes for jigsaw task
        self.jigsaw_patch_sizes = jigsaw_patch_sizes if jigsaw_patch_sizes else [(8, 8), (16, 16), (32, 32)]

        if self.use_jigsaw:
            # Create flexible patch embedding for jigsaw task
            self.flexi_patch_embed = FlexiPatchEmbed(
                img_size=self.patch_embed.img_size,
                patch_size=self.default_patch_size,
                in_chans=3,
                embed_dim=self.embed_dim
            )

            # Jigsaw prediction head
            self.jigsaw = torch.nn.Sequential(*[
                torch.nn.Linear(self.embed_dim, self.embed_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.embed_dim, self.embed_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.embed_dim, self.num_patches)
            ])

            # Create shared position embeddings that can be resized later
            # Note: we'll resize this according to the jigsaw patch size
            # self.jigsaw_pos_embed = nn.Parameter(
            #     torch.zeros(1, self.num_patches + 1, self.embed_dim)
            # )
            # trunc_normal_(self.jigsaw_pos_embed, std=0.02)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove

        # Keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]  # N, len_keep
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        target_masked = ids_keep

        return x_masked, target_masked

    def resize_pos_embed(self, pos_embed, num_patches):
        """
        Resize position embeddings to match the number of patches
        """
        pos_embed_cls, pos_embed_patch = pos_embed[:, :1], pos_embed[:, 1:]
        batch_size, seq_len, dim = pos_embed_patch.shape
        h = w = int(math.sqrt(seq_len))

        # Reshape to get 2D positional embedding
        pos_embed_patch = pos_embed_patch.reshape(batch_size, h, w, dim)

        # Compute new h, w for target number of patches
        new_h = new_w = int(math.sqrt(num_patches))

        # Resize position embeddings
        pos_embed_patch = F.interpolate(
            pos_embed_patch.permute(0, 3, 1, 2),  # [B, D, H, W]
            size=(new_h, new_w),
            mode='bilinear',
            align_corners=False
        ).permute(0, 2, 3, 1)  # [B, H', W', D]

        pos_embed_patch = pos_embed_patch.flatten(1, 2)  # [B, H'*W', D]

        # Concatenate with class token position embedding
        new_pos_embed = torch.cat((pos_embed_cls, pos_embed_patch), dim=1)

        return new_pos_embed

    def forward_jigsaw(self, x, patch_size=None):
        """
        Forward pass for jigsaw prediction branch with flexible patch size
        """
        B = x.shape[0]

        # If patch_size is specified, resize the patch embedding kernel
        if patch_size is not None and patch_size != self.default_patch_size:
            # Get embeddings with flexible patch size
            x = self.flexi_patch_embed(x, patch_size=patch_size)

            # Calculate new number of patches based on the patch size
            H, W = self.patch_embed.img_size
            P_H, P_W = patch_size
            num_patches = (H // P_H) * (W // P_W)

            # Resize position embeddings
            # pos_embed = self.resize_pos_embed(self.jigsaw_pos_embed, num_patches)
        else:
            # Use standard patch embeddings if no patch_size is specified
            x = self.patch_embed(x)
            # pos_embed = self.pos_embed
            num_patches = self.num_patches

        # Masking: length -> length * mask_ratio
        x, target = self.random_masking(x, self.mask_ratio)

        # Add position embeddings (excluding cls token position)
        # x = x + pos_embed[:, 1:, :][:, :x.size(1), :]

        # Append cls token
        # cls_token = self.cls_token + pos_embed[:, :1, :]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Apply Transformer blocks
        x = self.blocks(x)
        x = self.norm(x)

        # Predict patch positions
        x = self.jigsaw(x[:, 1:])

        return x.reshape(-1, num_patches), target.reshape(-1)

    def forward_cls(self, x):
        """
        Forward pass for classification branch (using fixed patch size)
        """
        # Add position embeddings (excluding cls token position)
        x = x + self.pos_embed[:, 1:, :]

        # Append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Apply Transformer blocks
        x = self.blocks(x)
        x = self.norm(x)

        # Classification prediction
        x = self.head(x[:, 0])

        return x

    def forward(self, x):
        """
        Main forward pass that combines both branches
        """
        # Standard patch embeddings for classification
        patch_embeds = self.patch_embed(x)

        # Classification branch (fixed patch size)
        pred_cls = self.forward_cls(patch_embeds)
        outs = Munch(sup=pred_cls)

        # Jigsaw branch (flexible patch size)
        if self.use_jigsaw:
            # Randomly choose a patch size from the list for jigsaw task
            if self.training:
                jigsaw_patch_size = random.choice(self.jigsaw_patch_sizes)
            else:
                jigsaw_patch_size = self.default_patch_size

            pred_jigsaw, targets_jigsaw = self.forward_jigsaw(x, patch_size=jigsaw_patch_size)
            outs.pred_jigsaw = pred_jigsaw
            outs.gt_jigsaw = targets_jigsaw
            outs.jigsaw_patch_size = jigsaw_patch_size

        return outs


class FlexiPatchEmbed(nn.Module):
    """
    Flexible patch embedding that can handle different patch sizes
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # Create the default patch embedding kernel
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

        # Calculate number of patches for default patch size
        self.num_patches = (self.img_size[0] // self.patch_size[0]) * (self.img_size[1] // self.patch_size[1])

    def resample_patch_embed(self, kernel, new_size):
        """
        Resample the patch embedding kernel to a new patch size
        Similar to the resample_patchemb function from FlexiViT
        """
        old_size = kernel.shape[2:4]
        if old_size == new_size:
            return kernel

        # Reshape kernel to [patch_H, patch_W, in_chans*out_dims]
        c_out, c_in, h, w = kernel.shape
        kernel_reshaped = kernel.permute(2, 3, 0, 1).reshape(h, w, -1)

        # Use bicubic interpolation to resize
        kernel_resized = F.interpolate(
            kernel_reshaped.permute(2, 0, 1).unsqueeze(0),
            size=new_size,
            mode='bilinear',
            align_corners=False
        ).squeeze(0).permute(1, 2, 0)

        # Reshape back to [c_out, c_in, new_h, new_w]
        kernel_resized = kernel_resized.reshape(new_size[0], new_size[1], c_in, c_out)
        kernel_resized = kernel_resized.permute(3, 2, 0, 1)

        return kernel_resized

    def forward(self, x, patch_size=None):
        """
        Forward function with support for dynamic patch sizes
        """
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})"

        if patch_size is None or patch_size == self.patch_size:
            # Use the default projection if patch size matches
            x = self.proj(x).flatten(2).transpose(1, 2)
        else:
            # Resample the projection kernel for a different patch size
            resampled_kernel = self.resample_patch_embed(
                self.proj.weight.data,
                (patch_size[0], patch_size[1])
            )

            # Create a temporary convolution with the resampled kernel
            temp_conv = nn.Conv2d(
                self.in_chans,
                self.embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
                bias=self.proj.bias is not None
            ).to(x.device)

            # Set the weights and bias
            temp_conv.weight.data = resampled_kernel
            if self.proj.bias is not None:
                temp_conv.bias.data = self.proj.bias.data

            # Apply the convolution
            x = temp_conv(x).flatten(2).transpose(1, 2)

        return x


# class JigsawVisionTransformer(VisionTransformer):
#     def __init__(self, mask_ratio, use_jigsaw, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.mask_ratio = mask_ratio
#         self.use_jigsaw = use_jigsaw
#
#         self.num_patches = self.patch_embed.num_patches
#
#         if self.use_jigsaw:
#             self.jigsaw = torch.nn.Sequential(*[torch.nn.Linear(self.embed_dim, self.embed_dim),
#                                               torch.nn.ReLU(),
#                                               torch.nn.Linear(self.embed_dim, self.embed_dim),
#                                               torch.nn.ReLU(),
#                                               torch.nn.Linear(self.embed_dim, self.num_patches)])
#             self.target = torch.arange(self.num_patches)
#
#     def random_masking(self, x, mask_ratio):
#         """
#         Perform per-sample random masking by per-sample shuffling.
#         Per-sample shuffling is done by argsort random noise.
#         x: [N, L, D], sequence
#         """
#         N, L, D = x.shape  # batch, length, dim
#         len_keep = int(L * (1 - mask_ratio))
#
#         noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
#
#         # sort noise for each sample
#         ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
#         # target = einops.repeat(self.target, 'L -> N L', N=N)
#         # target = target.to(x.device)
#
#         # keep the first subset
#         ids_keep = ids_shuffle[:, :len_keep] # N, len_keep
#         x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
#         target_masked = ids_keep
#
#         return x_masked, target_masked
#
#     def forward_jigsaw(self, x):
#         # masking: length -> length * mask_ratio
#         x, target = self.random_masking(x, self.mask_ratio)
#
#         # append cls token
#         cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
#         x = torch.cat((cls_tokens, x), dim=1)
#
#         # apply Transformer blocks
#         x = self.blocks(x)
#         x = self.norm(x)
#         x = self.jigsaw(x[:, 1:])
#         return x.reshape(-1, self.num_patches), target.reshape(-1)
#
#     def forward_cls(self, x):
#         # add pos embed w/o cls token
#         x = x + self.pos_embed[:, 1:, :]
#
#         # append cls token
#         cls_token = self.cls_token + self.pos_embed[:, :1, :]
#         cls_tokens = cls_token.expand(x.shape[0], -1, -1)
#         x = torch.cat((cls_tokens, x), dim=1)
#
#         x = self.blocks(x)
#         x = self.norm(x)
#         x = self.head(x[:, 0])
#         return x
#
#     def forward(self, x):
#         x = self.patch_embed(x)
#         pred_cls = self.forward_cls(x)
#         outs = Munch(sup=pred_cls)
#         if self.use_jigsaw:
#             pred_jigsaw, targets_jigsaw = self.forward_jigsaw(x)
#             outs.pred_jigsaw = pred_jigsaw
#             outs.gt_jigsaw = targets_jigsaw
#         return outs


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