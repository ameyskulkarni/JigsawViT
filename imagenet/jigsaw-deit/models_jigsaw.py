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

import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
from PIL import Image
import torch.nn.functional as F


class JigsawVisionTransformer(VisionTransformer):
    def __init__(self, mask_ratio, use_jigsaw, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("Positional arguments (*args):")
        for i, arg in enumerate(args):
            print(f"  args[{i}]: {arg}")

        print("\nKeyword arguments (**kwargs):")
        for key, value in kwargs.items():
            print(f"  {key}: {value}")
        self.mask_ratio = mask_ratio
        self.use_jigsaw = use_jigsaw

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
        N, L, D = x.shape  # batch, length, dim [128, 196, 384]
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
        # Shuffling patches for inference for classification head alone.
        x_shuffled = self.shuffle_patches(x, 16)
        # Batch size is 96
        x = self.patch_embed(x) # [128, 3, 224, 224] -> [128, 196, 384] Just for explanation, batch size is 96
        x_shuffled = self.patch_embed(x_shuffled)
        pred_cls = self.forward_cls(x_shuffled) # [128, 1000]. These are logits, not probabilities.
        outs = Munch(sup=pred_cls)
        if self.use_jigsaw:
            pred_jigsaw, targets_jigsaw = self.forward_jigsaw(x) # pred_jigsaw is resized from [128, 98, 196] and targets_jigsaw is resized from [128, 98]. The pred values are still logits.
            outs.pred_jigsaw = pred_jigsaw
            outs.gt_jigsaw = targets_jigsaw
        return outs

    # Function to display original and shuffled images
    def visualize_patch_shuffle(self, img, shuffled_img):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(img.permute(1, 2, 0))  # Convert to (H, W, C)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(shuffled_img.permute(1, 2, 0))  # Convert to (H, W, C)
        axes[1].set_title("Shuffled Patches")
        axes[1].axis("off")

        plt.show()

    def shuffle_patches(self, images, patch_size=16):
        """
        Extracts 16x16 patches from each image, shuffles them, and reconstructs the image.

        Args:
            images (torch.Tensor): Input batch of images of shape (B, C, H, W).
            patch_size (int): Size of each patch (default is 16x16).

        Returns:
            torch.Tensor: Batch of images with shuffled patches, same shape as input.
        """
        B, C, H, W = images.shape
        num_patches = (H // patch_size) * (W // patch_size)  # Total number of patches per image
        grid_size = H // patch_size  # Number of patches per row/column

        # Step 1: Reshape images into patches (B, num_patches, C, patch_size, patch_size)
        patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(B, num_patches, C, patch_size, patch_size)

        # Step 2: Shuffle patches randomly for each image in the batch
        shuffled_patches = patches.clone()
        for i in range(B):
            perm = torch.randperm(num_patches)  # Generate random permutation
            shuffled_patches[i] = patches[i][perm]  # Shuffle patches

        # Step 3: Reshape back to image format
        shuffled_patches = shuffled_patches.reshape(B, grid_size, grid_size, C, patch_size, patch_size)
        shuffled_patches = shuffled_patches.permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)

        return shuffled_patches


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