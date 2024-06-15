import json
import matplotlib.pyplot as plt
from pathlib import Path
from omegaconf import OmegaConf
from solo.methods import METHODS
from torchvision import transforms
import torch
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import cv2
import argparse
from tqdm import tqdm
from matplotlib.patches import Polygon
import colorsys
import random
import skimage.io
from skimage.measure import find_contours

def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image


def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask,(10,10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    print(f"{fname} saved.")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-path",
        type=str,
    )
    parser.add_argument("--reg-path", type=str)
    parser.add_argument("--threshold", type=float, default=None, help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    output_dir=Path("/tudelft.net/staff-umbrella/StudentsCVlab/adondera/outputs/attention_weights")
    output_dir.mkdir(exist_ok=True, parents=True)

    args = parser.parse_args()

    with open(f"/tudelft.net/staff-umbrella/StudentsCVlab/adondera/trained_models/mae-reg/bhgdj49u/args.json") as f:
        method_args = json.load(f)
    
    baseline_model = (
        METHODS["mae-reg"]
        .load_from_checkpoint(args.baseline_path, strict=False, cfg=OmegaConf.create(method_args))
        .backbone
    )

    reg_model = (
        METHODS["mae-reg"]
        .load_from_checkpoint(args.reg_path, strict=False, cfg=OmegaConf.create(method_args))
        .backbone
    )

    baseline_model = baseline_model.to(torch.device("cuda:0")).eval()
    reg_model = reg_model.to(torch.device("cuda:0")).eval()

    T_val = transforms.Compose(
            [
                transforms.Resize(256),  # resize shorter
                transforms.CenterCrop(224),  # take center crop
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
    )

    batch_size = 1
    # val_dataset = ImageFolder("/tudelft.net/staff-umbrella/StudentsCVlab/adondera/imagenet100/train", T_val)
    val_dataset = ImageFolder("/tudelft.net/staff-umbrella/StudentsCVlab/adondera/imagenet100/val", T_val)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=2,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
    )

    baseline_attention_weights = {}
    baseline_th_attention_weights = {}
    reg_attention_weights = {}
    reg_th_attention_weights = {}

    w_featmap = 224 // 16
    h_featmap = 224 // 16

    for block_idx, block in enumerate(baseline_model.blocks):

        def hook_fn_baseline(module, input, output, block_idx=block_idx):
            x = input[0]
            B, N, C = x.shape
            qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

            attn = (q @ k.transpose(-2, -1)) * module.scale
            attn = attn.softmax(dim=-1)
            
            nh = attn.shape[1]
            attn = attn[0, :, 0, 1:].reshape(nh, -1)

            if args.threshold is not None:
                # we keep only a certain percentage of the mass
                val, idx = torch.sort(attn)
                val /= torch.sum(val, dim=1, keepdim=True)
                cumval = torch.cumsum(val, dim=1)
                th_attn = cumval > (1 - args.threshold)
                idx2 = torch.argsort(idx)
                for head in range(nh):
                    th_attn[head] = th_attn[head][idx2[head]]
                th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
                # interpolate
                th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=16, mode="nearest")[0].cpu().numpy()
                baseline_th_attention_weights[block_idx] = th_attn

            attn = attn.reshape(nh, w_featmap, h_featmap)
            attn = nn.functional.interpolate(attn.unsqueeze(0), scale_factor=16, mode="nearest")[0].cpu().numpy()
            baseline_attention_weights[block_idx] = attn

        def hook_fn_reg(module, input, output, block_idx=block_idx):
            x = input[0]
            B, N, C = x.shape
            qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

            attn = (q @ k.transpose(-2, -1)) * module.scale
            attn = attn.softmax(dim=-1)
            
            nh = attn.shape[1]
            attn = attn[0, :, 0, 1:].reshape(nh, -1)

            if args.threshold is not None:
                # we keep only a certain percentage of the mass
                val, idx = torch.sort(attn)
                val /= torch.sum(val, dim=1, keepdim=True)
                cumval = torch.cumsum(val, dim=1)
                th_attn = cumval > (1 - args.threshold)
                idx2 = torch.argsort(idx)
                for head in range(nh):
                    th_attn[head] = th_attn[head][idx2[head]]
                th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
                # interpolate
                th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=16, mode="nearest")[0].cpu().numpy()
                reg_th_attention_weights[block_idx] = th_attn

            attn = attn.reshape(nh, w_featmap, h_featmap)
            attn = nn.functional.interpolate(attn.unsqueeze(0), scale_factor=16, mode="nearest")[0].cpu().numpy()
            reg_attention_weights[block_idx] = attn

        baseline_model.blocks[block_idx].attn.register_forward_hook(hook_fn_baseline)
        reg_model.blocks[block_idx].attn.register_forward_hook(hook_fn_reg)

    with torch.no_grad():
        for idx, (x, y) in tqdm(enumerate(val_loader), total=len(val_dataset) // batch_size):

            baseline_model(x.to(torch.device("cuda:0")))
            reg_model(x.to(torch.device("cuda:0")))

            break


    print(x)
    plt.imshow(x[0].permute(1,2,0))
    plt.savefig(output_dir / f"test.png")
    plt.clf()
    torchvision.utils.save_image(torchvision.utils.make_grid(x, normalize=True, scale_each=True), output_dir / "img.png")        
    for idx in [11]:
        for head_idx, _ in enumerate(baseline_attention_weights[idx]):
            plt.imsave(fname=output_dir / f"baseline_{idx}_{head_idx}.png", arr=baseline_attention_weights[idx][head_idx], format='png')
            plt.imsave(fname=output_dir / f"reg_{idx}_{head_idx}.png", arr=reg_attention_weights[idx][head_idx], format='png')

    if args.threshold is not None:
        image = skimage.io.imread(output_dir / "img.png")
        for j in range(12):
            display_instances(image, baseline_th_attention_weights[11][j], fname=output_dir / f"baseline_mask_th_{str(args.threshold)}_head_{j}.png", blur=False)
            display_instances(image, reg_th_attention_weights[11][j], fname=output_dir / f"reg_mask_th_{str(args.threshold)}_head_{j}.png", blur=False)
