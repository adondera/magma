import json
import matplotlib.pyplot as plt
from pathlib import Path
from omegaconf import OmegaConf
from solo.methods import METHODS
import pacmap
from torchvision import transforms
import torch
import random
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import umap
import numpy as np
import pandas as pd
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from solo.utils.embedding_propagation import get_similarity_matrix, get_laplacian

def normalize(X: np.ndarray):
    return (X - X.mean(axis=0)) / x.std(axis=0)

def plot_laplacian(x, name, title=""):
    weights_matrix = get_similarity_matrix(x, rbf_scale=1.0, scaling_factor=False)
    laplacian = get_laplacian(weights_matrix, normalized=True)
    sns.heatmap(torch.eye(laplacian.shape[0]) - laplacian.cpu())
    plt.title(title)
    plt.savefig(name, dpi=300, bbox_inches="tight")
    plt.clf()

def plot_laplacian_combined(x1, x2, name, title=""):
    weights_matrix = get_similarity_matrix(x1, rbf_scale=1.0, scaling_factor=False)
    laplacian1 = get_laplacian(weights_matrix, normalized=True)
    laplacian1 = torch.eye(laplacian1.shape[0]) - laplacian1.cpu()

    weights_matrix = get_similarity_matrix(x2, rbf_scale=1.0, scaling_factor=False)
    laplacian2 = get_laplacian(weights_matrix, normalized=True)
    laplacian2 = torch.eye(laplacian2.shape[0]) - laplacian2.cpu()

    v_min = min(torch.min(laplacian1), torch.min(laplacian2))
    v_max = min(torch.max(laplacian1), torch.max(laplacian2))

    fig, ax = plt.subplots(1, 2)
    sns.heatmap(laplacian1, ax=ax[0], cbar=False, vmin=v_min, vmax=v_max)
    sns.heatmap(laplacian2, ax=ax[1], vmin=v_min, vmax=v_max)
    print(name)
    print(laplacian1)
    print(laplacian2)
    fig.savefig(name, dpi=400, bbox_inches="tight")
    plt.clf()

def plot_laplacian_diff(x1, x2, name, title=""):
    weights_matrix = get_similarity_matrix(x1, rbf_scale=1.0, scaling_factor=False)
    laplacian1 = get_laplacian(weights_matrix, normalized=True)

    weights_matrix = get_similarity_matrix(x2, rbf_scale=1.0, scaling_factor=False)
    laplacian2 = get_laplacian(weights_matrix, normalized=True)

    sns.heatmap(torch.abs(laplacian1 - laplacian2).cpu())
    plt.title(title)
    plt.savefig(name, dpi=300, bbox_inches="tight")
    plt.clf()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-path",
        type=str,
    )
    parser.add_argument("--reg-path", type=str)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--classes", nargs='+', type=int, default=[])
    parser.add_argument("--layers", type=int, nargs='+', default=[10])

    output_dir=Path("/tudelft.net/staff-umbrella/StudentsCVlab/adondera/outputs/laplacian")
    output_dir.mkdir(exist_ok=True, parents=True)

    args = parser.parse_args()
    assert args.num_classes <= 10

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
    if not args.classes:
        classes = random.sample(range(100), k=args.num_classes)
    else:
        classes = args.classes

    print(classes)

    batch_size = 512
    # val_dataset = ImageFolder("/tudelft.net/staff-umbrella/StudentsCVlab/adondera/imagenet100/train", T_val)
    val_dataset = ImageFolder("/tudelft.net/staff-umbrella/StudentsCVlab/adondera/imagenet100/val", T_val)
    indices = [i for i in range(len(val_dataset)) if val_dataset.imgs[i][1] in classes]
    val_dataset = Subset(val_dataset, indices)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=2,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
    )

    baseline_extracted_features = {}
    reg_extracted_features = {}

    for layer in args.layers:
        def hook_fn_baseline(module, input, output, layer=layer):
            baseline_extracted_features[layer] = output[:, 0, :]
            # baseline_layer11_features.append(output[:, 1:, :].mean(dim=1).cpu().numpy())

        def hook_fn_reg(module, input, output, layer=layer):
            reg_extracted_features[layer] = output[:, 0, :]

        baseline_model.blocks[layer].register_forward_hook(hook_fn_baseline)
        reg_model.blocks[layer].register_forward_hook(hook_fn_reg)

    with torch.no_grad():
        for idx, (x, y) in tqdm(enumerate(val_loader), total=len(val_dataset) // batch_size):
            sorted, indices = torch.sort(y.cpu())

            baseline_features = baseline_model(x.to(torch.device("cuda:0"))).cpu()[indices].numpy()
            reg_features = reg_model(x.to(torch.device("cuda:0"))).cpu()[indices].numpy()

            for layer in args.layers:
                plot_laplacian(baseline_extracted_features[layer][indices], name=output_dir / f"baseline_layer{layer}_laplacian.png", title=f"MAE Layer {layer} Laplacian")
                plot_laplacian(reg_extracted_features[layer][indices], name=output_dir / f"reg_layer{layer}_laplacian.png", title=f"M-MAE Layer {layer} Laplacian")

                plot_laplacian_diff(baseline_extracted_features[layer][indices], reg_extracted_features[layer][indices], name=output_dir / f"baseline_vs_reg_layer{layer}_abs_diff_laplacian.png", title=f"MAE vs M-Mae Layer {layer} Laplacian Diff")

                plot_laplacian_combined(baseline_extracted_features[layer][indices], reg_extracted_features[layer][indices], name=output_dir / f"baseline_vs_reg_layer{layer}_laplacian_combined.png")

            break
    


