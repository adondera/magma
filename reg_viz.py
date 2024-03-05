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

def normalize(X: np.ndarray):
    return (X - X.mean(axis=0)) / x.std(axis=0)

def plot(embeddings, labels, name, has_layer_11=False, use_normalization=False):
    if use_normalization:
        if not has_layer_11:
            embeddings = normalize(embeddings)
        else:
            pass
    df = pd.DataFrame(embeddings, columns=["feat_1", "feat_2"])
    if has_layer_11:
        df["Y"] = np.tile(labels, 2)
        df["style"] = ["layer_12_cls"] * (len(df["Y"]) // 2) + ["layer_11_cls"] * (len(df["Y"]) // 2)
    else:
        df["Y"] = labels
        df["style"] = ["layer_12_cls"] * len(df["Y"])
    ax = sns.scatterplot(
                x="feat_1",
                y="feat_2",
                hue="Y",
                palette=sns.color_palette("tab10"),
                data=df,
                legend="full",
                style="style",
                alpha=0.7,
            )
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.savefig(name, dpi=300, bbox_inches="tight", bbox_extra_artists=[ax.get_legend()])
    plt.clf()

def plot_ax(embeddings, labels, ax,  has_layer_11=False, use_normalization=False, legend=False, title=""):
    if use_normalization:
        if not has_layer_11:
            embeddings = normalize(embeddings)
        else:
            pass
    df = pd.DataFrame(embeddings, columns=["feat_1", "feat_2"])
    if has_layer_11:
        df["Y"] = np.tile(labels, 2)
        df["style"] = ["layer_12_cls"] * (len(df["Y"]) // 2) + ["layer_11_cls"] * (len(df["Y"]) // 2)
    else:
        df["Y"] = labels
        df["style"] = ["layer_12_cls"] * len(df["Y"])
    ax.set_title(title)
    ax.set_axis_off()
    if legend:
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    return sns.scatterplot(
                x="feat_1",
                y="feat_2",
                hue="Y",
                palette=sns.color_palette("tab10"),
                data=df,
                legend=legend,
                style="style",
                alpha=0.7,
                ax = ax
            )

def plot_all(mae_embeddings, m_mae_embeddings, u_mae_embeddings, mu_mae_embeddings, labels, path):
    fig, axes = plt.subplots(2, 2)
    plot_ax(mae_embeddings, labels, axes[0, 0], title="MAE")
    legend_ax = plot_ax(m_mae_embeddings, labels, axes[0, 1], title="M-MAE", legend="full")
    plot_ax(u_mae_embeddings, labels, axes[1, 0], title="U-MAE")
    plot_ax(mu_mae_embeddings, labels, axes[1, 1], title="MU-MAE")
    # # Collect handles and labels from all subplots
    # handles, labels = [], []
    # for ax in axes.flat:
    #     h, l = ax.get_legend_handles_labels()
    #     handles.extend(h)
    #     labels.extend(l)
    # fig.legend(handles, labels, loc="upper right")
    fig.savefig(path, dpi=600, bbox_inches="tight", bbox_extra_artists=[legend_ax.get_legend()])



def print_metrics(features, labels, identifier=""):
    print(f"Metrics for: {identifier}")
    print(f"Silhouette score: {metrics.silhouette_score(features, labels)}")
    print(f"CHI score: {metrics.calinski_harabasz_score(features, labels)}")
    print(f"DBI score: {metrics.davies_bouldin_score(features, labels)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mae-path",
        type=str,
    )
    parser.add_argument("--m-mae-path", type=str)
    parser.add_argument("--u-mae-path", type=str)
    parser.add_argument("--mu-mae-path", type=str)
    parser.add_argument("--num-iterations", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--plot-layer-11", action="store_true", default=False)
    parser.add_argument("--normalize", action="store_true", default=False)
    parser.add_argument("--classes", nargs='+', type=int, default=[])

    output_dir=Path("/tudelft.net/staff-umbrella/StudentsCVlab/adondera/outputs/pacmap_plots")
    output_dir.mkdir(exist_ok=True, parents=True)

    args = parser.parse_args()
    assert args.num_classes <= 10

    with open(f"/tudelft.net/staff-umbrella/StudentsCVlab/adondera/trained_models/mae-reg/bhgdj49u/args.json") as f:
        method_args = json.load(f)
    
    mae_model = (
        METHODS["mae-reg"]
        .load_from_checkpoint(args.mae_path, strict=False, cfg=OmegaConf.create(method_args))
        .backbone
    ).to(torch.device("cuda:0")).eval()

    m_mae_model = (
        METHODS["mae-reg"]
        .load_from_checkpoint(args.m_mae_path, strict=False, cfg=OmegaConf.create(method_args))
        .backbone
    ).to(torch.device("cuda:0")).eval()

    u_mae_model = (
        METHODS["mae-reg"]
        .load_from_checkpoint(args.u_mae_path, strict=False, cfg=OmegaConf.create(method_args))
        .backbone
    ).to(torch.device("cuda:0")).eval()

    mu_mae_model = (
        METHODS["mae-reg"]
        .load_from_checkpoint(args.mu_mae_path, strict=False, cfg=OmegaConf.create(method_args))
        .backbone
    ).to(torch.device("cuda:0")).eval()

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
    val_dataset = ImageFolder("/tudelft.net/staff-umbrella/StudentsCVlab/adondera/imagenet100/train", T_val)
    # val_dataset = ImageFolder("/tudelft.net/staff-umbrella/StudentsCVlab/adondera/imagenet100/val", T_val)
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

    mae_features = []
    mae_layer11_features = []
    m_mae_features = []
    m_mae_layer11_features = []
    u_mae_features = []
    u_mae_layer11_features = []
    mu_mae_features = []
    mu_mae_layer11_features = []
    labels = []

    def hook_fn_mae(module, input, output):
        mae_layer11_features.append(output[:, 0, :].cpu().numpy())

    def hook_fn_m_mae(module, input, output):
        m_mae_features.append(output[:, 0, :].cpu().numpy())

    def hook_fn_u_mae(module, input, output):
        u_mae_features.append(output[:, 0, :].cpu().numpy())

    def hook_fn_mu_mae(module, input, output):
        mu_mae_features.append(output[:, 0, :].cpu().numpy())

    if args.plot_layer_11:
        mae_model.blocks[10].register_forward_hook(hook_fn_mae)
        m_mae_model.blocks[10].register_forward_hook(hook_fn_m_mae)
        u_mae_model.blocks[10].register_forward_hook(hook_fn_u_mae)
        mu_mae_model.blocks[10].register_forward_hook(hook_fn_mu_mae)

    suffix = "-with-layer11" if args.plot_layer_11 else ""

    with torch.no_grad():
        for idx, (x, y) in tqdm(enumerate(val_loader), total=len(val_dataset) // batch_size):
            y = y.cpu().numpy()
            index = np.isin(y, classes)
            labels.append(y[index])

            mae_features.append(mae_model(x.to(torch.device("cuda:0"))).cpu().numpy()[index])
            m_mae_features.append(m_mae_model(x.to(torch.device("cuda:0"))).cpu().numpy()[index])
            u_mae_features.append(u_mae_model(x.to(torch.device("cuda:0"))).cpu().numpy()[index])
            mu_mae_features.append(mu_mae_model(x.to(torch.device("cuda:0"))).cpu().numpy()[index])

            pacmap_mae_embeddings = pacmap.PaCMAP(n_components=2, verbose=True, random_state=42).fit_transform(np.concatenate(mae_features + mae_layer11_features))
            pacmap_m_mae_embeddings = pacmap.PaCMAP(n_components=2, verbose=True, random_state=42).fit_transform(np.concatenate(m_mae_features + m_mae_layer11_features))
            pacmap_u_mae_embeddings = pacmap.PaCMAP(n_components=2, verbose=True, random_state=42).fit_transform(np.concatenate(u_mae_features + u_mae_layer11_features))
            pacmap_mu_mae_embeddings = pacmap.PaCMAP(n_components=2, verbose=True, random_state=42).fit_transform(np.concatenate(mu_mae_features + mu_mae_layer11_features))

            plot_all(pacmap_mae_embeddings, pacmap_m_mae_embeddings, pacmap_u_mae_embeddings, pacmap_mu_mae_embeddings, np.concatenate(labels), output_dir / f"pacmap_all_{idx}{suffix}.png")

            # plot(pacmap_baseline_embeddings, np.concatenate(labels), output_dir / f"pacmap_baseline_{idx}{suffix}.png", has_layer_11=args.plot_layer_11, use_normalization=args.normalize)

            # plot(pacmap_reg_embeddings, np.concatenate(labels),  output_dir / f"pacmap_reg_{idx}{suffix}.png", has_layer_11=args.plot_layer_11, use_normalization=args.normalize)

            # plot(umap_baseline_embeddings, np.concatenate(labels), output_dir / f"umap_baseline_{idx}{suffix}.png", has_layer_11=args.plot_layer_11, use_normalization=args.normalize)

            # plot(umap_reg_embeddings, np.concatenate(labels),  output_dir / f"umap_reg_{idx}{suffix}.png", has_layer_11=args.plot_layer_11, use_normalization=args.normalize)

            # print_metrics(np.concatenate(baseline_features), np.concatenate(labels), "Baseline")
            # print_metrics(np.concatenate(reg_features), np.concatenate(labels), "Regularization")

            # if args.num_iterations and idx == args.num_iterations:
            #     break
    

    # pacmap_baseline_embeddings = pacmap.PaCMAP(n_components=2, verbose=True).fit_transform(np.concatenate(baseline_features))
    # pacmap_reg_embeddings = pacmap.PaCMAP(n_components=2, verbose=True).fit_transform(np.concatenate(reg_features))

    # plot(pacmap_baseline_embeddings, np.concatenate(labels))
    # plt.savefig("plots/baseline.png")
    # plt.clf()
    # plot(pacmap_reg_embeddings, np.concatenate(labels))
    # plt.savefig("plots/reg.png")
    # plt.clf()

