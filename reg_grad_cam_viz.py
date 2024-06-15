import json
from pathlib import Path

from omegaconf import OmegaConf
from solo.methods.base import BaseMethod

from torchvision import transforms
import torch
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.datasets import ImageFolder
import cv2
import argparse
import random
from solo.methods.linear import LinearModel
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
from pytorch_grad_cam.ablation_layer import AblationLayerVit
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
import numpy as np

def load_model(ckpt_path, method_args):
    backbone_model = BaseMethod._BACKBONES["vit_base"]
    backbone = backbone_model(method="mae-reg")
    assert ckpt_path.endswith(".ckpt") or ckpt_path.endswith(".pth") or ckpt_path.endswith(".pt")
    loaded_ckpt = torch.load(ckpt_path, map_location="cpu")
    if 'state_dict' in loaded_ckpt:
        state = loaded_ckpt['state_dict']
    elif 'model' in loaded_ckpt:
        state = loaded_ckpt['model']
    else:
        raise ValueError("Unsupported checkpoint")
    model = LinearModel(backbone, cfg=OmegaConf.create(method_args))
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Missing: {missing}")
    print(f"Unexpected: {unexpected}")
    return model

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-path",
        type=str,
    )
    parser.add_argument("--reg-path", type=str)
    parser.add_argument(
        '--method',
        type=str,
        default='gradcam',
        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')
    
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}
    

    output_dir=Path("/tudelft.net/staff-umbrella/StudentsCVlab/adondera/outputs/grad-cam")
    output_dir.mkdir(exist_ok=True, parents=True)

    args = parser.parse_args()

    with open(f"/tudelft.net/staff-umbrella/StudentsCVlab/adondera/trained_models/mae-reg/bhgdj49u/args.json") as f:
        method_args = json.load(f)
    
    backbone_model = BaseMethod._BACKBONES["vit_base"]
    backbone = backbone_model(method="mae-reg")
    ckpt_path = args.baseline_path
    assert ckpt_path.endswith(".ckpt") or ckpt_path.endswith(".pth") or ckpt_path.endswith(".pt")
    loaded_ckpt = torch.load(ckpt_path, map_location="cpu")
    if 'state_dict' in loaded_ckpt:
        state = loaded_ckpt['state_dict']
    elif 'model' in loaded_ckpt:
        state = loaded_ckpt['model']
    else:
        raise ValueError("Unsupported checkpoint")

    method_args["finetune"] = True
    baseline_model = load_model(args.baseline_path, method_args)
    reg_model = load_model(args.reg_path, method_args)

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

    T_val_unnormalized = transforms.Compose(
            [
                transforms.Resize(256),  # resize shorter
                transforms.CenterCrop(224),  # take center crop
                transforms.ToTensor(),
            ]
    )

    # val_dataset = ImageFolder("/tudelft.net/staff-umbrella/StudentsCVlab/adondera/imagenet100/train", T_val)
    val_dataset = ImageFolder("/tudelft.net/staff-umbrella/StudentsCVlab/adondera/imagenet100/val", T_val)


    index = random.choice(range(len(val_dataset)))
    path, label = val_dataset.samples[index]
    x, y = val_dataset[index]
    print(path)
    rgb_img = cv2.imread(path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255

    # rgb_img = T_val_unnormalized(rgb_img)

    input_tensor = preprocess_image(rgb_img, mean=IMAGENET_DEFAULT_MEAN,
                                    std=IMAGENET_DEFAULT_STD)
    
    print(x)
    print(input_tensor)
    # print(x-input_tensor)

    target_layers_baseline = [baseline_model.backbone.blocks[-1].norm1]
    target_layers_reg_model = [reg_model.backbone.blocks[-1].norm1]

    # print(baseline_model.backbone.blocks[-1])
    # print(baseline_model.backbone.blocks[-1].norm1.requires_grad_())

    if args.method == "ablationcam":
        cam_baseline = methods[args.method](model=baseline_model,
                                   target_layers=target_layers_baseline,
                                   reshape_transform=reshape_transform,
                                   ablation_layer=AblationLayerVit())

        cam_reg = methods[args.method](model=reg_model,
                                   target_layers=target_layers_reg_model,
                                   reshape_transform=reshape_transform,
                                   ablation_layer=AblationLayerVit())
    else:
        cam_baseline = methods[args.method](model=baseline_model,
                                   target_layers=target_layers_baseline,
                                   reshape_transform=reshape_transform)
        cam_reg = methods[args.method](model=reg_model,
                                   target_layers=target_layers_reg_model,
                                   reshape_transform=reshape_transform)

    targets = None


    grayscale_cam_baseline = cam_baseline(input_tensor=x.unsqueeze(0),
                        targets=targets,)
    grayscale_cam_baseline = grayscale_cam_baseline[0, :]
    cam_image_baseline = show_cam_on_image(rgb_img, grayscale_cam_baseline)
    cv2.imwrite(str(output_dir / f'{args.method}_cam_baseline.jpg'), cam_image_baseline)
    
    grayscale_cam_reg = cam_reg(input_tensor=x.unsqueeze(0),
                        targets=targets,)
    grayscale_cam_reg = grayscale_cam_reg[0, :]
    cam_image_reg = show_cam_on_image(rgb_img, grayscale_cam_reg)
    cv2.imwrite(str(output_dir / f'{args.method}_cam_reg.jpg'), cam_image_reg)
