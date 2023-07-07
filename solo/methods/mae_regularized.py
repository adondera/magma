# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import Any, Dict, List, Sequence

import omegaconf
import torch
import torch.nn as nn
from matplotlib import colors
from solo.losses.mae import mae_loss_func
from solo.methods.base import BaseMethod
from solo.losses.manifold_regularizer import ManifoldRegularizer
from solo.utils.embedding_propagation import get_similarity_matrix, get_laplacian
from solo.utils.misc import generate_2d_sincos_pos_embed, omegaconf_select
from solo.utils.weight_schedulers import TriangleScheduler, WarmupScheduler, ConstantScheduler
from solo.utils.metrics import weighted_mean, tensor_mean, get_heatmap
from timm.models.vision_transformer import Block


class MAEDecoder(nn.Module):
    def __init__(
        self,
        in_dim,
        embed_dim,
        depth,
        num_heads,
        num_patches,
        patch_size,
        mlp_ratio=4.0,
    ) -> None:
        super().__init__()

        self.num_patches = num_patches

        self.decoder_embed = nn.Linear(in_dim, embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # fixed sin-cos embedding
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )

        self.decoder_blocks = nn.Sequential(
            *[
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(depth)
            ]
        )

        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_pred = nn.Linear(embed_dim, patch_size**2 * 3, bias=True)

        # init all weights according to MAE's repo
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding

        decoder_pos_embed = generate_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.num_patches**0.5),
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        # unshuffle
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        x = self.decoder_blocks(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x


class MAE_REG(BaseMethod):
    def __init__(
        self,
        cfg: omegaconf.DictConfig,
    ):
        """Implements MAE (https://arxiv.org/abs/2111.06377).

        Extra cfg settings:
            method_kwargs:
                mask_ratio (float): percentage of image to mask.
                decoder_embed_dim (int): number of dimensions for the embedding in the decoder
                decoder_depth (int) depth of the decoder
                decoder_num_heads (int) number of heads for the decoder
                norm_pix_loss (bool): whether to normalize the pixels of each patch with their
                    respective mean and std for the loss. Defaults to False.
        """

        super().__init__(cfg)

        assert "vit" in self.backbone_name, "MAE only supports ViT as backbone."

        self.mask_ratio: float = cfg.method_kwargs.mask_ratio
        self.norm_pix_loss: bool = cfg.method_kwargs.norm_pix_loss
        self.layers = cfg.method_kwargs.layers

        # Scheduler params
        self.configure_reg_scheduler(cfg.method_kwargs.reg_scheduler)

        # gather backbone info from timm
        self._vit_embed_dim: int = self.backbone.pos_embed.size(-1)
        # if patch size is not available, defaults to 16 or 14 depending on backbone
        default_patch_size = 14 if self.backbone_name == "vit_huge" else 16
        self._vit_patch_size: int = self.backbone_args.get(
            "patch_size", default_patch_size
        )
        self._vit_num_patches: int = self.backbone.patch_embed.num_patches

        decoder_embed_dim: int = cfg.method_kwargs.decoder_embed_dim
        decoder_depth: int = cfg.method_kwargs.decoder_depth
        decoder_num_heads: int = cfg.method_kwargs.decoder_num_heads

        self.manifold_regularizer = ManifoldRegularizer(return_metrics=True)

        self.log_images = cfg.method_kwargs.log_images

        self.validation_mean_embeddings = {
            class_idx: torch.zeros(self.features_dim, device=self.device) for class_idx in range(self.num_classes)
        }
        self.validation_class_counts = {
            class_idx: 0 for class_idx in range(self.num_classes)
        }

        # decoder
        self.decoder = MAEDecoder(
            in_dim=self.features_dim,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            num_patches=self._vit_num_patches,
            patch_size=self._vit_patch_size,
            mlp_ratio=4.0,
        )

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg = super(MAE_REG, MAE_REG).add_and_assert_specific_cfg(cfg)

        assert not omegaconf.OmegaConf.is_missing(
            cfg, "method_kwargs.decoder_embed_dim"
        )
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.decoder_depth")
        assert not omegaconf.OmegaConf.is_missing(
            cfg, "method_kwargs.decoder_num_heads"
        )

        cfg.method_kwargs.mask_ratio = omegaconf_select(
            cfg, "method_kwargs.mask_ratio", 0.75
        )
        cfg.method_kwargs.norm_pix_loss = omegaconf_select(
            cfg,
            "method_kwargs.norm_pix_loss",
            False,
        )
        cfg.method_kwargs.layers = omegaconf_select(cfg, "method_kwargs.layers", [])
        cfg.method_kwargs.scheduler = omegaconf_select(
            cfg, "method_kwargs.scheduler", {"name": "constant", "weight": 1.0}
        )
        cfg.method_kwargs.log_images = omegaconf_select(cfg, "method_kwargs.log_images", False)

        return cfg
    
    def configure_reg_scheduler(self, scheduler):
        if scheduler.name == "triangle":
            self.reg_scheduler = TriangleScheduler(
                start_weight=scheduler.start_weight,
                max_weight=scheduler.max_weight,
                end_weight=scheduler.end_weight,
                start_epoch=scheduler.start_epoch,
                mid_epoch=scheduler.mid_epoch,
                end_epoch=scheduler.end_epoch,
            )
        elif scheduler.name == "warmup":
            self.reg_scheduler = WarmupScheduler(
                base_weight=scheduler.base_weight,
                warmup_epochs=scheduler.warmup_epochs,
                weight=scheduler.weight,
                reg_epochs=scheduler.reg_epochs,
            )
        else:
            self.reg_scheduler = ConstantScheduler(
                weight=scheduler.weight,
            )

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector and predictor parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [
            {"name": "decoder", "params": self.decoder.parameters()},
        ]
        return super().learnable_params + extra_learnable_params

    def forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """Performs forward pass of the online backbone, projector and predictor.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """

        # modified base forward
        if not self.no_channel_last:
            X = X.to(memory_format=torch.channels_last)
        out = {}

        handles = []
        for number, block in enumerate(self.backbone.blocks):

            def hook_fn(module, input, output, number=number):
                mean_token_representation = output[:, 1:, :].mean(dim=1)
                out.update({f"mean_block_{number}": mean_token_representation})
                out.update({f"cls_block_{number}": output[:, 0, :]})

            handle = block.register_forward_hook(hook_fn)
            handles.append(handle)

        if self.training:
            feats, patch_feats, mask, ids_restore = self.backbone(X, self.mask_ratio)
            pred = self.decoder(patch_feats, ids_restore)
            out.update({"mask": mask, "pred": pred})
        else:
            feats = self.backbone(X)

        logits = self.classifier(feats.detach())
        out.update({"logits": logits, "feats": feats})

        for handle in handles:
            handle.remove()
        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for MAE reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of MAE and classification loss.
        """

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        metrics = {}

        patch_size = self._vit_patch_size
        imgs = batch[1]
        reconstruction_loss = 0
        regularizer_loss = 0
        for i in range(self.num_large_crops):
            reconstruction_loss += mae_loss_func(
                imgs[i],
                out["pred"][i],
                out["mask"][i],
                patch_size,
                norm_pix_loss=self.norm_pix_loss,
            )
        reconstruction_loss /= self.num_large_crops

        last_block_number = len(self.backbone.blocks) - 1
        if last_block_number in self.layers:
            self.layers.remove(last_block_number)

        for layer in self.layers:
            if layer != last_block_number:
                loss_term, laplacian_metrics = self.manifold_regularizer.manifold_regularizer_loss(
                    out[f"mean_block_{layer}"][0],
                    out[f"mean_block_{last_block_number}"][0],
                )
                for key, value in laplacian_metrics.items():
                    metrics[f"{key}_Layer{layer}"] = value
                regularizer_loss += loss_term

        # Divide loss by the number of terms in the regularization loss
        if len(self.layers) > 0:
            regularizer_loss /= len(self.layers)

        metrics.update({
            "train_reconstruction_loss": reconstruction_loss,
            "train_regularization_loss": regularizer_loss,
        })

        for number, _ in enumerate(self.backbone.blocks):
            metrics.update(
                {
                    f"standard_deviation_cls_{number}": out[f"cls_block_{number}"][0]
                    .std(dim=0)
                    .mean()
                }
            )

        regularizer_weight = self.reg_scheduler(self.current_epoch)
        regularization_loss_scaled = regularizer_loss * regularizer_weight
        metrics.update(
            {
                "train_regularizer_weight": regularizer_weight,
                "train_regularization_loss_scaled": regularization_loss_scaled,
            }
        )

        self.log_dict(metrics, on_epoch=True, sync_dist=True)
        print(regularization_loss_scaled)
        return reconstruction_loss + class_loss + regularization_loss_scaled

    def validation_step(
        self, batch: List[torch.Tensor], batch_idx: int, dataloader_idx: int = None
    ) -> Dict[str, Any]:
        """Validation step for pytorch lightning. It does all the shared operations, such as
        forwarding a batch of images, computing logits and computing metrics.

        Args:
            batch (List[torch.Tensor]):a batch of data in the format of [img_indexes, X, Y].
            batch_idx (int): index of the batch.

        Returns:
            Dict[str, Any]: dict with the batch_size (used for averaging), the classification loss
                and accuracies.
        """

        X, targets = batch
        batch_size = targets.size(0)

        out = self.base_validation_step(X, targets)

        laplacian_matrices = {}
        similarity_matrices = {}
        for layer in self.layers:
            similarity_matrix = get_similarity_matrix(
                out[f"mean_block_{layer}"], rbf_scale=1.0, scaling_factor=False
            )
            laplacian_matrix = get_laplacian(similarity_matrix, normalized=True)
            similarity_matrices[f"SimilarityMatrix_Layer{layer}"] = similarity_matrix
            laplacian_matrices[f"LaplacianMatrix_Layer{layer}"] = laplacian_matrix

        # Necessary for the sanity check runs because of https://github.com/Lightning-AI/lightning/issues/13108
        if out['feats'].device != self.validation_mean_embeddings[0].device:
            self.validation_mean_embeddings = {
                class_idx: embedding.to(out['feats'].device) for class_idx, embedding in self.validation_mean_embeddings.items()
            }
            print("Got here")

        for idx, target in enumerate(targets):
            self.validation_mean_embeddings[target.item()] += out["feats"][idx]
            self.validation_class_counts[target.item()] += 1

        if self.knn_eval and not self.trainer.sanity_checking:
            self.knn.update(
                test_features=out.pop("feats").detach(), test_targets=targets.detach()
            )

        metrics = {
            "batch_size": batch_size,
            "val_loss": out["loss"],
            "val_acc1": out["acc1"],
            "val_acc5": out["acc5"],
        }
        metrics.update(laplacian_matrices)
        metrics.update(similarity_matrices)
        return metrics

    def validation_epoch_end(self, outs: List[Dict[str, Any]]):
        """Averages the losses and accuracies of all the validation batches.
        This is needed because the last batch can be smaller than the others,
        slightly skewing the metrics.

        Args:
            outs (List[Dict[str, Any]]): list of outputs of the validation step.
        """

        val_loss = weighted_mean(outs, "val_loss", "batch_size")
        val_acc1 = weighted_mean(outs, "val_acc1", "batch_size")
        val_acc5 = weighted_mean(outs, "val_acc5", "batch_size")

        log = {"val_loss": val_loss, "val_acc1": val_acc1, "val_acc5": val_acc5}

        if self.log_images:
            for layer in self.layers:
                laplacian_matrix = tensor_mean(
                    outs, f"LaplacianMatrix_Layer{layer}", "batch_size"
                )
                logged_laplacian_matrix = torch.eye(laplacian_matrix.shape[0], device=laplacian_matrix.device) - laplacian_matrix
                log.update({f"MaxValueLoggedLaplacianMatrix_Layer{layer}": logged_laplacian_matrix.max()})
                title = f"Laplacian matrix at layer {layer}. Epoch {self.current_epoch}"
                self.logger.log_image(
                    key=f"LaplacianMatrixLinearScale_Layer{layer}",
                    images=[get_heatmap(logged_laplacian_matrix, norm=colors.Normalize(vmin=0, vmax=0.1), title=title)],
                    caption=[title],
                    step = self.current_epoch,
                )
                self.logger.log_image(
                    key=f"LaplacianMatrixLogScale_Layer{layer}",
                    images=[get_heatmap(logged_laplacian_matrix, norm=colors.LogNorm(vmax=0.1), title=title)],
                    caption=[title],
                    step = self.current_epoch,
                )

                title = f"Similarity matrix at layer {layer}. Epoch {self.current_epoch}"
                similarity_matrix = tensor_mean(
                    outs, f"SimilarityMatrix_Layer{layer}", "batch_size"
                )
                log.update({f"MaxValueSimilarityMatrix_Layer{layer}": similarity_matrix.max()})
                self.logger.log_image(
                    key=f"SimilarityMatrixLinearScale_Layer{layer}",
                    images=[get_heatmap(similarity_matrix, norm=colors.Normalize(vmin=0, vmax=1), title=title)],
                    caption=[title],
                    step = self.current_epoch,
                )
                self.logger.log_image(
                    key=f"SimilarityMatrixLogScale_Layer{layer}",
                    images=[get_heatmap(similarity_matrix, norm=colors.LogNorm(vmax=1), title=title)],
                    caption=[title],
                    step = self.current_epoch,
                )
            
            # Log the mean embeddings laplacian/similarity matrices
            for class_idx in range(self.num_classes):
                # This can be 0 during the sanity check runs
                if self.validation_class_counts[class_idx]:
                    self.validation_mean_embeddings[class_idx] /= self.validation_class_counts[class_idx]
            tensors = torch.stack([v for (_, v) in sorted(self.validation_mean_embeddings.items(), key=lambda item: item[0])])
            similarity_matrix = get_similarity_matrix(tensors, rbf_scale=1.0, scaling_factor=False)
            laplacian_matrix = get_laplacian(similarity_matrix, normalized=True)
            title = f"Similarity matrix of mean embeddings per class. Epoch {self.current_epoch}"
            log.update({"MaxValueSimilarityMatrix": similarity_matrix.max()})
            self.logger.log_image(
                    key=f"MeanEmbeddingSimilarityMatrix",
                    images=[get_heatmap(similarity_matrix, norm=colors.Normalize(vmin=0, vmax=1), title=title)],
                    caption=[title],
                    step = self.current_epoch,
            )
            title = f"Laplacian matrix of mean embeddings per class. Epoch {self.current_epoch}"
            logged_laplacian_matrix = torch.eye(laplacian_matrix.shape[0], device=laplacian_matrix.device) - laplacian_matrix
            log.update({"MaxValueLoggedLaplacianMatrix": logged_laplacian_matrix.max()})
            self.logger.log_image(
                    key=f"MeanEmbeddingLaplacianMatrixLinearScale",
                    images=[get_heatmap(logged_laplacian_matrix, norm=colors.Normalize(vmin=0, vmax=1), title=title)],
                    caption=[title],
                    step = self.current_epoch,
            )
            self.logger.log_image(
                    key=f"MeanEmbeddingLaplacianMatrixLogScale",
                    images=[get_heatmap(logged_laplacian_matrix, norm=colors.LogNorm(vmax=0.1), title=title)],
                    caption=[title],
                    step = self.current_epoch,
            )
                

        if self.knn_eval and not self.trainer.sanity_checking:
            val_knn_acc1, val_knn_acc5 = self.knn.compute()
            log.update({"val_knn_acc1": val_knn_acc1, "val_knn_acc5": val_knn_acc5})

        self.log_dict(log, sync_dist=True)
        self.validation_mean_embeddings = {
            class_idx: torch.zeros(self.features_dim, device=self.device) for class_idx in range(self.num_classes)
        }
        self.validation_class_counts = {
            class_idx: 0 for class_idx in range(self.num_classes)
        }
