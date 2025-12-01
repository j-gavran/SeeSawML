import logging
import os
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn

from seesaw.models.flat_preprocessor import FlatPreprocessor
from seesaw.models.jagged_preprocessor import JaggedPreprocessor
from seesaw.models.layers import FlatJaggedFuser
from seesaw.models.transformers.set_attention import EventsSetDecoder, SetPredictorNet, SetTransformer


def build_object_padding_mask(valid_i: torch.Tensor, valid_j: torch.Tensor) -> torch.Tensor:
    # 1 = valid, 0 = invalid
    outer = torch.einsum("b i, b j -> b i j", valid_i.bool(), valid_j.bool())
    # 0 = valid, 1 = invalid
    outer = ~outer

    return rearrange(outer, "b i j -> b 1 i j")


def plot_attention_mask(
    mask: torch.Tensor, title: str | None = None, postfix: str | None = None, max_batch: int = 10
) -> None:
    if title is None:
        title = "Padding Mask"

    title += " (0=Valid, 1=Masked)"

    for b_i in range(mask.shape[0]):
        if b_i == max_batch:
            break

        mask_np = mask[b_i].squeeze(0).cpu().numpy()

        mask_dir = f"{os.environ['ANALYSIS_ML_RESULTS_DIR']}/masks/"
        os.makedirs(mask_dir, exist_ok=True)

        plt.figure(figsize=(8, 7))
        sns.heatmap(mask_np, cmap="viridis", linewidths=0.1, linecolor="k", cbar=True)

        plt.title(title)
        plt.xlabel("Key Positions")
        plt.ylabel("Query Positions")

        file_name = f"debug_attention_mask_{b_i}"
        if postfix is not None:
            file_name += f"_{postfix}"

        logging.info(f"Saving {file_name} attention mask to {mask_dir} for debugging...")

        plt.tight_layout()
        plt.savefig(mask_dir + f"{file_name}.png", dpi=300)
        plt.close()


class JaggedTransformer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        numer_idx: dict[str, np.ndarray],
        categ_idx: dict[str, np.ndarray],
        categories: dict[str, np.ndarray],
        numer_padding_tokens: dict[str, float | None],
        categ_padding_tokens: dict[str, int | None],
        object_dimensions: dict[str, int],
        heads: int = 8,
        encoder_depth: int = 2,
        decoder_depth: int | None = None,
        dim_head: int = 8,
        ff_hidden_mult: int = 4,
        dim_out: int = 1,
        act_out: str | None = None,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        seed_strategy: str = "pooling",
        set_predictor_dct: dict[str, Any] | None = None,
        cross_decoder_depth: int = 2,
        embedding_config_dct: dict[str, Any] | None = None,
        set_transform_events: bool = False,
        use_setnorm: bool = True,
        add_particle_types: bool = False,
        flat_model: nn.Module | None = None,
        flat_fuse: dict[str, Any] | None = None,
        sdp_backend: dict[str, bool] | None = None,
        debug_masks: bool = False,
    ) -> None:
        super().__init__()
        if set_transform_events and flat_model is not None:
            raise ValueError("Cannot use set transform on events and a flat model simultaneously.")

        self.debug_masks = debug_masks

        if set_predictor_dct is None:
            set_predictor_dct = {}

        events_numer_idx, events_categ_idx = numer_idx["events"], categ_idx["events"]
        total_events_features = len(events_numer_idx) + len(events_categ_idx)

        if not total_events_features > 0 and set_transform_events:
            raise ValueError("Set transform with events is set to True but no event features were provided!")

        self.set_transform_events = set_transform_events

        self.disable_decoder_masking = True

        if seed_strategy == "pooling":
            num_seeds = 1
        elif seed_strategy == "objects":
            num_seeds, self.disable_decoder_masking = sum(object_dimensions.values()), False
        elif seed_strategy == "particles":
            num_seeds = len(object_dimensions)
        else:
            raise ValueError(f"Invalid seed_strategy: {seed_strategy}, must be 'pooling', 'objects' or 'particles'.")

        self.num_seeds = num_seeds

        if embedding_config_dct is None:
            embedding_config_dct = {}

        self.jagged_preprocessor = JaggedPreprocessor(
            embedding_dim=embedding_dim,
            numer_idx=numer_idx,
            categ_idx=categ_idx,
            categories=categories,
            numer_padding_tokens=numer_padding_tokens,
            categ_padding_tokens=categ_padding_tokens,
            numer_feature_wise_linear=embedding_config_dct.get("numer_feature_wise_linear", False),
            reduction=embedding_config_dct.get("reduction", "mean"),
            conv1d_embedding=embedding_config_dct.get("conv1d_embedding", False),
            post_embeddings_dct=embedding_config_dct.get("post_embeddings_dct", None),
            ple_dct=embedding_config_dct.get("ple_config", None),
            add_particle_types=add_particle_types,
        )

        if self.set_transform_events:
            self.flat_preprocessor = FlatPreprocessor(
                embedding_dim=embedding_dim,
                numer_idx=events_numer_idx,
                categ_idx=events_categ_idx,
                categories=categories["events"],
                embedding_reduction=None,
                post_embeddings_dct=embedding_config_dct.get("post_embeddings_dct", None),
                ple_dct=embedding_config_dct.get("ple_config", None),
                disable_embeddings=False,
            )

            self.set_events_transformer = SetTransformer(
                dim=embedding_dim,
                encoder_depth=encoder_depth,
                decoder_depth=decoder_depth,
                heads=heads,
                dim_head=dim_head,
                mlp_dim=2 * ff_hidden_mult * embedding_dim,
                num_inducing_points=None,
                num_seeds=self.num_seeds,
                dim_out=dim_out,
                act_out=act_out,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                disable_decoder_masking=True,
                use_predict=False,
                use_setnorm=use_setnorm,
                sdp_backend=sdp_backend,
            )

            self.to_set_events_proj = nn.Sequential(
                Rearrange("b f e -> b 1 (f e)"),
                nn.Linear(total_events_features * embedding_dim, embedding_dim),
            )

        self.set_jagged_transformer = SetTransformer(
            dim=embedding_dim,
            encoder_depth=encoder_depth,
            decoder_depth=decoder_depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=2 * ff_hidden_mult * embedding_dim,
            num_inducing_points=None,
            num_seeds=self.num_seeds,
            dim_out=dim_out,
            act_out=act_out,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            act_predict=set_predictor_dct.get("act", "ReLU"),
            depth_predict=set_predictor_dct.get("depth", 1),
            pool_predict=set_predictor_dct.get("mean_pooling", False),
            disable_decoder_masking=self.disable_decoder_masking,
            use_predict=True if not self.set_transform_events else False,
            use_setnorm=use_setnorm,
            sdp_backend=sdp_backend,
        )

        self.set_decoder: nn.Module

        if self.set_transform_events and self.num_seeds > 1:
            self.set_decoder = EventsSetDecoder(
                dim=embedding_dim,
                depth=cross_decoder_depth,
                heads=heads,
                dim_head=dim_head,
                mlp_dim=2 * ff_hidden_mult * embedding_dim,
                num_seeds=self.num_seeds,
                dim_out=dim_out,
                act_out=act_out,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                act_predict=set_predictor_dct.get("act", "ReLU"),
                depth_predict=set_predictor_dct.get("depth", 1),
                pool_predict=set_predictor_dct.get("mean_pooling", False),
                disable_decoder_masking=self.disable_decoder_masking,
                sdp_backend=sdp_backend,
            )

        if self.set_transform_events and self.num_seeds == 1:
            self.set_decoder = SetPredictorNet(
                dim=embedding_dim,
                num_seeds=self.num_seeds,
                dim_out=dim_out,
                act_out=act_out,
                depth=set_predictor_dct.get("depth", 1),
                act=set_predictor_dct.get("act", "ReLU"),
                pool_predict=set_predictor_dct.get("mean_pooling", False),
                disable_reshape=True,
                ff_dropout=ff_dropout,
            )

        if flat_fuse is None:
            flat_fuse = {}

        self.flat_model = flat_model

        if flat_model or set_transform_events:
            self.fuser = FlatJaggedFuser(
                flat_fuse.get("mode", "add"),
                output_dim=dim_out,
                fuse_kwargs=flat_fuse.get("fuse_kwargs", None),
            )

    def forward(self, X_events: torch.Tensor, *Xs: torch.Tensor) -> torch.Tensor:
        x_jagged, x_jagged_valid = self.jagged_preprocessor(*Xs)

        x_jagged_object_mask = build_object_padding_mask(x_jagged_valid, x_jagged_valid)

        x_pooling_valid = torch.ones(x_jagged.shape[0], self.num_seeds, dtype=torch.bool, device=x_jagged.device)
        x_jagged_pooling_mask = build_object_padding_mask(x_pooling_valid, x_jagged_valid)

        if self.debug_masks:
            timestamp = int(time.time())
            plot_attention_mask(x_jagged_object_mask, title="Object Jagged Mask", postfix=f"{timestamp}_obj_jagged")
            plot_attention_mask(x_jagged_pooling_mask, title="Pooling Mask", postfix=f"{timestamp}_pool")

        x_jagged_set_out = self.set_jagged_transformer(
            x_jagged,
            attn_mask=x_jagged_object_mask,
            set_q_mask=x_jagged_valid,
            set_kv_mask=x_jagged_valid,
            pooling_attn_mask=x_jagged_pooling_mask,
            pooling_set_q_mask=x_pooling_valid,
            pooling_set_kv_mask=x_jagged_valid,
        )

        if self.flat_model is not None:
            flat_out = self.flat_model(X_events)
            return self.fuser(flat_out, x_jagged_set_out)

        if not self.set_transform_events:
            return x_jagged_set_out

        x_events = self.flat_preprocessor(X_events)

        if not self.disable_decoder_masking:
            x_events_valid = torch.ones(x_events.shape[0], self.num_seeds, dtype=torch.bool, device=x_events.device)
            x_events_mask = build_object_padding_mask(x_jagged_valid, x_events_valid)
        else:
            x_events_valid, x_events_mask = None, None

        if self.debug_masks and not self.disable_decoder_masking:
            plot_attention_mask(x_events_mask, title="Object Events Mask", postfix=f"{timestamp}_obj_events")  # type: ignore[arg-type]

        x_events = self.to_set_events_proj(x_events)
        x_events_set_out = self.set_events_transformer(x_events)

        if self.num_seeds > 1:
            x_enc_out = self.set_decoder(
                x_jagged_set_out,
                x_events_set_out,
                attn_mask=x_events_mask,
                set_q_mask=x_jagged_valid,
                set_kv_mask=x_events_valid,
            )
        else:
            x_fused = self.fuser(x_events_set_out, x_jagged_set_out)
            x_enc_out = self.set_decoder(x_fused)

        return x_enc_out
