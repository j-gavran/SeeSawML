from typing import Any

import numpy as np
import torch
from einops import rearrange
from torch import nn

from seesaw.models.activations import get_activation
from seesaw.models.jagged_preprocessor import JaggedPreprocessor
from seesaw.models.layers import FlatJaggedEmbeddingsFuser, FlatJaggedModelFuser
from seesaw.models.transformers.attention import AttentionBlock, build_adjacency_attention_mask
from seesaw.models.transformers.pairwise_features import (
    PairwiseFeaturesCalculator,
    PairwiseFeaturesEmbedder,
    ParticleAttentionConfig,
    ParticleFeatureSpec,
    derive_energy_from_mass,
)


class ParticleTransformer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        numer_idx: dict[str, np.ndarray],
        categ_idx: dict[str, np.ndarray],
        categories: dict[str, np.ndarray],
        numer_padding_tokens: dict[str, float | None],
        categ_padding_tokens: dict[str, int | None],
        object_dimensions: dict[str, int],
        act_out: str | None = None,
        heads: int = 8,
        dim_head: int = 32,
        particle_blocks: int = 3,
        class_blocks: int = 2,
        ff_hidden_mult: int = 4,
        dim_out: int = 1,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        embedding_config_dct: dict[str, Any] | None = None,
        add_particle_types: bool = False,
        flat_embeddings: nn.Module | None = None,
        flat_embeddings_fuse: dict[str, Any] | None = None,
        flat_model: nn.Module | None = None,
        flat_model_fuse: dict[str, Any] | None = None,
        particle_attention: ParticleAttentionConfig | None = None,
        first_attn_no_residual: bool = False,
        sdp_backend: dict[str, bool] | None = None,
    ) -> None:
        super().__init__()

        if embedding_config_dct is None:
            embedding_config_dct = {}

        if flat_embeddings_fuse is None:
            flat_embeddings_fuse = {}

        if flat_model_fuse is None:
            flat_model_fuse = {}

        self.flat_model = flat_model

        self.object_names = [name for name in numer_idx.keys() if name != "events"]
        self.object_name_to_tensor_idx = {name: idx for idx, name in enumerate(self.object_names)}
        self.object_slices: dict[str, slice] = {}
        start_idx = 0
        for name in self.object_names:
            length = object_dimensions[name]
            self.object_slices[name] = slice(start_idx, start_idx + length)
            start_idx += length

        # Pairwise attention setup
        self.particle_attention = particle_attention
        self._particle_attention_specs: list[ParticleFeatureSpec] = []
        self.pairwise_feature_dim: int | None = None

        if self.particle_attention is not None:
            spec_by_name = self.particle_attention.as_dict()

            self.pairwise_calculator = PairwiseFeaturesCalculator(quantities=self.particle_attention.quantities)

            self._particle_attention_specs = [spec_by_name[name] for name in self.object_names]
            self.pairwise_embedder = PairwiseFeaturesEmbedder(
                input_dim=len(self.particle_attention.quantities),
                embed_dims=self.particle_attention.embed_dims,
            )
            self.pairwise_feature_dim = self.pairwise_embedder.output_dim

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

        self.particle_blocks = nn.ModuleList(
            [
                AttentionBlock(
                    dim=embedding_dim,
                    heads=heads,
                    dim_head=dim_head,
                    mlp_dim=ff_hidden_mult * embedding_dim,
                    attn_dropout=attn_dropout,
                    ff_dropout=ff_dropout,
                    class_attention=False,
                    attention_residual=False if first_attn_no_residual and i == 0 else True,
                    sdp_backend=sdp_backend,
                )
                for i in range(particle_blocks)
            ]
        )

        self.final_particle_ln = nn.LayerNorm(embedding_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.class_blocks = nn.ModuleList(
            [
                AttentionBlock(
                    dim=embedding_dim,
                    heads=heads,
                    dim_head=dim_head,
                    mlp_dim=ff_hidden_mult * embedding_dim,
                    attn_dropout=attn_dropout,
                    ff_dropout=ff_dropout,
                    class_attention=True,
                    sdp_backend=sdp_backend,
                )
                for _ in range(class_blocks)
            ]
        )

        hidden = embedding_dim * ff_hidden_mult
        self.to_logits = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(hidden, dim_out),
            get_activation(act_out),
        )

        self.flat_embeddings = flat_embeddings
        self.disable_flat_embeddings_mask = True if embedding_config_dct.get("ple_config", None) is not None else False

        if flat_embeddings is not None:
            self.embeddings_fuser = FlatJaggedEmbeddingsFuser(
                flat_embeddings_fuse.get("mode", "add"),
                output_dim=embedding_dim,
                fuse_kwargs=flat_embeddings_fuse.get("fuse_kwargs", None),
            )

        self.flat_model = flat_model

        if flat_model is not None:
            self.model_fuser = FlatJaggedModelFuser(
                flat_model_fuse.get("mode", "add"),
                output_dim=dim_out,
                fuse_kwargs=flat_model_fuse.get("fuse_kwargs", None),
            )

    def forward(self, X_events: torch.Tensor, *Xs: torch.Tensor) -> torch.Tensor:
        x_jagged, x_jagged_valid = self.jagged_preprocessor(*Xs)
        particle_adjacency_mask = build_adjacency_attention_mask(x_jagged_valid)

        pairwise_bias: torch.Tensor | None = None

        if self.particle_attention is not None:
            pairwise_raw, pairwise_mask = self._build_pairwise_bias_raw(Xs, x_jagged_valid)
            pairwise_bias = self.pairwise_embedder(pairwise_raw, pairwise_mask.unsqueeze(1))

        if self.flat_embeddings is not None:
            x_flat = self.flat_embeddings(X_events)
            x_jagged = self.embeddings_fuser(
                x_flat, x_jagged, mask=None if self.disable_flat_embeddings_mask else x_jagged_valid
            )

        for block in self.particle_blocks:
            x_jagged = block(x_jagged, mask=particle_adjacency_mask, bias=pairwise_bias)

        x_jagged = self.final_particle_ln(x_jagged)

        cls_token = self.cls_token.expand(x_jagged.shape[0], -1, -1)

        cls_mask = torch.zeros(x_jagged.shape[0], 1, dtype=torch.bool, device=x_jagged.device)
        cls_mask = torch.cat([cls_mask, x_jagged_valid], dim=1)
        cls_mask = rearrange(cls_mask, "b n -> b 1 1 n")

        for block in self.class_blocks:
            cls_token = block(cls_token, x_jagged, mask=cls_mask)

        logits = self.to_logits(cls_token.squeeze(1))

        if self.flat_model is not None:
            flat_out = self.flat_model(X_events)
            return self.model_fuser(flat_out, logits)

        return logits

    def _build_pairwise_bias_raw(
        self,
        jagged_inputs: tuple[torch.Tensor, ...],
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build pairwise features (B, N, N, C) and corresponding pair mask."""

        pt_parts: list[torch.Tensor] = []
        eta_parts: list[torch.Tensor] = []
        phi_parts: list[torch.Tensor] = []
        energy_parts: list[torch.Tensor] = []
        mask_parts: list[torch.Tensor] = []

        for spec in self._particle_attention_specs:
            tensor_idx = self.object_name_to_tensor_idx[spec.object_name]
            x_obj = jagged_inputs[tensor_idx]

            mask_slice = self.object_slices[spec.object_name]
            obj_mask = valid_mask[:, mask_slice]

            pt = x_obj[:, :, spec.pt_index]
            eta = x_obj[:, :, spec.eta_index]
            phi = x_obj[:, :, spec.phi_index]

            pt = pt.masked_fill(obj_mask, 0.0)
            eta = eta.masked_fill(obj_mask, 0.0)
            phi = phi.masked_fill(obj_mask, 0.0)

            if spec.energy_index is not None:
                energy = x_obj[:, :, spec.energy_index]
            else:
                energy = derive_energy_from_mass(pt, eta, spec.rest_mass)  # type: ignore[arg-type]

            energy = energy.masked_fill(obj_mask, 0.0)

            pt_parts.append(pt)
            eta_parts.append(eta)
            phi_parts.append(phi)
            energy_parts.append(energy)
            mask_parts.append(obj_mask)

        pt_all = torch.cat(pt_parts, dim=1)
        eta_all = torch.cat(eta_parts, dim=1)
        phi_all = torch.cat(phi_parts, dim=1)
        energy_all = torch.cat(energy_parts, dim=1)
        is_padded_all = torch.cat(mask_parts, dim=1)

        pairwise_features, pairwise_mask = self.pairwise_calculator(
            pt=pt_all,
            eta=eta_all,
            phi=phi_all,
            energy=energy_all,
            mask=is_padded_all,
        )

        return pairwise_features, pairwise_mask

    @torch.no_grad()
    def get_pairwise_features_for_plotting(self, *Xs: torch.Tensor, max_events: int = 1) -> dict[str, Any] | None:
        """Extract pairwise features and metadata for visualization.

        Parameters
        ----------
        *Xs : torch.Tensor
            Jagged input tensors (same as forward, but without X_events).
        max_events : int
            Maximum number of events to return (default 1 for single-event plot).

        Returns
        -------
        dict or None
            Dictionary with keys:
            - "features": np.ndarray of shape (B, N, N, C) or (N, N, C) if max_events=1
            - "object_labels": list of labels like ["j1", "j2", ..., "e1", "μ1"]
            - "object_boundaries": cumulative boundaries [n_jets, n_jets+n_el, ...]
            - "participation_mask": np.ndarray (N, N) or None if all participate
            - Returns None if particle attention is not enabled
        """
        if self.particle_attention is None or self._particle_attention_specs is None:
            return None

        # Get valid mask from preprocessor
        _, x_jagged_valid = self.jagged_preprocessor(*Xs)

        # Build pairwise features (without participation mask for visualization)
        pairwise_raw, _ = self._build_pairwise_bias_raw(Xs, x_jagged_valid)

        # Limit to max_events
        pairwise_raw = pairwise_raw[:max_events]

        # Build object labels
        object_labels: list[str] = []
        object_boundaries: list[int] = []
        cumulative = 0

        label_prefixes = {
            "jets": "j",
            "electrons": "e",
            "muons": "mu",
        }

        for name in self.object_names:
            obj_slice = self.object_slices[name]
            n_objects = obj_slice.stop - obj_slice.start
            prefix = label_prefixes.get(name, name[0])
            for i in range(n_objects):
                object_labels.append(f"{prefix}{i + 1}")
            cumulative += n_objects
            object_boundaries.append(cumulative)

        # Build participation mask if subset
        participation_mask = None
        if self.particle_attention.participating_objects is not None:
            total_length = pairwise_raw.shape[1]
            participation_mask = np.zeros((total_length, total_length), dtype=bool)
            for obj_a in self.particle_attention.participating_objects:
                slice_a = self.object_slices.get(obj_a)
                if slice_a is None:
                    continue
                for obj_b in self.particle_attention.participating_objects:
                    slice_b = self.object_slices.get(obj_b)
                    if slice_b is None:
                        continue
                    participation_mask[slice_a, slice_b] = True

        features_np = pairwise_raw.cpu().numpy()
        if max_events == 1:
            features_np = features_np[0]

        return {
            "features": features_np,
            "object_labels": object_labels,
            "object_boundaries": object_boundaries,
            "participation_mask": participation_mask,
        }
