import logging
from typing import Any

import numpy as np
import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn

from seesawml.models.activations import get_activation
from seesawml.models.layers import FeatureWiseLinear, MeanPoolingLayer
from seesawml.models.masked_batchnorm import MaskedBatchNorm1d
from seesawml.models.ple import LearnablePiecewiseEncodingLayer, QuantilePiecewiseEncodingLayer


class JaggedNumEmbeddingModel(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_numerical_types: int,
        feature_wise_linear: bool = False,
        bias: bool = True,
        use_layernorm: bool = False,
    ) -> None:
        super().__init__()
        self.bias = bias
        self.use_layernorm = use_layernorm

        if feature_wise_linear:
            self.per_feature = True

            self.features_linear = FeatureWiseLinear(num_numerical_types, embedding_dim, bias=bias)
        else:
            self.per_feature = False

            self.weights = nn.Parameter(torch.empty(num_numerical_types, embedding_dim))
            if self.bias:
                self.biases = nn.Parameter(torch.empty(num_numerical_types, embedding_dim))

            self.reset_parameters()

        if self.use_layernorm:
            self.norm = nn.LayerNorm(embedding_dim)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weights, mean=0.0, std=0.02)

        if self.bias:
            nn.init.zeros_(self.biases)

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        if self.per_feature:
            x = self.features_linear(x_num)
        else:
            x = rearrange(x_num, "b o f -> b o f 1")

            if self.bias:
                x = x * self.weights + self.biases
            else:
                x = x * self.weights

        if self.use_layernorm:
            x = self.norm(x)

        return x


class JaggedPLENumEmbeddingModel(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_numerical_types: int,
        ple_dct: dict[str, Any],
        dataset_key: str | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.bias = bias

        learn_bins = ple_dct.get("learn_bins", False)
        uniform_bins = ple_dct.get("uniform_bins", False)

        act = ple_dct.get("act", None)
        dropout = ple_dct.get("dropout", 0.0)
        self.use_dropout = True if dropout > 0.0 else False
        self.use_layernorm = ple_dct.get("layernorm", False)

        if learn_bins and uniform_bins:
            raise ValueError("Only one of learn_bins or uniform_bins can be True!")

        ples: list[nn.Module] = []

        if learn_bins or uniform_bins:
            for i in range(num_numerical_types):
                learnable_ple = LearnablePiecewiseEncodingLayer(
                    bins=ple_dct["n_bins"],
                    embedding_dim=embedding_dim,
                    act=act,
                    learn_bins=True if learn_bins else False,
                    bias=bias,
                )
                ples.append(learnable_ple)
        else:
            for i in range(num_numerical_types):
                quantile_ple = QuantilePiecewiseEncodingLayer(
                    ple_file_hash_str=ple_dct["ple_file_hash_str"],
                    feature_idx=i,
                    embedding_dim=embedding_dim,
                    act=act,
                    bias=bias,
                    dataset_key=dataset_key,
                )
                ples.append(quantile_ple)

        self.ple = nn.ModuleList(ples)

        if self.use_layernorm:
            self.norm = nn.LayerNorm(embedding_dim)

        if self.use_dropout:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_value: torch.Tensor | None = None) -> torch.Tensor:
        embeddings = []
        for i in range(x.shape[2]):
            x_f = rearrange(x[:, :, i], "b o -> (b o)")
            emb = self.ple[i](x_f, pad_value)
            emb = rearrange(emb, "(b o) d -> b o d", b=x.shape[0], o=x.shape[1])
            embeddings.append(emb)

        x = torch.stack(embeddings, dim=2)

        if self.use_layernorm:
            x = self.norm(x)

        if self.use_dropout:
            x = self.dropout(x)

        return x


class JaggedCatEmbeddingModel(nn.Module):
    def __init__(self, embedding_sizes: list[tuple[int, int]], use_layernorm: bool = False) -> None:
        super().__init__()
        self.use_layernorm = use_layernorm

        if set(emb_dim for _, emb_dim in embedding_sizes) != {embedding_sizes[0][1]}:
            raise ValueError("All embedding dimensions must be the same!")

        self.embeddings = nn.ModuleList(
            [nn.Embedding(num_categories, emb_dim) for num_categories, emb_dim in embedding_sizes]
        )

        if self.use_layernorm:
            self.norm = nn.LayerNorm(embedding_sizes[0][1])

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        embedded = [emb(x_cat[:, :, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.stack(embedded, dim=2)

        if self.use_layernorm:
            x = self.norm(x)

        return x


class IdentityEmbedder(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(x, "b o f -> b o f 1").expand(-1, -1, -1, self.embedding_dim)


class MaskedSequential(nn.Module):
    def __init__(
        self,
        seq: nn.Sequential,
        first_mask_rearrange: Rearrange | None = None,
        second_mask_rearrange: Rearrange | None = None,
        skip_second_mask: bool = False,
    ) -> None:
        super().__init__()
        self.seq = seq
        self.first_mask_rearrange = first_mask_rearrange
        self.second_mask_rearrange = second_mask_rearrange
        self.skip_second_mask = skip_second_mask

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            return self.seq(x)

        if self.first_mask_rearrange is not None:
            first_mask = self.first_mask_rearrange(mask)
        else:
            first_mask = mask

        x = x.masked_fill(first_mask, 0.0)  # type: ignore[arg-type]

        x = self.seq(x)

        if self.skip_second_mask:
            return x

        if self.second_mask_rearrange is not None:
            second_mask = self.second_mask_rearrange(mask)
        else:
            second_mask = first_mask

        x = x.masked_fill(second_mask, 0.0)  # type: ignore[arg-type]

        return x


class Attention2DPoolingReduction(nn.Module):
    def __init__(self, embedding_dim: int, simple: bool = False) -> None:
        super().__init__()
        self.simple = simple

        self.conv = nn.Conv2d(embedding_dim, 1, kernel_size=(1, 1))
        self.softmax = nn.Softmax(dim=-1)

        if not self.simple:
            self.ln = nn.LayerNorm(embedding_dim)
            self.act = nn.GELU()
            self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, object_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (b, o, f, e), mask: (b, o)
        x = rearrange(x, "b o f e -> b e o f")  # (b, e, o, f)

        a = self.conv(x)  # (b, 1, o, f)

        if object_mask is not None:
            object_mask = rearrange(object_mask, "b o -> b 1 o 1")
            a = a.masked_fill(object_mask, float("-inf"))

        # normalize over f dimension only
        a = self.softmax(a)  # (b, 1, o, f)

        if object_mask is not None:
            a = torch.nan_to_num(a)

        y = (a * x).sum(dim=-1)  # (b, e, o, f) -> (b, e, o)

        y = rearrange(y, "b e o -> b o e")

        if not self.simple:
            y = self.dropout(self.ln(self.act(y)))

        return y


class MaskedBatchNorm1dJaggedEmbedding(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_rearrange = Rearrange("b o f -> b f o")
        self.batchnorm = MaskedBatchNorm1d(input_dim)

        self.seq = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=output_dim, kernel_size=1),
            Rearrange("b e o -> b o e"),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x_mask = rearrange(mask, "b o -> b o 1")

        bn_mask = ~rearrange(mask, "b o -> b 1 o")
        bn_mask = bn_mask.to(dtype=x.dtype)

        x = x.masked_fill(x_mask, 0.0)

        x = self.input_rearrange(x)
        x = self.batchnorm(x, bn_mask)
        x = self.seq(x)

        x = x.masked_fill(x_mask, 0.0)

        return x


class JaggedPreprocessor(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        numer_idx: dict[str, np.ndarray],
        categ_idx: dict[str, np.ndarray],
        categories: dict[str, np.ndarray],
        numer_padding_tokens: dict[str, float | None],
        categ_padding_tokens: dict[str, int | None],
        numer_feature_wise_linear: bool = False,
        reduction: str = "mean",
        conv1d_embedding: bool = False,
        post_embeddings_dct: dict[str, Any] | None = None,
        ple_dct: dict[str, Any] | None = None,
        add_particle_types: bool = False,
        valid_type_values: dict[str, list[int]] | None = None,
    ) -> None:
        super().__init__()
        if conv1d_embedding and ple_dct is not None:
            raise ValueError("conv1d_embedding and ple_dct cannot be used together!")

        # Store object names for type field lookup
        self.object_names = [k for k in numer_idx.keys() if k != "events"]

        # Setup type-based masking config
        self._setup_type_field_config(valid_type_values)

        self.add_particle_types = add_particle_types

        if add_particle_types:
            embedding_dim = embedding_dim - 1  # reserve 1 dim for particle type embedding

        self.use_ple = True if ple_dct is not None else False

        if conv1d_embedding:
            logging.info("Using object conv1d embedding, disabling other embeddings.")
            reduction = "conv1d"
            self.disable_embeddings = True
        else:
            self.disable_embeddings = False

        self._setup_numer_categ_indices(numer_idx, categ_idx)
        self._setup_padding_tokens(numer_padding_tokens, categ_padding_tokens)

        self._setup_embeddings(embedding_dim, numer_idx, categories, numer_feature_wise_linear, ple_dct=ple_dct)

        self._setup_projections(embedding_dim, numer_idx, categ_idx, reduction)
        self._setup_particle_types(add_particle_types, categories)

        self._setup_post_embeddings(embedding_dim, post_embeddings_dct)

    def _setup_type_field_config(
        self,
        valid_type_values: dict[str, list[int]] | None,
    ) -> None:
        if not valid_type_values:
            self.use_type_field_masking = False
            self._valid_type_tensors = nn.ParameterList()
            return

        self.use_type_field_masking = True
        self._logged_masking_stats = False  # One-time logging flag

        # ParameterList for ONNX compatibility (indexed by object position)
        self._valid_type_tensors = nn.ParameterList(
            [
                nn.Parameter(
                    torch.tensor(valid_type_values.get(name, []), dtype=torch.float32),
                    requires_grad=False,
                )
                for name in self.object_names
            ]
        )
        logging.info("[green]TYPE-BASED MASKING ENABLED[/green]")
        for name in self.object_names:
            vals = valid_type_values.get(name, [])
            if vals:
                logging.info(f"  {name}: valid_values={vals}")

    def _derive_mask_from_type(self, type_tensor: torch.Tensor, obj_idx: int) -> torch.Tensor:
        """Derive validity mask from type tensor. Returns True for invalid/masked.

        Parameters
        ----------
        type_tensor : torch.Tensor
            Type values tensor of shape (batch, max_objects) containing particle type values.
        obj_idx : int
            Index of the object type in self.object_names.

        Returns
        -------
        torch.Tensor
            Boolean mask where True = invalid/masked, False = valid.
        """
        valid_types = self._valid_type_tensors[obj_idx]
        is_valid = (type_tensor.unsqueeze(-1) == valid_types).any(dim=-1)
        return ~is_valid

    def _derive_mask_from_padding(self, x: torch.Tensor, obj_idx: int) -> torch.Tensor:
        """Fallback: derive mask where ALL numerical features equal padding token."""
        jagged_idx = self.numer_jagged_idx[obj_idx]
        if len(jagged_idx) == 0:
            # No numerical features - can't determine padding, assume all valid
            return torch.zeros(x.size(0), x.size(1), dtype=torch.bool, device=x.device)
        pad_token = self.numer_padding_tokens[obj_idx]
        return (x[:, :, jagged_idx] == pad_token).all(dim=-1)

    def _setup_numer_categ_indices(self, numer_idx: dict[str, np.ndarray], categ_idx: dict[str, np.ndarray]) -> None:
        numer_jagged_idx: list[np.ndarray | None] = []
        categ_jagged_idx: list[np.ndarray | None] = []

        for key in numer_idx:
            if key == "events":
                continue
            numer_idx_key = numer_idx[key]
            if len(numer_idx_key) > 0:
                numer_jagged_idx.append(numer_idx_key)
            else:
                numer_jagged_idx.append(None)

        for key in categ_idx:
            if key == "events":
                continue
            categ_idx_key = categ_idx[key]
            if len(categ_idx_key) > 0:
                categ_jagged_idx.append(categ_idx_key)
            else:
                categ_jagged_idx.append(None)

        numer_param_jagged_idx = []
        for idx in numer_jagged_idx:
            if idx is None:
                idx = np.array([], dtype=np.int64)

            param = nn.Parameter(torch.tensor(idx, dtype=torch.int64), requires_grad=False)
            numer_param_jagged_idx.append(param)

        self.numer_jagged_idx = nn.ParameterList(numer_param_jagged_idx)

        categ_param_jagged_idx = []
        for idx in categ_jagged_idx:
            if idx is None:
                idx = np.array([], dtype=np.int64)

            param = nn.Parameter(torch.tensor(idx, dtype=torch.int64), requires_grad=False)
            categ_param_jagged_idx.append(param)

        self.categ_jagged_idx = nn.ParameterList(categ_param_jagged_idx)

    def _setup_padding_tokens(
        self,
        numer_padding_tokens: dict[str, float | None],
        categ_padding_tokens: dict[str, int | None],
    ) -> None:
        self.numer_padding_tokens = nn.ParameterList()
        for t_num in numer_padding_tokens.values():
            if t_num is None:
                numer_param_token = -1.0
            else:
                numer_param_token = t_num

            param = nn.Parameter(torch.tensor(numer_param_token, dtype=torch.float32), requires_grad=False)
            self.numer_padding_tokens.append(param)

        self.categ_padding_tokens = nn.ParameterList()
        for t_cat in categ_padding_tokens.values():
            if t_cat is None:
                categ_param_token = -1
            else:
                categ_param_token = t_cat

            param = nn.Parameter(torch.tensor(categ_param_token, dtype=torch.int64), requires_grad=False)
            self.categ_padding_tokens.append(param)

    def _setup_embeddings(
        self,
        embedding_dim: int,
        numer_idx: dict[str, np.ndarray],
        categories: dict[str, np.ndarray],
        numer_feature_wise_linear: bool = False,
        ple_dct: dict[str, Any] | None = None,
    ) -> None:
        if self.disable_embeddings:
            return None

        numer_embedding_lst: list[nn.Module] = []

        # Use filtered indices (self.numer_jagged_idx) which excludes type field
        for i, obj_name in enumerate(self.object_names):
            filtered_idx = self.numer_jagged_idx[i]
            num_features = len(filtered_idx)

            numer_embedder: nn.Module

            if num_features == 0:
                numer_embedder = IdentityEmbedder(embedding_dim)
            else:
                if ple_dct is None:
                    numer_embedder = JaggedNumEmbeddingModel(
                        embedding_dim,
                        num_numerical_types=num_features,
                        feature_wise_linear=numer_feature_wise_linear,
                    )
                else:
                    numer_embedder = JaggedPLENumEmbeddingModel(
                        embedding_dim,
                        num_numerical_types=num_features,
                        ple_dct=ple_dct,
                        dataset_key=obj_name,
                    )

            numer_embedding_lst.append(numer_embedder)

        self.numer_jagged_embeddings = nn.ModuleList(numer_embedding_lst)

        categ_embeddings_lst: list[nn.Module] = []

        for obj_name, number_categ_arr in categories.items():
            if obj_name == "events":
                continue

            categ_embedder: nn.Module

            if number_categ_arr.shape[1] == 0:
                categ_embedder = IdentityEmbedder(embedding_dim)
                categ_embeddings_lst.append(categ_embedder)
                continue

            embedding_sizes = []

            for num_f in range(number_categ_arr.shape[1]):
                embedding_sizes.append((int(np.max(number_categ_arr[:, num_f])), embedding_dim))

            categ_embedder = JaggedCatEmbeddingModel(embedding_sizes)
            categ_embeddings_lst.append(categ_embedder)

        self.categ_jagged_embeddings = nn.ModuleList(categ_embeddings_lst)

    def _setup_post_embeddings(self, embedding_dim: int, post_embeddings_dct: dict[str, Any] | None) -> None:
        if post_embeddings_dct is None:
            self.post_embeddings = None
            return None

        post_embeddings: list[nn.Module] = []

        post_bn = post_embeddings_dct.get("batchnorm", False)
        post_layernorm = post_embeddings_dct.get("layernorm", False)
        post_act = post_embeddings_dct.get("act", None)
        post_dropout = post_embeddings_dct.get("dropout", 0.0)

        if post_bn and post_layernorm:
            raise ValueError("Only one of batchnorm or layernorm can be used in post embeddings!")

        if post_bn:
            post_embeddings.append(
                nn.Sequential(
                    Rearrange("b o e -> b e o"),
                    nn.BatchNorm1d(embedding_dim),
                    Rearrange("b e o -> b o e"),
                )
            )

        if post_layernorm:
            post_embeddings.append(nn.LayerNorm(embedding_dim))

        if post_act is not None:
            post_embeddings.append(get_activation(post_act))

        if post_dropout > 0.0:
            post_embeddings.append(nn.Dropout(post_dropout))

        if len(post_embeddings) == 0:
            self.post_embeddings = None
            return None

        self.post_embeddings = nn.Sequential(*post_embeddings)

    def _setup_projections(
        self, embedding_dim: int, numer_idx: dict[str, np.ndarray], categ_idx: dict[str, np.ndarray], reduction: str
    ) -> None:
        if self.add_particle_types:
            embedding_dim = embedding_dim + 1

        projections: list[nn.Module] = []

        self.masked_mean = False

        for k in numer_idx.keys():
            if k == "events":
                continue

            # input: (batch_size, n_objects, features)
            # itermediate: (batch_size, n_objects, features, embedding_dim)
            # output: (batch_size, n_objects, embedding_dim)

            numer_idx_k, categ_idx_k = numer_idx[k], categ_idx[k]
            total_features = len(numer_idx_k) + len(categ_idx_k)

            if reduction == "mean":
                self.masked_mean = True
                projections.append(MeanPoolingLayer(dim=2))
            elif reduction == "reshape":
                projections.append(
                    MaskedSequential(
                        seq=nn.Sequential(
                            Rearrange("b o f e -> b o (f e)"),
                            nn.Linear(total_features * embedding_dim, embedding_dim),
                        ),
                        first_mask_rearrange=Rearrange("b o -> b o 1 1") if not self.use_ple else None,
                        second_mask_rearrange=Rearrange("b o -> b o 1") if not self.use_ple else None,
                    )
                )
            elif reduction == "conv1d":
                projections.append(MaskedBatchNorm1dJaggedEmbedding(total_features, embedding_dim))
            elif reduction == "conv2d":
                projections.append(
                    MaskedSequential(
                        seq=nn.Sequential(
                            Rearrange("b o f e -> b f o e"),
                            nn.Conv2d(in_channels=total_features, out_channels=1, kernel_size=1),
                            Rearrange("b 1 o e -> b o e"),
                        ),
                        first_mask_rearrange=Rearrange("b o -> b o 1 1") if not self.use_ple else None,
                        second_mask_rearrange=Rearrange("b o -> b o 1") if not self.use_ple else None,
                    ),
                )
            elif reduction == "attn":
                projections.append(Attention2DPoolingReduction(embedding_dim))
            else:
                raise ValueError(f"Unknown reduction method: {reduction}!")

        self.projections = nn.ModuleList(projections)

    def _setup_particle_types(self, add_particle_types: bool, categories: dict[str, np.ndarray]) -> None:
        if not add_particle_types:
            return None

        logging.info("Adding particle type tokens to jagged objects.")

        if "events" in categories:
            num_particle_types = len(categories) - 1
        else:
            num_particle_types = len(categories)

        particle_type_tokens = []
        for _ in range(num_particle_types):
            token = nn.Parameter(torch.randn(1), requires_grad=True)
            particle_type_tokens.append(token)

        self.particle_type_tokens = nn.ParameterList(particle_type_tokens)

    def forward(
        self,
        *Xs: torch.Tensor,
        type_tensors: list[torch.Tensor | None] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        numer_jagged_embeddings: list[torch.Tensor] = []
        numer_jagged_masks: list[torch.Tensor] = []

        for i in range(len(Xs)):
            jagged_idx = self.numer_jagged_idx[i]

            x_full = Xs[i]
            obj_name = self.object_names[i]

            # Derive mask (type-based or padding-based)
            if self.use_type_field_masking and type_tensors is not None and type_tensors[i] is not None:
                # Type tensors passed from dataloader
                mask_all = self._derive_mask_from_type(type_tensors[i], i)
                # One-time logging of masking statistics (first batch only)
                if not self._logged_masking_stats and x_full.shape[0] > 0:
                    valid_types = self._valid_type_tensors[i]
                    n_masked = mask_all.sum().item()
                    n_total = mask_all.numel()
                    pct = 100.0 * n_masked / n_total if n_total > 0 else 0
                    logging.info(
                        f"[green][{obj_name}] Type masking: {n_masked}/{n_total} "
                        f"({pct:.1f}%) masked, valid_types={valid_types.tolist()}[/green]"
                    )
            else:
                # Fallback to padding-based masking
                if not self._logged_masking_stats and self.use_type_field_masking:
                    logging.warning(
                        f"[yellow][{obj_name}] Type masking enabled but no type_tensor provided, "
                        f"falling back to padding-based masking[/yellow]"
                    )
                mask_all = self._derive_mask_from_padding(x_full, i)

            x = x_full[:, :, jagged_idx]

            if self.disable_embeddings:
                embedded = x
            elif self.use_ple:
                # PLE embeddings need padding token for special handling
                padding_token = self.numer_padding_tokens[i]
                embedded = self.numer_jagged_embeddings[i](x, padding_token)
            else:
                embedded = self.numer_jagged_embeddings[i](x)

            numer_jagged_masks.append(mask_all)
            numer_jagged_embeddings.append(embedded)

        # Set flag after logging all objects in first batch
        if self.use_type_field_masking and not self._logged_masking_stats:
            self._logged_masking_stats = True

        categ_jagged_embeddings: list[torch.Tensor] = []
        categ_jagged_masks: list[torch.Tensor] = []

        for i in range(len(Xs)):
            jagged_idx = self.categ_jagged_idx[i]

            x = Xs[i]
            x = x[:, :, jagged_idx].to(torch.int64)

            # Use same mask as numerical features (already derived from type field)
            mask_all = numer_jagged_masks[i]

            if self.disable_embeddings:
                embedded = x
            else:
                embedded = self.categ_jagged_embeddings[i](x)

            categ_jagged_masks.append(mask_all)
            categ_jagged_embeddings.append(embedded)

        jagged_inputs: list[torch.Tensor] = []
        jagged_masks: list[torch.Tensor] = []

        for i in range(len(Xs)):
            numer_embed, categ_embed = numer_jagged_embeddings[i], categ_jagged_embeddings[i]
            numer_mask, categ_mask = numer_jagged_masks[i], categ_jagged_masks[i]

            jagged_cat = torch.cat((numer_embed, categ_embed), dim=2)
            jagged_mask = numer_mask & categ_mask

            if self.add_particle_types:
                particle_type_token = self.particle_type_tokens[i]
                b, o, f, _ = jagged_cat.shape
                particle_type_embedding = particle_type_token.expand(b, o, f, 1)

                particle_type_mask = rearrange(jagged_mask, "b o -> b o 1 1")
                particle_type_embedding = particle_type_embedding.masked_fill(particle_type_mask, 0.0)

                jagged_cat = torch.cat((jagged_cat, particle_type_embedding), dim=-1)

            if self.masked_mean:
                proj_jagged_mask = rearrange(jagged_mask, "b o -> b o 1 1") if not self.use_ple else None
                jagged_cat_proj = self.projections[i](jagged_cat, proj_jagged_mask)
            else:
                jagged_cat_proj = self.projections[i](jagged_cat, jagged_mask if not self.use_ple else None)

            jagged_inputs.append(jagged_cat_proj)
            jagged_masks.append(jagged_mask)

        x_jagged = torch.cat(jagged_inputs, dim=1)

        if self.post_embeddings is not None:
            x_jagged = self.post_embeddings(x_jagged)

        x_jagged_masks = torch.cat(jagged_masks, dim=1)  # padded: True, valid: False

        return x_jagged, x_jagged_masks
