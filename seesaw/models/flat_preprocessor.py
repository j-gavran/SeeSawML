import logging
import math
from typing import Any

import numpy as np
import torch
from einops import rearrange
from torch import nn

from seesaw.models.activations import get_activation
from seesaw.models.layers import FeatureWiseLinear
from seesaw.models.ple import LearnablePiecewiseEncodingLayer, QuantilePiecewiseEncodingLayer


class FlatNumEmbeddingModel(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_numerical_types: int,
        feature_wise_linear: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.bias = bias

        if feature_wise_linear:
            self.per_feature = True

            self.features_linear = FeatureWiseLinear(num_numerical_types, embedding_dim, bias=bias)
        else:
            self.per_feature = False

            self.weights = nn.Parameter(torch.empty(num_numerical_types, embedding_dim))
            if self.bias:
                self.biases = nn.Parameter(torch.empty(num_numerical_types, embedding_dim))

            self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))

        if self.bias:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.biases, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.per_feature:
            return self.features_linear(x)

        x = rearrange(x, "b n -> b n 1")

        if self.bias:
            return x * self.weights + self.biases
        else:
            return x * self.weights


class FlatPLENumEmbeddingModel(nn.Module):
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

        if learn_bins and uniform_bins:
            raise ValueError("Only one of learn_bins or uniform_bins can be True!")

        ples: list[nn.Module] = []

        if learn_bins or uniform_bins:
            for i in range(num_numerical_types):
                learnable_ple = LearnablePiecewiseEncodingLayer(
                    bins=ple_dct["n_bins"],
                    embedding_dim=embedding_dim,
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
                    dataset_key=dataset_key,
                )
                ples.append(quantile_ple)

        self.ple = nn.ModuleList(ples)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = [self.ple[i](x[:, i]) for i in range(x.shape[1])]
        return torch.stack(embeddings, dim=1)


class FlatCatEmbeddingModel(nn.Module):
    def __init__(self, embedding_sizes: list[tuple[int, int]]) -> None:
        super().__init__()

        if set(emb_dim for _, emb_dim in embedding_sizes) != {embedding_sizes[0][1]}:
            raise ValueError("All embedding dimensions must be the same!")

        self.embeddings = nn.ModuleList(
            [nn.Embedding(num_categories, emb_dim) for num_categories, emb_dim in embedding_sizes]
        )

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        embedded = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        return torch.stack(embedded, dim=1)


class FlatPreprocessor(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        numer_idx: np.ndarray,
        categ_idx: np.ndarray,
        categories: np.ndarray | None,
        embedding_reduction: str | None = "mean",
        numer_feature_wise_linear: bool = False,
        post_embeddings_dct: dict[str, Any] | None = None,
        ple_dct: dict[str, Any] | None = None,
        disable_embeddings: bool = False,
    ) -> None:
        super().__init__()
        self.disable_embeddings = disable_embeddings

        self.mean_reduction = embedding_reduction == "mean"
        self.reshape_reduction = embedding_reduction == "reshape"
        self.conv1d_reduction = embedding_reduction == "conv1d"
        self.none_reduction = embedding_reduction == "none" or embedding_reduction is None

        self.has_numer, self.has_categ = False, False

        if len(numer_idx) > 0:
            self.has_numer = True
            self.numer_flat_idx = nn.Parameter(torch.tensor(numer_idx, dtype=torch.int64), requires_grad=False)

        if len(categ_idx) > 0:
            self.has_categ = True
            self.categ_flat_idx = nn.Parameter(torch.tensor(categ_idx, dtype=torch.int64), requires_grad=False)

        if not self.disable_embeddings:
            self._setup_embeddings(embedding_dim, numer_idx, categories, numer_feature_wise_linear, ple_dct)
            self._setup_post_embeddings(embedding_dim, post_embeddings_dct)
        else:
            logging.warning("Embeddings are disabled in FlatPreprocessor!")

        if self.conv1d_reduction:
            self.conv1d_reduction_layer = nn.Conv1d(in_channels=embedding_dim, out_channels=1, kernel_size=1)

        self._get_output_dim(numer_idx, categ_idx, embedding_dim)

    def _setup_embeddings(
        self,
        embedding_dim: int,
        numer_idx: np.ndarray,
        categories: np.ndarray | None,
        numer_feature_wise_linear: bool = False,
        ple_dct: dict[str, Any] | None = None,
    ) -> None:
        if self.has_numer:
            self.numer_embeddings: nn.Module
            if ple_dct is not None:
                self.numer_embeddings = FlatPLENumEmbeddingModel(
                    embedding_dim,
                    len(numer_idx),
                    ple_dct,
                    dataset_key="events",
                )
            else:
                self.numer_embeddings = FlatNumEmbeddingModel(
                    embedding_dim,
                    len(numer_idx),
                    feature_wise_linear=numer_feature_wise_linear,
                )

        if self.has_categ:
            if categories is None:
                raise ValueError("Categories must be provided if categ_idx is not empty!")

            categ_sizes = [(num_cat, embedding_dim) for num_cat in categories]
            self.categ_embeddings = FlatCatEmbeddingModel(categ_sizes)

    def _setup_post_embeddings(self, embedding_dim: int, post_embeddings_dct: dict[str, Any] | None) -> None:
        if post_embeddings_dct is None:
            self.post_embeddings = None
            return None

        post_embeddings: list[nn.Module] = []

        post_act = post_embeddings_dct.get("act", None)
        post_layernorm = post_embeddings_dct.get("layernorm", False)
        post_dropout = post_embeddings_dct.get("dropout", 0.0)

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

    def _get_output_dim(self, numer_idx: np.ndarray, categ_idx: np.ndarray, embedding_dim: int) -> None:
        n_features = len(numer_idx) + len(categ_idx)

        if self.disable_embeddings or self.conv1d_reduction:
            self._output_dim = n_features
            return None

        output_dim = 0

        if self.has_numer:
            if self.reshape_reduction:
                output_dim += embedding_dim * len(numer_idx)
            else:
                output_dim = embedding_dim

        if self.has_categ:
            if self.reshape_reduction:
                output_dim += embedding_dim * len(categ_idx)
            else:
                output_dim = embedding_dim

        self._output_dim = output_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.disable_embeddings:
            xs = []
            if self.has_numer:
                x_numer = x[:, self.numer_flat_idx]
                xs.append(x_numer)

            if self.has_categ:
                x_categ = x[:, self.categ_flat_idx]
                xs.append(x_categ)

            x = torch.cat(xs, dim=1)
            return x

        embeddings = []

        if self.has_numer:
            x_numer = x[:, self.numer_flat_idx]
            numer_embedded = self.numer_embeddings(x_numer)
            embeddings.append(numer_embedded)

        if self.has_categ:
            x_categ = x[:, self.categ_flat_idx].to(torch.int64)
            categ_embedded = self.categ_embeddings(x_categ)
            embeddings.append(categ_embedded)

        x_embedded = torch.cat(embeddings, dim=1)

        if self.post_embeddings is not None:
            x_embedded = self.post_embeddings(x_embedded)

        if self.mean_reduction:
            x_embedded = torch.mean(x_embedded, dim=1)
        elif self.reshape_reduction:
            x_embedded = rearrange(x_embedded, "b f e -> b (f e)")
        elif self.conv1d_reduction:
            x_embedded = rearrange(x_embedded, "b f e -> b e f")
            x_embedded = self.conv1d_reduction_layer(x_embedded).squeeze(dim=1)
        elif self.none_reduction:
            pass
        else:
            raise RuntimeError("Invalid embedding_reduction!")

        return x_embedded
