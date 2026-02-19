from typing import Any

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from seesawml.models.activations import get_activation
from seesawml.models.flat_preprocessor import FlatPreprocessor
from seesawml.models.jagged_preprocessor import JaggedPreprocessor
from seesawml.models.layers import FlatJaggedEmbeddingsFuser, FlatJaggedModelFuser


def _handle_layers_params(
    encoder_layers: int | list[int],
    decoder_layers: int | list[int],
    n_hidden: int | None,
    embedding_dim: int | None,
    output_dim: int,
) -> tuple[list[int], list[int], int]:
    if n_hidden is None:
        if type(encoder_layers) is not list:
            raise ValueError("Provide a list of encoder_layers.")

        if type(decoder_layers) is not list:
            raise ValueError("Provide a list of decoder_layers.")

        if embedding_dim is None:
            embedding_dim = encoder_layers[0]

        from_list = True
    else:
        if type(encoder_layers) is not int:
            raise ValueError("Provide an integer number of encoder_layers.")

        if type(decoder_layers) is not int:
            raise ValueError("Provide an integer number of decoder_layers.")

        if embedding_dim is None:
            embedding_dim = n_hidden

        from_list = False

    encoder_layers_dim: list[int]
    decoder_layers_dim: list[int]

    if from_list:
        encoder_layers_dim = [embedding_dim] + encoder_layers  # type: ignore
        decoder_layers_dim = decoder_layers + [output_dim]  # type: ignore
    else:
        encoder_layers_dim = [embedding_dim] + [n_hidden] * encoder_layers  # type: ignore
        decoder_layers_dim = [n_hidden] * decoder_layers + [output_dim]  # type: ignore

    return encoder_layers_dim, decoder_layers_dim, embedding_dim


def _setup_encoder(
    encoder_layers_dim: list[int], act: str, use_batchnorm: bool, dropout: float, **act_kwargs: Any
) -> nn.Sequential:
    phi: list[nn.Module] = []

    # https://jduarte.physics.ucsd.edu/iaifi-summer-school/1.3_deep_sets.html
    for layer in range(len(encoder_layers_dim) - 1):
        n_in, n_out = encoder_layers_dim[layer], encoder_layers_dim[layer + 1]
        phi.append(nn.Conv1d(n_in, n_out, kernel_size=1))

        if use_batchnorm:
            phi.append(nn.BatchNorm1d(n_out))

        phi.append(get_activation(act, **act_kwargs))

        if dropout > 0.0:
            phi.append(nn.Dropout(dropout))

    return nn.Sequential(*phi)


def _setup_decoder(
    decoder_layers_dim: list[int], act: str, act_out: str | None, use_batchnorm: bool, dropout: float, **act_kwargs: Any
) -> nn.Sequential:
    rho: list[nn.Module] = []

    for layer in range(len(decoder_layers_dim) - 1):
        n_in, n_out = decoder_layers_dim[layer], decoder_layers_dim[layer + 1]
        rho.append(nn.Linear(n_in, n_out))

        if layer < len(decoder_layers_dim) - 2:
            if use_batchnorm:
                rho.append(nn.BatchNorm1d(n_out))

            rho.append(get_activation(act, **act_kwargs))

            if dropout > 0.0:
                rho.append(nn.Dropout(dropout))
        else:
            rho.append(get_activation(act_out, **act_kwargs))

    return nn.Sequential(*rho)


class FlatDeepSets(nn.Module):
    def __init__(
        self,
        numer_idx: np.ndarray,
        categ_idx: np.ndarray,
        output_dim: int,
        encoder_layers: int | list[int] = 2,
        decoder_layers: int | list[int] = 2,
        n_hidden: int | None = 128,
        embedding_dim: int | None = None,
        act: str = "ReLU",
        act_out: str | None = None,
        use_batchnorm: bool = True,
        dropout: float = 0.0,
        categories: np.ndarray | None = None,
        embedding_config_dct: dict[str, Any] | None = None,
        **act_kwargs: Any,
    ) -> None:
        super().__init__()

        encoder_layers_dim, decoder_layers_dim, embedding_dim = _handle_layers_params(
            encoder_layers, decoder_layers, n_hidden, embedding_dim, output_dim
        )

        if embedding_config_dct is None:
            embedding_config_dct = {}

        self.flat_preprocessor = FlatPreprocessor(
            embedding_dim=embedding_dim,
            numer_idx=numer_idx,
            categ_idx=categ_idx,
            categories=categories,
            embedding_reduction=None,
            numer_feature_wise_linear=embedding_config_dct.get("numer_feature_wise_linear", False),
            post_embeddings_dct=embedding_config_dct.get("post_embeddings_dct", None),
            ple_dct=embedding_config_dct.get("ple_config", None),
            disable_embeddings=False,
        )

        self.phi = _setup_encoder(encoder_layers_dim, act, use_batchnorm, dropout, **act_kwargs)

        self.rho = _setup_decoder(decoder_layers_dim, act, act_out, use_batchnorm, dropout, **act_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flat_preprocessor(x)  # (batch_size, features) -> (batch_size, features, embeddings)

        # embeddings as channels and features as elements in the set
        # conv1d wants (batch_size, channels, length of signal sequence)
        # conv slides along last dimension
        x = rearrange(x, "b f e -> b e f")

        out = self.phi(x)
        out = torch.mean(out, dim=-1)
        out = self.rho(out)

        return out


class JaggedDeepsets(nn.Module):
    def __init__(
        self,
        numer_idx: dict[str, np.ndarray],
        categ_idx: dict[str, np.ndarray],
        categories: dict[str, np.ndarray],
        output_dim: int,
        object_dimensions: dict[str, int],
        numer_padding_tokens: dict[str, float | None],
        categ_padding_tokens: dict[str, int | None],
        encoder_layers: int | list[int] = 2,
        decoder_layers: int | list[int] = 2,
        n_hidden: int | None = 128,
        embedding_dim: int | None = None,
        act: str = "ReLU",
        act_out: str | None = None,
        use_batchnorm: bool = True,
        dropout: float = 0.0,
        mean_pooling: bool = True,
        embedding_config_dct: dict[str, Any] | None = None,
        add_particle_types: bool = False,
        flat_embeddings: nn.Module | None = None,
        flat_embeddings_fuse: dict[str, Any] | None = None,
        flat_model: nn.Module | None = None,
        flat_model_fuse: dict[str, Any] | None = None,
        valid_type_values: dict[str, list[int]] | None = None,
        **act_kwargs,
    ) -> None:
        super().__init__()

        if embedding_config_dct is None:
            embedding_config_dct = {}

        if flat_embeddings_fuse is None:
            flat_embeddings_fuse = {}

        if flat_model_fuse is None:
            flat_model_fuse = {}

        objects_idx = []
        for i, dim in enumerate(object_dimensions.values()):
            objects_idx.extend([i] * dim)

        self.object_idx = torch.nn.Parameter(torch.tensor(objects_idx, dtype=torch.int64), requires_grad=False)
        self.max_object_idx = max(objects_idx) + 1

        encoder_layers_dim, decoder_layers_dim, embedding_dim = _handle_layers_params(
            encoder_layers, decoder_layers, n_hidden, embedding_dim, output_dim
        )

        self.mean_pooling = mean_pooling

        if not mean_pooling:
            decoder_layers_dim[0] = decoder_layers_dim[0] * len(object_dimensions)

        if flat_embeddings_fuse.get("mode", None) == "cat":
            jagged_embedding_dim = embedding_dim // 2
        else:
            jagged_embedding_dim = embedding_dim

        self.jagged_preprocessor = JaggedPreprocessor(
            embedding_dim=jagged_embedding_dim,
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
            valid_type_values=valid_type_values,
        )

        self.phi = _setup_encoder(encoder_layers_dim, act, use_batchnorm, dropout, **act_kwargs)

        self.rho = _setup_decoder(
            decoder_layers_dim,
            act,
            act_out,
            use_batchnorm,
            dropout,
            **act_kwargs,
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

        if flat_model:
            self.fuser = FlatJaggedModelFuser(
                flat_model_fuse["mode"],
                output_dim=output_dim,
                fuse_kwargs=flat_model_fuse.get("fuse_kwargs", None),
            )

    def forward(
        self,
        X_events: torch.Tensor,
        *Xs: torch.Tensor,
        type_tensors: list[torch.Tensor | None] | None = None,
    ) -> torch.Tensor:
        # N x (batch_size, n_objects, features) -> (batch_size, n_objects, embedding_dim)
        x_jagged, x_jagged_valid = self.jagged_preprocessor(*Xs, type_tensors=type_tensors)

        if self.flat_embeddings is not None:
            x_flat = self.flat_embeddings(X_events)
            x_jagged = self.embeddings_fuser(
                x_flat, x_jagged, mask=None if self.disable_flat_embeddings_mask else x_jagged_valid
            )

        x = rearrange(x_jagged, "b o e -> b e o")

        out = self.phi(x)  # (batch_size, embedding_dim, n_objects)

        object_idx = self.object_idx.repeat(x.shape[0], 1)
        object_idx[x_jagged_valid] = self.max_object_idx
        object_idx = object_idx.unsqueeze(1).expand(-1, out.shape[1], -1)

        shapes = (out.shape[0], out.shape[1], self.max_object_idx + 1)
        mean_out = torch.zeros(shapes, dtype=out.dtype, device=out.device)
        out = torch.scatter_reduce(mean_out, dim=-1, index=object_idx, src=out, reduce="mean", include_self=False)

        out = out[:, :, :-1]

        if self.mean_pooling:
            out = torch.mean(out, dim=-1)
        else:
            out = rearrange(out, "b e t -> b (e t)")

        out = self.rho(out)

        if self.flat_model is not None:
            flat_out = self.flat_model(X_events)
            out = self.fuser(flat_out, out)

        return out
