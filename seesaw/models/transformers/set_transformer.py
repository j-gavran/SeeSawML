from typing import Any

import numpy as np
import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn

from seesaw.models.jagged_preprocessor import JaggedPreprocessor
from seesaw.models.layers import FlatJaggedEmbeddingsFuser, FlatJaggedModelFuser
from seesaw.models.mlp import FeedForwardLayer
from seesaw.models.transformers.attention import Attend
from seesaw.models.transformers.ff_blocks import GeGLUNet


class SetNorm(nn.Module):
    def __init__(self, feature_dim: int, eps: float = 1e-5) -> None:
        """Equivariant layer normalization for sets.

        References
        ----------
        [1] - https://arxiv.org/abs/2206.11925
        [2] - https://github.com/rajesh-lab/deep_permutation_invariant

        """
        super().__init__()
        self.eps = eps

        self.weights = nn.Parameter(torch.empty(feature_dim))
        torch.nn.init.constant_(self.weights, 1.0)

        self.biases = nn.Parameter(torch.empty(feature_dim))
        torch.nn.init.constant_(self.biases, 0.0)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b, n, f = x.shape

        if mask is None:
            mask = torch.ones((b, n), dtype=torch.bool, device=x.device)

        num_valid = torch.clamp(mask.sum(dim=1) * f, min=1.0)

        means = (x * mask.unsqueeze(-1)).sum(dim=[1, 2]) / num_valid
        means = means.reshape(b, 1, 1)

        std = torch.sqrt(((x - means).square() * mask.unsqueeze(-1)).sum(dim=[1, 2]) / num_valid + self.eps)
        std = std.reshape(b, 1, 1)

        out = (x - means) / std
        out = out * self.weights + self.biases

        return out


class MaskedLayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return super().forward(x)


class SetAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        attn_dropout: float = 0.1,
        normalize_q: bool = True,
        use_setnorm: bool = True,
        sdp_backend: dict[str, bool] | None = None,
    ) -> None:
        super().__init__()
        self.heads = heads

        self.scale = dim_head**-0.5

        inner_dim = dim_head * heads

        self.normalize_q = normalize_q

        if self.normalize_q:
            self.q_norm = SetNorm(dim) if use_setnorm else MaskedLayerNorm(dim)

        self.kv_norm = SetNorm(dim) if use_setnorm else MaskedLayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.attend = Attend(heads, attn_dropout, sdp_backend)

        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        set_q_mask: torch.Tensor | None = None,
        set_kv_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if context is None:
            context = x

        if self.normalize_q:
            x = self.q_norm(x, mask=set_q_mask)

        context = self.kv_norm(context, mask=set_kv_mask)

        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        attn = self.attend(q, k, v, self.scale, attn_mask)

        out = rearrange(attn, "b h n d -> b n (h d)")

        return self.to_out(out)


class SetAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        normalize_q: bool = True,
        use_setnorm: bool = True,
        attention_residual: bool = True,
        sdp_backend: dict[str, bool] | None = None,
    ) -> None:
        super().__init__()
        self.attention_residual = attention_residual

        self.attn = SetAttention(
            dim,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            normalize_q=normalize_q,
            use_setnorm=use_setnorm,
            sdp_backend=sdp_backend,
        )
        self.ff = GeGLUNet(dim, mult=mlp_dim // dim, dropout=ff_dropout, use_layernorm=False, output_dropout=False)

        self.norm = SetNorm(dim) if use_setnorm else MaskedLayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        set_q_mask: torch.Tensor | None = None,
        set_kv_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.attn(x, context=context, attn_mask=attn_mask, set_q_mask=set_q_mask, set_kv_mask=set_kv_mask)

        if self.attention_residual:
            h = x + h

        h_norm = self.norm(h, mask=set_q_mask)
        out = h + self.ff(h_norm)
        return out


class PooledMultiheadAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        num_seeds: int = 1,
        use_setnorm: bool = True,
        sdp_backend: dict[str, bool] | None = None,
    ) -> None:
        super().__init__()
        self.seed_vectors = nn.Parameter(torch.empty(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.seed_vectors)

        self.attn = SetAttentionBlock(
            dim,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            normalize_q=True,
            use_setnorm=use_setnorm,
            sdp_backend=sdp_backend,
        )

    def forward(
        self,
        z: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        set_q_mask: torch.Tensor | None = None,
        set_kv_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seed_vectors = repeat(self.seed_vectors, "1 k d -> b k d", b=z.shape[0])

        z = self.attn(seed_vectors, context=z, attn_mask=attn_mask, set_q_mask=set_q_mask, set_kv_mask=set_kv_mask)

        return z


class SetPredictorNet(nn.Module):
    def __init__(
        self,
        dim: int,
        num_seeds: int,
        inner_dim: int | None = None,
        dim_out: int = 1,
        act_out: str | None = None,
        depth: int = 3,
        act: str = "ReLU",
        pool_predict: bool = False,
        disable_reshape: bool = False,
        ff_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if inner_dim is None:
            inner_dim = dim

        self.pool_predict = pool_predict

        set_predictor: list[nn.Module] = []

        if not pool_predict and not disable_reshape:
            set_predictor.append(Rearrange("b k d -> b (k d)"))

        if num_seeds > 1 and not pool_predict:
            n_in = dim * num_seeds
        else:
            n_in = dim

        for i in range(depth):
            is_last = i == depth - 1
            set_predictor.append(
                FeedForwardLayer(
                    n_in=n_in if i == 0 else inner_dim,
                    n_out=dim_out if is_last else inner_dim,
                    act=act_out if is_last else act,
                    use_batchnorm=False,
                    dropout=0.0 if is_last else ff_dropout,
                )
            )

        self.set_predictor = nn.Sequential(*set_predictor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pool_predict:
            x = x.mean(dim=1)

        return self.set_predictor(x)


class SetTransformerModel(nn.Module):
    def __init__(
        self,
        dim: int,
        encoder_depth: int,
        decoder_depth: int | None,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        num_seeds: int = 1,
        dim_out: int = 1,
        act_out: str | None = None,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        act_predict: str = "ReLU",
        depth_predict: int = 2,
        pool_predict: bool = False,
        use_predict: bool = True,
        use_setnorm: bool = True,
        first_attn_no_residual: bool = False,
        sdp_backend: dict[str, bool] | None = None,
    ) -> None:
        """Set Transformer: Attention-based permutation invariant neural network for sets.

        References
        ----------
        - Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks: https://arxiv.org/abs/1810.00825
        - Set Transformer in PyTorch: https://github.com/TropComplique/set-transformer and https://github.com/juho-lee/set_transformer

        """
        super().__init__()
        self.encoder = nn.ModuleList()

        for i in range(encoder_depth):
            self.encoder.append(
                SetAttentionBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    mlp_dim=mlp_dim,
                    attn_dropout=attn_dropout,
                    ff_dropout=ff_dropout,
                    use_setnorm=use_setnorm,
                    attention_residual=False if first_attn_no_residual and i == 0 else True,
                    sdp_backend=sdp_backend,
                )
            )

        self.norm = SetNorm(dim) if use_setnorm else MaskedLayerNorm(dim)

        self.pooling = PooledMultiheadAttention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            num_seeds=num_seeds,
            use_setnorm=use_setnorm,
            sdp_backend=sdp_backend,
        )

        if decoder_depth is None or decoder_depth == 0 or num_seeds == 1:
            self.use_decoder = False
        else:
            self.use_decoder = True

        if self.use_decoder:
            self.decoder = nn.ModuleList()
            for _ in range(decoder_depth):  # type: ignore[arg-type]
                self.decoder.append(
                    SetAttentionBlock(
                        dim=dim,
                        heads=heads,
                        dim_head=dim_head,
                        attn_dropout=attn_dropout,
                        ff_dropout=ff_dropout,
                        mlp_dim=mlp_dim,
                        use_setnorm=use_setnorm,
                        sdp_backend=sdp_backend,
                    )
                )

        self.use_predict = use_predict

        if self.use_predict:
            self.predict = SetPredictorNet(
                dim=dim,
                num_seeds=num_seeds,
                inner_dim=mlp_dim,
                dim_out=dim_out,
                act_out=act_out,
                depth=depth_predict,
                act=act_predict,
                pool_predict=pool_predict,
                ff_dropout=ff_dropout,
            )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        set_q_mask: torch.Tensor | None = None,
        set_kv_mask: torch.Tensor | None = None,
        pooling_attn_mask: torch.Tensor | None = None,
        pooling_set_q_mask: torch.Tensor | None = None,
        pooling_set_kv_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for block in self.encoder:
            x = block(x, attn_mask=attn_mask, set_q_mask=set_q_mask, set_kv_mask=set_kv_mask)

        x = self.norm(x, mask=set_q_mask)
        x = self.pooling(x, attn_mask=pooling_attn_mask, set_q_mask=pooling_set_q_mask, set_kv_mask=pooling_set_kv_mask)

        if self.use_decoder:
            for block in self.decoder:
                x = block(x, attn_mask=None, set_q_mask=None, set_kv_mask=None)

        if self.use_predict:
            x = self.predict(x)

        return x


class SetTransformer(nn.Module):
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
        embedding_config_dct: dict[str, Any] | None = None,
        use_setnorm: bool = False,
        add_particle_types: bool = False,
        flat_embeddings: nn.Module | None = None,
        flat_embeddings_fuse: dict[str, Any] | None = None,
        flat_model: nn.Module | None = None,
        flat_model_fuse: dict[str, Any] | None = None,
        first_attn_no_residual: bool = False,
        sdp_backend: dict[str, bool] | None = None,
        valid_type_values: dict[str, list[int]] | None = None,
    ) -> None:
        super().__init__()

        if set_predictor_dct is None:
            set_predictor_dct = {}

        if embedding_config_dct is None:
            embedding_config_dct = {}

        if flat_embeddings_fuse is None:
            flat_embeddings_fuse = {}

        if flat_model_fuse is None:
            flat_model_fuse = {}

        if seed_strategy == "pooling":
            num_seeds = 1
        elif seed_strategy == "objects":
            num_seeds = sum(object_dimensions.values())
        elif seed_strategy == "particles":
            num_seeds = len(object_dimensions)
        else:
            raise ValueError(f"Invalid seed_strategy: {seed_strategy}, must be 'pooling', 'objects' or 'particles'.")

        self.num_seeds = num_seeds

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

        self.set_jagged_transformer = SetTransformerModel(
            dim=embedding_dim,
            encoder_depth=encoder_depth,
            decoder_depth=decoder_depth,
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
            use_predict=True,
            use_setnorm=use_setnorm,
            first_attn_no_residual=first_attn_no_residual,
            sdp_backend=sdp_backend,
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

    def forward(
        self,
        X_events: torch.Tensor,
        *Xs: torch.Tensor,
        type_tensors: list[torch.Tensor | None] | None = None,
    ) -> torch.Tensor:
        x_jagged, x_jagged_valid = self.jagged_preprocessor(*Xs, type_tensors=type_tensors)
        x_jagged_object_mask = rearrange(x_jagged_valid, "b i -> b 1 1 i")

        x_pooling_valid = torch.ones(x_jagged.shape[0], self.num_seeds, dtype=torch.bool, device=x_jagged.device)

        x_jagged_pooling_mask = x_jagged_object_mask.expand(-1, 1, self.num_seeds, -1)

        if self.flat_embeddings is not None:
            x_flat = self.flat_embeddings(X_events)
            x_jagged = self.embeddings_fuser(
                x_flat, x_jagged, mask=None if self.disable_flat_embeddings_mask else x_jagged_valid
            )

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
            return self.model_fuser(flat_out, x_jagged_set_out)

        return x_jagged_set_out
