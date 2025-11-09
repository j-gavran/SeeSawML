from typing import Any

import numpy as np
import torch
from einops import repeat
from torch import nn

from seesaw.models.activations import get_activation
from seesaw.models.flat_preprocessor import FlatPreprocessor
from seesaw.models.transformers.attention import AttentionEncoder


class EventTransformer(nn.Module):
    def __init__(
        self,
        numer_idx: np.ndarray,
        categ_idx: np.ndarray,
        embedding_dim: int = 32,
        transformer_depth: int = 6,
        ff_hidden_mult: int = 4,
        heads: int = 8,
        dim_head: int = 16,
        dim_out: int = 1,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        act_out: str | None = None,
        categories: np.ndarray | None = None,
        flash: bool = False,
        embedding_config_dct: dict[str, Any] | None = None,
        remove_first_attn_residual: bool = False,
        remove_first_attn_layernorm: bool = True,
        use_cls_token: bool = True,
    ) -> None:
        """Transformer classifier model for event data.

        Parameters
        ----------
        numer_idx : np.ndarray
            Indices of numerical features.
        categ_idx : np.ndarray
            Indices of categorical features.
        embedding_dim : int
            Embedding dimension, paper set at 32.
        transformer_depth : int
            Number of transformer blocks, paper recommended 6.
        ff_hidden_mult : int
            Feed forward hidden layer dimension multiplier, by default 4.
        heads : int
            Attention heads, paper recommends 8.
        dim_head : int, optional
            Dimension of the head, by default 16.
        dim_out : int, optional
            Binary prediction, but could be anything, by default 1.
        attn_dropout : float, optional
            Post-attention dropout, by default 0.0.
        ff_dropout : float, optional
            Feed forward dropout, by default 0.0.
        act_out : str, optional
            Activation function for the output layer, by default None. If None, Identity is applied.
        categories : np.ndarray, optional
            Array of categories, by default None. If None, no categorical embeddings are used.
        flash : bool, optional
            If True, uses flash attention for the multi-head attention block, by default False.
        embedding_config_dct : dict[str, Any] | None, optional
            Configuration dictionary for embeddings, by default None.
        remove_first_attn_residual : bool, optional
            Whether to remove the residual connection in the first attention block, by default False.
        remove_first_attn_layernorm : bool, optional
            Whether to remove the layernorm in the first attention block, by default True.
        use_cls_token : bool, optional
            Whether to use a classification token, by default True.

        References
        ----------
        [1] - Revisiting Deep Learning Models for Tabular Data: https://arxiv.org/abs/2106.11959
        [2] - https://github.com/lucidrains/tab-transformer-pytorch

        """
        super().__init__()

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

        self.encoder = AttentionEncoder(
            dim=embedding_dim,
            depth=transformer_depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=ff_hidden_mult * embedding_dim,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            first_attn_no_residual=remove_first_attn_residual,
            first_attn_no_layernorm=remove_first_attn_layernorm,
            use_flash=flash,
        )

        self.use_cls_token = use_cls_token
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        self.to_logits = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, dim_out),
            get_activation(act_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flat_preprocessor(x)

        if self.use_cls_token:
            b = x.shape[0]
            cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
            x = torch.cat((cls_tokens, x), dim=1)

        x = self.encoder(x)

        if self.use_cls_token:
            x = x[:, 0]
        else:
            x = torch.mean(x, dim=1)

        logits = self.to_logits(x)

        return logits
