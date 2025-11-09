import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from seesaw.models.activations import get_activation
from seesaw.models.flat_preprocessor import FlatPreprocessor
from seesaw.models.layers import ResidualFeedForwardLayer


class ResNet(nn.Module):
    def __init__(
        self,
        numer_idx: np.ndarray,
        categ_idx: np.ndarray,
        output_dim: int,
        num_blocks: int | None = None,
        n_hidden: int | None = None,
        block_size: int = 2,
        res_layers: list[list[int]] | None = None,
        act: str = "ReLU",
        act_out: str | None = None,
        use_batchnorm: bool = True,
        dropout: float = 0.0,
        categories: np.ndarray | None = None,
        embedding_config_dct: dict[str, Any] | None = None,
        disable_embeddings: bool = False,
        **act_kwargs: Any,
    ) -> None:
        """Implementation of ResNet model.

        This module implements the following:

        ResNet(x) = Prediction(ResNetBlock(...(ResNetBlock(Linear(x)))))
        ResNetBlock(x) = x + Dropout(Linear(Dropout(Act(Linear(BatchNorm(x))))))
        Prediction(x) = Linear(Act(BatchNorm(x)))

        Parameters
        ----------
        output_dim : int
            The number of output features.
        num_blocks : int, optional
            The number of residual blocks, by default None.
        n_hidden : int, optional
            The number of hidden units in the residual blocks, by default None.
        block_size : int, optional
            The number of layers in each residual block, by default 2.
        res_layers : list[list[int]]
            The dimensions of the residual layers. For example [[x, y, z], [z, y, x]] will create two residual layers,
            or 4 blocks, with the given dimensions: x -> y -> z -> y -> x.
        act : str, optional
            The activation function to use in the hidden layers, by default "ReLU".
        act_out : str, optional
            The activation function to use in the output layer, by default "Identity".
        use_batchnorm : bool, optional
            Whether to use batch normalization, by default True.
        dropout : float, optional
            The dropout probability, by default 0.0.
        categories : np.ndarray, optional
            Array of categories for each categorical feature, by default None. If provided, the network will include an
            embedding layer before the first hidden layer.
        embedding_config_dct : dict[str, Any] | None, optional
            Configuration dictionary for embeddings, by default None.
        disable_embeddings : bool, optional
            If True, embeddings are disabled and raw features are used, by default False.
        act_kwargs : Any, optional
            Additional arguments to pass to the activation function. Does not apply to the output layer.

        References
        ----------
        [1] - Identity Mappings in Deep Residual Networks: https://arxiv.org/abs/1603.05027.
        [2] - Revisiting Deep Learning Models for Tabular Data: https://arxiv.org/abs/2106.11959v2

        """
        super().__init__()

        if res_layers is None:
            if num_blocks is None or n_hidden is None:
                raise ValueError("num_blocks and n_hidden must be provided if res_layers is None.")

            res_layers = [[n_hidden] * (block_size + 1)] * num_blocks
        else:
            if num_blocks is not None or n_hidden is not None:
                logging.warning("Ignoring num_blocks and n_hidden as res_layers is provided.")

        embedding_dim = res_layers[0][0]

        if embedding_config_dct is None:
            embedding_config_dct = {}

        self.flat_preprocessor = FlatPreprocessor(
            embedding_dim=embedding_dim,
            numer_idx=numer_idx,
            categ_idx=categ_idx,
            categories=categories,
            embedding_reduction=embedding_config_dct.get("reduction", "mean"),
            numer_feature_wise_linear=embedding_config_dct.get("numer_feature_wise_linear", False),
            post_embeddings_dct=embedding_config_dct.get("post_embeddings_dct", None),
            ple_dct=embedding_config_dct.get("ple_config", None),
            disable_embeddings=disable_embeddings,
        )

        embedding_output_dim = self.flat_preprocessor.output_dim

        self.layers = nn.ModuleList()

        if embedding_output_dim != embedding_dim:
            self.layers.add_module("input_layer", nn.Linear(embedding_output_dim, embedding_dim))

        for i, layers_dim in enumerate(res_layers):
            res_layer = ResidualFeedForwardLayer(
                layers_dim=layers_dim,
                act=act,
                use_batchnorm=use_batchnorm,
                dropout=dropout,
                **act_kwargs,
            )
            self.layers.add_module(f"res_layer{i}", res_layer)

        last_dim = res_layers[-1][-1]
        output_layer = nn.Sequential(
            nn.Linear(last_dim, output_dim),
            get_activation(act_out),
        )

        self.layers.add_module("output_layer", output_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flat_preprocessor(x)

        for layer in self.layers:
            x = layer(x)

        return x
