import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from seesawml.models.flat_preprocessor import FlatPreprocessor
from seesawml.models.layers import FeedForwardLayer


class MLP(nn.Module):
    def __init__(
        self,
        numer_idx: np.ndarray,
        categ_idx: np.ndarray,
        output_dim: int | None = None,
        n_layers: int | None = None,
        n_hidden: int | None = None,
        layers_dim: list[int] | None = None,
        act: str = "ReLU",
        act_out: str | None = None,
        use_batchnorm: bool = False,
        dropout: float = 0.0,
        categories: np.ndarray | None = None,
        embedding_config_dct: dict[str, Any] | None = None,
        disable_embeddings: bool = False,
        **act_kwargs: Any,
    ) -> None:
        """Multi-layer perceptron (MLP) neural network.

        Note
        ----
        There are two ways to define the network architecture: either by specifying the input dimension, number of
        hidden layers, and number of hidden units in each layer, or by providing a list of layer dimensions.

        Parameters
        ----------
        numer_idx : np.ndarray
            Indices of numerical features in X.
        categ_idx : np.ndarray
            Indices of categorical features in X.
        output_dim : int, optional
            Dimension of the output tensor, by default None.
        n_layers : int, optional
            Number of hidden layers, by default None.
        n_hidden : int, optional
            Number of hidden units in each layer, by default None.
        layers_dim : list[int], optional
            List of layer dimensions.
        act : str, optional
            Activation function, by default "ReLU". See `torch.nn` for available activations.
        act_out : str | None, optional
            Activation function for the output layer, by default None. If None, Identity is applied.
        use_batchnorm : bool, optional
            Whether to use batch normalization, by default False.
        dropout : float, optional
            Dropout rate, by default 0.0.
        categories : np.ndarray, optional
            Array of categories for each categorical feature, by default None. If provided, the network will include an
            embedding layer before the first hidden layer.
        embedding_config_dct : dict[str, Any] | None, optional
            Configuration dictionary for embeddings, by default None.
        disable_embeddings : bool, optional
            If True, embeddings are disabled and raw features are used, by default False.
        act_kwargs : Any, optional
            Additional arguments to pass to the activation function. Does not apply to the output layer.
        """
        super().__init__()

        if layers_dim is None:
            if n_layers is None or n_hidden is None or output_dim is None:
                raise ValueError("n_layers, n_hidden and output_dim must be provided if layers_dim is None.")

            layers_dim = [len(numer_idx) + len(categ_idx)] + [n_hidden] * n_layers + [output_dim]
        else:
            if n_layers is not None or n_hidden is not None:
                logging.warning("Ignoring n_layers and n_hidden as layers_dim is provided.")

        embedding_dim = layers_dim[1]

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

        layers_dim[0] = self.flat_preprocessor.output_dim

        self.layers = nn.ModuleList()

        input_layer = FeedForwardLayer(
            n_in=layers_dim[0],
            n_out=layers_dim[1],
            act=act,
            use_batchnorm=use_batchnorm,
            dropout=dropout,
            act_kwargs=act_kwargs,
        )
        self.layers.add_module("input_layer", input_layer)

        for i in range(1, len(layers_dim) - 2):
            n_in, n_out = layers_dim[i], layers_dim[i + 1]
            layer = FeedForwardLayer(
                n_in=n_in,
                n_out=n_out,
                act=act,
                use_batchnorm=use_batchnorm,
                dropout=dropout,
                act_kwargs=act_kwargs,
            )
            self.layers.add_module(f"layer{i}", layer)

        output_layer = FeedForwardLayer(
            n_in=layers_dim[-2],
            n_out=layers_dim[-1],
            act=act_out,
            use_batchnorm=False,
            dropout=0.0,
        )
        self.layers.add_module("output_layer", output_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flat_preprocessor(x)

        for layer in self.layers:
            x = layer(x)

        return x
