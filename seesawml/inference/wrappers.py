"""Inference wrappers with embedded scaling for ONNX export.

This module provides wrapper classes that embed feature scaling and categorical
encoding inside the model, making them part of the ONNX graph. This allows C++
inference code to pass raw features directly without preprocessing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import nn

# Type field is always the last column in jagged inputs
TYPE_FIELD_INDEX = -1

# Small epsilon to prevent division by zero in scaling
_SCALING_EPS = 1e-8


def split_features_and_type(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract features and type field from jagged tensor.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (batch, max_objects, features+1) where
        the last column is the type field.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        features : Tensor of shape (batch, max_objects, features)
        type_tensor : Tensor of shape (batch, max_objects)
    """
    return x[:, :, :TYPE_FIELD_INDEX], x[:, :, TYPE_FIELD_INDEX]


@dataclass
class NumericalScalerConfig:
    """Configuration for numerical feature scaling."""

    scaler_type: Literal["standard", "minmax"] | None = None
    idx: list[int] = field(default_factory=list)
    mu: list[float] = field(default_factory=list)
    std: list[float] = field(default_factory=list)
    min: list[float] = field(default_factory=list)
    max: list[float] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> NumericalScalerConfig:
        return cls(
            scaler_type=d.get("scaler_type", "standard"),
            idx=d.get("idx", []),
            mu=d.get("mu", []),
            std=d.get("std", []),
            min=d.get("min", []),
            max=d.get("max", []),
        )


@dataclass
class CategoricalScalerConfig:
    """Configuration for categorical feature encoding."""

    idx: list[int] = field(default_factory=list)
    categories: list[dict[float, int]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> CategoricalScalerConfig:
        return cls(
            idx=d.get("idx", []),
            categories=d.get("categories", []),
        )


@dataclass
class CollectionScalerConfig:
    """Combined scaling configuration for a single collection."""

    numerical: NumericalScalerConfig = field(default_factory=NumericalScalerConfig)
    categorical: CategoricalScalerConfig = field(default_factory=CategoricalScalerConfig)

    @classmethod
    def from_dict(cls, d: dict) -> CollectionScalerConfig:
        return cls(
            numerical=NumericalScalerConfig(
                scaler_type=d.get("scaler_type", "standard"),
                idx=d.get("numer_idx", []),
                mu=d.get("numer_mu", []),
                std=d.get("numer_std", []),
                min=d.get("numer_min", []),
                max=d.get("numer_max", []),
            ),
            categorical=CategoricalScalerConfig(
                idx=d.get("categ_idx", []),
                categories=d.get("categories", []),
            ),
        )


class ScalingModule(nn.Module):
    """Base class for inference wrappers with scaling operations.

    Provides methods to register scaling buffers and apply transformations.
    """

    def _register_numerical_buffers(
        self,
        config: NumericalScalerConfig,
        prefix: str = "",
    ) -> None:
        """Register numerical scaling buffers.

        Parameters
        ----------
        config : NumericalScalerConfig
            Numerical scaler configuration.
        prefix : str
            Prefix for buffer names (e.g., "jets_" for collection-specific).
        """
        self.register_buffer(f"{prefix}numer_idx", torch.tensor(config.idx, dtype=torch.long))

        scaler_type = config.scaler_type or "standard"

        if scaler_type == "standard":
            if len(config.idx) > 0 and (len(config.mu) == 0 or len(config.std) == 0):
                raise ValueError(
                    f"Missing standard-scaler statistics for '{prefix or 'global'}' numerical features. "
                    "Expected non-empty 'mu' and 'std' when numerical indices are provided."
                )

            self.register_buffer(f"{prefix}numer_mu", torch.tensor(config.mu, dtype=torch.float32))
            self.register_buffer(f"{prefix}numer_std", torch.tensor(config.std, dtype=torch.float32))
        elif scaler_type == "minmax":
            if len(config.idx) > 0 and (len(config.min) == 0 or len(config.max) == 0):
                raise ValueError(
                    f"Missing min-max scaler statistics for '{prefix or 'global'}' numerical features. "
                    "Expected non-empty 'min' and 'max' when numerical indices are provided."
                )

            self.register_buffer(f"{prefix}numer_min", torch.tensor(config.min, dtype=torch.float32))
            range_val = [mx - mn for mx, mn in zip(config.max, config.min)]
            self.register_buffer(f"{prefix}numer_range", torch.tensor(range_val, dtype=torch.float32))
        else:
            raise ValueError(f"Unknown scaler_type '{scaler_type}' for '{prefix or 'global'}' numerical features.")

    def _register_categorical_buffers(
        self,
        config: CategoricalScalerConfig,
        prefix: str = "",
    ) -> None:
        """Register categorical encoding buffers.

        Parameters
        ----------
        config : CategoricalScalerConfig
            Categorical scaler configuration.
        prefix : str
            Prefix for buffer names.
        """
        self.register_buffer(f"{prefix}categ_idx", torch.tensor(config.idx, dtype=torch.long))

        for i, cat_map in enumerate(config.categories):
            if cat_map:
                keys = torch.tensor(list(cat_map.keys()), dtype=torch.float32)
                vals = torch.tensor(list(cat_map.values()), dtype=torch.long)
                self.register_buffer(f"{prefix}categ_keys_{i}", keys)
                self.register_buffer(f"{prefix}categ_vals_{i}", vals)

    def _apply_numerical_scaling(
        self,
        x: torch.Tensor,
        numer_idx: torch.Tensor,
        scaler_type: str,
        prefix: str = "",
    ) -> torch.Tensor:
        """Apply numerical scaling to features.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor. Shape (B, F) for flat, (B, N, F) for jagged.
        numer_idx : torch.Tensor
            Indices of numerical features.
        scaler_type : str
            Type of scaling: "standard" or "minmax".
        prefix : str
            Prefix for accessing buffers.

        Returns
        -------
        torch.Tensor
            Scaled tensor (cloned).
        """
        if len(numer_idx) == 0:
            return x

        x = x.clone()
        is_jagged = x.dim() == 3

        if is_jagged:
            x_numer = x[:, :, numer_idx]
        else:
            x_numer = x[:, numer_idx]

        if scaler_type == "standard":
            mu = getattr(self, f"{prefix}numer_mu")
            std = getattr(self, f"{prefix}numer_std")
            x_numer = (x_numer - mu) / (std + _SCALING_EPS)
        elif scaler_type == "minmax":
            min_val = getattr(self, f"{prefix}numer_min")
            range_val = getattr(self, f"{prefix}numer_range")
            # Handle zero-range features (min == max): output 0 instead of unbounded values
            zero_range_mask = range_val == 0
            safe_range = torch.where(zero_range_mask, torch.ones_like(range_val), range_val)
            x_numer = (x_numer - min_val) / (safe_range + _SCALING_EPS)
            # Set zero-range features to 0 (they carry no information)
            x_numer = torch.where(zero_range_mask, torch.zeros_like(x_numer), x_numer)

        if is_jagged:
            x[:, :, numer_idx] = x_numer
        else:
            x[:, numer_idx] = x_numer

        return x

    def _apply_categorical_encoding(
        self,
        x: torch.Tensor,
        categ_idx: torch.Tensor,
        n_categ: int,
        prefix: str = "",
        wrapper_name: str = "InferenceWrapper",
    ) -> torch.Tensor:
        """Apply categorical encoding to features.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor. Shape (B, F) for flat, (B, N, F) for jagged.
        categ_idx : torch.Tensor
            Indices of categorical features.
        n_categ : int
            Number of categorical features.
        prefix : str
            Prefix for accessing buffers.
        wrapper_name : str
            Name for warning messages.

        Returns
        -------
        torch.Tensor
            Encoded tensor (cloned).
        """
        if len(categ_idx) == 0:
            return x

        x = x.clone()
        is_jagged = x.dim() == 3

        for i in range(n_categ):
            col_idx = categ_idx[i]
            keys = getattr(self, f"{prefix}categ_keys_{i}", None)
            vals = getattr(self, f"{prefix}categ_vals_{i}", None)

            if keys is None or vals is None:
                continue

            if is_jagged:
                col_vals = x[:, :, col_idx]
            else:
                col_vals = x[:, col_idx]

            matched = torch.zeros_like(col_vals, dtype=torch.bool)
            encoded = torch.zeros_like(col_vals, dtype=torch.float32)

            for k, v in zip(keys, vals):
                mask = col_vals == k
                matched = matched | mask
                encoded = torch.where(mask, v.float(), encoded)

            # Warn about unmatched values (only works in PyTorch, not ONNX runtime)
            if not torch.jit.is_scripting() and not matched.all():
                unmatched_vals = col_vals[~matched].unique()
                logging.warning(
                    f"[{wrapper_name}] Categorical feature {i} has "
                    f"{(~matched).sum().item()} unknown values (mapped to 0): "
                    f"{unmatched_vals.tolist()[:5]}..."
                )

            if is_jagged:
                x[:, :, col_idx] = encoded
            else:
                x[:, col_idx] = encoded

        return x


class FlatInferenceWrapper(ScalingModule):
    """Wrapper for flat (events-only) models with embedded scaling.

    Parameters
    ----------
    base_model : nn.Module
        The trained PyTorch model.
    numer_config : NumericalScalerConfig
        Numerical scaler configuration.
    categ_config : CategoricalScalerConfig
        Categorical scaler configuration.
    num_classes : int
        Number of output classes.

    Notes
    -----
    Output shape is (B,) for binary classification, (B, C) for multiclass.
    Unknown categorical values will raise an error.
    """

    def __init__(
        self,
        base_model: nn.Module,
        numer_config: NumericalScalerConfig,
        categ_config: CategoricalScalerConfig,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.scaler_type = numer_config.scaler_type or "standard"
        self.n_categ_features = len(categ_config.categories)

        self._register_numerical_buffers(numer_config)
        self._register_categorical_buffers(categ_config)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass with scaling and probability output.

        Parameters
        ----------
        X : torch.Tensor
            Raw input tensor of shape (B, F).

        Returns
        -------
        torch.Tensor
            Probabilities of shape (B,) for binary, (B, C) for multiclass.
        """
        X_scaled = self._apply_numerical_scaling(X, self.numer_idx, self.scaler_type)
        X_scaled = self._apply_categorical_encoding(
            X_scaled, self.categ_idx, self.n_categ_features, wrapper_name="FlatInferenceWrapper"
        )
        logits = self.base_model(X_scaled)

        if self.num_classes > 2:
            return torch.softmax(logits, dim=-1)
        return torch.sigmoid(logits).squeeze(-1)


class JaggedInferenceWrapper(ScalingModule):
    """Wrapper for jagged-only models with embedded scaling.

    Parameters
    ----------
    base_model : nn.Module
        The trained PyTorch model (expects jagged inputs).
    collection_configs : dict[str, CollectionScalerConfig]
        Per-collection scaling configurations.
    num_classes : int
        Number of output classes.

    Notes
    -----
    This wrapper is for jagged-only models (no events features).
    Input tensors have type field as last column: shape (B, max_obj, features+1).
    Collections are passed in the order defined by collection_names.
    """

    def __init__(
        self,
        base_model: nn.Module,
        collection_configs: dict[str, CollectionScalerConfig],
        num_classes: int,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.collection_names = list(collection_configs.keys())
        self._preserve_padding_masks = False
        self._collection_pad_tokens: dict[str, float] = {}
        self._collection_padding_numer_idx: dict[str, torch.Tensor] = {}

        # Validate and register per-collection buffers
        self._collection_scaler_types: dict[str, str] = {}
        self._collection_n_categ: dict[str, int] = {}

        for name, config in collection_configs.items():
            prefix = f"{name}_"
            scaler_type = config.numerical.scaler_type or "standard"
            self._collection_scaler_types[name] = scaler_type
            self._collection_n_categ[name] = len(config.categorical.categories)

            self._register_numerical_buffers(config.numerical, prefix)
            self._register_categorical_buffers(config.categorical, prefix)

        self._setup_padding_mask_preservation()

    def _setup_padding_mask_preservation(self) -> None:
        """Preserve padded rows when type-field masking is disabled in the base model."""
        prep = getattr(self.base_model, "jagged_preprocessor", None)
        if prep is None:
            return

        if bool(getattr(prep, "use_type_field_masking", False)):
            return

        object_names = list(getattr(prep, "object_names", []))
        numer_padding_tokens = getattr(prep, "numer_padding_tokens", None)
        numer_jagged_idx = getattr(prep, "numer_jagged_idx", None)
        if numer_padding_tokens is None or numer_jagged_idx is None:
            return

        for i, name in enumerate(object_names):
            if i >= len(numer_padding_tokens) or i >= len(numer_jagged_idx):
                continue

            pad_token = numer_padding_tokens[i]
            if isinstance(pad_token, torch.Tensor):
                pad_value = float(pad_token.detach().cpu().item())
            else:
                pad_value = float(pad_token)

            idx_tensor = numer_jagged_idx[i]
            if isinstance(idx_tensor, torch.Tensor):
                idx_values = idx_tensor.detach().cpu().to(torch.long)
            else:
                idx_values = torch.tensor(idx_tensor, dtype=torch.long)

            self._collection_pad_tokens[name] = pad_value
            self._collection_padding_numer_idx[name] = idx_values

        self._preserve_padding_masks = len(self._collection_pad_tokens) > 0
        if self._preserve_padding_masks:
            logging.info("  Padding-preserving jagged scaling enabled (type-field masking disabled).")

    def _scale_collection(self, x: torch.Tensor, collection_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Scale a collection and extract type tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, N, F+1) where last column is type field.
        collection_name : str
            Name of the collection.

        Returns
        -------
        x_scaled : torch.Tensor
            Scaled features (excluding type field).
        type_tensor : torch.Tensor
            Type values of shape (B, N).

        Raises
        ------
        KeyError
            If collection_name is not in configured collections.
        """
        if collection_name not in self._collection_scaler_types:
            raise KeyError(
                f"Unknown collection '{collection_name}'. Available: {list(self._collection_scaler_types.keys())}"
            )

        # Extract features and type tensor using shared helper
        x_features, type_tensor = split_features_and_type(x)
        pad_mask: torch.Tensor | None = None

        prefix = f"{collection_name}_"
        scaler_type = self._collection_scaler_types[collection_name]
        numer_idx = getattr(self, f"{prefix}numer_idx")
        categ_idx = getattr(self, f"{prefix}categ_idx")
        n_categ = self._collection_n_categ[collection_name]

        if self._preserve_padding_masks and collection_name in self._collection_padding_numer_idx:
            pad_numer_idx = self._collection_padding_numer_idx[collection_name].to(
                device=x_features.device, dtype=torch.long
            )
            if len(pad_numer_idx) > 0:
                pad_value = self._collection_pad_tokens[collection_name]
                numer_values = torch.index_select(x_features, dim=2, index=pad_numer_idx)
                # Handle NaN padding tokens: NaN == NaN is always False in IEEE 754
                # Use math.isnan for Python floats; pad_value is always float from _setup_padding_mask_preservation
                if pad_value != pad_value:  # NaN != NaN is True (IEEE 754 compliant, works with both float and tensor)
                    pad_mask = torch.isnan(numer_values).all(dim=-1, keepdim=True)
                else:
                    pad_mask = (numer_values == pad_value).all(dim=-1, keepdim=True)

        # Apply scaling
        x_scaled = self._apply_numerical_scaling(x_features, numer_idx, scaler_type, prefix)
        x_scaled = self._apply_categorical_encoding(
            x_scaled, categ_idx, n_categ, prefix, f"JaggedInferenceWrapper/{collection_name}"
        )

        # Keep fully padded rows unchanged so padding-token masking remains valid downstream.
        if pad_mask is not None:
            x_scaled = torch.where(pad_mask, x_features, x_scaled)

        return x_scaled, type_tensor

    def forward(self, *Xs: torch.Tensor) -> torch.Tensor:
        """Forward pass with per-collection scaling.

        Parameters
        ----------
        *Xs : torch.Tensor
            Collection tensors in order of collection_names.
            Each has shape (B, max_obj, features+1) with type as last column.

        Returns
        -------
        torch.Tensor
            Probabilities of shape (B,) for binary, (B, C) for multiclass.

        Raises
        ------
        ValueError
            If number of input tensors doesn't match number of collections.
        """
        if len(Xs) != len(self.collection_names):
            raise ValueError(
                f"Expected {len(self.collection_names)} input tensors for collections "
                f"{self.collection_names}, but got {len(Xs)}"
            )

        scaled_Xs = []
        type_tensors = []

        for i, X in enumerate(Xs):
            X_scaled, type_t = self._scale_collection(X, self.collection_names[i])
            scaled_Xs.append(X_scaled)
            type_tensors.append(type_t)

        # Forward through base model
        # Base model expects: X_events (None for jagged-only), *Xs, type_tensors=...
        logits = self.base_model(None, *scaled_Xs, type_tensors=type_tensors)

        if self.num_classes > 2:
            return torch.softmax(logits, dim=-1)
        return torch.sigmoid(logits).squeeze(-1)


class RawJaggedWrapper(nn.Module):
    """Thin wrapper for raw jagged models (no scaling, just signature adaptation).

    Parameters
    ----------
    base_model : nn.Module
        The trained PyTorch model.
    collection_names : list[str]
        Names of collections in order.

    Notes
    -----
    This wrapper adapts the model signature for ONNX export:
    - Accepts collection tensors with type field as last column
    - Passes None for X_events (jagged-only models)
    - Returns raw logits
    """

    def __init__(self, base_model: nn.Module, collection_names: list[str]) -> None:
        super().__init__()
        self.base_model = base_model
        self.collection_names = collection_names

    def forward(self, *Xs: torch.Tensor) -> torch.Tensor:
        """Forward pass extracting type tensors from input.

        Parameters
        ----------
        *Xs : torch.Tensor
            Collection tensors with type field as last column.

        Returns
        -------
        torch.Tensor
            Raw logits.

        Raises
        ------
        ValueError
            If number of input tensors doesn't match number of collections.
        """
        if len(Xs) != len(self.collection_names):
            raise ValueError(
                f"Expected {len(self.collection_names)} input tensors for collections "
                f"{self.collection_names}, but got {len(Xs)}"
            )

        scaled_Xs = []
        type_tensors = []

        for X in Xs:
            features, type_t = split_features_and_type(X)
            scaled_Xs.append(features)
            type_tensors.append(type_t)

        return self.base_model(None, *scaled_Xs, type_tensors=type_tensors)
