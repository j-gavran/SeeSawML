"""Inference and ONNX export utilities for SeeSaw models.

This package provides reusable components for model inference and ONNX export
that can be used across different model types (signal, fakes, etc.).

Subpackages
-----------
onnx
    ONNX export utilities including metadata, validation, and input building.

Modules
-------
wrappers
    Inference wrappers with embedded scaling for ONNX export.
"""

from seesawml.inference.wrappers import (
    TYPE_FIELD_INDEX,
    CategoricalScalerConfig,
    CollectionScalerConfig,
    FlatInferenceWrapper,
    JaggedInferenceWrapper,
    NumericalScalerConfig,
    RawJaggedWrapper,
    ScalingModule,
    split_features_and_type,
)

__all__ = [
    # Constants
    "TYPE_FIELD_INDEX",
    # Config dataclasses
    "NumericalScalerConfig",
    "CategoricalScalerConfig",
    "CollectionScalerConfig",
    # Wrappers
    "ScalingModule",
    "FlatInferenceWrapper",
    "JaggedInferenceWrapper",
    "RawJaggedWrapper",
    # Utilities
    "split_features_and_type",
]
