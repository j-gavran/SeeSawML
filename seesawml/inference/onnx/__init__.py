"""ONNX export utilities for SeeSaw models.

This package provides modular components for exporting PyTorch models to ONNX format.

Modules
-------
constants
    Shared constants for ONNX export (opset version, tolerances).
utils
    ONNX validation and testing utilities.
metadata
    ONNX metadata generation and serialization.
scaler_utils
    Scaler comparison and validation utilities.
config_extractors
    Extract scaler configurations from training reports.
input_builders
    Build example inputs for ONNX export tracing.
export
    Core export logic with validation.
"""

from seesawml.inference.onnx.constants import (
    DEFAULT_MAX_OBJECTS,
    ONNX_ATOL,
    ONNX_OPSET_VERSION,
    ONNX_PRODUCER_NAME,
    ONNX_PRODUCER_VERSION,
    ONNX_RTOL,
)
from seesawml.inference.onnx.utils import test_onnx_export
from seesawml.inference.onnx.metadata import ONNXMetadata, build_metadata, save_metadata
from seesawml.inference.onnx.scaler_utils import (
    categorical_maps_equal,
    categorical_scalers_equivalent,
    infer_numerical_scaler_type,
    numerical_scalers_equivalent,
    select_uniform_scaler_from_list,
    tensor_values_equal,
)
from seesawml.inference.onnx.config_extractors import (
    build_collection_config,
    extract_categ_scaler_config,
    extract_numer_scaler_config,
)
from seesawml.inference.onnx.input_builders import (
    build_jagged_example_inputs,
    create_collection_example_input,
    get_collection_max_objects,
)
from seesawml.inference.onnx.export import (
    ExportResult,
    export_onnx_model,
    extract_valid_type_values,
    prepare_model_for_export,
    to_tensor_list,
    validate_onnx_model,
)

__all__ = [
    # constants
    "DEFAULT_MAX_OBJECTS",
    "ONNX_ATOL",
    "ONNX_RTOL",
    "ONNX_OPSET_VERSION",
    "ONNX_PRODUCER_NAME",
    "ONNX_PRODUCER_VERSION",
    # utils
    "test_onnx_export",
    # metadata
    "ONNXMetadata",
    "build_metadata",
    "save_metadata",
    # scaler_utils
    "tensor_values_equal",
    "infer_numerical_scaler_type",
    "categorical_maps_equal",
    "numerical_scalers_equivalent",
    "categorical_scalers_equivalent",
    "select_uniform_scaler_from_list",
    # config_extractors
    "extract_numer_scaler_config",
    "extract_categ_scaler_config",
    "build_collection_config",
    # input_builders
    "get_collection_max_objects",
    "create_collection_example_input",
    "build_jagged_example_inputs",
    # export
    "ExportResult",
    "to_tensor_list",
    "validate_onnx_model",
    "export_onnx_model",
    "prepare_model_for_export",
    "extract_valid_type_values",
]
