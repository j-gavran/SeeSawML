"""Shared constants for ONNX export."""

# Default max objects when object_slices not found
DEFAULT_MAX_OBJECTS = 10

# ONNX export comparison tolerances
ONNX_ATOL = 1e-5
ONNX_RTOL = 1e-4

# ONNX opset version (17 required for modern PyTorch ops)
ONNX_OPSET_VERSION = 17

# Producer name for ONNX metadata
ONNX_PRODUCER_NAME = "seesaw"
ONNX_PRODUCER_VERSION = "1.0"
