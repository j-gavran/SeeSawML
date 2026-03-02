"""Generic ONNX export utilities.

This module provides reusable components for exporting PyTorch models to ONNX format,
including validation, output comparison, and error handling.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import torch

from seesawml.inference.onnx.constants import ONNX_ATOL, ONNX_OPSET_VERSION, ONNX_RTOL
from seesawml.inference.onnx.utils import test_onnx_export


@dataclass
class ExportResult:
    """Result of an ONNX export operation.

    Attributes
    ----------
    success : bool
        Whether export succeeded.
    save_path : str
        Path where ONNX model was saved (or attempted).
    error_message : str | None
        Error message if export failed.
    """

    success: bool
    save_path: str
    error_message: str | None = None


def to_tensor_list(x: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
    """Convert model output to a list of tensors.

    Parameters
    ----------
    x : torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]
        Model output as single tensor, list, or tuple.

    Returns
    -------
    list[torch.Tensor]
        Output tensors as a list.

    Raises
    ------
    TypeError
        If x is not a tensor, list, or tuple.
    """
    if isinstance(x, torch.Tensor):
        return [x]
    if isinstance(x, (list, tuple)):
        return list(x)
    raise TypeError(f"Unsupported output type: {type(x)}")


def validate_onnx_model(save_path: str) -> tuple[bool, str | None]:
    """Validate ONNX model structure using onnx.checker.

    Parameters
    ----------
    save_path : str
        Path to the ONNX model file.

    Returns
    -------
    tuple[bool, str | None]
        (is_valid, error_message) - True if valid, error message if not.
    """
    try:
        import onnx

        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)
        return True, None
    except ImportError:
        logging.warning("onnx package not installed, skipping structural validation")
        return True, None
    except Exception as e:
        return False, f"ONNX validation failed: {e}"


def export_onnx_model(
    model: torch.nn.Module,
    example_input: torch.Tensor | tuple[torch.Tensor, ...],
    save_path: str,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]],
    atol: float = ONNX_ATOL,
    rtol: float = ONNX_RTOL,
    opset_version: int = ONNX_OPSET_VERSION,
) -> ExportResult:
    """Export model to ONNX and verify output matches.

    Parameters
    ----------
    model : torch.nn.Module
        Model to export.
    example_input : torch.Tensor | tuple[torch.Tensor, ...]
        Example input(s) for tracing.
    save_path : str
        Path to save the ONNX model.
    input_names : list[str]
        Names for ONNX inputs.
    output_names : list[str]
        Names for ONNX outputs.
    dynamic_axes : dict[str, dict[int, str]]
        Dynamic axes specification.
    atol : float
        Absolute tolerance for output comparison.
    rtol : float
        Relative tolerance for output comparison.
    opset_version : int
        ONNX opset version.

    Returns
    -------
    ExportResult
        Result containing success status and path.
    """
    logging.info(f"Exporting to {save_path}")

    with torch.no_grad():
        if isinstance(example_input, tuple):
            torch_output = model(*example_input)
        else:
            torch_output = model(example_input)

    args = example_input if isinstance(example_input, tuple) else (example_input,)
    try:
        torch.onnx.export(
            model,
            args=args,
            f=save_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            dynamo=False,
            fallback=True,  # Allow fallback to TorchScript for unsupported ops
        )
    except (RuntimeError, TypeError, ValueError) as e:
        error_message = f"ONNX export failed: {type(e).__name__}: {e}"
        logging.critical(error_message)
        return ExportResult(success=False, save_path=save_path, error_message=error_message)

    logging.info(f"Model saved to {save_path}")

    # Validate ONNX model structure
    is_valid, validation_error = validate_onnx_model(save_path)
    if not is_valid:
        logging.critical(validation_error)
        if os.path.exists(save_path):
            os.remove(save_path)
        return ExportResult(success=False, save_path=save_path, error_message=validation_error)

    # Test ONNX inference and compare outputs
    onnx_output = test_onnx_export(example_input, save_path)

    torch_out_list = [t.detach().cpu() for t in to_tensor_list(torch_output)]
    onnx_out_list = [t.detach().cpu() for t in to_tensor_list(onnx_output)]

    # Use both absolute and relative tolerance for robust comparison
    success = len(torch_out_list) == len(onnx_out_list) and all(
        torch.allclose(t, o, atol=atol, rtol=rtol) for t, o in zip(torch_out_list, onnx_out_list)
    )

    if success:
        logging.info(f"Export successful: {save_path}")
        return ExportResult(success=True, save_path=save_path)

    # Log detailed error info
    error_lines = [f"Export verification failed: {save_path}"]
    for i, (t, o) in enumerate(zip(torch_out_list, onnx_out_list)):
        max_diff = (t - o).abs().max().item()
        mean_diff = (t - o).abs().mean().item()
        error_lines.append(f"  Output {i}: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")
        error_lines.append(f"    PyTorch: min={t.min():.4f}, max={t.max():.4f}, shape={t.shape}")
        error_lines.append(f"    ONNX:    min={o.min():.4f}, max={o.max():.4f}, shape={o.shape}")

    error_message = "\n".join(error_lines)
    logging.critical(error_message)

    # Clean up failed export
    if os.path.exists(save_path):
        os.remove(save_path)
        logging.warning(f"Removed failed export: {save_path}")

    return ExportResult(success=False, save_path=save_path, error_message=error_message)


def prepare_model_for_export(torch_model: torch.nn.Module) -> torch.nn.Module:
    """Prepare model for ONNX export by deep copying and configuring for ONNX compatibility.

    Parameters
    ----------
    torch_model : torch.nn.Module
        Original model (will NOT be modified).

    Returns
    -------
    torch.nn.Module
        Copy of model ready for export.

    Notes
    -----
    Configures:
    - torch.compile wrapper (unwrapped)
    - SDP attention (uses manual attention with nan_to_num for 100% masking)
    - Pairwise attention features (switched to ONNX-compatible full-matrix mode)
    """
    import copy

    # Deep copy to avoid mutating the original
    model = copy.deepcopy(torch_model)

    # Unwrap torch.compile if present
    if hasattr(model, "_orig_mod") and model._orig_mod is not None:
        logging.info("Detected torch.compile wrapper; unwrapping for ONNX export.")
        model = model._orig_mod

    model = model.eval().cpu()

    # Enable ONNX-compatible mode for pairwise features (full N×N matrix instead of tril_indices)
    _enable_pairwise_onnx_mode(model)

    # Disable SDP attention for ONNX export - use manual attention with nan_to_num
    # to properly handle 100% masking (all -inf softmax inputs)
    try:
        from seesawml.models.transformers.attention import Attend

        sdp_disabled_count = 0
        for module in model.modules():
            if isinstance(module, Attend) and module.use_sdp:
                module.use_sdp = False
                sdp_disabled_count += 1
        if sdp_disabled_count > 0:
            logging.info(f"  Disabled SDP attention in {sdp_disabled_count} modules for ONNX export")
    except ImportError:
        pass  # Attend class not available (e.g., for simpler models)

    return model


def _enable_pairwise_onnx_mode(model: torch.nn.Module) -> None:
    """Enable ONNX-compatible mode for pairwise feature modules.

    Sets onnx_compatible=True on PairwiseFeaturesCalculator and PairwiseFeaturesEmbedder,
    switching from tril_indices (not ONNX-traceable) to full N×N matrix computation.

    Parameters
    ----------
    model : torch.nn.Module
        Model to configure (modified in-place).
    """
    try:
        from seesawml.models.transformers.pairwise_features import (
            PairwiseFeaturesCalculator,
            PairwiseFeaturesEmbedder,
        )
    except ImportError:
        return  # Pairwise features not available

    calculator_count = 0
    embedder_count = 0

    for module in model.modules():
        if isinstance(module, PairwiseFeaturesCalculator):
            if not module.onnx_compatible:
                module.onnx_compatible = True
                calculator_count += 1
        elif isinstance(module, PairwiseFeaturesEmbedder):
            if not module.onnx_compatible:
                module.onnx_compatible = True
                embedder_count += 1

    if calculator_count > 0 or embedder_count > 0:
        logging.info(
            f"  Enabled ONNX-compatible pairwise features: "
            f"{calculator_count} calculator(s), {embedder_count} embedder(s)"
        )


def extract_valid_type_values(torch_model: torch.nn.Module) -> dict[str, list[float]]:
    """Extract valid type values from model's jagged preprocessor.

    These values are used to set the type field in example inputs,
    preventing 100% masking which causes attention softmax NaN.

    Parameters
    ----------
    torch_model : torch.nn.Module
        Model with jagged_preprocessor attribute.

    Returns
    -------
    dict[str, list[float]]
        Mapping of collection name to valid type values.
    """
    valid_type_values: dict[str, list[float]] = {}

    if not hasattr(torch_model, "jagged_preprocessor"):
        return valid_type_values

    prep = torch_model.jagged_preprocessor
    if not hasattr(prep, "_valid_type_tensors") or not hasattr(prep, "object_names"):
        return valid_type_values

    for i, obj_name in enumerate(prep.object_names):
        if i < len(prep._valid_type_tensors) and len(prep._valid_type_tensors[i]) > 0:
            valid_type_values[obj_name] = prep._valid_type_tensors[i].tolist()
            logging.info(f"  Found valid_types for {obj_name}: {valid_type_values[obj_name]}")

    return valid_type_values
