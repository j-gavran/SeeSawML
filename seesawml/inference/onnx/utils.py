"""ONNX validation and testing utilities."""

from __future__ import annotations

import onnxruntime
import torch


def test_onnx_export(
    example_input: torch.Tensor | tuple[torch.Tensor, ...],
    save_path: str,
) -> list[torch.Tensor] | torch.Tensor:
    """Test ONNX export by running inference and returning outputs.

    Parameters
    ----------
    example_input : torch.Tensor | tuple[torch.Tensor, ...]
        Single tensor or tuple of tensors for multi-input models.
    save_path : str
        Path to the ONNX model file.

    Returns
    -------
    list[torch.Tensor] | torch.Tensor
        Single tensor for single-output models, list of tensors for multi-output.
        Output dtype is preserved from ONNX model (converted via torch.from_numpy).

    Raises
    ------
    ValueError
        If number of input tensors doesn't match ONNX model's expected inputs.

    Notes
    -----
    Uses CPUExecutionProvider for inference. Input tensors are converted to
    numpy arrays automatically. Output tensors are converted back using
    torch.from_numpy which preserves the original dtype from the ONNX model.
    """
    if isinstance(example_input, tuple):
        onnx_input = list(example_input)
    else:
        onnx_input = [example_input]

    ort_session = onnxruntime.InferenceSession(save_path, providers=["CPUExecutionProvider"])

    # Validate input count matches ONNX model expectations
    expected_inputs = ort_session.get_inputs()
    if len(onnx_input) != len(expected_inputs):
        expected_names = [inp.name for inp in expected_inputs]
        raise ValueError(
            f"Input count mismatch: got {len(onnx_input)} tensors, "
            f"but ONNX model expects {len(expected_inputs)} inputs: {expected_names}"
        )

    onnxruntime_input = {}
    for k, v in zip(expected_inputs, onnx_input):
        onnxruntime_input[k.name] = v.numpy() if isinstance(v, torch.Tensor) else v

    onnxruntime_outputs = ort_session.run(None, onnxruntime_input)

    outputs_as_tensors = [torch.from_numpy(o) for o in onnxruntime_outputs]

    if len(outputs_as_tensors) == 1:
        return outputs_as_tensors[0]

    return outputs_as_tensors
