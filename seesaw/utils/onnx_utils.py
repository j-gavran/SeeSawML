import logging
import os
from typing import Any

import onnxruntime
import torch
from f9columnar.utils.helpers import dump_json

from seesaw.models.utils import load_reports


def get_metadata(
    checkpoint_path: str,
    model_save_name: str,
    dataset_name: str,
    onnx_dir: str,
    extra_metadata: dict | None = None,
) -> dict[str, Any]:
    reports = load_reports(checkpoint_path, add_selection=True)

    selection = reports["selection"][dataset_name]

    metadata = {
        "column_names": selection["offset_used_columns"],
        "numer_column_idx": [int(i) for i in selection["offset_numer_columns_idx"]],
        "categ_column_idx": [int(i) for i in selection["offset_categ_columns_idx"]],
    }

    if type(extra_metadata) is dict:
        metadata.update(extra_metadata)

    # Attach class label mapping if available in reports
    # Expected format: { "ttH_cc": 0, "ttH_bb": 1, ... }
    class_labels = reports.get("class_labels", None)
    if class_labels is not None and len(class_labels) > 0:
        try:
            # Ensure JSON-serializable content and deterministic ordering helpers
            cls_map: dict[str, int] = {str(k): int(v) for k, v in class_labels.items()}
            cls_order: list[str] = [k for k, _ in sorted(cls_map.items(), key=lambda kv: kv[1])]
            inverse_cls_map: dict[str, str] = {str(v): k for k, v in cls_map.items()}

            metadata["class_labels"] = cls_map
            metadata["class_labels_order"] = cls_order
            metadata["inverse_class_labels"] = inverse_cls_map
            metadata["num_classes"] = len(cls_map)
        except Exception:
            logging.warning("Failed to serialize class_labels for ONNX metadata.", exc_info=True)

    scaler_dct = reports.get("numer_feature_scaler_dct", None)
    categ_scaler_dct = reports.get("categ_feature_scaler_dct", None)

    if scaler_dct is None:
        scalers = reports.get("numer_scaler", None)
    else:
        scalers = scaler_dct[dataset_name]

    if categ_scaler_dct is None:
        categ_scalers = reports.get("categ_scaler", None)
    else:
        categ_scalers = categ_scaler_dct[dataset_name]

    if scalers is not None:
        if type(scalers) is not list:
            scalers = [scalers]

        scalers_lst = []

        for scaler in scalers:
            scaler_type = scaler.scaler_type
            scaler_metadata = {"scaler_type": scaler_type}

            if scaler_type == "standard":
                scaler_metadata["mu"] = [float(i) for i in scaler.scaler.mean_]
                scaler_metadata["std"] = [float(i) for i in scaler.scaler.scale_]
            elif scaler_type == "minmax":
                scaler_metadata["min"] = [float(i) for i in scaler.scaler.data_min_]
                scaler_metadata["max"] = [float(i) for i in scaler.scaler.data_max_]
            else:
                raise ValueError(f"Scaler type {scaler_type} not supported for ONNX inference!")

            scalers_lst.append(scaler_metadata)

        if len(scalers_lst) == 1:
            metadata["numer_scaler"] = scalers_lst[0]
        else:
            metadata["numer_scaler"] = scalers_lst

    else:
        metadata["numer_scaler"] = None

    if categ_scalers is not None:
        if type(categ_scalers) is not list:
            categ_scalers = [categ_scalers]

        categ_scalers_lst = []

        for categ_scaler in categ_scalers:
            scaler_type = categ_scaler.scaler_type
            categ_scaler_metadata = {"scaler_type": scaler_type}

            if scaler_type == "categorical":
                categ_scaler_metadata["categories"] = categ_scaler.categories
                categ_scaler_metadata["max_offset"] = categ_scaler.max_offset
                categ_scaler_metadata["inverse_categories"] = categ_scaler.inverse_categories
                categ_scaler_metadata["inverse_max_offset"] = categ_scaler.inverse_max_offset
            else:
                raise ValueError(f"Scaler type {scaler_type} not supported for ONNX inference!")

            categ_scalers_lst.append(categ_scaler_metadata)

        if len(categ_scalers_lst) == 1:
            metadata["categ_scaler"] = categ_scalers_lst[0]
        else:
            metadata["categ_scaler"] = categ_scalers_lst

    else:
        metadata["categ_scaler"] = None

    metadata_save_path = os.path.join(onnx_dir, f"{model_save_name}_metadata.json")
    logging.info(f"Saved metadata to {metadata_save_path}")
    dump_json(metadata, metadata_save_path)

    return metadata


def test_onnx_export(example_input: torch.Tensor, save_path: str) -> list[torch.Tensor] | torch.Tensor:
    onnx_input = [example_input]

    ort_session = onnxruntime.InferenceSession(save_path, providers=["CPUExecutionProvider"])

    onnxruntime_input = {}
    for k, v in zip(ort_session.get_inputs(), onnx_input):
        onnxruntime_input[k.name] = v.numpy() if isinstance(v, torch.Tensor) else v

    onnxruntime_outputs = ort_session.run(None, onnxruntime_input)

    outputs_as_tensors = [torch.tensor(o, dtype=torch.float32) for o in onnxruntime_outputs]

    if len(outputs_as_tensors) == 1:
        return outputs_as_tensors[0]

    return outputs_as_tensors
