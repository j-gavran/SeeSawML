"""Build example inputs for ONNX export tracing."""

from __future__ import annotations

import logging
from typing import Any

import torch

from seesawml.inference.wrappers import TYPE_FIELD_INDEX, CategoricalScalerConfig
from seesawml.inference.onnx.constants import DEFAULT_MAX_OBJECTS


def get_collection_max_objects(
    coll_name: str,
    object_slices: dict[str, slice],
) -> int:
    """Get max objects for a collection from object slices.

    Parameters
    ----------
    coll_name : str
        Collection name.
    object_slices : dict[str, slice]
        Object slices from model.

    Returns
    -------
    int
        Max objects for this collection.
    """
    obj_slice = object_slices.get(coll_name)
    if obj_slice is not None:
        return obj_slice.stop - obj_slice.start
    logging.warning(f"  No object_slice found for {coll_name}, using default {DEFAULT_MAX_OBJECTS}")
    return DEFAULT_MAX_OBJECTS


def create_collection_example_input(
    coll_name: str,
    max_objects: int,
    coll_selection: dict[str, Any],
    valid_type_values: dict[str, list[float]],
    categorical_config: CategoricalScalerConfig | None,
    raw_model: bool,
) -> torch.Tensor:
    """Create example input tensor for a single collection.

    Parameters
    ----------
    coll_name : str
        Collection name.
    max_objects : int
        Maximum objects in this collection.
    coll_selection : dict[str, Any]
        Selection info for this collection.
    valid_type_values : dict[str, list[float]]
        Valid type values per collection.
    categorical_config : CategoricalScalerConfig | None
        Categorical config for scaling-embedded exports.
    raw_model : bool
        Whether this is a raw (no scaling) export.

    Returns
    -------
    torch.Tensor
        Example input of shape (1, max_objects, n_features).

    Notes
    -----
    For scaling-embedded exports, categorical features are filled with known
    raw values from the category mapping to ensure proper tracing.
    For raw exports, categorical indices are kept at 0 (valid encoded category).
    """
    n_features = len(coll_selection.get("offset_used_columns", [])) + 1
    example_input = torch.zeros(1, max_objects, n_features, dtype=torch.float32)

    # Fill numerical features with non-trivial values
    numer_idx = list(coll_selection.get("offset_numer_columns_idx", []))
    if len(numer_idx) > 0:
        example_input[:, :, numer_idx] = torch.randn(1, max_objects, len(numer_idx))

    # For scaling-embedded exports, drive categorical paths with known raw values
    if not raw_model and categorical_config is not None:
        categ_idx = list(coll_selection.get("offset_categ_columns_idx", []))
        for feat_idx, col_idx in enumerate(categ_idx):
            if feat_idx >= len(categorical_config.categories):
                continue
            cat_map = categorical_config.categories[feat_idx]
            if not isinstance(cat_map, dict) or len(cat_map) == 0:
                continue
            raw_values = torch.tensor(list(cat_map.keys()), dtype=torch.float32)
            repeats = (max_objects + len(raw_values) - 1) // len(raw_values)
            tiled = raw_values.repeat(repeats)[:max_objects]
            example_input[0, :, col_idx] = tiled

    # Set type field (last column) to valid type values
    valid_types = valid_type_values.get(coll_name)
    if not valid_types:
        logging.warning(
            f"No valid_types found for collection '{coll_name}', using [0.0]. "
            "This may cause attention softmax NaN if the model uses type-field masking."
        )
        valid_types = [0.0]
    for obj_idx in range(max_objects):
        example_input[:, obj_idx, TYPE_FIELD_INDEX] = valid_types[obj_idx % len(valid_types)]

    return example_input


def build_jagged_example_inputs(
    collection_names: list[str],
    selection: dict[str, Any],
    object_slices: dict[str, slice],
    valid_type_values: dict[str, list[float]],
    categorical_configs: dict[str, CategoricalScalerConfig] | None = None,
    raw_model: bool = False,
) -> tuple[list[str], list[torch.Tensor], dict[str, dict[int, str]]]:
    """Build example inputs for jagged model export.

    Parameters
    ----------
    collection_names : list[str]
        Names of collections to build inputs for.
    selection : dict[str, Any]
        Selection info from reports.
    object_slices : dict[str, slice]
        Object slices from model.
    valid_type_values : dict[str, list[float]]
        Valid type values per collection.
    categorical_configs : dict[str, CategoricalScalerConfig] | None
        Categorical configs for scaling-embedded exports.
    raw_model : bool
        Whether this is a raw (no scaling) export.

    Returns
    -------
    tuple[list[str], list[torch.Tensor], dict[str, dict[int, str]]]
        Input names, example inputs, and dynamic axes.
    """
    input_names: list[str] = []
    example_inputs: list[torch.Tensor] = []
    dynamic_axes: dict[str, dict[int, str]] = {}

    for coll_name in collection_names:
        logging.info(f"  Processing collection: {coll_name}")

        max_objects = get_collection_max_objects(coll_name, object_slices)
        logging.info(f"    max_objects: {max_objects}")

        coll_selection = selection.get(coll_name, {})
        categorical_config = categorical_configs.get(coll_name) if categorical_configs else None

        example_input = create_collection_example_input(
            coll_name=coll_name,
            max_objects=max_objects,
            coll_selection=coll_selection,
            valid_type_values=valid_type_values,
            categorical_config=categorical_config,
            raw_model=raw_model,
        )

        input_name = f"X_{coll_name}"
        input_names.append(input_name)
        example_inputs.append(example_input)
        dynamic_axes[input_name] = {0: "batch_size", 1: f"max_objects_{coll_name}"}

    return input_names, example_inputs, dynamic_axes
