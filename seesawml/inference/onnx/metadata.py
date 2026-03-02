"""ONNX model metadata generation and serialization."""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import torch
from f9columnar.utils.helpers import dump_json

from seesawml.models.utils import load_reports
from seesawml.utils.constants import PARTICLE_PREFIX_MAP


@dataclass
class ScalingConfig:
    """Scaling configuration for a single feature set."""

    scaler_type: Literal["standard", "minmax"] = "standard"
    numer_idx: list[int] = field(default_factory=list)
    numer_mu: list[float] = field(default_factory=list)
    numer_std: list[float] = field(default_factory=list)
    numer_min: list[float] = field(default_factory=list)
    numer_max: list[float] = field(default_factory=list)
    categ_idx: list[int] = field(default_factory=list)
    categories: list[dict[float, int]] = field(default_factory=list)


@dataclass
class ONNXMetadata:
    """Metadata for ONNX model inference.

    Attributes
    ----------
    model_type : str
        Either "flat" or "jagged".
    output : str
        Output type: "probabilities" (scaling embedded) or "logits" (raw).
    inputs : list[str]
        Input tensor names (e.g., ["X"] or ["X_jets", "X_leptons"]).
    class_labels : dict[str, int]
        Mapping from class name to label index.
    scaling : dict | None
        Scaling config (only for raw models without embedded scaling).
    feature_names : dict[str, list[str]]
        Feature names per collection in the order used by the model.
        Required for TNAnalysis to build features in the correct order.
    max_objects : dict[str, int] | None
        Maximum objects per collection for jagged models (e.g., {"jets": 15}).
    custom_groups : dict[str, list[str]] | None
        Custom class groupings for computing group-level scores (e.g., {"ttH": ["ttH_bb", "ttH_cc"]}).
    """

    model_type: Literal["flat", "jagged"]
    output: Literal["probabilities", "logits"]
    inputs: list[str]
    class_labels: dict[str, int] = field(default_factory=dict)
    scaling: dict[str, Any] | None = None
    feature_names: dict[str, list[str]] = field(default_factory=dict)
    max_objects: dict[str, int] | None = None
    custom_groups: dict[str, list[str]] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, excluding None values and empty dicts."""
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None and v != {}}


def build_metadata(
    checkpoint_path: str,
    scaling_embedded: bool,
    collection_names: list[str] | None = None,
) -> ONNXMetadata:
    """Build ONNX metadata from checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to the model checkpoint.
    scaling_embedded : bool
        Whether scaling is embedded in the ONNX model.
    collection_names : list[str] | None
        Collection names for jagged models. None for flat models.

    Returns
    -------
    ONNXMetadata
        Populated metadata object.

    Raises
    ------
    ValueError
        If feature columns cannot be found for any collection.
    """
    reports = load_reports(checkpoint_path, add_selection=True)
    selection = reports.get("selection", {})
    is_jagged = collection_names is not None

    # Determine model type and inputs
    model_type: Literal["flat", "jagged"] = "jagged" if is_jagged else "flat"
    if is_jagged and collection_names:
        inputs = [f"X_{name}" for name in collection_names]
    else:
        inputs = ["X"]

    # Output type
    output: Literal["probabilities", "logits"] = "probabilities" if scaling_embedded else "logits"

    # Class labels
    class_labels: dict[str, int] = {}
    raw_labels = reports.get("class_labels")
    if raw_labels:
        try:
            class_labels = {str(k): int(v) for k, v in raw_labels.items()}
        except Exception:
            logging.warning("Failed to parse class_labels from checkpoint.", exc_info=True)

    # Feature names (always populated - required for TNAnalysis dynamic ordering)
    feature_names: dict[str, list[str]] = {}
    if is_jagged and collection_names:
        for coll_name in collection_names:
            coll_sel = selection.get(coll_name, {})
            cols = coll_sel.get("offset_used_columns", [])
            if not cols:
                raise ValueError(
                    f"No feature columns found for collection '{coll_name}'. Ensure the checkpoint has selection info."
                )
            # Append type field (always last column in ONNX model input)
            prefix = PARTICLE_PREFIX_MAP.get(coll_name)
            if prefix:
                type_field = f"{prefix}_type"
                cols = list(cols) + [type_field]  # Create new list, don't modify original
            feature_names[coll_name] = cols
    else:
        events_sel = selection.get("events", {})
        cols = events_sel.get("offset_used_columns", [])
        if not cols:
            raise ValueError("No feature columns found for 'events'. Ensure the checkpoint has selection info.")
        feature_names["events"] = cols

    # Scaling config (only for raw models)
    scaling: dict[str, Any] | None = None
    if not scaling_embedded:
        scaling = _extract_scaling_config(selection, collection_names)

    # Max objects per collection (for jagged models)
    max_objects: dict[str, int] | None = None
    if is_jagged and collection_names:
        max_objects = {}
        for coll_name in collection_names:
            coll_sel = selection.get(coll_name, {})
            n_obj = coll_sel.get("n_objects")
            if n_obj is not None:
                max_objects[coll_name] = int(n_obj)

    # Custom groups (from checkpoint hyper_parameters -> dataset_conf)
    custom_groups: dict[str, list[str]] | None = None
    try:
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        hyper_params = loaded.get("hyper_parameters", {})
        dataset_conf = hyper_params.get("dataset_conf", {})

        # Try custom_groups first, then scores (legacy name)
        raw_groups = dataset_conf.get("custom_groups") or dataset_conf.get("scores")
        if raw_groups:
            custom_groups = {str(k): list(v) for k, v in dict(raw_groups).items()}
            logging.info(f"  Found custom_groups: {list(custom_groups.keys())}")
    except Exception:
        logging.debug("Could not extract custom_groups from checkpoint.", exc_info=True)

    return ONNXMetadata(
        model_type=model_type,
        output=output,
        inputs=inputs,
        class_labels=class_labels,
        scaling=scaling,
        feature_names=feature_names,
        max_objects=max_objects,
        custom_groups=custom_groups,
    )


def _extract_scaling_config(
    selection: dict[str, Any],
    collection_names: list[str] | None,
) -> dict[str, Any]:
    """Extract scaling configuration from selection info.

    Parameters
    ----------
    selection : dict
        Selection/preprocessing info from checkpoint.
    collection_names : list[str] | None
        Collection names for jagged, None for flat.

    Returns
    -------
    dict
        Scaling configuration.
    """
    if collection_names is not None:
        # Jagged: per-collection scaling
        collections = {}
        for coll_name in collection_names:
            coll_sel = selection.get(coll_name, {})
            collections[coll_name] = {
                "columns": coll_sel.get("offset_used_columns", []),
                "scaler_type": coll_sel.get("scaler_type", "standard"),
                "numer_idx": coll_sel.get("numer_idx", []),
                "numer_mu": coll_sel.get("numer_mu", []),
                "numer_std": coll_sel.get("numer_std", []),
                "numer_min": coll_sel.get("numer_min", []),
                "numer_max": coll_sel.get("numer_max", []),
                "categ_idx": coll_sel.get("categ_idx", []),
                "categories": coll_sel.get("categories", []),
            }
        return {"collections": collections}
    else:
        # Flat: single scaling config
        events_sel = selection.get("events", {})
        return {
            "columns": events_sel.get("offset_used_columns", []),
            "scaler_type": events_sel.get("scaler_type", "standard"),
            "numer_idx": events_sel.get("numer_idx", []),
            "numer_mu": events_sel.get("numer_mu", []),
            "numer_std": events_sel.get("numer_std", []),
            "numer_min": events_sel.get("numer_min", []),
            "numer_max": events_sel.get("numer_max", []),
            "categ_idx": events_sel.get("categ_idx", []),
            "categories": events_sel.get("categories", []),
        }


def save_metadata(
    checkpoint_path: str,
    model_save_name: str,
    onnx_dir: str,
    scaling_embedded: bool,
    collection_names: list[str] | None = None,
) -> dict[str, Any]:
    """Build and save ONNX metadata to JSON file.

    Parameters
    ----------
    checkpoint_path : str
        Path to the model checkpoint.
    model_save_name : str
        Base name for the metadata file.
    onnx_dir : str
        Directory to save metadata.
    scaling_embedded : bool
        Whether scaling is embedded in the ONNX model.
    collection_names : list[str] | None
        Collection names for jagged models.

    Returns
    -------
    dict
        The saved metadata dictionary.

    Raises
    ------
    ValueError
        If feature columns cannot be found for any collection.
    """
    metadata = build_metadata(
        checkpoint_path=checkpoint_path,
        scaling_embedded=scaling_embedded,
        collection_names=collection_names,
    )

    metadata_dict = metadata.to_dict()
    save_path = os.path.join(onnx_dir, f"{model_save_name}_metadata.json")
    dump_json(metadata_dict, save_path)
    logging.info(f"Saved ONNX metadata to {save_path}")

    return metadata_dict
