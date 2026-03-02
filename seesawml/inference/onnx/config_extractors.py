"""Extract scaler configurations from training reports."""

from __future__ import annotations

import logging
from typing import Any

from seesawml.inference.wrappers import (
    CategoricalScalerConfig,
    CollectionScalerConfig,
    NumericalScalerConfig,
)
from seesawml.inference.onnx.scaler_utils import (
    infer_numerical_scaler_type,
    select_uniform_scaler_from_list,
)


def extract_numer_scaler_config(reports: dict[str, Any], dataset_name: str) -> NumericalScalerConfig:
    """Extract numerical scaler configuration from training reports.

    Parameters
    ----------
    reports : dict[str, Any]
        Training reports loaded from checkpoint.
    dataset_name : str
        Name of the dataset (e.g., "events", "jets").

    Returns
    -------
    NumericalScalerConfig
        Typed scaler configuration.

    Notes
    -----
    Handles both single scalers and per-object scaler lists.
    For scaler lists, all non-None entries must be equivalent.
    """
    scaler_dct = reports.get("numer_feature_scaler_dct")
    selection = reports.get("selection", {})

    scaler = scaler_dct.get(dataset_name) if scaler_dct else reports.get("numer_scaler")

    # Handle list of scalers (allow only if all non-None are equivalent)
    if isinstance(scaler, list):
        scaler = select_uniform_scaler_from_list(scaler, dataset_name=dataset_name, scaler_kind="numerical")

    idx = list(selection.get(dataset_name, {}).get("offset_numer_columns_idx", []))

    if scaler is None:
        logging.info(
            f"  No numerical scaler found for '{dataset_name}'; numerical features will be passed through unchanged."
        )
        return NumericalScalerConfig(scaler_type=None, idx=[])

    scaler_type = infer_numerical_scaler_type(scaler)

    if scaler_type == "standard":
        return NumericalScalerConfig(
            scaler_type="standard",
            idx=idx,
            mu=list(scaler.scaler.mean_),
            std=list(scaler.scaler.scale_),
        )
    elif scaler_type == "minmax":
        return NumericalScalerConfig(
            scaler_type="minmax",
            idx=idx,
            min=list(scaler.scaler.data_min_),
            max=list(scaler.scaler.data_max_),
        )

    logging.info(
        f"  Unsupported or missing numerical scaler type for '{dataset_name}'; "
        "numerical features will be passed through unchanged."
    )
    return NumericalScalerConfig(scaler_type=None, idx=[])


def extract_categ_scaler_config(reports: dict[str, Any], dataset_name: str) -> CategoricalScalerConfig:
    """Extract categorical scaler configuration from training reports.

    Parameters
    ----------
    reports : dict[str, Any]
        Training reports loaded from checkpoint.
    dataset_name : str
        Name of the dataset (e.g., "events", "jets").

    Returns
    -------
    CategoricalScalerConfig
        Typed scaler configuration.

    Notes
    -----
    Handles both single scalers and per-object scaler lists.
    For scaler lists, all non-None entries must be equivalent.
    """
    categ_scaler_dct = reports.get("categ_feature_scaler_dct")
    selection = reports.get("selection", {})

    categ_scaler = categ_scaler_dct.get(dataset_name) if categ_scaler_dct else reports.get("categ_scaler")

    # Handle list of scalers (allow only if all non-None are equivalent)
    if isinstance(categ_scaler, list):
        categ_scaler = select_uniform_scaler_from_list(
            scalers=categ_scaler, dataset_name=dataset_name, scaler_kind="categorical"
        )

    idx = list(selection.get(dataset_name, {}).get("offset_categ_columns_idx", []))

    if categ_scaler is None:
        return CategoricalScalerConfig(idx=idx)

    categories = categ_scaler.categories if hasattr(categ_scaler, "categories") else []
    return CategoricalScalerConfig(idx=idx, categories=categories)


def build_collection_config(reports: dict[str, Any], collection_name: str) -> CollectionScalerConfig:
    """Build complete scaler configuration for a collection.

    Parameters
    ----------
    reports : dict[str, Any]
        Training reports.
    collection_name : str
        Name of the collection.

    Returns
    -------
    CollectionScalerConfig
        Combined numerical and categorical configuration.
    """
    return CollectionScalerConfig(
        numerical=extract_numer_scaler_config(reports, collection_name),
        categorical=extract_categ_scaler_config(reports, collection_name),
    )
