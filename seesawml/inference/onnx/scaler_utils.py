"""Scaler comparison and validation utilities for ONNX export."""

from __future__ import annotations

import logging
from typing import Any, Callable

import torch


def tensor_values_equal(a: Any, b: Any) -> bool:
    """Compare scalar/array-like values for exact equality.

    Parameters
    ----------
    a : Any
        First value (scalar, list, or tensor).
    b : Any
        Second value (scalar, list, or tensor).

    Returns
    -------
    bool
        True if values are exactly equal (same shape and values).
    """
    ta = torch.as_tensor(a)
    tb = torch.as_tensor(b)
    return ta.shape == tb.shape and bool(torch.equal(ta, tb))


def infer_numerical_scaler_type(scaler: Any) -> str | None:
    """Infer scaler type from wrapper metadata or underlying scaler attributes.

    Parameters
    ----------
    scaler : Any
        Scaler object, typically a wrapper with `.scaler` attribute
        pointing to sklearn-style scaler.

    Returns
    -------
    str | None
        "standard" for StandardScaler (has mean_, scale_),
        "minmax" for MinMaxScaler (has data_min_, data_max_),
        None if type cannot be determined.
    """
    scaler_type = getattr(scaler, "scaler_type", None)
    if scaler_type is not None:
        return scaler_type

    sc = getattr(scaler, "scaler", None)
    if sc is None:
        return None
    if hasattr(sc, "mean_") and hasattr(sc, "scale_"):
        return "standard"
    if hasattr(sc, "data_min_") and hasattr(sc, "data_max_"):
        return "minmax"
    return None


def categorical_maps_equal(
    categories_a: list[dict[float, int]] | Any,
    categories_b: list[dict[float, int]] | Any,
) -> bool:
    """Compare categorical map lists for equality.

    Parameters
    ----------
    categories_a : list[dict[float, int]] | Any
        First list of category mappings (raw_value -> encoded_index).
    categories_b : list[dict[float, int]] | Any
        Second list of category mappings.

    Returns
    -------
    bool
        True if all mappings are equal (order-independent for dict keys).
    """
    if len(categories_a) != len(categories_b):
        return False

    for map_a, map_b in zip(categories_a, categories_b):
        if isinstance(map_a, dict) and isinstance(map_b, dict):
            norm_a = sorted((float(k), int(v)) for k, v in map_a.items())
            norm_b = sorted((float(k), int(v)) for k, v in map_b.items())
            if norm_a != norm_b:
                return False
        else:
            if not tensor_values_equal(map_a, map_b):
                return False
    return True


def numerical_scalers_equivalent(scaler_a: Any, scaler_b: Any) -> bool:
    """Check if two numerical scalers have equivalent parameters.

    Parameters
    ----------
    scaler_a : Any
        First numerical scaler.
    scaler_b : Any
        Second numerical scaler.

    Returns
    -------
    bool
        True if both scalers have same type and parameters.
    """
    type_a = infer_numerical_scaler_type(scaler_a)
    type_b = infer_numerical_scaler_type(scaler_b)

    if type_a != type_b:
        return False

    base_a = getattr(scaler_a, "scaler", None)
    base_b = getattr(scaler_b, "scaler", None)
    if base_a is None or base_b is None:
        return False

    if type_a == "standard":
        return tensor_values_equal(base_a.mean_, base_b.mean_) and tensor_values_equal(base_a.scale_, base_b.scale_)
    if type_a == "minmax":
        return tensor_values_equal(base_a.data_min_, base_b.data_min_) and tensor_values_equal(
            base_a.data_max_, base_b.data_max_
        )

    # Unknown scaler type: treat as non-equivalent to avoid silent wrong scaling.
    return False


def categorical_scalers_equivalent(scaler_a: Any, scaler_b: Any) -> bool:
    """Check if two categorical scalers have equivalent mappings.

    Parameters
    ----------
    scaler_a : Any
        First categorical scaler with `.categories` attribute.
    scaler_b : Any
        Second categorical scaler.

    Returns
    -------
    bool
        True if both scalers have identical category mappings.
    """
    categories_a = getattr(scaler_a, "categories", [])
    categories_b = getattr(scaler_b, "categories", [])
    return categorical_maps_equal(categories_a, categories_b)


# Dispatch table for scaler equivalence checking
_EQUIV_FUNCS: dict[str, Callable[[Any, Any], bool]] = {
    "numerical": numerical_scalers_equivalent,
    "categorical": categorical_scalers_equivalent,
}


def select_uniform_scaler_from_list(
    scalers: list[Any],
    dataset_name: str,
    scaler_kind: str,
) -> Any | None:
    """Select a scaler from list only if all non-None entries are equivalent.

    Parameters
    ----------
    scalers : list[Any]
        List of scalers, possibly containing None values.
    dataset_name : str
        Name of dataset for error messages.
    scaler_kind : str
        Either "numerical" or "categorical".

    Returns
    -------
    Any | None
        First non-None scaler if all are equivalent, None if all are None.

    Raises
    ------
    ValueError
        If non-None scalers have different parameters or unknown scaler_kind.
    """
    non_none_scalers = [s for s in scalers if s is not None]
    if len(non_none_scalers) == 0:
        return None
    if len(non_none_scalers) == 1:
        return non_none_scalers[0]

    equiv_func = _EQUIV_FUNCS.get(scaler_kind)
    if equiv_func is None:
        raise ValueError(f"Unknown scaler kind: {scaler_kind}. Valid: {list(_EQUIV_FUNCS)}")

    reference = non_none_scalers[0]
    equivalent = all(equiv_func(reference, s) for s in non_none_scalers[1:])

    if not equivalent:
        raise ValueError(
            f"Found non-equivalent per-object {scaler_kind} scalers for '{dataset_name}'. "
            "In-model ONNX scaling currently requires the same scaler parameters across object slots."
        )

    logging.info(
        f"  Found {len(non_none_scalers)} per-object {scaler_kind} scalers for '{dataset_name}'; "
        "verified equivalent and using shared parameters."
    )
    return reference
