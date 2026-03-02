"""ONNX export for signal/background classifiers.

Model type (flat/jagged) is auto-detected from the checkpoint.

Usage:
    python -m seesaw.signal.models.model_to_onnx onnx_config.checkpoint_path=/path/to/checkpoint.ckpt
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import hydra
import torch
from omegaconf import DictConfig

from seesawml.inference import (
    FlatInferenceWrapper,
    JaggedInferenceWrapper,
    RawJaggedWrapper,
)
from seesawml.inference.onnx import (
    ExportResult,
    build_collection_config,
    build_jagged_example_inputs,
    export_onnx_model,
    extract_categ_scaler_config,
    extract_numer_scaler_config,
    extract_valid_type_values,
    prepare_model_for_export,
    save_metadata,
)
from seesawml.models.utils import load_reports
from seesawml.utils.loggers import setup_logger


def _save_metadata_if_success(
    result: ExportResult,
    checkpoint_path: str,
    model_save_name: str,
    onnx_dir: str,
    in_model_scaling: bool,
    collection_names: list[str] | None,
) -> None:
    """Save ONNX metadata only if export succeeded.

    Parameters
    ----------
    result : ExportResult
        Export result to check.
    checkpoint_path : str
        Path to source checkpoint.
    model_save_name : str
        Name for saved model.
    onnx_dir : str
        Output directory.
    in_model_scaling : bool
        Whether scaling is embedded.
    collection_names : list[str] | None
        Collection names for jagged models, None for flat.
    """
    if result.success:
        save_metadata(
            checkpoint_path=checkpoint_path,
            model_save_name=model_save_name,
            onnx_dir=onnx_dir,
            scaling_embedded=in_model_scaling,
            collection_names=collection_names,
        )


def _detect_model_type(checkpoint_path: str) -> tuple[bool, dict[str, Any]]:
    """Detect if checkpoint contains a jagged (multi-collection) model.

    Parameters
    ----------
    checkpoint_path : str
        Path to the checkpoint file.

    Returns
    -------
    tuple[bool, dict[str, Any]]
        (is_jagged, reports) - True if jagged model, plus loaded reports.

    Raises
    ------
    ValueError
        If checkpoint has no 'selection' in reports (corrupt or incompatible).
    """
    reports = load_reports(checkpoint_path, add_selection=True)
    selection = reports.get("selection")

    # Validate selection exists
    if selection is None:
        raise ValueError(
            f"No 'selection' found in checkpoint reports for {checkpoint_path}. "
            "Checkpoint may be corrupt or from an incompatible version."
        )

    # Jagged models have collections beyond "events"
    collections = [k for k in selection.keys() if k != "events"]
    is_jagged = len(collections) > 0

    logging.info(f"  Detected model type: {'jagged' if is_jagged else 'flat'}")
    if is_jagged:
        logging.info(f"  Collections: {collections}")

    return is_jagged, reports


def _load_model_from_checkpoint(
    checkpoint_path: str,
    jagged: bool,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load model directly from checkpoint without external config.

    Parameters
    ----------
    checkpoint_path : str
        Path to the checkpoint file.
    jagged : bool
        Whether this is a jagged (multi-collection) model.

    Returns
    -------
    tuple[torch.nn.Module, dict[str, Any]]
        Model and hyperparameters from checkpoint.
    """
    from omegaconf import open_dict

    from seesawml.models.nn_modules import BaseLightningModule

    logging.info(f"Loading model from checkpoint: {checkpoint_path}")

    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = loaded["state_dict"]
    hyper_params = loaded["hyper_parameters"]

    # Remove dynamic model references if present
    for key in ["num_model", "den_model"]:
        if key in hyper_params:
            hyper_params.pop(key)

    # Disable torch.compile (if model_conf exists)
    if "model_conf" in hyper_params:
        with open_dict(hyper_params["model_conf"]):
            if "architecture_config" in hyper_params["model_conf"]:
                hyper_params["model_conf"]["architecture_config"]["compile"] = False

    # Determine model class based on detected type
    if jagged:
        from seesawml.signal.training.sig_bkg_trainer import SigBkgFullNNClassifier

        model_class = SigBkgFullNNClassifier
    else:
        from seesawml.signal.training.sig_bkg_trainer import SigBkgEventsNNClassifier

        model_class = SigBkgEventsNNClassifier

    # Fix typo in old checkpoints: qunatile_bins_param -> quantile_bins_param
    fixed_state_dict = {}
    for key, value in state_dict.items():
        fixed_key = key.replace("qunatile_bins_param", "quantile_bins_param")
        fixed_state_dict[fixed_key] = value

    model: BaseLightningModule = model_class(**hyper_params)
    model.load_state_dict(fixed_state_dict)
    model.eval()

    return model.model, hyper_params


def _generate_model_name(
    checkpoint_path: str,
    timestamp: str | None,
    name_suffix: str | None,
    jagged: bool,
) -> str:
    """Generate model save name from checkpoint and options.

    Parameters
    ----------
    checkpoint_path : str
        Path to the checkpoint.
    timestamp : str | None
        Optional timestamp override.
    name_suffix : str | None
        Optional suffix.
    jagged : bool
        Whether this is a jagged model.

    Returns
    -------
    str
        Generated model name.
    """
    checkpoint_name = os.path.basename(checkpoint_path)
    checkpoint_split = checkpoint_name.split("_")

    # Robust parsing: handle non-standard checkpoint names
    if len(checkpoint_split) >= 2:
        model_save_name = f"{checkpoint_split[0]}_{checkpoint_split[1]}_"
    else:
        # Fallback: use stem without extension
        stem = os.path.splitext(checkpoint_name)[0]
        logging.warning(f"Non-standard checkpoint name '{checkpoint_name}', using '{stem}_'")
        model_save_name = f"{stem}_"

    if timestamp is None:
        timestamp = time.asctime(time.localtime())

    model_save_name += timestamp.replace("  ", "_").replace(" ", "_").replace(":", "_")

    if name_suffix is not None:
        model_save_name += f"_{name_suffix}"

    if jagged:
        model_save_name += "_jagged"

    return model_save_name


def convert_from_checkpoint(
    checkpoint_path: str,
    onnx_dir: str,
    timestamp: str | None = None,
    name_suffix: str | None = None,
    in_model_scaling: bool = True,
) -> ExportResult:
    """Export model to ONNX from checkpoint path.

    Parameters
    ----------
    checkpoint_path : str
        Path to the trained model checkpoint.
    onnx_dir : str
        Directory to save ONNX models.
    timestamp : str | None
        Timestamp for model naming.
    name_suffix : str | None
        Additional suffix for model name.
    in_model_scaling : bool
        If True, embed scaling in model (outputs probabilities).
        If False, export raw model (outputs logits, requires external scaling).

    Returns
    -------
    ExportResult
        Result containing success status and save path.

    Notes
    -----
    Model type (flat/jagged) is auto-detected from the checkpoint.
    """
    logging.info("Exporting model to ONNX format!")
    logging.info(f"  Checkpoint: {checkpoint_path}")

    # Auto-detect model type from checkpoint (also loads reports)
    jagged, reports = _detect_model_type(checkpoint_path)

    # Generate model save name
    model_save_name = _generate_model_name(checkpoint_path, timestamp, name_suffix, jagged)

    # Load model and hyperparameters
    torch_model, hyper_params = _load_model_from_checkpoint(checkpoint_path, jagged=jagged)

    # Prepare model for export (deep copy + disable incompatible features)
    export_model = prepare_model_for_export(torch_model)

    logging.info("Loading training reports...")

    # Determine number of classes
    class_labels = reports.get("class_labels", {})
    num_classes = len(class_labels) if class_labels else 2
    logging.info(f"  Number of classes: {num_classes}")

    if jagged:
        return _export_jagged_classifier(
            export_model,
            reports,
            num_classes,
            model_save_name,
            onnx_dir,
            checkpoint_path,
            in_model_scaling,
        )
    else:
        return _export_flat_classifier(
            export_model,
            reports,
            num_classes,
            model_save_name,
            onnx_dir,
            checkpoint_path,
            in_model_scaling,
        )


def _export_flat_classifier(
    torch_model: torch.nn.Module,
    reports: dict[str, Any],
    num_classes: int,
    model_save_name: str,
    onnx_dir: str,
    checkpoint_path: str,
    in_model_scaling: bool,
) -> ExportResult:
    """Export flat (events-only) classifier to ONNX.

    Parameters
    ----------
    torch_model : torch.nn.Module
        Model prepared for export.
    reports : dict[str, Any]
        Training reports with selection and scaler info.
    num_classes : int
        Number of output classes.
    model_save_name : str
        Name for the saved model file.
    onnx_dir : str
        Output directory.
    checkpoint_path : str
        Path to checkpoint (for metadata).
    in_model_scaling : bool
        Whether to embed scaling in the ONNX model.

    Returns
    -------
    ExportResult
        Result with success status and path.
    """
    logging.info(f"Exporting flat classifier (in_model_scaling={in_model_scaling})...")

    # Log PLE usage for flat models
    preprocessor = getattr(torch_model, "preprocessor", None)
    if preprocessor is not None:
        use_ple = getattr(preprocessor, "use_ple", False)
        if use_ple:
            logging.info("  PLE ENABLED (piecewise linear encoding for numerical features)")
        else:
            logging.info("  PLE DISABLED (standard numerical embeddings)")

    # Get column names from selection to determine input size
    selection = reports.get("selection", {})
    column_names = selection.get("events", {}).get("offset_used_columns", [])
    example_input = torch.randn(1, len(column_names))

    save_path = os.path.join(onnx_dir, f"{model_save_name}.onnx")

    if in_model_scaling:
        logging.info("Exporting model with embedded scaling...")
        logging.info("  Extracting scaler configurations...")

        numer_config = extract_numer_scaler_config(reports, "events")
        categ_config = extract_categ_scaler_config(reports, "events")

        logging.info(f"  Numerical scaler: {numer_config.scaler_type or 'none'}")
        logging.info(f"  Categorical features: {len(categ_config.idx)}")

        wrapped_model = FlatInferenceWrapper(
            base_model=torch_model,
            numer_config=numer_config,
            categ_config=categ_config,
            num_classes=num_classes,
        )
        wrapped_model.eval()

        dynamic_axes = {"X": {0: "batch_size"}, "probabilities": {0: "batch_size"}}
        result = export_onnx_model(
            wrapped_model,
            example_input,
            save_path,
            input_names=["X"],
            output_names=["probabilities"],
            dynamic_axes=dynamic_axes,
        )
    else:
        logging.info("Exporting raw model (without scaling)...")

        with torch.no_grad():
            test_output = torch_model(example_input)

        if isinstance(test_output, torch.Tensor):
            output_names = ["logits"]
            dynamic_axes = {"X": {0: "batch_size"}, "logits": {0: "batch_size"}}
        else:
            output_names = [f"logits_{i}" for i in range(len(test_output))]
            dynamic_axes = {"X": {0: "batch_size"}}
            for name in output_names:
                dynamic_axes[name] = {0: "batch_size"}

        result = export_onnx_model(
            torch_model,
            example_input,
            save_path,
            input_names=["X"],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

    _save_metadata_if_success(result, checkpoint_path, model_save_name, onnx_dir, in_model_scaling, None)

    logging.info("Flat classifier export complete!")
    logging.info("To check the graph upload the onnx file to https://netron.app/.")

    return result


def _export_jagged_classifier(
    torch_model: torch.nn.Module,
    reports: dict[str, Any],
    num_classes: int,
    model_save_name: str,
    onnx_dir: str,
    checkpoint_path: str,
    in_model_scaling: bool,
) -> ExportResult:
    """Export jagged (multi-collection) classifier to ONNX.

    Parameters
    ----------
    torch_model : torch.nn.Module
        Model prepared for export.
    reports : dict[str, Any]
        Training reports with selection and scaler info.
    num_classes : int
        Number of output classes.
    model_save_name : str
        Name for the saved model file.
    onnx_dir : str
        Output directory.
    checkpoint_path : str
        Path to checkpoint (for metadata).
    in_model_scaling : bool
        Whether to embed scaling in the ONNX model.

    Returns
    -------
    ExportResult
        Result with success status and path.
    """
    logging.info(f"Exporting jagged classifier (in_model_scaling={in_model_scaling})...")

    selection = reports.get("selection", {})
    collection_names = [k for k in selection.keys() if k != "events"]
    logging.info(f"  Collections: {collection_names}")

    # Log PLE usage
    jagged_preprocessor = getattr(torch_model, "jagged_preprocessor", None)
    if jagged_preprocessor is not None:
        use_ple = getattr(jagged_preprocessor, "use_ple", False)
        if use_ple:
            logging.info("  PLE ENABLED (piecewise linear encoding for numerical features)")
        else:
            logging.info("  PLE DISABLED (standard numerical embeddings)")

    # Log pairwise attention
    particle_attention = getattr(torch_model, "particle_attention", None)
    if particle_attention is not None:
        logging.info("  PAIRWISE ATTENTION ENABLED")
    else:
        logging.info("  PAIRWISE ATTENTION DISABLED")

    # Get object slices and valid type values
    object_slices = getattr(torch_model, "object_slices", {})
    valid_type_values = extract_valid_type_values(torch_model)

    if in_model_scaling:
        logging.info("Exporting model with embedded scaling...")

        # Build collection configs
        collection_configs = {coll_name: build_collection_config(reports, coll_name) for coll_name in collection_names}

        # Build example inputs with known raw categorical values.
        categorical_configs = {name: cfg.categorical for name, cfg in collection_configs.items()}
        input_names, example_inputs, dynamic_axes = build_jagged_example_inputs(
            collection_names,
            selection,
            object_slices,
            valid_type_values,
            categorical_configs=categorical_configs,
            raw_model=False,
        )

        wrapped_model = JaggedInferenceWrapper(
            base_model=torch_model,
            collection_configs=collection_configs,
            num_classes=num_classes,
        )
        wrapped_model.eval()

        save_path = os.path.join(onnx_dir, f"{model_save_name}.onnx")
        dynamic_axes["probabilities"] = {0: "batch_size"}
        result = export_onnx_model(
            wrapped_model,
            tuple(example_inputs),
            save_path,
            input_names=input_names,
            output_names=["probabilities"],
            dynamic_axes=dynamic_axes,
        )
    else:
        logging.info("Exporting raw model (without scaling)...")

        # Build example inputs that keep categorical slots as valid encoded indices.
        input_names, example_inputs, dynamic_axes = build_jagged_example_inputs(
            collection_names,
            selection,
            object_slices,
            valid_type_values,
            categorical_configs=None,
            raw_model=True,
        )

        raw_wrapper = RawJaggedWrapper(base_model=torch_model, collection_names=collection_names)
        raw_wrapper.eval()

        save_path = os.path.join(onnx_dir, f"{model_save_name}.onnx")
        dynamic_axes["logits"] = {0: "batch_size"}
        result = export_onnx_model(
            raw_wrapper,
            tuple(example_inputs),
            save_path,
            input_names=input_names,
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
        )

    _save_metadata_if_success(result, checkpoint_path, model_save_name, onnx_dir, in_model_scaling, collection_names)

    logging.info("Jagged classifier export complete!")

    return result


@hydra.main(
    config_path=os.path.join(os.environ["ANALYSIS_ML_CONFIG_DIR"], "signal"),
    config_name="onnx_export_config",
    version_base=None,
)
def main(config: DictConfig) -> None:
    """Hydra entry point for ONNX export.

    Parameters
    ----------
    config : DictConfig
        Hydra configuration with onnx_config section containing:
        - checkpoint_path: Path to model checkpoint (required)
        - output_dir: ONNX output directory (optional)
        - timestamp: Model name timestamp (optional)
        - name_suffix: Model name suffix (optional)
        - in_model_scaling: Whether to embed scaling (default: True)

    Raises
    ------
    ValueError
        If checkpoint_path is not provided.
    FileNotFoundError
        If checkpoint file does not exist.
    RuntimeError
        If ONNX export fails.
    """
    setup_logger(config.min_logging_level)

    onnx_cfg = config.onnx_config

    # Validate checkpoint path
    checkpoint_path = onnx_cfg.checkpoint_path
    if checkpoint_path is None:
        raise ValueError("onnx_config.checkpoint_path is required!")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Set output directory
    onnx_dir = onnx_cfg.output_dir
    if onnx_dir is None:
        onnx_dir = os.path.join(os.environ["ANALYSIS_ML_MODELS_DIR"], "onnx_export")
    os.makedirs(onnx_dir, exist_ok=True)

    result = convert_from_checkpoint(
        checkpoint_path=checkpoint_path,
        onnx_dir=onnx_dir,
        timestamp=onnx_cfg.get("timestamp", None),
        name_suffix=onnx_cfg.get("name_suffix", None),
        in_model_scaling=onnx_cfg.get("in_model_scaling", True),
    )

    if not result.success:
        raise RuntimeError(f"ONNX export failed: {result.error_message}")


if __name__ == "__main__":
    main()
