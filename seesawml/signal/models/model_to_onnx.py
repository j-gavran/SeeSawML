import logging
import os
import time

import hydra
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from seesawml.signal.training.sig_bkg_trainer import load_sig_bkg_model
from seesawml.utils.helpers import setup_analysis_dirs
from seesawml.utils.loggers import setup_logger
from seesawml.utils.onnx_utils import get_metadata, test_onnx_export


def _to_tensor_list(x: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
    if isinstance(x, torch.Tensor):
        return [x]
    if isinstance(x, (list, tuple)):
        return list(x)
    raise TypeError(f"Unsupported output type: {type(x)}")


def convert_sig_bkg_classifier(
    config: DictConfig,
    onnx_dir: str,
    timestamp: str | None = None,
    name_suffix: str | None = None,
    extra_metadata: dict | None = None,
) -> None:
    logging.info("Exporting model to ONNX format!")

    model_config = config.model_config

    # get save name
    checkpoint = os.path.basename(model_config.load_checkpoint)
    checkpoint_split = checkpoint.split("_")
    model_save_name = f"{checkpoint_split[0]}_{checkpoint_split[1]}_"

    if timestamp is None:
        timestamp = time.asctime(time.localtime())

    model_save_name += timestamp.replace("  ", "_").replace(" ", "_").replace(":", "_")

    if name_suffix is not None:
        model_save_name += f"_{name_suffix}"

    # get model
    # Force-disable torch.compile in the loaded module by overriding saved hyperparameters
    with open_dict(model_config):
        if "architecture_config" not in model_config:
            model_config["architecture_config"] = {}
        model_config["architecture_config"]["compile"] = False

    lightning_module, checkpoint_path = load_sig_bkg_model(config, events_only=True)
    torch_model = lightning_module.model.eval().cpu()
    # If the model is wrapped by torch.compile/dynamo, unwrap to the original nn.Module for export
    # Prefer attribute `_orig_mod` that is set by dynamo wrappers; otherwise, keep as-is.
    if hasattr(torch_model, "_orig_mod") and torch_model._orig_mod is not None:
        orig = torch_model._orig_mod
        logging.info("Detected torch.compile/dynamo wrapper; unwrapping _orig_mod for ONNX export.")
        torch_model = orig.eval().cpu()  # type: ignore[union-attr]
    else:
        logging.info("No _orig_mod found; exporting the model as-is.")

    # get metadata
    # prepare extra metadata and attach custom_groups from dataset_config if present
    prepared_extra_metadata = {} if extra_metadata is None else dict(extra_metadata)
    try:
        ds_conf = config.dataset_config
        groups_cfg = None
        if "custom_groups" in ds_conf:
            groups_cfg = OmegaConf.to_container(ds_conf.custom_groups, resolve=True)
        elif "scores" in ds_conf:
            groups_cfg = OmegaConf.to_container(ds_conf.scores, resolve=True)

        if groups_cfg:
            prepared_extra_metadata["custom_groups"] = {str(k): list(v) for k, v in dict(groups_cfg).items()}  # type: ignore[arg-type]
    except Exception:
        logging.warning("Failed to attach custom_groups from dataset_config.", exc_info=True)

    metadata = get_metadata(
        checkpoint_path=checkpoint_path,
        model_save_name=model_save_name,
        dataset_name="events",
        onnx_dir=onnx_dir,
        extra_metadata=prepared_extra_metadata,
    )

    save_path = os.path.join(onnx_dir, f"{model_save_name}.onnx")

    # get example input
    column_names = metadata["column_names"]

    example_input = torch.rand(1, len(column_names))

    # test torch model
    logging.info(f"Example input: {example_input}")
    with torch.no_grad():
        torch_output = torch_model(example_input)

    logging.info(f"Torch test output: {torch_output}")

    # Determine output names and dynamic axes based on model output structure
    if isinstance(torch_output, torch.Tensor):
        output_names = ["output"]
        dynamic_axes = {"X": {0: "batch_size"}, "output": {0: "batch_size"}}
    elif isinstance(torch_output, (tuple, list)):
        output_names = [f"output_{i}" for i in range(len(torch_output))]
        dynamic_axes = {"X": {0: "batch_size"}}
        for name in output_names:
            dynamic_axes[name] = {0: "batch_size"}
    else:
        raise TypeError(f"Unsupported model output type for ONNX export: {type(torch_output)}")

    # Use legacy exporter path (dynamo=False) to avoid interactions with compiled models/tracing.
    # dynamic_axes is supported in the legacy path for variable batch size.
    torch.onnx.export(
        torch_model,
        args=(example_input,),
        f=save_path,
        input_names=["X"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False,
        fallback=True,
    )

    logging.info(f"Model saved to {save_path}")

    # test_model
    onnx_output = test_onnx_export(example_input, save_path)
    logging.info(f"ONNX test output: {onnx_output}")

    torch_out_list = [t.detach().cpu() for t in _to_tensor_list(torch_output)]
    onnx_out_list = [t.detach().cpu() for t in _to_tensor_list(onnx_output)]

    success = len(torch_out_list) == len(onnx_out_list) and all(
        torch.allclose(t, o) for t, o in zip(torch_out_list, onnx_out_list)
    )

    if success:
        logging.info("[green]Model conversion successful!")
        logging.info("To check the graph upload the onnx file to https://netron.app/.")
    else:
        logging.critical("[red]Model conversion failed!")


@hydra.main(
    config_path=os.path.join(os.environ["ANALYSIS_ML_CONFIG_DIR"], "signal"),
    config_name="training_config",
    version_base=None,
)
def main(config: DictConfig) -> None:
    setup_logger(config.min_logging_level)
    setup_analysis_dirs(config, verbose=False)

    onnx_dir = os.path.join(os.environ["ANALYSIS_ML_MODELS_DIR"], "onnx_export")
    os.makedirs(onnx_dir, exist_ok=True)

    convert_sig_bkg_classifier(config, onnx_dir)


if __name__ == "__main__":
    main()
