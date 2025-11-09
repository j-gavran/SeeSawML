import logging
import os
import time

import hydra
import torch
from omegaconf import DictConfig, open_dict

from seesaw.fakes.training.fakes_trainer import load_ratio_model
from seesaw.fakes.training.run_pt_sliced_training import TRIGGER_PT
from seesaw.utils.helpers import setup_analysis_dirs
from seesaw.utils.loggers import setup_logger
from seesaw.utils.onnx_utils import get_metadata, test_onnx_export

TRIGGER_PT_REVERSE = {v: k for k, v in TRIGGER_PT.items()}


def convert_ratio_model(
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
    model_name = model_config.name

    if model_name == "ratioModel":
        lightning_module, checkpoint_path = load_ratio_model(config)
        torch_model = lightning_module.model.eval().cpu()  # type: ignore[union-attr]
    else:
        raise ValueError(f"Model {model_name} not supported for ONNX conversion.")

    # get metadata
    metadata = get_metadata(checkpoint_path, model_save_name, "events", onnx_dir, extra_metadata=extra_metadata)

    # check if pt slice is present
    if metadata.get("pt_slice", None) is not None:
        pt_low, pt_high = TRIGGER_PT_REVERSE[metadata["pt_slice"][0]], TRIGGER_PT_REVERSE[metadata["pt_slice"][1]]
        model_save_name += f"_pt_{pt_low}_{pt_high}"

    save_path = os.path.join(onnx_dir, f"{model_save_name}.onnx")

    # get example input
    column_names = metadata["column_names"]

    example_input = torch.rand(1, len(column_names))

    # test torch model
    logging.info(f"Example input: {example_input}")
    with torch.no_grad():
        torch_output = torch_model(example_input)

    logging.info(f"Torch test output: {torch_output}")

    # export model
    torch.onnx.export(
        torch_model,
        args=(example_input,),
        f=save_path,
        input_names=["X"],
        output_names=["output"],
        dynamic_axes={"X": {0: "batch_size"}, "output": {0: "batch_size"}},  # variable length axes
    )

    logging.info(f"Model saved to {save_path}")

    # test_model
    onnx_output = test_onnx_export(example_input, save_path)
    logging.info(f"ONNX test output: {onnx_output}")

    if torch.allclose(torch_output, onnx_output):  # type: ignore[arg-type]
        logging.info("[green]Model conversion successful!")
        logging.info("To check the graph upload the onnx file to https://netron.app/.")
    else:
        logging.critical("[red]Model conversion failed!")


def convert_pt_sliced_ratio_model(config: DictConfig, onnx_dir: str) -> None:
    model_config = config.model_config

    model_checkpoints = model_config.pt_sliced_model.checkpoints

    if model_config.pt_sliced_model.get("model_save_path", None):
        model_save_paths = [model_config.pt_sliced_model.model_save_path] * len(model_checkpoints)
    else:
        model_save_paths = model_config.pt_sliced_model.model_save_paths

    timestamp = time.asctime(time.localtime())

    for checkpoint, save_path in zip(model_checkpoints, model_save_paths):
        with open_dict(config):
            config.model_config.load_checkpoint = checkpoint
            config.model_config.training_config.model_save_path = save_path
            config.model_config.pt_sliced_model = None

        convert_ratio_model(config, onnx_dir, timestamp=timestamp)


@hydra.main(
    config_path=os.path.join(os.environ["ANALYSIS_ML_CONFIG_DIR"], "fakes"),
    config_name="training_config",
    version_base=None,
)
def main(config: DictConfig) -> None:
    setup_logger(config.min_logging_level)
    setup_analysis_dirs(config, verbose=False)

    onnx_dir = os.path.join(os.environ["ANALYSIS_ML_MODELS_DIR"], "onnx_export")
    os.makedirs(onnx_dir, exist_ok=True)

    if config.model_config.get("load_checkpoint", None) is not None:
        convert_ratio_model(config, onnx_dir)
    elif config.model_config.get("pt_sliced_model", None) is not None:
        convert_pt_sliced_ratio_model(config, onnx_dir)
    else:
        raise ValueError("No model configuration found for ONNX conversion!")


if __name__ == "__main__":
    main()
