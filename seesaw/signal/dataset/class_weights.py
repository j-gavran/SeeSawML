import hashlib
import logging
import os

import hydra
from f9columnar.ml.dataloader_helpers import get_hdf5_metadata
from f9columnar.ml.hdf5_dataloader import get_ml_hdf5_dataloader
from f9columnar.utils.helpers import dump_pickle
from omegaconf import DictConfig

from seesaw.signal.utils import get_classifier_labels
from seesaw.utils.helpers import setup_analysis_dirs
from seesaw.utils.loggers import get_batch_progress, setup_logger


def calculate_weights(
    config: DictConfig,
    labels: dict[str, int],
    stage: str,
    dataset_kwargs: dict,
    dataloader_kwargs: dict,
    norm: bool = False,
) -> dict[int, float]:
    inverse_labels = {v: k for k, v in labels.items()}

    hdf5_dataloader, _, _ = get_ml_hdf5_dataloader(
        name="classWeights",
        files=config.dataset_config.files,
        column_names=config.dataset_config.features,
        stage_split_piles=config.dataset_config.stage_split_piles,
        stage=stage,
        dataset_kwargs=dataset_kwargs,
        dataloader_kwargs=dataloader_kwargs,
    )

    progress = get_batch_progress()
    progress.start()
    bar = progress.add_task("Processing batches", total=None)

    counts = {label: 0 for label in labels.values()}

    for batch in hdf5_dataloader:
        ds_batch, _ = batch

        y = ds_batch["events"].y
        for label in labels.values():
            counts[label] += (y == label).sum().item()

        progress.update(bar, advance=1)

    progress.stop()

    total = sum(counts.values())

    logging.info("Calculating class weights...")
    logging.info("Counts per class:")
    for label, count in counts.items():
        logging.info(f"  {inverse_labels[label]}: {count}, {count / total:.2%} of total events")

    num_classes = len(counts)
    logging.info(f"Total number of classes: {num_classes}.")

    if norm:
        inv_weights = {label: 1 / count if count > 0 else 0.0 for label, count in counts.items()}
        weight_sum = sum(inv_weights[label] * counts[label] for label in counts)
        class_weights = {label: (inv_weights[label] / weight_sum) for label in counts}

        total_weight = sum(class_weights[label] * counts[label] for label in counts)
        if not abs(total_weight - 1.0) < 1e-6:
            logging.warning(f"Normalized class weights do not sum to 1.0, got {total_weight:.6f} instead.")
    else:
        class_weights = {label: total / (count * num_classes) if count > 0 else 0.0 for label, count in counts.items()}

    logging.info("Class weights:")
    for label, weight in class_weights.items():
        if norm:
            logging.info(f"  {inverse_labels[label]}: {weight:.3e}")
        else:
            logging.info(f"  {inverse_labels[label]}: {weight:.3f}")

    return class_weights


@hydra.main(
    config_path=os.path.join(os.environ["ANALYSIS_ML_CONFIG_DIR"], "signal"),
    config_name="training_config",
    version_base=None,
)
def main(config: DictConfig) -> None:
    setup_logger(config.min_logging_level)
    setup_analysis_dirs(config, verbose=False)

    classes = config.dataset_config.get("classes", None)

    if classes is None:
        raise ValueError("No classes defined in the dataset configuration.")

    metadata = get_hdf5_metadata(config.dataset_config.files, resolve_path=True)
    start_labels = metadata.get("labels", None)

    if start_labels is None:
        raise ValueError("No labels found in the metadata of the provided files!")

    dataset_kwargs = {}
    dataset_kwargs["max_label"] = max(start_labels.values())

    labels, remap_labels, _ = get_classifier_labels(classes, start_labels)

    dataset_kwargs["remap_labels"] = remap_labels

    dataloader_kwargs = dict(config.dataset_config.dataloader_kwargs)

    if dataloader_kwargs.get("batch_size", None) is not None:
        logging.warning("Batch size is set, consider setting it to null.")

    norm = config.dataset_config.get("norm_class_weights", False)
    if norm:
        logging.info("Normalizing class weights to sum to 1.")

    stage_class_weights = {}
    for stage, stage_piles in config.dataset_config.stage_split_piles.items():
        if stage_piles == 0:
            logging.info(f"[yellow]Skipping {stage} dataset as no piles are assigned.[/yellow]")
            continue

        logging.info(f"[green]Starting dataloader for {stage} dataset...[/green]")
        class_weights = calculate_weights(
            config,
            labels,
            stage=stage,
            dataset_kwargs=dataset_kwargs,
            dataloader_kwargs=dataloader_kwargs,
            norm=norm,
        )
        stage_class_weights[stage] = class_weights

    class_weights_dir = os.path.join(os.environ["ANALYSIS_ML_RESULTS_DIR"], "class_weights")
    os.makedirs(class_weights_dir, exist_ok=True)

    hash_name = "".join(str(classes)) + str(config.dataset_config.files)
    class_weights_file_name = hashlib.md5(hash_name.encode()).hexdigest()

    class_weights_file_path = os.path.join(class_weights_dir, f"{class_weights_file_name}.p")
    logging.info(f"Saving class weights to {class_weights_file_path}.")
    dump_pickle(class_weights_file_path, stage_class_weights)


if __name__ == "__main__":
    main()
