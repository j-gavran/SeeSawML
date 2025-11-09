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

    inverse_labels = {v: k for k, v in labels.items()}

    dataset_kwargs["remap_labels"] = remap_labels

    dataloader_kwargs = dict(config.dataset_config.dataloader_kwargs)

    if dataloader_kwargs.get("batch_size", None) is not None:
        logging.warning("Batch size is set, consider setting it to null.")

    norm = config.dataset_config.get("norm_class_weights", False)
    if norm:
        logging.info("Normalizing class weights to sum to 1.")

    logging.info("[green]Initializing dataloaders...[/green]")
    hdf5_dataloader, _, _ = get_ml_hdf5_dataloader(
        name="classWeights",
        files=config.dataset_config.files,
        column_names=config.dataset_config.features,
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

    inv_counts_sum = sum(1 / count for count in counts.values())

    class_weights = {}
    for label, count in counts.items():
        if count > 0:
            if norm:
                class_weights[label] = (1 / count) / inv_counts_sum
            else:
                class_weights[label] = total / (count * num_classes)
        else:
            logging.warning(f"Class {inverse_labels[label]} has no samples. Assigning weight of 0.")
            class_weights[label] = 0.0

    logging.info("Class weights:")
    for label, weight in class_weights.items():
        logging.info(f"  {inverse_labels[label]}: {weight:.4f}")

    class_weights_dir = os.path.join(os.environ["ANALYSIS_ML_RESULTS_DIR"], "class_weights")
    os.makedirs(class_weights_dir, exist_ok=True)

    hash_name = "".join(str(classes)) + str(config.dataset_config.files)
    class_weights_file_name = hashlib.md5(hash_name.encode()).hexdigest()

    class_weights_file_path = os.path.join(class_weights_dir, f"{class_weights_file_name}.p")
    logging.info(f"Saving class weights to {class_weights_file_path}.")
    dump_pickle(class_weights_file_path, class_weights)


if __name__ == "__main__":
    main()
