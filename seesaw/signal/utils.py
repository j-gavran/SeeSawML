import numpy as np
import torch
from f9columnar.ml.hdf5_dataloader import (
    MLHdf5Iterator,
    StackedDatasets,
    WeightedBatch,
    WeightedDatasetBatch,
    remap_labels_lookup,
)
from omegaconf import DictConfig, ListConfig
from sklearn.preprocessing import OneHotEncoder


def handle_events_signal_dataset(
    stacked_datasets: StackedDatasets, ml_iterator: MLHdf5Iterator
) -> WeightedDatasetBatch | None:
    weighted_datase_batch = WeightedDatasetBatch()

    events_ds = stacked_datasets["events"]

    X = events_ds.X
    y: np.ndarray = events_ds.get_extra("label_type")  # type: ignore[assignment]
    w: np.ndarray = events_ds.get_extra("weights")  # type: ignore[assignment]

    imbalanced_sampler = ml_iterator.imbalanced_sampler
    remap_labels: dict[int, int] | None = ml_iterator.dataset_kwargs.get("remap_labels", None)

    y_classes: np.ndarray | None = None

    if remap_labels is not None:
        max_label = ml_iterator.dataset_kwargs["max_label"]
        y, mask_unmapped = remap_labels_lookup(y, max_label, remap_labels)

        if y.shape[0] == 0:
            return None

        X, w = X[mask_unmapped], w[mask_unmapped]
        y_classes = y.copy()

    X, y, w = X.astype(np.float32), y.astype(np.float32), w.astype(np.float32)

    if y_classes is not None:
        y_classes = y_classes.astype(np.float32)

    if imbalanced_sampler is not None:
        if len(np.unique(y)) >= 2:
            if y_classes is not None:
                X = np.concatenate([X, y_classes[:, None], w[:, None]], axis=1)
            else:
                X = np.concatenate([X, w[:, None]], axis=1)

            X, y = imbalanced_sampler.fit(X, y)

            if y_classes is not None:
                X, w, y_classes = X[:, :-2], X[:, -1], X[:, -2]
            else:
                X, w = X[:, :-1], X[:, -1]
        else:
            return None

    if remap_labels is not None:
        label_values = np.unique(list(remap_labels.values()))

        if len(label_values) > 2:
            enc = OneHotEncoder(handle_unknown="ignore")
            enc.fit(label_values[:, None])
            y = enc.transform(y[:, None]).toarray()

    if ml_iterator.shuffle:
        shuffle_idx = np.random.permutation(len(y))
        X[:], y[:], w[:] = X[shuffle_idx], y[shuffle_idx], w[shuffle_idx]

        if y_classes is not None:
            y_classes[:] = y_classes[shuffle_idx]

    weighted_datase_batch["events"] = WeightedBatch(X, y, w, y_classes)

    return weighted_datase_batch


def handle_full_signal_dataset(stacked_datasets: StackedDatasets, ml_iterator: MLHdf5Iterator) -> WeightedDatasetBatch:
    if ml_iterator.imbalanced_sampler is not None:
        raise NotImplementedError("Imbalanced sampling is not implemented for full signal dataset!")

    weighted_dataset_batch_dct = {}

    X: np.ndarray = stacked_datasets["events"].X
    y: np.ndarray = stacked_datasets["events"].get_extra("label_type")  # type: ignore[assignment]
    w: np.ndarray = stacked_datasets["events"].get_extra("weights")  # type: ignore[assignment]

    remap_labels: dict[int, int] | None = ml_iterator.dataset_kwargs.get("remap_labels", None)
    mask_unmapped: np.ndarray | None = None
    y_classes: np.ndarray | None = None

    if remap_labels is not None:
        max_label = ml_iterator.dataset_kwargs["max_label"]
        y, mask_unmapped = remap_labels_lookup(y, max_label, remap_labels)
        X, w = X[mask_unmapped], w[mask_unmapped]
        y_classes = y.copy()

        label_values = np.unique(list(remap_labels.values()))

        if len(label_values) > 2:
            enc = OneHotEncoder(handle_unknown="ignore")
            enc.fit(label_values[:, None])
            y = enc.transform(y[:, None]).toarray()

    if ml_iterator.shuffle:
        shuffle_idx = np.random.permutation(len(y))

        X = X[shuffle_idx]
        y = y[shuffle_idx]
        w = w[shuffle_idx]

        if y_classes is not None:
            y_classes = y_classes[shuffle_idx]
    else:
        shuffle_idx = None

    y = y.astype(np.float32)

    weighted_dataset_batch_dct["events"] = WeightedBatch(X, y, w, y_classes)

    for ds_name, ds in stacked_datasets.items():
        if ds_name == "events":
            continue

        X_other = ds.X.astype(np.float32)

        if mask_unmapped is not None:
            if mask_unmapped.shape[0] != X_other.shape[0]:
                raise RuntimeError(
                    f"Length mismatch: {ds_name} has {X_other.shape[0]}, events mask has {mask_unmapped.shape[0]}"
                )
            X_other = X_other[mask_unmapped]

        if shuffle_idx is not None:
            if shuffle_idx.shape[0] != X_other.shape[0]:
                raise RuntimeError(
                    f"Shuffle mismatch: {ds_name} length {X_other.shape[0]} vs events {shuffle_idx.shape[0]}"
                )
            X_other = X_other[shuffle_idx]

        weighted_dataset_batch_dct[ds_name] = WeightedBatch(X_other, None, None, None)

    weighted_dataset_batch = WeightedDatasetBatch()

    for k in stacked_datasets.keys():
        weighted_dataset_batch[k] = weighted_dataset_batch_dct[k]

    return weighted_dataset_batch


def get_classifier_labels(
    config_classes: ListConfig, labels: dict[str, int]
) -> tuple[dict[str, int], dict[int, int], dict[str, int]]:
    """Get classifier labels from the configuration.

    Parameters
    ----------
    config_classes : ListConfig
        Configuration containing class labels.
    labels : dict[str, int]
        HDF5 metadata labels.

    Returns
    -------
    tuple[dict[str, int], dict[int, int], dict[str, int]]
        A tuple containing:
        - class_labels: A dictionary mapping new class names to their new indices.
        - remap_class_labels: A dictionary mapping original label indices to new indices.
        - remap_class_labels_names: A dictionary mapping original class names to their new indices.
    """
    class_labels, remap_class_labels, remap_class_labels_names = {}, {}, {}

    for i, config_key in enumerate(config_classes):
        if isinstance(config_key, str):
            class_labels[config_key] = i
            remap_class_labels[labels[config_key]] = i
            remap_class_labels_names[config_key] = i
        elif isinstance(config_key, DictConfig):
            if len(config_key.keys()) != 1:
                raise ValueError(f"Expected a single key in the class configuration: {config_key}.")

            k = str(list(config_key.keys())[0])
            class_labels[k] = i

            for v in list(config_key.values())[0]:
                remap_class_labels[labels[str(v)]] = i
                remap_class_labels_names[str(v)] = i
        else:
            raise ValueError(f"Unsupported class key type: {type(config_key)}.")

    return class_labels, remap_class_labels, remap_class_labels_names


def multiclass_discriminant(X: np.ndarray, epsilon: float = 1e-12, clip: bool = False) -> list[np.ndarray]:
    ds = []

    for i in range(X.shape[1]):
        numerator = X[:, i]
        denominator = np.sum(X, axis=1) - X[:, i] + epsilon
        ratio = numerator / denominator

        if clip:
            ratio = np.clip(ratio, epsilon, None)
        else:
            ratio = ratio + epsilon

        d = np.log(ratio)
        ds.append(d)

    return ds


def torch_multiclass_discriminant(X: torch.Tensor, epsilon: float = 1e-12, clip: bool = False) -> list[torch.Tensor]:
    ds = []

    for i in range(X.shape[1]):
        numerator = X[:, i]
        denominator = torch.sum(X, dim=1) - X[:, i] + epsilon
        ratio = numerator / denominator

        if clip:
            ratio = torch.clamp(ratio, min=epsilon)
        else:
            ratio = ratio[ratio != 0]

        d = torch.log(ratio)
        ds.append(d)

    return ds


def multiclass_group_discriminant(
    X: np.ndarray, index_groups: list[list[int]], epsilon: float = 1e-12, clip: bool = False
) -> list[np.ndarray]:
    ds: list[np.ndarray] = []

    total_sum = np.sum(X, axis=1)

    for group_indices in index_groups:
        numerator = np.sum(X[:, group_indices], axis=1)
        denominator = total_sum - numerator + epsilon
        ratio = numerator / denominator

        if clip:
            ratio = np.clip(ratio, epsilon, None)
        else:
            ratio = ratio + epsilon

        d = np.log(ratio)
        ds.append(d)

    return ds
