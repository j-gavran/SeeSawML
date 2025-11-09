import numpy as np
import torch
from f9columnar.ml.hdf5_dataloader import MLHdf5Iterator, StackedDatasets, WeightedBatch, WeightedDatasetBatch

from seesaw.fakes.models.loss import DensityRatio


def handle_fakes_dataset(stacked_datasets: StackedDatasets, ml_iterator: MLHdf5Iterator) -> WeightedDatasetBatch | None:
    weighted_datase_batch = WeightedDatasetBatch()

    events_ds = stacked_datasets["events"]
    X = events_ds.X
    w: np.ndarray = events_ds.get_extra("weights")  # type: ignore[assignment]
    y: np.ndarray = events_ds.get_extra("data_type")  # type: ignore[assignment]

    y_lt: np.ndarray

    particle_types = ml_iterator.selection["events"].extra_columns
    if "el_type" in particle_types:
        y_lt = events_ds.get_extra("el_type")  # type: ignore[assignment]
    elif "mu_type" in particle_types:
        y_lt = events_ds.get_extra("mu_type")  # type: ignore[assignment]
    elif "toy_type" in particle_types:
        y_lt = events_ds.get_extra("toy_type")  # type: ignore[assignment]
    else:
        raise RuntimeError("Events fakes dataset must contain 'el_type', 'mu_type' or 'toy_type'.")

    pt_cut = ml_iterator.dataset_kwargs.get("pt_cut", None)
    if pt_cut is not None:
        pt_min, pt_max = pt_cut[0], pt_cut[1]
        pt_idx = ml_iterator.dataset_kwargs["pt_idx"]

        pt = X[:, pt_idx]
        mask = (pt >= pt_min) & (pt <= pt_max)
        X, y, w, y_lt = X[mask], y[mask], w[mask], y_lt[mask]

    tight_label, loose_label = 0, 2
    new_tight_label, new_loose_label = 1, 0

    y_lt[y_lt == tight_label] = new_tight_label  # tight
    y_lt[y_lt == loose_label] = new_loose_label  # loose

    dataset_kwargs = ml_iterator.dataset_kwargs
    use_mc, use_data = dataset_kwargs["use_mc"], dataset_kwargs["use_data"]
    use_tight, use_loose = dataset_kwargs["use_tight"], dataset_kwargs["use_loose"]

    mc_label, data_label = 0, 1

    # handle data and mc combinations
    if use_data and use_mc:
        pass
    elif use_data and not use_mc:
        mask = y == data_label
        X, y, w, y_lt = X[mask], y[mask], w[mask], y_lt[mask]
    elif not use_data and use_mc:
        mask = y == mc_label
        X, y, w, y_lt = X[mask], y[mask], w[mask], y_lt[mask]
    else:
        raise RuntimeError("Need one or both of data or mc!")

    # handle loose and tight combinations
    if use_loose and use_tight:
        pass
    elif use_loose and not use_tight:
        mask = y_lt == new_loose_label
        X, y, w, y_lt = X[mask], y[mask], w[mask], y_lt[mask]
    elif not use_loose and use_tight:
        mask = y_lt == new_tight_label
        X, y, w, y_lt = X[mask], y[mask], w[mask], y_lt[mask]
    else:
        raise RuntimeError("Need one or both of loose or tight!")

    if ml_iterator.shuffle:
        shuffle_idx = np.random.permutation(len(y))
        X, y = X[shuffle_idx], y[shuffle_idx]
        w, y_lt = w[shuffle_idx], y_lt[shuffle_idx]

    X, y, w, y_lt = X.astype(np.float32), y.astype(np.float32), w.astype(np.float32), y_lt.astype(np.float32)

    weighted_datase_batch["events"] = WeightedBatch(X, y, w, y_lt)

    return weighted_datase_batch


def nn_reweight(
    model: torch.nn.Module,
    X: torch.Tensor,
    density_ratio: DensityRatio,
    return_f: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    with torch.no_grad():
        f = model(X)
        r = density_ratio(f)

    if return_f:
        return r, f
    else:
        return r


def get_num_den_weights(
    model: torch.nn.Module,
    X: torch.Tensor,
    density_ratio: DensityRatio,
    data_label: int = 1,
    mc_label: int = 0,
    is_data: bool = True,
    return_intermediate: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor:
    if return_intermediate:
        r, f = nn_reweight(model, X, density_ratio, return_f=True)
    else:
        r = nn_reweight(model, X, density_ratio, return_f=False)

    if mc_label == 1 and data_label == 0:
        if is_data:
            w = 1.0 - r
        else:
            w = 1.0 / r - 1.0

    elif mc_label == 0 and data_label == 1:
        if is_data:
            w = 1.0 - 1.0 / r
        else:
            w = r - 1.0

    else:
        raise ValueError(f"Invalid labels: mc_label={mc_label}, data_label={data_label}")

    if return_intermediate:
        return w, r, f
    else:
        return w
