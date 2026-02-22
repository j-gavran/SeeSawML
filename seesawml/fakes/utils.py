import numpy as np
import torch
from f9columnar.ml.hdf5_dataloader import MLHdf5Iterator, StackedDatasets, WeightedBatch, WeightedDatasetBatch

from seesawml.fakes.models.loss import DensityRatio
from seesawml.models.ensembles import torch_predict_from_ensemble_logits


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
    model: torch.nn.Module, X: torch.Tensor, density_ratio: DensityRatio
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the density ratio from a trained classifier.

    Runs the model in inference mode and converts the raw network output
    to a density ratio using the link function defined by `density_ratio`.

    Parameters
    ----------
    model : torch.nn.Module
        Trained classifier that maps input features to a scalar output.
    X : torch.Tensor
        Input feature tensor of shape `(N, D)`.
    density_ratio : DensityRatio
        Converts the raw model output `f` to the density ratio `r`
        via the appropriate link function for the training loss.

    Returns
    -------
    r : torch.Tensor
        Estimated density ratio of shape `(N,)`.
    f : torch.Tensor
        Raw model output of shape `(N,)`.
    """
    with torch.no_grad():
        f = model(X)
        r = density_ratio(f)

    return r, f


def get_num_den_weights(
    model: torch.nn.Module,
    X: torch.Tensor,
    density_ratio: DensityRatio,
    data_label: int = 1,
    mc_label: int = 0,
    is_data: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Derive numerator/denominator subtraction weights from the density ratio.

    Given the density ratio `r = p_num / p_den` estimated by a trained
    classifier, computes the weight `w`. The formula depends on which class
    (data or MC) the numerator corresponds to and whether the input
    `X` comes from data or MC.

    Parameters
    ----------
    model : torch.nn.Module
        Trained classifier that maps input features to a scalar output.
    X : torch.Tensor
        Input feature tensor of shape `(N, D)`.
    density_ratio : DensityRatio
        Converts the raw model output to the density ratio.
    data_label : int, optional
        Label assigned to data events during training, by default 1.
    mc_label : int, optional
        Label assigned to MC events during training, by default 0.
    is_data : bool, optional
        `True` if `X` consists of data events, `False` for MC,
        by default `True`.

    Returns
    -------
    w : torch.Tensor
        Subtraction weights of shape `(N,)`.
    r : torch.Tensor
        Estimated density ratio of shape `(N,)`.
    f : torch.Tensor
        Raw model output of shape `(N,)`.

    Raises
    ------
    ValueError
        If `(mc_label, data_label)` is not one of the two expected
        combinations `(0, 1)` or `(1, 0)`.
    """
    r, f = nn_reweight(model, X, density_ratio)

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

    return w, r, f


def nn_reweight_with_errors(
    model: torch.nn.Module, X: torch.Tensor, density_ratio: DensityRatio
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the density ratio and its uncertainty from an ensemble model.

    Runs the ensemble model, averages the per-member logits (log-ratios) via
    :func:`torch_predict_from_ensemble_logits`, then propagates the epistemic
    uncertainty through the density-ratio link function via
    `DensityRatio.ratio_with_errors`.

    Parameters
    ----------
    model : torch.nn.Module
        Trained ensemble classifier (e.g. wrapping `StackedEnsembleNetWrapper`
        with `use_log_var=False`) whose forward pass returns a
        `(channels, batch, 1)` tensor of per-member logit predictions.
    X : torch.Tensor
        Input feature tensor of shape `(N, D)`.
    density_ratio : DensityRatio
        Converts the raw model output and its uncertainty to the density
        ratio and its propagated uncertainty.

    Returns
    -------
    r_mean : torch.Tensor
        Mean of the estimated density ratio, shape `(N,)`.
    r_std : torch.Tensor
        Standard deviation of the estimated density ratio, shape `(N,)`.
    f_mean : torch.Tensor
        Mean of the raw model output, shape `(N,)`.
    f_std : torch.Tensor
        Total standard deviation of the raw model output, shape `(N,)`.
    """
    with torch.no_grad():
        model_output = model(X)  # (channels, batch, output_dim)

        if density_ratio.name == "bce":
            f_mean, f_std = torch_predict_from_ensemble_logits(model_output)
        else:
            raise ValueError(f"Unsupported loss for ensemble reweighting: {density_ratio.name}")

        r_mean, r_std = density_ratio.ratio_with_errors(f_mean, f_std)

    return r_mean, r_std, f_mean, f_std


def sample_subtraction_weights(
    model: torch.nn.Module,
    X: torch.Tensor,
    channels: int,
    mc_label: int = 0,
    data_label: int = 1,
    is_data: bool = True,
    training: bool = True,
) -> torch.Tensor:
    """Sample per-channel subtraction weights for ensemble ratio training.

    For each of the `channels` ratio-ensemble members, independently draws
    one complete subtraction-ensemble member and uses its logit output for
    all events:

        k  ~  Uniform{0, ..., channels_sub - 1}
        f_k(x) = logits_{k}(x)

    Sampling a complete member (rather than per-event Gaussian noise)
    preserves the spatial correlation of the subtraction model uncertainty:
    all events seen by ratio member k share the same subtraction network,
    so the uncertainty on integrated quantities (yields, bin counts) is
    correctly propagated.  During validation (`training=False`) every
    channel receives the mean logit across subtraction members.

    Parameters
    ----------
    model : torch.nn.Module
        Trained subtraction ensemble (`StackedEnsembleNetWrapper`) whose
        forward pass returns a `(channels_sub, batch, 1)` logit tensor.
    X : torch.Tensor
        Input feature tensor of shape `(N, D)`.
    channels : int
        Number of ratio-ensemble members (output first dimension).
    mc_label : int, optional
        Label assigned to MC events during subtraction training, by default 0.
    data_label : int, optional
        Label assigned to data events during subtraction training, by default 1.
    is_data : bool, optional
        `True` if `X` consists of data events, by default `True`.
    training : bool, optional
        Whether to sample random members (`True`) or use the member mean
        (`False`), by default `True`.

    Returns
    -------
    torch.Tensor
        Per-channel subtraction weights of shape `(channels, N)`.
    """
    with torch.no_grad():
        logits = model(X).squeeze(-1)  # (channels_sub, N)

    channels_sub = logits.shape[0]

    if training:
        member_idx = torch.randint(0, channels_sub, (channels,), device=X.device)
        f_k = logits[member_idx]  # (channels, N) — spatially correlated
    else:
        f_k = torch.mean(logits, dim=0).unsqueeze(0).expand(channels, -1)  # (channels, N)

    if mc_label == 0 and data_label == 1:
        if is_data:
            return 1.0 - torch.exp(-f_k)  # w = 1 - 1/r = 1 - exp(-f)
        else:
            return torch.exp(f_k) - 1.0  # w = r - 1 = exp(f) - 1
    elif mc_label == 1 and data_label == 0:
        if is_data:
            return 1.0 - torch.exp(f_k)  # w = 1 - r = 1 - exp(f)
        else:
            return torch.exp(-f_k) - 1.0  # w = 1/r - 1 = exp(-f) - 1
    else:
        raise ValueError(f"Invalid labels: mc_label={mc_label}, data_label={data_label}")


def get_num_den_weights_with_errors(
    model: torch.nn.Module,
    X: torch.Tensor,
    density_ratio: DensityRatio,
    data_label: int = 1,
    mc_label: int = 0,
    is_data: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Derive subtraction weights with propagated uncertainties.

    Equivalent to :func:`get_num_den_weights`, but uses an ensemble
    model that exposes predictive uncertainties.  The density-ratio
    uncertainty is propagated through the weight formula analytically
    using first-order error propagation, and the result is returned
    as a pair of tensors `(w_mean, w_std)`.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model returning `(f_mean, f_std)` — the mean
        and standard deviation of the network output.
    X : torch.Tensor
        Input feature tensor of shape `(N, D)`.
    density_ratio : DensityRatio
        Converts the raw model output and its uncertainty to the density
        ratio and its propagated uncertainty.
    data_label : int, optional
        Label assigned to data events during training, by default 1.
    mc_label : int, optional
        Label assigned to MC events during training, by default 0.
    is_data : bool, optional
        `True` if `X` consists of data events, `False` for MC,
        by default `True`.

    Returns
    -------
    w_mean : torch.Tensor
        Mean of the subtraction weights, shape `(N,)`.
    w_std : torch.Tensor
        Standard deviation of the subtraction weights, shape `(N,)`.
    r_mean : torch.Tensor
        Mean of the estimated density ratio, shape `(N,)`.
    r_std : torch.Tensor
        Standard deviation of the estimated density ratio, shape `(N,)`.
    f_mean : torch.Tensor
        Mean of the raw model output, shape `(N,)`.
    f_std : torch.Tensor
        Standard deviation of the raw model output, shape `(N,)`.

    Raises
    ------
    ValueError
        If `(mc_label, data_label)` is not one of the two expected
        combinations `(0, 1)` or `(1, 0)`.
    """
    r_mean, r_std, f_mean, f_std = nn_reweight_with_errors(model, X, density_ratio)

    # first-order error propagation: w_std = |dw/dr| * r_std
    if mc_label == 1 and data_label == 0:
        if is_data:  # w = 1 - r -> |dw/dr| = 1
            w_mean = 1.0 - r_mean
            w_std = r_std
        else:  # w = 1/r - 1 -> |dw/dr| = 1/r^2
            w_mean = 1.0 / r_mean - 1.0
            w_std = r_std / r_mean**2

    elif mc_label == 0 and data_label == 1:
        if is_data:  # w = 1 - 1/r -> |dw/dr| = 1/r^2
            w_mean = 1.0 - 1.0 / r_mean
            w_std = r_std / r_mean**2
        else:  # w = r - 1 -> |dw/dr| = 1
            w_mean = r_mean - 1.0
            w_std = r_std

    else:
        raise ValueError(f"Invalid labels: mc_label={mc_label}, data_label={data_label}")

    return w_mean, w_std, r_mean, r_std, f_mean, f_std
