import numpy as np
import torch
from f9columnar.ml.dataloader_helpers import ColumnSelection
from f9columnar.ml.scalers import NumericalFeatureScaler
from omegaconf import DictConfig
from torch import nn


class CrackVetoModel(nn.Module):
    def __init__(self, eta_idx: int, eta_crack_start: float, eta_crack_end: float) -> None:
        """Glue together the eta regions around the crack.

        Parameters
        ----------
        eta_idx : int
            Index of the eta feature in the input tensor.
        eta_crack_start : float
            Value where the crack starts.
        eta_crack_end : float
            Value where the crack ends.

        Note
        ----
        - Module assumes a crack veto has been applied in the preprocessig step.
        - Module modifies the input tensor in place when gluing the eta values.
        - Eta values are expected to be already scaled if feature scaling is applied.

        """
        super().__init__()
        self.eta_idx: torch.Tensor
        self.eta_crack_end: torch.Tensor
        self.eta_crack_width: torch.Tensor

        self.register_buffer("eta_idx", torch.tensor([eta_idx], dtype=torch.int64))
        self.register_buffer("eta_crack_end", torch.tensor([eta_crack_end], dtype=torch.float32))
        self.register_buffer("eta_crack_width", torch.tensor([eta_crack_end - eta_crack_start], dtype=torch.float32))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        eta = X[:, self.eta_idx]
        abs_eta = torch.abs(eta)

        shift = (abs_eta > self.eta_crack_end).to(X.dtype) * self.eta_crack_width
        shift = torch.where(eta > 0, shift, -shift)

        eta_glued = eta - shift

        X_out = X.clone()
        X_out[:, self.eta_idx] = eta_glued

        return X_out


def get_numer_scaler(
    dataset_config: DictConfig, numer_column_names: list[str], extra_hash: str = ""
) -> NumericalFeatureScaler | None:
    scaler_type = dataset_config.feature_scaling.scaler_type
    scaler_path = dataset_config.feature_scaling.save_path

    numer_scaler = NumericalFeatureScaler(scaler_type, save_path=scaler_path).load(
        column_names=numer_column_names, postfix="events_0", extra_hash=extra_hash
    )

    if numer_scaler is None:
        return None

    return numer_scaler


def get_crack_veto_model(dataset_conf: DictConfig, selection: ColumnSelection) -> nn.Module:
    numer_columns = selection["events"].numer_columns

    etas_idx = []
    for i, c in enumerate(numer_columns):
        if "eta" in c:
            etas_idx.append(i)

    if len(etas_idx) != 1:
        raise RuntimeError("Crack veto can only be applied when there is exactly one eta feature!")

    eta_idx = etas_idx[0]

    numer_scaler = get_numer_scaler(dataset_conf, selection["events"].numer_columns, extra_hash=dataset_conf.files)

    eta_low, eta_high = 1.37, 1.52

    if numer_scaler is None:
        eta_low_scaled, eta_high_scaled = eta_low, eta_high
    else:
        X = np.ones((2, len(numer_columns)), dtype=np.float32)
        X[0, eta_idx], X[1, eta_idx] = eta_low, eta_high

        X_scaled = numer_scaler.transform(X)

        eta_low_scaled, eta_high_scaled = float(X_scaled[0, eta_idx]), float(X_scaled[1, eta_idx])

    model = CrackVetoModel(eta_idx, eta_low_scaled, eta_high_scaled)

    return model
