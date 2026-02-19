import torch

VALID_LOSSES = [
    "bce",
    "mse",
    "logistic",
    "lsif",
    "kliep",
    "square",
    "exp",
    "savage",
    "tangent",
    "bregman",
    "gaussian_nll",
]


class DensityRatioLoss:
    def __init__(
        self,
        name: str,
        class_weight: float | None = None,
        w_lambda: float | None = None,
        ess_lambda: float | None = None,
    ) -> None:
        """Collection of losses for density ratio estimation.

        Parameters
        ----------
        name : str
            Name of the loss function and its corresponding density ratio estimator.
        class_weight : float, optional
            Weight of the positive class, by default None.
        w_lambda : float, optional
            Weight for the weight decay penalty, by default None.
        ess_lambda : float, optional
            Weight for the effective sample size penalty, by default None.

        Note
        ----
        Setting "bce" is the same as using "logistic" but implemented with BCEWithLogitsLoss.

        References
        ----------
        [1] - Loss functions for classification: https://en.wikipedia.org/wiki/Loss_functions_for_classification
        [2] - Linking losses for density ratio and class-probability estimation: https://proceedings.mlr.press/v48/menon16.html
        [3] - Binary Losses for Density Ratio Estimation: https://arxiv.org/abs/2407.01371

        """
        if name not in VALID_LOSSES:
            raise ValueError(f"Loss function {name} not supported!")

        if (w_lambda is not None or ess_lambda is not None) and name != "bce":
            raise ValueError("w_lambda and ess_lambda are only supported for 'bce' loss.")

        self.name = name
        self.class_weight = class_weight

        self.w_lambda: float | None = None

        if w_lambda is not None:
            self.w_lambda = w_lambda

        self.ess_lambda: float | None = None

        if ess_lambda is not None:
            self.ess_lambda = ess_lambda

        if self.w_lambda is not None or self.ess_lambda is not None:
            if name not in {"bce", "gaussian_nll"}:
                raise ValueError("Weight decay and ESS penalties not supported for this loss function!")

    def _handle_penalty(self, w: torch.Tensor | None, y_model: torch.Tensor, loss: torch.Tensor) -> torch.Tensor:
        if w is None:
            if loss.dim() == 3:
                return torch.mean(loss.squeeze(-1))

            return torch.mean(loss)

        if y_model.dim() == 3 and (self.w_lambda is not None or self.ess_lambda is not None):
            y_model = torch.mean(y_model[..., 0], dim=0)  # mean over ensembles

        if self.w_lambda is not None:
            w_penalty = torch.mean(y_model**2)

        if self.ess_lambda is not None:
            r = torch.exp(y_model)
            ess = (torch.sum(r) ** 2) / (torch.sum(r**2) + 1e-8)
            ess_penalty = 1 / (ess + 1e-8)

        if loss.dim() == 3:
            base_loss = torch.mean(w * loss.squeeze(-1))
        else:
            base_loss = torch.mean(w * loss)

        if self.w_lambda is not None and self.ess_lambda is not None:
            return base_loss + self.w_lambda * w_penalty + self.ess_lambda * ess_penalty
        elif self.w_lambda is not None:
            return base_loss + self.w_lambda * w_penalty
        elif self.ess_lambda is not None:
            return base_loss + self.ess_lambda * ess_penalty
        else:
            return base_loss

    def loss(self, y_model: torch.Tensor, y_target: torch.Tensor, w: torch.Tensor | None = None) -> torch.Tensor:
        if self.name == "bce":
            if y_model.dim() == 3:
                y_target = y_target.unsqueeze(0).unsqueeze(-1).expand_as(y_model)

            loss = torch.nn.BCEWithLogitsLoss(reduction="none")(y_model, y_target)
            return self._handle_penalty(w, y_model, loss)

        if self.name == "lsif" or self.name == "kliep":
            y_model = torch.sigmoid(y_model)
            y_model = torch.clamp(y_model, 1e-7, 1 - 1e-7)

        if self.name == "mse":
            loss = torch.nn.MSELoss(reduction="none")(y_model, y_target)

            if w is not None:
                return torch.mean(w * loss)
            else:
                return torch.mean(loss)

        elif self.name == "lsif":
            loss1 = y_model
            loss0 = 0.5 * y_model**2

        elif self.name == "kliep":
            loss1 = torch.log(y_model)
            loss0 = y_model

        elif self.name == "logistic":
            loss1 = torch.log(1 + torch.exp(-y_model))
            loss0 = torch.log(1 + torch.exp(y_model))

        elif self.name == "square":
            loss1 = (1 - y_model) ** 2
            loss0 = (1 + y_model) ** 2

        elif self.name == "exp":
            loss1 = torch.exp(-y_model)
            loss0 = torch.exp(y_model)

        elif self.name == "savage":
            loss1 = 1 / (1 + torch.exp(y_model)) ** 2
            loss0 = 1 / (1 + torch.exp(-y_model)) ** 2

        elif self.name == "tangent":
            loss1 = (2 * torch.arctan(y_model) - 1) ** 2
            loss0 = (2 * torch.arctan(-y_model) - 1) ** 2

        elif self.name == "bregman":
            loss1 = -y_model
            loss0 = 0.5 * y_model * (torch.log(2 * y_model) - 1)

        else:
            raise ValueError(f"Loss function {self.name} not supported!")

        if self.class_weight is None:
            a, b = y_target, 1 - y_target
        else:
            a = y_target / self.class_weight
            b = (1 - y_target) / (1 - self.class_weight)

        loss = a * loss1 + b * loss0

        if w is not None:
            return torch.mean(w * loss)
        else:
            return torch.mean(loss)

    def __call__(self, y_model: torch.Tensor, y_target: torch.Tensor, w: torch.Tensor | None = None) -> torch.Tensor:
        return self.loss(y_model, y_target, w)


class DensityRatio:
    def __init__(self, name: str) -> None:
        if name not in VALID_LOSSES:
            raise ValueError(f"Loss function {name} not supported!")

        self.name = name

    @staticmethod
    def _ratio_link(psi_inv: torch.Tensor) -> torch.Tensor:
        return psi_inv / (1 - psi_inv)

    def ratio(self, y_model: torch.Tensor) -> torch.Tensor:
        if self.name == "lsif" or self.name == "kliep":
            y_model = torch.sigmoid(y_model)
            y_model = torch.clamp(y_model, 1e-7, 1 - 1e-7)

        if self.name == "logistic" or self.name == "bce":
            r = torch.exp(y_model)

        elif self.name == "mse":
            r = y_model

        elif self.name == "lsif" or self.name == "kliep":
            r = self._ratio_link(y_model)

        elif self.name == "square":
            link = 2 * y_model - 1
            r = self._ratio_link(link)

        elif self.name == "exp":
            link = 1 / (1 + torch.exp(-2 * y_model))
            r = self._ratio_link(link)

        elif self.name == "savage":
            link = torch.exp(y_model) / (1 + torch.exp(y_model))
            r = self._ratio_link(link)

        elif self.name == "tangent":
            link = torch.arctan(y_model) + 0.5
            r = self._ratio_link(link)

        elif self.name == "bregman":
            r = 0.5 * torch.log(2 * y_model)

        else:
            raise ValueError(f"Loss function {self.name} not supported!")

        return r

    def ratio_with_errors(self, y_model: torch.Tensor, y_std: torch.Tensor, use_probability: bool = False):
        """Compute the density ratio and propagate uncertainties via the delta method.

        Supports two input spaces controlled by `use_probability`:

        - **Log-ratio space** (default, `use_probability=False`):
          The input `y_model` is treated as the log-ratio log(r). The density
          ratio and its uncertainty are obtained via r = exp(y) with
          sigma_r = r * sigma_y. Use this with `predict_from_ensemble_logits`
          or `predict_from_ensemble` (gaussian_nll).

        - **Probability space** (`use_probability=True`):
          The input `y_model` is treated as a class probability p in (0, 1).
          The density ratio is r = p / (1 - p) with sigma_r = sigma_p / (1 - p)^2.
          Use after sigmoid output of the model.

        Parameters
        ----------
        y_model : torch.Tensor
            Model prediction: log-ratio (default) or probability (if use_probability=True).
        y_std : torch.Tensor
            Standard deviation of the model prediction in the same space as y_model.
        use_probability : bool, optional
            If True, interpret y_model as a probability and use the p/(1-p) link.
            If False (default), interpret y_model as a log-ratio and use exp(y).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (r_mean, r_std): density ratio estimate and its propagated uncertainty.
        """
        if self.name != "bce":
            raise ValueError("Ratio with errors is only supported for 'bce' loss!")

        if use_probability:
            mean = torch.clamp(y_model, 1e-7, 1 - 1e-7)
            r_mean = mean / (1 - mean)
            r_std = y_std / (1 - mean) ** 2

            return r_mean, r_std
        else:
            r_mean = torch.exp(y_model)
            r_std = r_mean * y_std

            return r_mean, r_std

    def __call__(self, y_model: torch.Tensor) -> torch.Tensor:
        return self.ratio(y_model)
