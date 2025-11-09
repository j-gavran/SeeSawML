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

    def loss(self, y_model: torch.Tensor, y_target: torch.Tensor, w: torch.Tensor | None = None) -> torch.Tensor:
        if self.name == "bce":
            loss = torch.nn.BCEWithLogitsLoss(reduction="none")(y_model, y_target)

            if w is not None:
                if self.w_lambda is not None:
                    w_penalty = torch.mean(y_model**2)

                if self.ess_lambda is not None:
                    ess = (torch.sum(y_model) ** 2) / (torch.sum(y_model**2) + 1e-8)
                    ess_penalty = 1 / (ess + 1e-8)

                base_loss = torch.mean(w * loss)

                if self.w_lambda is not None and self.ess_lambda is not None:
                    return base_loss + self.w_lambda * w_penalty + self.ess_lambda * ess_penalty
                elif self.w_lambda is not None:
                    return base_loss + self.w_lambda * w_penalty
                elif self.ess_lambda is not None:
                    return base_loss + self.ess_lambda * ess_penalty
                else:
                    return base_loss
            else:
                return torch.mean(loss)

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

    def __call__(self, y_model: torch.Tensor) -> torch.Tensor:
        return self.ratio(y_model)
