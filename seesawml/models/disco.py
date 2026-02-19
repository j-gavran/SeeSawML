import torch


def distance_corr(
    var_1: torch.Tensor,
    var_2: torch.Tensor,
    power: float = 2.0,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Distance correlation (DisCo), a measure quantifying non-linear correlations.

    Usage: Add to your loss function. total_loss = BCE_loss + lambda * distance_corr

    See: https://arxiv.org/abs/2001.05310

    Parameters
    ----------
    var_1: torch.Tensor
        First variable to decorrelate (e.g., mass)
    var_2: torch.Tensor
        Second variable to decorrelate (e.g., classifier output)
    weights: torch.Tensor | None
        Per-example weights. If None, uniform weights are assumed.
    power: float
        Exponent used in calculating the distance correlation. Default is 2.0.
    """
    if var_1.ndim == 0 or var_2.ndim == 0:
        return torch.tensor(0.0, dtype=var_1.dtype, device=var_1.device)

    if weights is None:
        normedweight = torch.ones_like(var_1)
    else:
        normedweight = weights * len(weights) / weights.sum()

    xx = var_1.view(-1, 1).repeat(1, len(var_1)).view(len(var_1), len(var_1))
    yy = var_1.repeat(len(var_1), 1).view(len(var_1), len(var_1))
    amat = (xx - yy).abs()

    xx = var_2.view(-1, 1).repeat(1, len(var_2)).view(len(var_2), len(var_2))
    yy = var_2.repeat(len(var_2), 1).view(len(var_2), len(var_2))
    bmat = (xx - yy).abs()

    amatavg = torch.mean(amat * normedweight, dim=1)
    Amat = (
        amat
        - amatavg.repeat(len(var_1), 1).view(len(var_1), len(var_1))
        - amatavg.view(-1, 1).repeat(1, len(var_1)).view(len(var_1), len(var_1))
        + torch.mean(amatavg * normedweight)
    )

    bmatavg = torch.mean(bmat * normedweight, dim=1)
    Bmat = (
        bmat
        - bmatavg.repeat(len(var_2), 1).view(len(var_2), len(var_2))
        - bmatavg.view(-1, 1).repeat(1, len(var_2)).view(len(var_2), len(var_2))
        + torch.mean(bmatavg * normedweight)
    )

    ABavg = torch.mean(Amat * Bmat * normedweight, dim=1)
    AAavg = torch.mean(Amat * Amat * normedweight, dim=1)
    BBavg = torch.mean(Bmat * Bmat * normedweight, dim=1)

    if power == 1.0:
        dCorr = (torch.mean(ABavg * normedweight)) / torch.sqrt(
            (torch.mean(AAavg * normedweight) * torch.mean(BBavg * normedweight))
        )
    elif power == 2.0:
        dCorr = (torch.mean(ABavg * normedweight)) ** 2 / (
            torch.mean(AAavg * normedweight) * torch.mean(BBavg * normedweight)
        )
    else:
        dCorr = (
            (torch.mean(ABavg * normedweight))
            / torch.sqrt((torch.mean(AAavg * normedweight) * torch.mean(BBavg * normedweight)))
        ) ** power

    if torch.isnan(dCorr):
        dCorr = torch.tensor(0.0, dtype=var_1.dtype, device=var_1.device)

    return dCorr
