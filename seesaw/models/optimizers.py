import logging
from typing import Any, Iterator, Type

import torch
from pytorch_optimizer import get_supported_optimizers, load_optimizer
from torch.nn.parameter import Parameter


def get_optimizer(optimizer_name: str, prefer_torch: bool = True) -> Type[torch.optim.Optimizer]:
    try:
        torch_optimizer = getattr(torch.optim, optimizer_name)
    except AttributeError:
        torch_optimizer = None

    # see: https://github.com/kozistr/pytorch_optimizer
    if optimizer_name.lower() in get_supported_optimizers():
        lib_optimizer = load_optimizer(optimizer_name)
    else:
        lib_optimizer = None

    if prefer_torch and torch_optimizer is not None:
        return torch_optimizer
    elif lib_optimizer is not None:
        return lib_optimizer
    else:
        logging.info(f"Supported pytorch-optimizer optimizers: {get_supported_optimizers()}")
        raise ValueError(f"Optimizer {optimizer_name} not found in either PyTorch or library optimizers!")


def select_nn_optimizer(
    optimizer_name: str, parameters: Iterator[Parameter], optimizer_params: dict[str, Any] | None = None
) -> torch.optim.Optimizer:
    if optimizer_params is None:
        optimizer_params = {}

    optimizer = get_optimizer(optimizer_name)

    return optimizer(parameters, **optimizer_params)
