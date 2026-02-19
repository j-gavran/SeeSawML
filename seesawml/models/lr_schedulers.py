import logging
import math
from typing import Any, Type

import torch
from pytorch_optimizer import get_supported_lr_schedulers, load_lr_scheduler
from torch.optim.lr_scheduler import LRScheduler


class AttentionWarmupScheduler(LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        d_model: float,
        n_warmup_steps: int,
        lr_mul: float = 1.0,
        min_lr: float = 1e-6,
        freeze_step: int | None = None,
        last_epoch: int = -1,
    ) -> None:
        """Implementation of Eq. 3 from https://arxiv.org/abs/1706.03762.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer to apply the scheduler to.
        d_model : float
            The model dimension.
        n_warmup_steps : int
            The number of warmup steps.
        lr_mul : float, optional
            A multiplier for the learning rate, by default 1.0.
        min_lr : float, optional
            The minimum learning rate, by default 1e-6.
        freeze_step : int | None, optional
            The step at which to freeze the learning rate, by default None.
        last_epoch : int, optional
            The index of the last epoch, by default -1.
        """
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.min_lr = min_lr
        self.freeze_step = freeze_step

        self.n_steps = 0

        super().__init__(optimizer, last_epoch)
        self.init_lr()

    def init_lr(self) -> None:
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self) -> list[float]:
        lr = (
            self.lr_mul
            * (self.d_model**-0.5)
            * min(self.n_steps ** (-0.5), self.n_steps * self.n_warmup_steps ** (-1.5))
        )

        return [lr for _ in self.base_lrs]

    def step(self, epoch: int | None = None) -> None:
        self.n_steps += 1
        lrs = self.get_lr()

        if self.freeze_step is not None and self.n_steps >= self.freeze_step:
            return None

        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group["lr"] = lr


class SqrtExpWarmupScheduler(LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        a: float,
        c: float,
        lr_mul: float = 1.0,
        min_lr: float = 1e-6,
        freeze_step: int | None = None,
        last_epoch: int = -1,
    ) -> None:
        """Simple sqrt(x)exp(-x) scheduler.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer to apply the scheduler to.
        a : float
            The scaling factor for the learning rate.
        c : float
            The decay factor for the learning rate.
        lr_mul : float, optional
            A multiplier for the learning rate, by default 1.0.
        min_lr : float, optional
            The minimum learning rate, by default 1e-6.
        freeze_step : int | None, optional
            The step at which to freeze the learning rate, by default None.
        last_epoch : int, optional
            The index of the last epoch, by default -1.
        """
        self.lr_mul = lr_mul
        self.a = a
        self.c = c
        self.min_lr = min_lr
        self.freeze_step = freeze_step

        self.n_steps = 0

        super().__init__(optimizer, last_epoch)
        self.init_lr()

    def init_lr(self) -> None:
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self) -> list[float]:
        lr = self.a * math.sqrt(self.n_steps * self.c) * math.exp(-self.n_steps * self.c)
        return [lr for _ in self.base_lrs]

    def step(self, epoch: int | None = None) -> None:
        self.n_steps += 1
        lrs = self.get_lr()

        if self.freeze_step is not None and self.n_steps >= self.freeze_step:
            return None

        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group["lr"] = lr


class CosineWarmupScheduler(LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        n_warmup_steps: int,
        T_max: int,
        max_lr: float = 1e-3,
        min_lr: float = 1e-6,
        freeze_step: int | None = None,
        last_epoch: int = -1,
    ) -> None:
        """Cosine scheduler with linear warmup.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer to apply the scheduler to.
        n_warmup_steps : int
            The number of warmup steps.
        T_max : int
            The maximum number of iterations.
        max_lr : float, optional
            The maximum learning rate, by default 1e-3.
        min_lr : float, optional
            The minimum learning rate, by default 1e-6.
        freeze_step : int | None, optional
            The step at which to freeze the learning rate, by default None.
        last_epoch : int, optional
            The index of the last epoch, by default -1.
        """
        self.n_warmup_steps = n_warmup_steps
        self.T_max = T_max
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.freeze_step = freeze_step
        self.n_steps = 0

        super().__init__(optimizer, last_epoch)
        self._init_base_lrs()

    def _init_base_lrs(self) -> None:
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            self.base_lrs.append(param_group.get("lr", 1e-3))
            param_group["lr"] = self.min_lr

    def get_lr(self) -> list[float]:
        lrs = []
        for _ in self.base_lrs:
            if self.n_steps < self.n_warmup_steps:
                lr = self.min_lr + (self.max_lr - self.min_lr) * (self.n_steps / self.n_warmup_steps)
            else:
                t = (self.n_steps - self.n_warmup_steps) % self.T_max
                cosine_decay = 0.5 * (1 + math.cos(math.pi * t / self.T_max))
                lr = self.min_lr + (self.max_lr - self.min_lr) * cosine_decay
            lrs.append(lr)

        return lrs

    def step(self, epoch: int | None = None) -> None:
        if self.freeze_step is not None and self.n_steps >= self.freeze_step:
            return

        self.n_steps += 1
        lrs = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group["lr"] = lr


class LinearWarmupScheduler(LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        n_warmup_steps: int,
        min_lr: float = 1e-6,
        max_lr: float = 1e-3,
        last_epoch: int = -1,
    ) -> None:
        """Linear warmup scheduler from min_lr to max_lr and then constant max_lr.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer to apply the scheduler to.
        n_warmup_steps : int
            The number of warmup steps.
        min_lr : float, optional
            The minimum learning rate, by default 1e-6.
        max_lr : float, optional
            The maximum learning rate, by default 1e-3.
        last_epoch : int, optional
            The index of the last epoch, by default -1.
        """
        self.n_warmup_steps = n_warmup_steps
        self.min_lr = min_lr
        self.max_lr = max_lr

        self.n_steps = 0

        super().__init__(optimizer, last_epoch)
        self.init_lr()

    def init_lr(self) -> None:
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self) -> list[float]:
        if self.n_steps < self.n_warmup_steps:
            lr = self.min_lr + (self.max_lr - self.min_lr) * (self.n_steps / self.n_warmup_steps)
        else:
            lr = self.max_lr

        return [lr for _ in self.base_lrs]

    def step(self, epoch: int | None = None) -> None:
        self.n_steps += 1
        lrs = self.get_lr()

        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group["lr"] = lr


def get_lr_scheduler(scheduler_name: str, prefer_torch: bool = True) -> Type[LRScheduler]:
    if scheduler_name == "AttentionWarmup":
        return AttentionWarmupScheduler

    if scheduler_name == "SqrtExpWarmup":
        return SqrtExpWarmupScheduler

    if scheduler_name == "CosineWarmup":
        return CosineWarmupScheduler

    if scheduler_name == "LinearWarmup":
        return LinearWarmupScheduler

    try:
        torch_scheduler = getattr(torch.optim.lr_scheduler, scheduler_name)
    except AttributeError:
        torch_scheduler = None

    # see: https://github.com/kozistr/pytorch_optimizer
    if scheduler_name.lower() in get_supported_lr_schedulers():
        lib_scheduler = load_lr_scheduler(scheduler_name)
    else:
        lib_scheduler = None

    if prefer_torch and torch_scheduler is not None:
        return torch_scheduler
    elif lib_scheduler is not None:
        return lib_scheduler
    else:
        logging.info(f"Supported pytorch-optimizer lr_schedulers: {get_supported_lr_schedulers()}")
        raise ValueError(f"Scheduler {scheduler_name} not found in either PyTorch or pytorch_optimizer!")


def select_nn_lr_scheduler(
    scheduler_name: str, optimizer: torch.optim.Optimizer, scheduler_params: dict[str, Any] | None = None
) -> LRScheduler:
    if scheduler_params is None:
        scheduler_params = {}

    scheduler = get_lr_scheduler(scheduler_name)

    return scheduler(optimizer, **scheduler_params)
