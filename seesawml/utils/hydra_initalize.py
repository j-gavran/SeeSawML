from typing import Any

from hydra import compose, version
from hydra import initialize as _initialize
from hydra._internal.hydra import Hydra
from hydra._internal.utils import detect_calling_file_or_module_from_stack_frame, detect_task_name
from hydra.initialize import get_gh_backup, restore_gh_from_backup
from omegaconf import DictConfig


class absolute_initialize:
    def __init__(
        self,
        config_path: str,
        job_name: str | None = None,
        caller_stack_depth: int = 1,
        version_base: str | None = None,
    ) -> None:
        self._gh_backup = get_gh_backup()

        version.setbase(version_base)

        calling_file, calling_module = detect_calling_file_or_module_from_stack_frame(caller_stack_depth + 1)
        if job_name is None:
            job_name = detect_task_name(calling_file=calling_file, calling_module=calling_module)

        Hydra.create_main_hydra_file_or_module(
            calling_file=calling_file,
            calling_module=calling_module,
            config_path=config_path,
            job_name=job_name,
        )

    def __enter__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        restore_gh_from_backup(self._gh_backup)


class relative_initialize(_initialize):
    def __init__(
        self,
        config_path: str,
        job_name: str | None = None,
        caller_stack_depth: int = 1,
        version_base: str | None = None,
    ) -> None:
        super().__init__(config_path, job_name, caller_stack_depth, version_base)


def get_hydra_config(
    config_path,
    config_name="training_config",
    allow_absolute: bool = True,
    overrides: list[str] | None = None,
) -> DictConfig:
    if allow_absolute:
        with absolute_initialize(config_path, version_base=None):
            config = compose(config_name=config_name, overrides=overrides)
    else:
        with relative_initialize(config_path, version_base=None):
            config = compose(config_name=config_name, overrides=overrides)

    return config
