import datetime
import logging
import logging.handlers
import os
import time
import warnings
from collections import deque
from datetime import timedelta
from pathlib import Path
from typing import Any, Deque

from lightning.pytorch.callbacks import RichProgressBar
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks.progress.rich_progress import (
    ProcessingSpeedColumn,
    RichProgressBarTheme,
)
from pytorch_lightning.trainer import Trainer
from rich.console import RenderableType
from rich.highlighter import ReprHighlighter
from rich.logging import RichHandler
from rich.progress import (
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TextColumn,
    TimeElapsedColumn,
)
from rich.style import Style
from rich.text import Text

FILE_NAME_FORMAT = "{year:04d}{month:02d}{day:02d}-" + "{hour:02d}{minute:02d}{second:02d}.log"
DATE_FORMAT = "%d %b %Y | %H:%M:%S"
LOGGER_FORMAT = "%(asctime)s | %(message)s"
LOGGING_LEVEL = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}
CUSTOM_LOGGERS = [
    logging.getLogger("lightning.pytorch"),
    logging.getLogger("lightning.fabric"),
    logging.getLogger("lightning.pytorch.core"),
]


class LoggerConfig:
    def __init__(self, handlers, log_format, date_format=None, level="info"):
        self.handlers = handlers
        self.log_format = log_format
        self.date_format = date_format
        self.level = LOGGING_LEVEL[level.lower()]


class CustomReprHighlighter(ReprHighlighter):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.highlights.pop(1)


def get_logger_config(
    min_level: str,
    override: bool,
    logger_path: str = os.path.join(os.environ.get("ANALYSIS_ML_LOGS_DIR", "."), "stdout"),
    file_name_format: str = FILE_NAME_FORMAT,
    date_format: str = DATE_FORMAT,
    logger_format: str = LOGGER_FORMAT,
    append_custom_loggers: list[logging.Logger] | None = None,
    **kwargs: Any,
) -> LoggerConfig:
    t = datetime.datetime.now()

    file_name = file_name_format.format(
        year=t.year,
        month=t.month,
        day=t.day,
        hour=t.hour,
        minute=t.minute,
        second=t.second,
    )

    Path(logger_path).mkdir(parents=True, exist_ok=True)

    file_name = os.path.join(logger_path, file_name)

    output_file_handler = logging.handlers.RotatingFileHandler(file_name, maxBytes=1024**2, backupCount=100)

    handler_format = logging.Formatter(logger_format, datefmt=date_format)
    output_file_handler.setFormatter(handler_format)

    rich_handler = RichHandler(
        rich_tracebacks=True,
        tracebacks_show_locals=True,
        show_time=False,
        markup=True,
        highlighter=CustomReprHighlighter(),
        **kwargs,
    )

    if override:
        custom_loggers = CUSTOM_LOGGERS + (append_custom_loggers or [])

        for log in custom_loggers:
            log.handlers.clear()
            log.addHandler(output_file_handler)
            log.addHandler(rich_handler)

    return LoggerConfig(
        handlers=[rich_handler, output_file_handler],
        log_format=logger_format,
        date_format=date_format,
        level=min_level,
    )


def reset_loggers() -> None:
    logger = logging.getLogger()

    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])


def filter_lightning_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module="lightning.pytorch.trainer.configuration_validator",
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module="lightning.pytorch.loops.utilities",
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module="lightning.pytorch.loops.optimization.automatic",
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module="lightning.pytorch.callbacks.model_checkpoint",
    )


def setup_logger(
    min_level: str = "info",
    override: bool = True,
    only_logger: bool = True,
    ignore_lightning_warnings: bool = True,
    **kwargs: Any,
) -> None:
    if only_logger:
        reset_loggers()

    if ignore_lightning_warnings:
        filter_lightning_warnings()

    logger_config = get_logger_config(min_level, override, **kwargs)

    logging.basicConfig(
        level=logger_config.level,
        format=logger_config.log_format,
        datefmt=logger_config.date_format,
        handlers=logger_config.handlers,
    )


def log_hydra_config(config: DictConfig) -> None:
    log_file = logging.getLogger().handlers[1].baseFilename  # type: ignore

    log_dir, log_name = os.path.dirname(log_file), os.path.basename(log_file).split(".")[0]

    hydra_dir = os.path.join("/".join(log_dir.split("/")[:-1]), "hydra")
    os.makedirs(hydra_dir, exist_ok=True)

    hydra_yaml = os.path.join(hydra_dir, f"{log_name}.yaml")

    with open(hydra_yaml, "w") as f:
        OmegaConf.save(config, f)


def get_batch_progress() -> Progress:
    progress_columns = (
        SpinnerColumn(spinner_name="aesthetic", speed=1.0, style="bold green"),
        TextColumn("[progress.description]{task.description} |"),
        BatchesProcessedColumn(style="white"),
        TextColumn("|"),
        TextColumn("Elapsed:"),
        TimeElapsedColumn(),
    )
    return Progress(*progress_columns, auto_refresh=True)


class BatchesProcessedColumn(ProgressColumn):
    def __init__(self, style: str | Style) -> None:
        self.style = style
        super().__init__()

    def render(self, task: Any) -> RenderableType:
        return Text(f"Batch: {int(task.completed)}", style=self.style)


class CustomTimeColumn(ProgressColumn):
    max_refresh = 0.5

    def __init__(self, style: str | Style) -> None:
        self.style = style
        super().__init__()
        self.current_epoch, self.max_epochs = 0, 0

    def render(self, task: Task) -> Text:
        elapsed = task.finished_time if task.finished else task.elapsed
        elapsed_delta = "-:--:--" if elapsed is None else str(timedelta(seconds=int(elapsed)))

        return Text(f"{elapsed_delta}", style=self.style)


class RemainingTimeColumn(ProgressColumn):
    max_refresh = 0.5

    def __init__(self, style: str | Style) -> None:
        self.style = style
        super().__init__()
        self.remaining_delta = 0.0
        self.current_epoch = 0

    def set_remaining_time(self, remaining_time: float, current_epoch: int) -> None:
        self.remaining_delta = remaining_time
        self.current_epoch = current_epoch

    def render(self, task: Task) -> Text:
        if task.finished:
            return Text("Done", style=self.style)

        if self.current_epoch == 0:
            return Text("Remaining: ?")

        return Text(f"Remaining: {datetime.timedelta(seconds=int(self.remaining_delta))}", style=self.style)


class TotalTimeColumn(ProgressColumn):
    def __init__(self, style: str | Style) -> None:
        self.style = style
        super().__init__()
        self.start_time = time.time()

    def render(self, task: Task) -> Text:
        if task.finished:
            return Text("Total: Done", style=self.style)

        total_delta = time.time() - self.start_time

        return Text(f"Total: {datetime.timedelta(seconds=int(total_delta))}", style=self.style)


class CustomRichProgressBar(RichProgressBar):
    def __init__(self, *args, **kwargs) -> None:
        theme = RichProgressBarTheme(
            description="white",
            progress_bar="white",
            progress_bar_finished="white",
            progress_bar_pulse="white",
            batch_progress="white",
            time="white",
            processing_speed="underline",
            metrics="white",
        )
        super().__init__(*args, theme=theme, **kwargs)
        self.current_epoch = 0
        self.max_epochs = 0
        self.epoch_durations: Deque[float] = deque(maxlen=10)
        self.epoch_start_time = 0.0

    def on_train_epoch_start(self, *args, **kwargs) -> None:
        super().on_train_epoch_start(*args, **kwargs)
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, *args, **kwargs) -> None:
        super().on_train_epoch_end(*args, **kwargs)
        duration = time.time() - self.epoch_start_time
        self.epoch_durations.append(duration)

    def estimate_remaining_time(self, current_epoch: int) -> float:
        if current_epoch < 0 or not self.epoch_durations:
            return 0.0

        avg_time = sum(self.epoch_durations) / len(self.epoch_durations)
        remaining_epochs = self.max_epochs - (current_epoch + 1)
        return max(0.0, remaining_epochs * avg_time)

    def configure_columns(self, trainer: Trainer) -> list:
        self.max_epochs = trainer.max_epochs
        return [
            TextColumn("[progress.description]{task.description}"),
            SpinnerColumn(style="green", spinner_name="aesthetic"),
            BatchesProcessedColumn(style=self.theme.batch_progress),
            TextColumn("|"),
            TextColumn("Batch elapsed:"),
            CustomTimeColumn(style=self.theme.time),
            TextColumn("|"),
            TotalTimeColumn(style=self.theme.time),
            TextColumn("|"),
            RemainingTimeColumn(style=self.theme.time),
            TextColumn("|"),
            TextColumn("Speed:"),
            ProcessingSpeedColumn(style=self.theme.processing_speed),
            TextColumn("|"),
        ]

    def get_metrics(self, *args, **kwargs) -> dict:
        current_epoch = self.trainer.current_epoch

        if self.current_epoch != current_epoch:
            self.current_epoch = current_epoch
            remaining_delta = self.estimate_remaining_time(self.current_epoch)

            time_column = self.progress.columns[9]
            time_column.set_remaining_time(remaining_delta, self.current_epoch)

        metrics = super().get_metrics(*args, **kwargs)
        metrics.pop("v_num", None)

        return metrics
