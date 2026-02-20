import os
import platform
from pathlib import Path

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

console = Console()

EXPORTED = {
    "ANALYSIS_ML_CODE_DIR": "Analysis code",
    "ANALYSIS_ML_OUTPUT_DIR": "Output",
    "ANALYSIS_ML_CONFIG_DIR": "Configuration",
    "ANALYSIS_ML_DATA_DIR": "Data",
    "ANALYSIS_ML_RESULTS_DIR": "Results",
    "ANALYSIS_ML_MODELS_DIR": "Saved models",
    "ANALYSIS_ML_LOGS_DIR": "Logs",
    "ANALYSIS_ML_NTUPLES_DIR": "Ntuples",
    "ANALYSIS_ML_PYTHON": "Python environment",
    "ANALYSIS_ML_TORCH": "PyTorch installation",
    "ANALYSIS_COLUMNAR_DEV": "Columnar analysis utils",
}

OPTIONAL = {"ANALYSIS_ML_NTUPLES_DIR", "ANALYSIS_ML_PYTHON", "ANALYSIS_ML_TORCH", "ANALYSIS_COLUMNAR_DEV"}


def print_motd() -> None:
    current_file_path = Path(__file__).resolve()
    motd_file = current_file_path.parent / "motd.txt"

    if motd_file.exists():
        motd = motd_file.read_text().rstrip()
        console.print(Text(motd))

    console.print()
    console.print("[bold]Documentation:[/bold] https://seesawml.docs.cern.ch/")
    console.print("[bold]Repository:[/bold] https://gitlab.cern.ch/atlas-dch-seesaw-analyses/SeeSawML")
    console.print()


def check_exported_paths() -> None:
    table = Table(
        title="",
        box=box.SIMPLE_HEAVY,
        show_edge=False,
        header_style="bold cyan",
    )

    table.add_column("Variable", style="bold")
    table.add_column("Description")
    table.add_column("Path / Status")

    for variable, desc in EXPORTED.items():
        value = os.environ.get(variable)

        if variable in OPTIONAL:
            opt = True
        else:
            opt = False

        if value is None:
            table.add_row(variable, desc, "[red]Not set[/red]" if not opt else "[yellow]Not set (optional)[/yellow]")
        else:
            table.add_row(variable, desc, f"[green]{value}[/green]")

    console.print(table)


def create_exported_dirs() -> None:
    for variable in EXPORTED.keys():
        if variable in {
            "ANALYSIS_ML_CODE_DIR",
            "ANALYSIS_ML_OUTPUT_DIR",
        }:
            continue

        if "DIR" not in variable:
            continue

        path = os.environ.get(variable)
        if path:
            Path(path).mkdir(parents=True, exist_ok=True)


def check_python() -> None:
    console.print(
        f"[bold]Python[/bold]  {platform.python_version()}",
        style="cyan",
    )


def check_pytorch() -> None:
    console.print("[bold]PyTorch[/bold]", style="cyan")

    try:
        import torch
    except ImportError:
        console.print("  [red]Not installed[/red]")
        return None

    console.print(f"  Version: [green]{torch.__version__}[/green]")

    if torch.cuda.is_available():
        console.print(f"  CUDA:    [green]{torch.version.cuda}[/green]")
        console.print(f"  GPU:     [green]{torch.cuda.get_device_name()}[/green]")
    else:
        console.print("  CUDA:    [yellow]Not available[/yellow]")


def main() -> None:
    console.print()
    print_motd()
    check_exported_paths()
    create_exported_dirs()
    console.print()
    check_python()
    check_pytorch()
    console.print()


if __name__ == "__main__":
    main()
