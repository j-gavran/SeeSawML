import os
import platform

EXPORTED = {
    "ANALYSIS_ML_CODE_DIR": "analysis code directory",
    "ANALYSIS_ML_OUTPUT_DIR": "output directory",
    "ANALYSIS_ML_NTUPLES_DIR": "Ntuples directory",
    "ANALYSIS_ML_CONFIG_DIR": "configuration from",
    "ANALYSIS_ML_DATA_DIR": "data from",
    "ANALYSIS_ML_RESULTS_DIR": "results directory",
    "ANALYSIS_ML_MODELS_DIR": "saved models directory",
    "ANALYSIS_ML_LOGS_DIR": "logs directory",
}


def print_motd() -> None:
    current_file_path = os.path.abspath(__file__)
    motd_file = os.path.join(os.path.dirname(current_file_path), "motd.txt")

    with open(motd_file, "r") as f:
        motd = f.read()

    print(motd)


def check_exported_paths() -> None:
    for variable, desc in EXPORTED.items():
        value = os.environ.get(variable, None)
        if value is None:
            print(f"{variable} is not set!")
        else:
            print(f"Using {desc}: {value}")


def create_exported_dirs() -> None:
    for variable in EXPORTED.keys():
        if variable == "ANALYSIS_ML_CODE_DIR" or variable == "ANALYSIS_ML_OUTPUT_DIR":
            continue

        path = os.environ.get(variable, None)

        if path is not None:
            os.makedirs(path, exist_ok=True)


def check_pytorch() -> None:
    print("Checking PyTorch installation...")

    try:
        import torch
    except ImportError:
        print("Failed to import PyTorch!")
        return None

    print(f"Using PyTorch: {torch.__version__}")

    if torch.cuda.is_available():
        print(f"Found GPU: {torch.cuda.get_device_name()}")
    else:
        print("No GPU detected.")


def check_python() -> None:
    print(f"Using Python: {platform.python_version()}")


def main() -> None:
    print()
    print_motd()
    check_exported_paths()
    create_exported_dirs()
    check_python()
    check_pytorch()


if __name__ == "__main__":
    main()
