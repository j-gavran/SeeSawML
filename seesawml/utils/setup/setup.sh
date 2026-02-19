#!/bin/bash

if [ -z "$ANALYSIS_ML_CODE_DIR" ]; then
    echo "Error: ANALYSIS_ML_CODE_DIR is not set!" >&2
    return 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# use UV_PROJECT_ENVIRONMENT to set custom venv path, otherwise defaults to .venv in project dir
if [ -n "$ANALYSIS_ML_VENV_PATH" ]; then
    export UV_PROJECT_ENVIRONMENT="$ANALYSIS_ML_VENV_PATH"
fi

# ANALYSIS_ML_TORCH: user can set this to specify which torch variant to install
TORCH_EXTRA="torch-${ANALYSIS_ML_TORCH:-cu128}"
# ANALYSIS_ML_PYTHON: python version to use (default: 3.12)
PYTHON_VERSION="${ANALYSIS_ML_PYTHON:-3.12}"

echo "Setting up ML environment with uv (Python ${PYTHON_VERSION}, ${TORCH_EXTRA})"

VENV_PATH="${UV_PROJECT_ENVIRONMENT:-${ANALYSIS_ML_CODE_DIR}/modules/SeeSawML/.venv}"

# ANALYSIS_COLUMNAR_DEV: path to a local F9Columnar clone for editable development
if [ -n "$ANALYSIS_COLUMNAR_DEV" ]; then
    uv sync --python "$PYTHON_VERSION" --group dev --extra "$TORCH_EXTRA" --inexact --project "${ANALYSIS_ML_CODE_DIR}/modules/SeeSawML"
    source "$VENV_PATH/bin/activate"

    echo "Installing F9Columnar in editable mode from ${ANALYSIS_COLUMNAR_DEV}"
    uv pip install -e "$ANALYSIS_COLUMNAR_DEV"
else
    uv sync --python "$PYTHON_VERSION" --group dev --extra "$TORCH_EXTRA" --extra f9columnar --project "${ANALYSIS_ML_CODE_DIR}/modules/SeeSawML"
    source "$VENV_PATH/bin/activate"
fi

seesaw_ml_setup

if [ -n "$ANALYSIS_ML_LOGS_DIR" ]; then
    function track {
        local port=5000
        while [[ $# -gt 0 ]]; do
            case "$1" in
                -p|--port)
                    port="$2"
                    shift 2
                    ;;
                *)
                    echo "Usage: track [-p PORT]"
                    return 1
                    ;;
            esac
        done
        mlflow ui --host 0.0.0.0 --allowed-hosts "mlflow.internal:5000,localhost:*" --backend-store-uri "sqlite:///${ANALYSIS_ML_LOGS_DIR}/mlruns/mlflow.sqlite" -p "$port"
    }

    echo "Use \`track -p <PORT>\` to start the MLFlow UI"
fi
