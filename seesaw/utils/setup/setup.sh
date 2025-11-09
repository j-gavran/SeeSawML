#!/bin/bash

if [ -z "$ANALYSIS_ML_CODE_DIR" ]; then
    echo "Error: ANALYSIS_ML_CODE_DIR is not set!" >&2
    return 1
fi

if [ -n "$ANALYSIS_ML_VENV_PATH" ]; then
    VENV_PATH="$ANALYSIS_ML_VENV_PATH"
else
    VENV_PATH="${ANALYSIS_ML_CODE_DIR}/.venv"
    UV_EXE=$(find "${ANALYSIS_ML_CODE_DIR}/modules/TNAnalysis" -mindepth 2 -type f -name "uv" | head -n 1)

    if [[ -n "$UV_EXE" ]]; then
        echo "Using TNAnalysis virtual environment"
    else
        echo "No TNAnalysis virtual environment found!"
        return 1
    fi
fi

if [[ $(hostname) == *lxplus* ]]; then
    export IS_LXPLUS=1
else
    export IS_LXPLUS=0
fi

ACTIVATE_SCRIPT="$VENV_PATH/bin/activate"

if [ -d "$VENV_PATH" ]; then
    echo "Activating ML virtual environment at $VENV_PATH"
    source "$ACTIVATE_SCRIPT"
else
    if [ -z "$UV_EXE" ]; then
        if command -v python3.12 >/dev/null 2>&1; then
            echo "Using python3.12 to create the virtual environment"
            PYTHON_BIN="python3.12"
        else
            PYTHON_BIN="python3"
        fi

        echo "Creating virtual environment at $VENV_PATH..."
        $PYTHON_BIN -m venv "$VENV_PATH"

        source "$ACTIVATE_SCRIPT"
        echo "Virtual environment created and activated"
    fi
fi

if [ -z "$UV_EXE" ]; then
    if pip list | grep -i seesawml >/dev/null 2>&1; then
        echo "Found SeeSawML installation"
    else
        echo "Installing SeeSawML..."
        if [ "$IS_LXPLUS" = 0 ]; then
            $PYTHON_BIN -m pip install -e "${ANALYSIS_ML_CODE_DIR}/modules/SeeSawML"
        else
            $PYTHON_BIN -m pip install --no-deps --no-cache-dir -e "${ANALYSIS_ML_CODE_DIR}/modules/SeeSawML"
        fi
    fi
else
    if $UV_EXE pip list | grep -i seesawml >/dev/null 2>&1; then
        echo "Found SeeSawML installation"
    else
        echo "Installing SeeSawML with TNAnalysis uv..."
        if [ "$IS_LXPLUS" = 0 ]; then
            $UV_EXE pip install -e "${ANALYSIS_ML_CODE_DIR}/modules/SeeSawML"
        else
            $UV_EXE pip install --no-deps --no-cache-dir -e "${ANALYSIS_ML_CODE_DIR}/modules/SeeSawML"
        fi
    fi
fi

seesaw_ml_setup

if [ -n "$ANALYSIS_ML_LOGS_DIR" ] && [ "$IS_LXPLUS" = 0 ]; then
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
        mlflow ui --backend-store-uri "file://${ANALYSIS_ML_LOGS_DIR}/mlruns" -p "$port"
    }

    echo "Use \`track -p <PORT>\` to start the MLFlow UI"
fi
