export ANALYSIS_ML_CODE_DIR="path/to/your/analysis" # Path to the analysis code directory (repository)
export ANALYSIS_ML_OUTPUT_DIR="path/to/your/output" # Use eos if running with condor

export ANALYSIS_ML_VENV_PATH="${ANALYSIS_ML_OUTPUT_DIR}/venv" # Path to the virtual environment, set to TNAnalysis venv if not set

export ANALYSIS_ML_CONFIG_DIR="${ANALYSIS_ML_CODE_DIR}/seesaw/config"
export ANALYSIS_ML_DATA_DIR="${ANALYSIS_ML_OUTPUT_DIR}/data" # Use eos if running with condor
export ANALYSIS_ML_RESULTS_DIR="${ANALYSIS_ML_OUTPUT_DIR}/results"
export ANALYSIS_ML_MODELS_DIR="${ANALYSIS_ML_RESULTS_DIR}/models"
export ANALYSIS_ML_LOGS_DIR="${ANALYSIS_ML_RESULTS_DIR}/logs"

export ANALYSIS_ML_NTUPLES_DIR="path/to/your/ntuples" # Directory for HDF5 converter input, if not set will be ANALYSIS_ML_DATA_DIR, use eos if running with condor

source "${ANALYSIS_ML_CODE_DIR}/modules/SeeSawML/seesaw/utils/setup/setup.sh"
