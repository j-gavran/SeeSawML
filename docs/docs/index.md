# Welcome to SeeSawML documentation

SeeSawML is a modular Python ML-toolkit for HEP ML analyses based on PyTorch Lightning. It was developed for the LRSM
and SeeSaw Type-II/III ATLAS Exotics analysis, but can be used for any ML-based HEP analysis. It includes a complete ML
pipeline from data loading and preprocessing to model training, evaluation, visualization, hyperparameter optimization,
and exporting to ONNX format. The package is focused on signal vs background classification and fake lepton estimation tasks.

The package workflow is:

1. **HDF5 conversion**: Convert ROOT dataset into HDF5 format using the provided data converter pipeline.
2. **Preprocessing**: Apply feature scaling, calculate class weights and other preprocessing steps to prepare data for training.
3. **Model training**: Train ML models using PyTorch Lightning with configurable architectures and training parameters.
4. **Evaluation and visualization**: Evaluate trained models on validation/test datasets and generate performance plots.
5. **Hyperparameter optimization**: Optimize model hyperparameters using Optuna integration.
6. **ONNX export**: Export trained models to ONNX format for deployment.

Every step of the workflow is configurable via YAML configuration files, making it easy to adapt to different datasets and analysis needs.

![workflow](images/workflow.svg)

Data conversion, loading, and preprocessing are handled by the [`F9Columnar`](https://gitlab.cern.ch/ijs-f9-ljubljana/F9Columnar)
library, while model architectures and training logic are implemented in SeeSawML.

## Contributing

All contributions are welcome! If you find a bug or have a feature request, please open an issue on the [GitLab repository](https://gitlab.cern.ch/atlas-dch-seesaw-analyses/SeeSawML/-/issues/). If you would like to contribute code, please fork the repository and submit a merge request with your changes. Please make sure to follow the existing code style and run pre-commit hooks before submitting your changes. If you add new features, please also update the documentation accordingly on [this](https://gitlab.cern.ch/atlas-dch-seesaw-analyses/seesaw-ml-docs) site.

## Releases

Tags are labeled as `vX.Y.Z`. Release will be made only for major (`X`) and minor version (`Y`) updates. Patch updates will be merged to the `main` branch without a formal release. Before starting a new study, make sure that you are using the latest release version for the most up-to-date features and fixes. Documentation follows `main` branch and may include features not yet available in the latest release. Features under development (not yet in a release) are flaged as such in the documentation.
