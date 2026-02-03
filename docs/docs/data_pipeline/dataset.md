Training dataset configuration options for saved HDF5 datasets are specified in the `dataset_config/` directory. These configurations define features, labels, and other dataset properties used during model training and evaluation.

## Dataset Configuration

- `name: str`: Name of the dataset configuration.
- `files: list[str] | str`: List of file paths or a single file path to the HDF5 dataset files. If `"*"` is used, all files in `ANALYSIS_ML_DATA_DIR` directory will be included.
- `features: list[str]`: List of feature names to be used as input for the model. Both flat and jagged features can be included.
- `stage_split_piles: dict[str, int | list[int]]`: Dictionary defining the split of data into different stages (e.g., training, validation, testing) based on number of HDF5 piles. It should hold that $N_\mathrm{train} + N_\mathrm{val} + N_\mathrm{test} \leq N_\mathrm{total}$.
    - `train: int | list[int]`: Number of piles or list of pile indices for the training set.
    - `val: int | list[int]`: Number of piles or list of pile indices for the validation set.
    - `test: int | list[int]`: Number of piles or list of pile indices for the test set.
    - `calib: int | list[int]`: (Optional) Number of piles or list of pile indices for the calibration set.
- `dataset_kwargs: dict[str, Any]`: Additional keyword arguments to be passed to the Dataset when initializing.
    - `imbalanced_sampler: str | None`: Type of imbalanced sampler to use (`RandomUnderSampler`, `RandomOverSampler`, or `RandomResamplerToTarget`), by default `null` (no sampler).
    - `imbalanced_sampler_kwargs: dict[str, Any]`: Additional keyword arguments to be passed to the imbalanced sampler during initialization.
    - `drop_last: bool`: Whether to drop the last incomplete batch if the dataset size is not divisible by the batch size.

!!! Example "Imbalanced Sampling"
    Example config for oversampling:

    ```yaml
    dataset_kwargs:
      imbalanced_sampler: RandomOverSampler # or RandomUnderSampler
      imbalanced_sampler_kwargs:
        sampling_strategy: not majority
    ```
    For all alvailable options, see the [imbalanced-learn documentation](https://imbalanced-learn.org/stable/index.html).

!!! Info "RandomResamplerToTarget"
    `RandomResamplerToTarget` is a custom sampler that resamples each class to an exact target count *per pile* specified in the sampling strategy. This is particularly useful when you want precise control over class distribution in each HDF5 pile:

    - If a class has *more* samples than the target count in a pile, it randomly undersamples (without replacement)
    - If a class has *fewer* samples than the target count in a pile, it randomly oversamples (with replacement)
    - If a class has *exactly* the target count in a pile, it keeps all samples as-is

    The resampling is applied independently to each pile, and the total number of events will be: `target_count × number_of_piles`.

    **Usage Example:**
    ```yaml
    dataset_kwargs:
      imbalanced_sampler: RandomResamplerToTarget
      imbalanced_sampler_kwargs:
        sampling_strategy:
          ttH_cc: 5000
          ttH_bb: 5000
          ttbar_enhanced_HF1b: 3000
          ttbar_enhanced_HF2b: 3000
    ```

    In this example, each pile will contain exactly 5000 `ttH_cc` events and 5000 `ttH_bb` events, and 3000 events each for the ttbar classes. If using 10 piles for training, the total dataset will have 50000 `ttH_cc` events, 50000 `ttH_bb` events, and 30000 events per ttbar class.

## Dataloader Configuration

Settings for the [`DataLoader`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) used to load the dataset during training and evaluation can be specified in the `dataloader_kwargs` field.

- `dataloader_kwargs: dict[str, Any]`: Additional keyword arguments to be passed to the DataLoader when loading the dataset.
    - `batch_size: int`: Number of samples per batch to load.
    - `prefetch_factor: int`: Number of batches to prefetch.
    - `num_workers: int`: Number of subprocesses to use for data loading. Set to -1 to use all available CPU cores.

## Feature Scaling

- `feature_scaling: dict[str, Any | dict[str, Any]]`: Dictionary specifying the scaling method for each feature.
    - `numer_scaler_type: str | None`: Type of scaler to use for numerical features. Supported scalers are listed below. If set to `null`, no numerical scaling is applied.
    - `categ_scaler_type: str | None`: Type of scaler to use for categorical features. Only `categorical` (label encoding) is supported. If set to `null`, no categorical scaling is applied.
    - `n_max: int | None`: Maximum number of samples to use for fitting the scaler. If set to `null`, all samples are used.
    - `save_path: str`: Path where to save the fitted scaler.
    - `scaler_params: dict[str, Any]`: Additional parameters to be passed to the scaler during initialization. `partial_fit` is set in this field for supported scalers.
- `ple_bins: int | None`: Number of bins to use for PLE (piecewise linear encoding). If set to `null`, PLE is not applied.

!!! Info "Feature Scaling Details"
    Feature scaling is split into continuous (numerical) and discrete (categorical) feature scaling. Numerical features are scaled using the specified scaler, while categorical features are scaled using a label encoder (that also supports `partial_fit`). Categorical features are automatically detected based on their data type in the HDF5 metadata. If scaling is enabled for numerical features, label encoding is applied to categorical features automatically as well.

    Features list can contain *special* features that are used for specific purposes. These include all features ending with `_type`, `weights` feature, and any feature that includes `mlmass` in its name.

    For example, `label_type` feature is used to determine the label of each event based on the configuration. Similarly `sig_type` encodes if the event is signal or background. The `weights` feature is used to save MC or data weights. Features containing `mlmass` are used for parameterized neural networks, where the model is conditioned on the mass value.

!!! Info "Supported Scalers"
    List of supported scalers:

    - `minmax`: Min-Max scaler
    - `maxabs`: Max-Abs scaler
    - `standard`: Standard scaler (Z-score normalization)
    - `robust`: Robust scaler (using median and interquartile range)
    - `quantile`: Quantile normalizer
    - `power`: Power transform
    - `logit`: Logit transform
    - `standard_logit`: Logit followed by standard scaler

    Only `minmax` and `standard` scalers support `partial_fit`, meaning they can be fitted in batches. Other scalers require all data to be available at once for fitting.

!!! Note "Piecewise Linear Encoding (PLE)"
    Piecewise Linear Encoding (PLE) does not require feature scaling. It is a model layer applied directly on the raw feature values during model training.

!!! Example "Feature Scaling Configuration"
    Example config for feature scaling:

    ```yaml
    feature_scaling:
    numer_scaler_type: standard
    categ_scaler_type: categorical
    n_max: null
    save_path: null # default: ANALYSIS_ML_RESULTS_DIR/scalers/
    scaler_params:
      partial_fit: true
      copy: true
    ```

!!! Warning "Scaler Compatibility with TNAnalysis"
    TNAnalysis only supports `minmax`, `standard`, and `categorical` scaling methods.

## Signal Specific Configuration

- `use_mc_weights: bool`: Whether to use Monte Carlo weights in the loss function during training.
- `use_class_weights: bool`: Whether to use class weights in the loss function during training to handle class imbalance.
- `norm_class_weights: bool`: Whether to normalize class weights so that they sum to 1.
- `classes: list[str] | dict[str, list[str]]`: Class names for classification tasks. A dictionary mapping each label to its possible classes can be provided. For binary classification, a list of two class names is sufficient. Labels should match those defined in the HDF5 metadata.

!!! Tip "Obtaining Class Weights and Quantile Bins"
    Scalers can be obtained by running `scale_signal` before training. Class weights can be calculated by running `calculate_class_weights` before training. Qauntile bins can be calculated by running `calculate_quantiles` before training.

!!! Note "Class Weights Calculation"
    Class weights are calculated based on the frequency of each class in the training dataset. They help to mitigate class imbalance by assigning higher weights to less frequent classes during training. They are calculated as:
    $$
    w_i = \frac{N}{n_i \, C} ,
    $$
    where $w_i$ is the weight for class $i$, $N$ is the total number of samples, $n_i$ is the number of samples in class $i$ and $C$ is the total number of classes. If `norm_class_weights` is set to `true`, the weights are normalized so that they sum to 1 in the specific dataset split.

!!! Example "Class Configuration Examples"
    `classes` configuration for binary classification:

    ```yaml
    classes:
    - signal:
      - typeIIseesaw_fast_1000
    - background:
      - ttbar_inclusive
      - singletop_inclusive
      - raretop
      - ttH
      - ttW
      - ttZ
      - Zjets
      - Wjets
      - dijet
      - diboson
    ```

    `classes` configuration for multi-class classification:

    ```yaml
    classes:
      - ttH_cc
      - ttH_bb
      - ttZ_bb
      - ttZ_cc
      - ttbar_enhanced_HF1b
      - ttbar_enhanced_HF2b
      - ttbar_enhanced_HF1c
      - ttbar_enhanced_HF2c
      - ttbar_enhanced_HFlight
    ```

    Additionally, labels can be merged into classes any way desired, for example:

    ```yaml
    classes:
      - signal:
        - typeIIseesaw_fast_1000
      - Zjets
      - Wjets
      - top:
        - ttbar_inclusive
        - singletop_inclusive
        - raretop
      - ttbar:
        - ttH
        - ttW
        - ttZ
      - other:
       - dijet
       - diboson
    ```

    Internally this creates a lookup table mapping each label to its corresponding class and applies it on the fly during dataset loading.

## Fakes Specific Configuration

- `dataset_kwargs: dict[str, Any]`: Additional keyword arguments to be passed to the Dataset when initializing.
    - `drop_last: bool`: Whether to drop the last incomplete batch if the dataset size is not divisible by the batch size.
    - `use_data_in_ratio: bool`: Whether to use data events in the ratio reweighting.
    - `pt_cut: dict[str, float] | None`: Dictionary specifying $p_\mathrm{T}$ cut values for different objects. If set to `null`, no cuts are applied. Keys are `min` and `max`. Only supported for electrons.
- `crack_veto: bool`: Whether to apply crack veto for electrons, by default `false`.
