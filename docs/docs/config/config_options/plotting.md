Plotting configuration is specified via YAML files that define the plots to be generated. These files can be found in the `plotting_config/` directory. All the YAML files are composed together inside a main plotting configuration `plotting_config.yaml` and imported to the `training_config.yaml` under the `plotting` section.

## Signal vs Background Plots

### Score Plot

Configuration for tracker score plots when training. Available in `score_plot.yaml`.

- `sigmoid_plot: dict[str, Any]`: Configuration for sigmoid score plots (in range $[0, 1]$).
    - `x_min: float`: Minimum x-axis value.
    - `x_max: float`: Maximum x-axis value.
    - `n_bins: int`: Number of bins.
    - `normalised_weighted: bool`: Plot normalised weighted distributions.
    - `normalised_unweighted: bool`: Plot normalised unweighted distributions.
    - `unnormalised_weighted: bool`: Plot unnormalised weighted distributions.
    - `unnormalised_unweighted: bool`: Plot unnormalised unweighted distributions.
- `logit_plot: dict[str, Any]`: Configuration for logit score plots (in range $(-\infty, +\infty)$).
    - `x_min: float`: Minimum x-axis value.
    - `x_max: float`: Maximum x-axis value.
    - `n_bins: int`: Number of bins.
    - `normalised_weighted: bool`: Plot normalised weighted distributions.
    - `normalised_unweighted: bool`: Plot normalised unweighted distributions.
    - `unnormalised_weighted: bool`: Plot unnormalised weighted distributions.
    - `unnormalised_unweighted: bool`: Plot unnormalised unweighted distributions.

### Signal plot

Configuration for distribution plots of signal and background samples. Available in `signal_plot.yaml`.

- `nbins: int`: Number of bins.
- `scale: bool`: Whether to use feature scaling.
- `rescale: bool`: Whether to rescale features.
- `weighted: bool`: Whether to use MC weights.
- `signal_name: str | None`: Name of the signal class.
- `closure: bool`: Perform classification closure test, model must be provided in the config.
- `cut_low: float`: Lower cut on classifier score.
- `cut_high: float`: Upper cut on classifier score.
- `ml_mass: list[float] | None`: list of ML mass values to set at inference for parameterized models
- `disc: bool`: Whether to plot discriminator distributions instead of classifier score.
- `is_multiclass: bool`: Force multiclass classification mode.

!!! Tip
    Run `plot_signal` command to generate signal vs background plots after training. Class labels can be set in the `dataset_config.classes`.

## Fakes Plots

### Closure Plot

Configuration for closure plots available in `closure_plot.yaml`.

- `closure_type: str`: Type of closure plot, can be `all`, `binned`, or `ml`.
- `atlas_label: str | None`: ATLAS label to be used in the plot.
- `variabels: dict[str, dict[str, Any]]`: Dictionary defining the variables to be plotted.
    - `<variable_name>: str`: Name of the variable to plot.
        - `x_min: float`: Minimum x-axis value.
        - `x_max: float`: Maximum x-axis value.
        - `nbins: int`: Number of bins.
        - `logx: bool`: Whether to use logarithmic x-axis.
        - `logy: bool`: Whether to use logarithmic y-axis.
- `ff_standard_binning: dict[str, list[float]]`: Standard binning for fake factor plots.
    - `pt_bins: list[float]`: List of $p_\mathrm{T}$ bin edges.
    - `eta_bins: list[float]`: List of $|\eta|$ bin edges.
    - `met_bins: list[float]`: List of $E_\mathrm{T}^\mathrm{miss}$ bin edges.

### Model Fake Factor Plot

Configuration for model fake factor plots available in `model_ff_plot.yaml`.

- `variables: dict[str, dict[str, Any]]`: Dictionary defining the variables to be plotted.
    - `<variable_name>: str`: Name of the variable to plot.
        - `x_min: float`: Minimum x-axis value.
        - `x_max: float`: Maximum x-axis value.
        - `nbins: int`: Number of bins.
        - `logx: bool`: Whether to use logarithmic x-axis.
        - `log_scale: bool`: Whether to apply logarithmic scale on plot.
        - `bins: list[float] | None`: Custom bin edges. If provided, other binning options are ignored.
- `use_density: bool`: Whether to use density for the fake factor calculation. If `false`, will use logits.
- `pt_eta_density_plot: bool`: Whether to plot $p_\mathrm{T}$ vs $|\eta|$ density plots.
- `pt_met_density_plot: bool`: Whether to plot $p_\mathrm{T}$ vs $E_\mathrm{T}^\mathrm{miss}$ density plots.
- `met_eta_density_plot: bool`: Whether to plot $E_\mathrm{T}^\mathrm{miss}$ vs $|\eta|$ density plots.
- `crack_veto: bool`: Whether to apply crack veto in $\eta$ on the grid of points
- `plot_crack_veto: bool`: Whether to plot the crack veto region in $\eta$.
- `binned_ff_file_path: str | None`: Path to the binned fake factor file for comparison. If `null`, will use `ANALYSIS_ML_RESULTS_DIR/closure/binned_ff.p`.
- `atlas_label: str | None`: ATLAS label to be used in the plot.

### Subtraction Plot

Configuration for subtraction plots available in `subtraction_plot.yaml`. This config is used for tracker plotting when training.

- `variables: dict[str, dict[str, Any]]`: Dictionary defining the variables to be plotted.
    - `<variable_name>: str`: Name of the variable to plot.
        - `x_min: float`: Minimum x-axis value.
        - `x_max: float`: Maximum x-axis value.
        - `nbins: int`: Number of bins.
        - `logx: bool`: Whether to use logarithmic x-axis.
        - `logy: bool`: Whether to use logarithmic y-axis.
- `weights: dict[str, list[float]]`: Reweighting plots.
    - `data_prescales: list[float]`
    - `data_out: list[float]`
    - `data_density: list[float]`
    - `data_sub: list[float]`
    - `data_reweighted: list[float]`
    - `mc_weights: list[float]`
    - `mc_out: list[float]`
    - `mc_density: list[float]`
    - `mc_sub: list[float]`
    - `mc_reweighted: list[float]`
- `atlas_label: str | None`: ATLAS label to be used in the plot.

### Toy Plot

Configuration for toy plots available in `toy_plot.yaml`.

- `dataset_bins: int`: Number of bins for dataset distributions.
- `weight_bins: int`: Number of bins for fake factor weight.
- `ff_bins: list[float]`: List of fake factor bin edges.
- `pt_min: float`: Minimum $p_\mathrm{T}$ value.
- `pt_max: float`: Maximum $p_\mathrm{T}$ value.
- `mc_variation: float`: MC variation to be applied.
- `atlas_label: str | None`: ATLAS label to be used in the plot.
