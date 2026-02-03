There are several model architectures available in the SeeSawML. Each model may have its own specific configuration options. The configuration for each model type can be found in the `model_config/` directory. There are two top level YAML fields that are common to all models: `architecture_config` and `training_config`. The `architecture_config` field contains options specific to the model type.

## Compiling Models

All models can be compiled using PyTorch's `torch.compile` feature for improved performance during training and inference. Compilation can be enabled by setting the `compile` field to `true` in the model's `architecture_config`. Additional keyword arguments for `torch.compile` can be specified in the `compile_kwargs` field. See [PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.compile.html) for more details. The options are the same for all models described above:

- `compile: bool`: Whether to compile the model using `torch.compile` for improved performance, by default `false`.
- `compile_kwargs: dict[str | Any]`: Additional keyword arguments to pass to `torch.compile`, by default `null`.

## Combining Models and Embeddings

### Combining Model Outputs

Flat and jagged model outputs can be combined by merging (*fusing*) them together. This can be done from any jagged model config by setting a flat model in the model configuration YAML:

```yaml
defaults:
  - _self_
  - <model_config>@flat_model_config
```

For example, `<model_config>` can be `mlp`, `res_net`, `deep_sets` or `event_transformer`. The flat model will be built and its output will be fused to the jagged model output. The flat model will use the flat event features only while the jagged model will use the jagged object features only. This allows to build hybrid models that can leverage both flat and jagged features.

Model fusion can be controlled by `architecture_config.flat_fuse` from any jagged model config:

```yaml
flat_fuse:
  mode: ...
  fuse_kwargs: null
```

Valid modes are:

- `add`: Element-wise addition of flat and jagged model outputs.
- `cat`: Concatenation of flat and jagged model outputs followed by a linear layer to project to the desired output dimension.
- `learn`: Learnable weighted addition of flat and jagged model outputs as
$$
\mathrm{output} = \alpha \cdot \mathrm{flat\_output} + (1 - \alpha) \cdot \mathrm{jagged\_output},
$$
where $\alpha$ is a learnable parameter between 0 and 1.
- `gate`: Gated addition of flat and jagged model outputs. First calculates a gate value $g = \sigma(\mathrm{Linear}[\mathrm{flat\_output}, \mathrm{jagged\_output}])$ where $\mathrm{Linear}$ is a linear layer and $\sigma$ is the sigmoid activation function. Then combines the outputs as
$$
\mathrm{output} = g \cdot \mathrm{flat\_output} + (1 - g) \cdot \mathrm{jagged\_output}.
$$
- `attn`: Attention-based fusion of flat and jagged model outputs. Uses multi-head attention mechanism to combine the outputs. Additional parameters for attention can be specified in `fuse_kwargs` such as number of heads, dimension of each head, dropout, etc. Jagged features are used as queries while flat features are used as keys and values.

### Combining Embeddings

Flat embeddings can be joined with the jagged embeddings by fusing them together in any jagged model configuration. That is, flat features can be embedded and then combined with jagged feature embeddings before feeding into the main model architecture.

This can be done by setting the following in the jagged model `arhitecture_config`:

- `flat_embeddings: dict[str | Any] | None`: Equivalent to `architecture_config.embeddings` but for flat features, by default `null`.
- `post_flat_embeddings: dict[str | Any] | None`: Equivalent to `architecture_config.post_embeddings` but for flat features, by default `null`.
- `flat_embeddings_fuse: dict[str | Any] | None`: Configuration for fusing flat embeddings with jagged embeddings, by default `null`.
    - `mode: str`: Fusion mode (`sum`, `mean`, `add`, `cat`, `cat_proj`, `learn`, `gate`, `attn` or `res_attn`).
    - `fuse_kwargs: dict[str | Any] | None`: Additional parameters for fusion method, by default `null`.

## Model Calibration

Temperature, vector or matrix scaling (see: [https://arxiv.org/abs/1706.04599](https://arxiv.org/abs/1706.04599)) can be applied to the trained model outputs for calibration. Calibration parameters are learned on a separate validation dataset after training the main model. Calibration configuration is specified under the `calibration_config` field in the model configuration file. It requires a separate validation dataset for calibration that can be specified in the `dataset_config.stage_split_piles` as a `calib` field. The calibration config supports the following options:

- `method: str`: Calibration method to use (`temperature`, `vector` or `matrix`).
- `fit_all: bool`: Whether to fit calibration parameters on all validation data at once or in batches.
- `set_temperature: float | None`: If provided, sets the temperature parameter to this value instead of fitting it, by default `null`.
- `calibration_params: dict[str | Any] | None`: Additional parameters for calibration method, by default `null`.
    - `optimizer: str`: Optimizer to use for fitting calibration parameters, `lbfgs` or `lbfgs_line_search`, by default `lbfgs`.
    - `lr: float`: Learning rate for optimizer, by default `0.01`.
    - `max_iter: int`: Maximum number of iterations for optimizer, by default `200`.
    - `is_binary: bool`: Whether the task is binary classification, by default `false`. Only temperature scaling is supported for binary classification.

Post-hoc calibration is applied after the main model training (given that a `calib` split exists) and does not affect the training process itself. It can be performed by setting a model checkpoint to `load_checkpoint` along with a calibration configuration described above. Calibration can be fitted using the `calibrate_signal` command after training the main model. This will produce a new checkpoint (ending with `_calib.ckpt`) with calibrated model weights that can be used for inference.
