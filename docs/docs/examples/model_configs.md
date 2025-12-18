# Model Configuration Examples

This page provides example configurations for different models implemented in SeeSawML. This is a good starting point for users looking to set up their own model configurations. The given models can be used as templates and modified according to specific requirements. Note that these configurations are not exhaustive and users are encouraged to explore and customize them further. Also note that the given configurations may not represent the best performing models for specific tasks, but rather serve as illustrative examples.

## MLP

```yaml
name: sigBkgClassifier

load_checkpoint: null
continue_training: false

architecture_config:
  model: MLP

  n_layers: 4
  n_hidden: 256
  act: ReLU
  act_out: null
  batchnorm: false
  dropout: 0.0

  disable_embeddings: false
  embeddings:
    ple: null
    reduction: mean
  post_embeddings: null

  compile: true
  compile_kwargs:
    fullgraph: true

training_config:
  loss: bce
  disco: null

  optimizer:
    optimizer_name: AdamW
    optimizer_params:
      lr: 1.0e-3
      weight_decay: 1.0e-5

  scheduler:
    scheduler_name: ReduceLROnPlateau
    interval: epoch
    scheduler_params:
      factor: 0.5
      patience: 3

  reduce_lr_on_epoch: 1.0

  max_epochs: 100
  early_stop_patience: 10
  monitor: val_loss
  save_top_k: 5
  log_train_memory: false
  model_save_path: null # set to ANALYSIS_ML_MODELS_DIR/checkpoints if null
```

## ResNet

```yaml
name: sigBkgClassifier

load_checkpoint: null
continue_training: false

architecture_config:
  model: ResNet

  n_layers: 6
  n_hidden: 256
  act: ReLU
  act_out: null
  batchnorm: true
  dropout: 0.1

  disable_embeddings: false
  embeddings:
    ple: null
    reduction: mean
  post_embeddings: null

  compile: true
  compile_kwargs:
    fullgraph: true

training_config:
  loss: bce
  disco: null

  optimizer:
    optimizer_name: Adam
    optimizer_params:
      lr: 1.0e-3
      weight_decay: 1.0e-5

  scheduler:
    scheduler_name: ReduceLROnPlateau
    interval: epoch
    scheduler_params:
      factor: 0.5
      patience: 5

  reduce_lr_on_epoch: 1.0

  max_epochs: 100
  early_stop_patience: 10
  monitor: val_loss
  save_top_k: 5
  log_train_memory: false
  model_save_path: null
```

## DeepSets

```yaml
name: sigBkgClassifier

load_checkpoint: null
continue_training: false

architecture_config:
  model: DeepSets

  embedding_dim: 128
  encoder_layers: 5
  decoder_layers: 5
  n_hidden: 256
  act: GELU
  act_out: null
  batchnorm: true
  dropout: 0.1

  mean_pooling: true

  embeddings:
    ple: null
    conv1d_embedding: false
    numer_feature_wise_linear: false
    reduction: mean

  post_embeddings:
    batchnorm: false
    act: GELU

  flat_fuse:
    mode: cat
    fuse_kwargs: null

  compile: true
  compile_kwargs:
    fullgraph: false

training_config:
  loss:
    loss_name: ce

  optimizer:
    optimizer_name: AdamW
    optimizer_params:
      lr: 3.0e-4
      weight_decay: 1.0e-4

  scheduler:
    scheduler_name: ReduceLROnPlateau
    interval: epoch
    scheduler_params:
      factor: 0.5
      patience: 5

  # scheduler:
  #   scheduler_name: CosineAnnealingWarmRestarts
  #   interval: step
  #   reduce_lr_on_epoch: 0.95
  #   scheduler_params:
  #     T_0: 289

  reduce_lr_on_epoch: 1.0

  max_epochs: 100
  early_stop_patience: 10
  monitor: val_F1
  monitor_mode: max
  save_top_k: 5
  log_train_memory: false
  model_save_path: null

defaults:
  - _self_
  - mlp@flat_model_config
```

## EventTransformer

```yaml
name: sigBkgClassifier

load_checkpoint: null
continue_training: false

architecture_config:
  model: EventTransformer

  embedding_dim: 64
  transformer_depth: 6
  heads: 8
  dim_head: 16
  attn_dropout: 0.1

  remove_first_attn_residual: false
  remove_first_attn_layernorm: true

  ff_hidden_mult: 4
  ff_dropout: 0.1

  use_cls_token: true

  embeddings:
    numer_feature_wise_linear: false

  post_embeddings: null

  sdp_backend:
    enable_cudnn: true
    enable_mem_efficient: true
    enable_flash: false
    enable_math: false

  compile: true
  compile_kwargs:
    fullgraph: true

training_config:
  loss: ce

  optimizer:
    optimizer_name: AdamW
    optimizer_params:
      lr: 3.0e-4
      weight_decay: 1.0e-4

  scheduler:
    scheduler_name: AttentionWarmup
    scheduler_params:
      lr_mul: 1.0
      d_model: 64
      n_warmup_steps: 1024
      freeze_step: null

  reduce_lr_on_epoch: 1.0

  max_epochs: 100
  early_stop_patience: 10
  monitor: val_loss
  save_top_k: 5
  log_train_memory: false
  model_save_path: null
```

## SetTransformer

```yaml
name: sigBkgClassifier

load_checkpoint: null
continue_training: false

architecture_config:
  model: SetTransformer

  embedding_dim: 128
  heads: 8
  dim_head: 16
  attn_dropout: 0.1
  use_setnorm: false

  ff_hidden_mult: 2
  ff_dropout: 0.1

  first_attn_no_residual: false

  add_particle_types: true

  seed_strategy: pooling # pooling, particles or objects

  # used if seed strategy is not pooling
  encoder_depth: 4
  decoder_depth: 2

  set_predictor:
    depth: 2
    act: GELU
    mean_pooling: true

  # flat_embeddings:
  #   reduction: reshape
  #   numer_feature_wise_linear: false
  #   ple: null

  # post_flat_embeddings: null

  # flat_embeddings_fuse:
  #   mode: gate

  embeddings:
    reduction: reshape
    conv1d_embedding: false
    numer_feature_wise_linear: false

    ple: null
      # learn_bins: false
      # uniform_bins: false
      # act: ReLU
      # dropout: 0.1
      # layernorm: false

  post_embeddings: null

  compile: true
  compile_kwargs:
    fullgraph: true

training_config:
  loss:
    loss_name: ce
    loss_params:
      label_smoothing: 0.1

  disco: null

  optimizer:
    optimizer_name: AdamW
    optimizer_params:
      lr: 3.0e-4
      weight_decay: 1.0e-5

  # scheduler:
  #   scheduler_name: AttentionWarmup
  #   scheduler_params:
  #     lr_mul: 1.0
  #     d_model: 256
  #     n_warmup_steps: 609
  #     freeze_step: null

  # scheduler:
  #   scheduler_name: CosineWarmup
  #   interval: epoch
  #   scheduler_params:
  #     n_warmup_steps: 1
  #     T_max: 30
  #     max_lr: 1e-3
  #     min_lr: 1e-9

  # scheduler:
  #   scheduler_name: CosineAnnealingWarmRestarts
  #   interval: step
  #   scheduler_params:
  #     T_0: 800

  reduce_lr_on_epoch: 1.0

  max_epochs: 100
  early_stop_patience: 10
  monitor: val_F1
  monitor_mode: max
  save_top_k: 10
  log_train_memory: false
  model_save_path: null

calibration_config:
  method: temperature
  fit_all: true
  set_temperature: 0.3
  calibration_params:
    optimizer: lbfgs_line_search
    lr: 0.01
    max_iter: 1000
```

## ParticleTransformer

```yaml
name: sigBkgClassifier

load_checkpoint: null
continue_training: false

architecture_config:
  model: ParticleTransformer

  embedding_dim: 64
  heads: 8
  dim_head: 16
  attn_dropout: 0.1
  ff_hidden_mult: 4
  ff_dropout: 0.1

  particle_blocks: 4
  class_blocks: 2

  add_particle_types: true
  first_attn_no_residual: false

  embeddings:
    reduction: mean

    ple:
      learn_bins: false
      uniform_bins: false
      act: GELU
      dropout: 0.1
      layernorm: true

  flat_embeddings:
    reduction: mean
    numer_feature_wise_linear: false
    ple:
      learn_bins: false
      uniform_bins: false
      act: GELU
      dropout: 0.1
      layernorm: true

  post_flat_embeddings: null

  flat_embeddings_fuse:
    mode: gate

  particle_attention:
    embedding_layers: 2
    embedding_dim: 64
    objects:
      - jets
      - electrons
      - muons
    quantities:
      - delta_r
      - kt
      - z
      - m2

  sdp_backend:
    enable_cudnn: true
    enable_mem_efficient: true
    enable_flash: false
    enable_math: false

  compile: true
  compile_kwargs:
    fullgraph: true

training_config:
  loss:
    loss_name: ce
    loss_params:
      label_smoothing: 0.01

  disco: null

  optimizer:
    optimizer_name: AdamW
    optimizer_params:
      lr: 3.0e-4
      weight_decay: 1.0e-2

  scheduler:
    scheduler_name: AttentionWarmup
    scheduler_params:
      lr_mul: 1.0
      d_model: 512
      n_warmup_steps: 1024
      freeze_step: null

  # scheduler:
  #   scheduler_name: CosineWarmup
  #   interval: epoch
  #   scheduler_params:
  #     n_warmup_steps: 1
  #     T_max: 30
  #     max_lr: 1e-3
  #     min_lr: 1e-9

  max_epochs: 100
  early_stop_patience: 10
  monitor: val_loss
  monitor_mode: min
  save_top_k: 5
  log_train_memory: false
  model_save_path: null
```
