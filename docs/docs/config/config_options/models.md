There are several model architectures available in the library. Each model may have its own specific configuration options. The configuration for each model type can be found in the `model_config/` directory. There are two top level YAML fields that are common to all models: `architecture_config` and `training_config`. The `architecture_config` field contains options specific to the model type, while the `training_config` field contains options related to the training process and is common across all models.

## Common Configuration

Besides the mentioned top level fields, there are some common configuration options that apply to all models and are specified at the top level of the model configuration file.

- `name: str`: Name of the model.
- `load_checkpoint: str | None`: Name of the checkpoint file to load the model weights from. If `null`, the model will be trained from scratch. Path to this checkpoint is configured in the `training_config.model_save_path`.
- `continue_training: bool`: Whether to continue training from the loaded checkpoint. If set to `true`, needs to be used in conjunction with `load_checkpoint`.

## Training Configuration

- `loss: dict[str | Any] | str`: Specifies the loss function to be used during training.
    - `loss_name: str`: The name of the loss function if `loss` is provided as a dictionary.
    - `loss_params: dict[str | Any]`: Parameters specific to the chosen loss function if `loss` is provided as a dictionary.

!!! Info
    Supported loss functions for signal vs background classification include:

    - `BCEWithLogitsLoss`: Binary classification loss with logits (**good baseline**).
    - `MSELoss`: Mean Squared Error loss for regression tasks.
    - `CrossEntropyLoss`: Standard cross-entropy loss for multi-class classification (**good baseline**).
    - `SigmoidFocalLoss`: Focal loss for addressing class imbalance in binary classification.
    - `MulticlassFocalLoss`: Focal loss for multi-class classification tasks.
    - All the losses from `pytorch_optimizer` package listed [here](https://pytorch-optimizers.readthedocs.io/en/latest/loss/).

    Supported loss functions for fakes estimation include different options, see [here](https://gitlab.cern.ch/atlas-dch-seesaw-analyses/SeeSawML/-/blob/main/seesaw/fakes/models/loss.py?ref_type=heads#L3).

- `disco: dict[str | Any] | None`: Distance correlation regularization to reduce mass sculpting from [https://arxiv.org/abs/2001.05310](https://arxiv.org/abs/2001.05310).
    - `variables: list[str]`: List of variable names to apply distance correlation regularization on.
    - `lambda: float`: Regularization strength.
    - `power: float`: Power to which the distance correlation is raised, by default `1.0`.
    - `weighted: bool`: Whether to use class or MC weights in the distance correlation calculation, by default `false`.
    - `multiclass_reduction: str`: Reduction method for multiclass distance correlation (`logits` or `entropy`), by default `logits`.

- `optimizer: dict[str | Any]`: Specifies the optimizer to be used during training.
    - `optimizer_name: str`: The name of the optimizer.
    - `optimizer_params: dict[str | Any]`: Parameters specific to the chosen optimizer.

!!! Info
    Supported optimizers include all optimizers from PyTorch listed [here](https://docs.pytorch.org/docs/stable/optim.html#algorithms) as well as those from `pytorch_optimizer` package listed [here](https://pytorch-optimizers.readthedocs.io/en/latest/optimizer/). A good default choice is `Adam` or `AdamW` with a learning rate of `3e-4`.

- `scheduler: dict[str | Any] | None`: Learning rate scheduler to adjust the learning rate during training.
    - `scheduler_name: str`: The name of the scheduler.
    - `interval: str`: Interval for scheduler step (`epoch` or `step`).
    - `scheduler_params: dict[str | Any]`: Parameters specific to the chosen scheduler.

!!! Info
    Supported schedulers include all schedulers from PyTorch listed [here](https://docs.pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate) as well as those from `pytorch_optimizer` package listed [here](https://pytorch-optimizers.readthedocs.io/en/latest/lr_scheduler/). Additionally, the following custom schedulers are available:

    - `AttentionWarmup`: Linear warmup followed by a decay.
    - `SqrtExpWarmup`: Combines square root decay with exponential warmup.
    - `CosineWarmup`: Cosine annealing with linear warmup.
    - `LinearWarmup`: Linear warmup followed by a constant learning rate.

- `reduce_lr_on_epoch: float | None`: If set, reduces the learning rate by a factor every epoch. Can be used as an alternative to schedulers.
- `max_epochs: int`: Maximum number of training epochs.
- `early_stop_patience: int | None`: Number of epochs with no improvement after which training will be stopped. If `null`, early stopping is disabled.
- `gradient_clip_val: float | None`: Maximum norm for gradient clipping. If `null`, gradient clipping is disabled.
- `monitor: str`: Metric to monitor for early stopping and learning rate scheduling, by default `val_loss`.
- `monitor_mode: str`: Mode for monitoring the metric (`min` or `max`), by default `min`.
- `save_top_k: int`: Number of best models to save based on the monitored metric.
- `log_train_memory: bool`: Whether to log memory usage during training, by default `false`.
- `model_save_path: str | None`: Path where to save the model checkpoints. If `null`, defaults to `ANALYSIS_ML_MODELS_DIR/checkpoints`.

!!! Info
    Fakes models additionally support the following training configuration options:

    - `w_lambda: float`: Regularization strength for density estimation.
    - `ess_lambda: float`: Regularization strength for effective sample size.

!!! Example
    A simple binary classifier training configuration might look like this:

    ```yaml
    training_config:
      loss: bce

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

      max_epochs: 100
      early_stop_patience: 10
      monitor: val_loss
      save_top_k: 5
      model_save_path: null
    ```

## Model Types

### Embeddings

In SeeSawML, embeddings are used to transform input physics variables into a higher-dimensional space, enabling the model to capture complex relationships within the data. There are two types of embeddings available: flat embeddings and jagged embeddings, each separated for continuous (numerical) and discrete (categorical) features. Categorical features are always embedded using a PyTorch `nn.Embedding` layer, while for numerical features there are three options:

1. Linear transformation of type `A*X + b` where `A` and `b` are learnable parameters.
2. Feature-wise linear transformation where each feature has its own linear layer.
3. Piecewise linear encoding (PLE) as described in [https://arxiv.org/abs/2203.05556](https://arxiv.org/abs/2203.05556), where multiple binning schemes are possible: learnable, uniform, or quantile-based.

<figure markdown="span">
  ![Embedding NN](../../images/mlp_emb.png){ width="900" }
  <figcaption>Embeddings for a MLP model with both numerical and categorical features.</figcaption>
</figure>

Embeddings configuration is specified under the `architecture_config` and `architecture_config.embeddings` field in the model configuration file.

!!! Info
    Embeddings can be used with any model architecture in SeeSawML and they are part of the model. They are not limited to specific model types. They serve as a preprocessing step to transform input features before they are fed into the main model architecture. If `embedding_dim` is not specified explicitly, it defaults to the first hidden layer dimension of the model.

!!! Note
    Embeddings can also be disabled altogether by setting `disable_embeddings: true` in the `architecture_config` for flat models only. In this case, the input features are used directly without any embedding transformation. This option is not available for jagged models as they require embeddings to handle variable-length sequences.

!!! Tip
    Quantile bins for piecewise linear encoding can be precomputed using the `calculate_quantiles` command before training the model. It can be enabled by setting the number of bins in `dataset_config.ple_bins`.

- `reduction: str`: Reduction method for combining multiple feature embeddings. For flat embeddings: `mean`, `reshape`, `conv1d`, `attn` or `none`, by default `mean`. For jagged embeddings: `mean`, `reshape`, `conv2d` or `attn`, by default `mean`.
- `numer_feature_wise_linear: bool`: Whether to use feature-wise linear layers for numerical feature embeddings, by default `false`.
- `ple: bool | None | dict[str, Any]`: Configuration for piecewise linear encoding (PLE) for numerical features, by default `null`, which means PLE is disabled.
    - If set to `true`, uses default PLE configuration.
    - If set to a dictionary, allows to specify custom PLE parameters:
        - `learn_bins: bool`: Whether to use learnable bin edges, by default `false`.
        - `uniform_bins: bool`: Whether to use uniform binning, by default `false`.
        - `act: str`: Activation function to use after PLE layer, by default `null`.
        - `dropout: float`: Dropout rate to apply after PLE layer, by default `0.0`.
        - `layernorm: bool`: Whether to apply layer normalization after PLE layer, by default `false`.

For jagged embeddings, the following additional field is available:

- `conv1d_embedding: bool`: Whether to use a 1D convolutional layer for projecting jagged feature embeddings per object. This option will disable all other embeddings and build a 1D convolutional layer with batch normalization and ReLU activation over the feature dimension for each object, by default `false`.

!!! Note
    `reduction` specifies how to combine multiple embeddings per feature into a single embedding vector. Input to the embedding module is expected to have shape $(B, F)$ where $B$ is the batch size and $F$ is the number of features. The output of an embedding module is of shape $(B, F, E)$ where $E$ is the embedding dimension. If `reduction` is set to `mean`, the output will be averaged over the feature dimension resulting in shape $(B, E)$. If set to `reshape`, the output will be reshaped to $(B, F\cdot E)$. If set to `conv1d`, a 1D convolution will be applied over the feature dimension to combine the embeddings to get $(B, E)$. If set to `none`, the output will retain its shape $(B, F, E)$.

    For jagged features, the output of an embedding is $(B, P, F, E)$ and `reduction` is applied over the feature dimension $F$ to get $(B, P, E)$.

The `architecture_config` can also take the `post_embeddings` field to specify additional layers after the embedding layer.

- `post_embeddings: dict[str | Any] | None`: Additional layers to apply after the embedding layer, by default `null`.
    - `act: str`: Activation function to use.
    - `layernorm: bool`: Whether to apply layer normalization, by default `false`.
    - `dropout: float`: Dropout rate, by default `0.0`.
    - `batchnorm: bool`: Whether to apply batch normalization, by default `false`. Only for jagged embeddings.

!!! Note
    Embeddings are part of the model architecture and are trained jointly with the rest of the model parameters. They are defined in the [`flat_preprocessor`](https://seesawml.docs.cern.ch/api/models/preprocessing/#seesaw.models.flat_preprocessor) and [`jagged_preprocessor`](https://seesawml.docs.cern.ch/api/models/preprocessing/#seesaw.models.jagged_preprocessor) modules in the codebase. In both cases, the numerical and categorical features are handled separately and then concatenated to form the final embedding output.

    More specifically in the jagged case, the input to is of shape $K \times (B, P_k, F_k)$ where $K$ is the number of objects (e.g. jets, electrons, etc.), $B$ is the batch size, $P$ is the maximum number of particles (padded) and $F$ is the number of features per object. In this case $P$ and $F$ dimensions can vary for each object $k \in K$. The output of the embedding module is of shape $K \times (B, P_k, F_k, E)$ where $E$ is the feature embedding dimension per particle. After the per object embedding, a projection layer (`reduction`) is applied to get the final shape of $K \times (B, P_k, E)$ which is then concatenated over the object dimension to get a final sequence of shape $(B, \sum_{k}^K P_k, E)$ which is then fed into the transformer model.

### Feedforward Neural Networks

Most commonly used model architecture in SeeSawML is the feedforward neural network (FNN) also known as multi-layer perceptron (MLP). It consists of multiple fully connected layers with non-linear activation functions in between. A more advanced variant of the MLP is the ResNet architecture which includes skip connections between layers to improve gradient flow during training.

#### MLP

- `model: MLP`: Specifies the model architecture.
- `n_layers: int`: Number of hidden layers.
- `n_hidden: int`: Number of hidden units per layer.
- `act: str`: Activation function to use.
- `act_out: str | None`: Activation function for the output layer, by default `null`, which means no activation.
- `batchnorm: bool`: Whether to apply batch normalization after each layer, by default `false`.
- `dropout: float`: Dropout rate to apply after each layer, by default `0.0`.

#### ResNet

- `model: ResNet`: Specifies the model architecture.
- `n_layers: int`: Number of hidden layers.
- `n_hidden: int`: Number of hidden units per layer.
- `block_size: int`: Number of layers per residual block, by default `2`.
- `act: str`: Activation function to use.
- `act_out: str | None`: Activation function for the output layer, by default `null`, which means no activation.
- `batchnorm: bool`: Whether to apply batch normalization after each layer, by default `true`.
- `dropout: float`: Dropout rate to apply after each layer, by default `0.1`.

!!! Info
    The code implements a pre-activation ResNet architecture where the activation and normalization layers are applied before the linear layer in each block. This has been shown to improve training stability and performance compared to the original post-activation ResNet design.

### Deep Sets

Implements deep sets architecture from [https://arxiv.org/abs/1703.06114](https://arxiv.org/abs/1703.06114). The model can be used for both flat and jagged input features. For flat features, the input is embedded using the flat embedding module and then passed through the deep sets layers. The same applies for jagged features using the jagged embedding module.

- `model: DeepSets`: Specifies the model architecture.
- `embedding_dim: int | None`: Dimension of the embedding. If `null`, defaults to the first hidden layer dimension of the model.
- `encoder_layers: int | list][int]`: Number of hidden layers in the encoder MLP. If a list is provided, it specifies the number of units per layer.
- `decoder_layers: int | list[int]`: Number of hidden layers in the decoder MLP. If a list is provided, it specifies the number of units per layer.
- `n_hidden: int | None`: Number of hidden units per layer if `encoder_layers` and `decoder_layers` are specified as integers. If `null`, must provide a list for each.
- `act: str`: Activation function to use.
- `act_out: str | None`: Activation function for the output layer, by default `null`, which means no activation.
- `batchnorm: bool`: Whether to apply batch normalization after each layer, by default `true`.
- `dropout: float`: Dropout rate to apply after each layer, by default `0.0`.
- `mean_pooling: bool`: Whether to use mean pooling before the decoder MLP, by default `true`. If `false` reshapes the output before feeding to MLP.
- `add_particle_types: bool`: Whether to add particle type embeddings to the jagged input features, by default `false`. For jagged model only.

!!! Info
    Deep sets architecture is designed to handle set-structured data and is permutation invariant to the order of the input features. The model consists of an encoder that processes each element in the set independently, followed by a pooling operation (mean pooling) to aggregate the information from all elements. The aggregated representation is then passed through a decoder MLP to produce the final output.

    Encoder is implemented as a stack of 1D convolutional layers with kernel size 1, which is equivalent to applying a fully connected layer to each element in the set independently. Each convolutional layer is followed by an activation function, batch normalization (if enabled), and dropout (if specified). After the encoder, mean pooling is applied across the set dimension to obtain a fixed-size representation. This representation is then fed into the decoder MLP.

    For jagged input features, the model handles variable-length sequences through masking using [`scatter_mean`](https://docs.pytorch.org/docs/stable/generated/torch.scatter_reduce.html) operation to perform mean pooling only over valid (non-padded) elements. Additionally, masking is also performed in jagged embeddings to ensure padded values do not contribute to the embedding output.

### Transformers

#### Feature Tokenizer Transformer

Implements feature tokenzier transformer from [https://arxiv.org/abs/2106.11959](https://arxiv.org/abs/2106.11959). The model is used for flat input features only. Each feature is first embedded using the embedding module described above and then passed through a series of transformer encoder layers. The output of the transformer is then pooled or a cls token is used and passed through a final MLP for classification.

<figure markdown="span">
  ![Embedding NN](../../images/ftt.png){ width="900" }
  <figcaption>Feature tokenizer transformer architecture.</figcaption>
</figure>

- `model: EventTransformer`: Specifies the model architecture.
- `embedding_dim: int`: Dimension of the embedding, by default `32`.
- `transformer_depth: int`: Number of transformer encoder layers, by default `6`.
- `heads: int`: Number of attention heads, by default `8`.
- `dim_head: int`: Dimension of each attention head, by default `16`.
- `attn_dropout: float`: Dropout rate for attention weights, by default `0.1`.
- `remove_first_attn_residual: bool`: Whether to remove the residual connection from the first attention layer, by default `false`.
- `remove_first_attn_layernorm: bool`: Whether to remove the layer normalization from the first attention layer, by default `false`.
- `ff_hidden_mult: int`: Multiplier for the hidden dimension of the feedforward layers, by default `4`.
- `ff_dropout: float`: Dropout rate for feedforward layers, by default `0.1`.
- `use_cls_token: bool`: Whether to use a cls token for classification, by default `true`. If `false`, mean pooling is used.

!!! Info

    All the implemented models use pre-norm transformer architecture where layer normalization is applied before the attention and feedforward layers. This has been shown to improve training stability and performance compared to post-norm transformers originally proposed in the "Attention is All You Need" paper.

#### Set Transformer

Implements the Set Transformer model from [https://arxiv.org/abs/1810.00825](https://arxiv.org/abs/1810.00825). The model is used for jagged input features. Each object (e.g. jet, electron, etc.) is first embedded using the jagged embedding module described above and then passed through a series of Set Transformer encoder layers. The output of the transformer is then pooled and passed through a final MLP for classification. The model is permutation invariant to the order of the input objects.

- `model: SetTransformer`: Specifies the model architecture.
- `embedding_dim: int`: Dimension of the embedding, by default `64`.
- `heads: int`: Number of attention heads, by default `8`.
- `dim_head: int`: Dimension of each attention head, by default `16`.
- `attn_dropout: float`: Dropout rate for attention weights, by default `0.0`.
- `use_setnorm: bool`: Whether to use set normalization, by default `true`. If `false`, layer normalization is used instead.
- `ff_hidden_mult: int`: Multiplier for the hidden dimension of the feedforward layers, by default `4`.
- `ff_dropout: float`: Dropout rate for feedforward layers, by default `0.0`.
- `seed_strategy: str`: Strategy for selecting number of seed vectors (`pooling`, `particles` or `objects`), by default `pooling`.
- `set_predictor: dict[str | Any]`: Configuration for the final set predictor MLP.
    - `depth: int`: Number of hidden layers.
    - `act`: str`: Activation function to use.
    - `mean_pooling: bool`: Whether to use mean pooling before the MLP, by default `false`. If `false` reshapes the output before feeding to MLP.
- `encoder_depth: int`: Number of encoder layers, by default `2`.
- `decoder_depth: int`: Number of decoder layers, if number of seed vectors is greater than 1, by default `2`.
- `add_particle_types: bool`: Whether to add particle type embeddings to jagged features, by default `false`.
- `first_attn_no_residual: bool`: Whether to remove the residual connection from the first attention layer, by default `false`.

!!! Info

    Set transformer aggregates features by applying multihead attention (MHA) on a learnable set of $k$ seed vectors $S \in \mathbb{R}^{k \times d}$. Pooling by Multihead Attention (PMA) with $k$ seed vectors is defined as:
    $$
    \mathrm{PMA}(S, Z) = \mathrm{MHA}(S, Z, Z)
    $$
    where $Z \in \mathbb{R}^{n \times d}$ is the encoder output. If $k=1$, the output is pooled to a single vector, otherwise the output is of shape $k \times d$.

    Set transformer is build with an encoder and a decoder. The encoder, $X \rightarrow Z \in \mathbb{R}^{n \times d}$, consists of multiple attention blocks. After the encoder transforms data $X$ into features $Z$ and PMA performs pooling into representation $P\in \mathbb{R}^{k \times d}$, the decoder aggregates them into a single or a set of vectors which is fed into a feed-forward network to get final output.

    The final model can be summarized as (omitting batch dimension):
    $$
    \mathrm{Encoder}:\quad \mathrm{MHA}(X, X, X) \rightarrow Z \in \mathbb{R}^{n \times d}
    $$
    $$
    \mathrm{Pooling}:\quad \mathrm{PMA}(S, Z, Z) \rightarrow P \in \mathbb{R}^{k \times d}
    $$
    $$
    \mathrm{Decoder}:\quad \mathrm{MHA}(P, P, P) \rightarrow Y \in \mathbb{R}^{k \times d} \text{ if } k>1 \text{ else } Y = P
    $$
    $$
    \mathrm{Reshaping}:\quad Y \rightarrow Y^\prime \in \mathbb{R}^{1\times k \cdot d} (\text{ or pooled } \mathbb{R}^{1\times d}) \text{ if } k>1 \text{ else } Y^\prime = Y
    $$
    $$
    \mathrm{Prediction}:\quad \mathrm{FFN}(Y^\prime) \rightarrow \mathrm{output} \in \mathbb{R}^{1 \times o}
    $$


!!! Note
    The number of seed vectors $k$ in the implementation depends on the `seed_strategy`:

    - `pooling`: $k=1$ for single output.
    - `particles`: $k$ is set to the maximum number of particles across all objects.
    - `objects`: $k$ is set to the number of objects (e.g., jets, electrons, etc.).

    If $k>1$ a decoder layer is constructed after the encoder using `encoder_depth` that has the same structure as the encoder.

#### Particle Transformer

Implements particle transformers as described in [https://arxiv.org/abs/2202.03772](https://arxiv.org/abs/2202.03772). The model uses pairwise interactions between particles to capture complex relationships in the data through a modified attention mechanism that incorporates physics-motivated features. The model can also be viewed as a graph neural network on a fully-connected graph, in which each node corresponds to a particle, and the interactions are the edge features.

<figure markdown="span">
  ![ParT Architecture](../../images/part.png){ width="800" }
  <figcaption>Overview of the Particle Transformer model.</figcaption>
</figure>

##### Architecture Overview

The Particle Transformer consists of two main stages:

1. **Particle Attention Blocks**: Process all particles (jets and leptons) together using self-attention enhanced with pairwise interaction features. These blocks learn to model relationships between all particles in the event.

2. **Class Attention Blocks**: A learnable CLS token attends to the encoded particle representations to aggregate information for the final classification. This is similar to the CLS token in BERT-like architectures.

##### Pairwise Feature Engineering

For each pair of particles $i$ and $j$ with 4-momenta $p_i$ and $p_j$, the following physics-inspired pairwise features are computed:

- **$\Delta R$**: Angular separation in the $(\eta, \phi)$ plane:
$$
\Delta R = \sqrt{(\eta_i - \eta_j)^2 + (\phi_i - \phi_j)^2}
$$

- **$k_T$**: Relative transverse momentum (used in jet clustering algorithms):
$$
k_T = \min(p_{\mathrm{T},i}, p_{\mathrm{T},j}) \cdot \Delta R
$$

- **$z$**: Energy sharing fraction:
$$
z = \frac{\min(p_{\mathrm{T},i}, p_{\mathrm{T},j})}{p_{\mathrm{T},i} + p_{\mathrm{T},j}}
$$

- **$m^2$**: Squared invariant mass of the pair:
$$
m^2 =(E_i + E_j)^2 - ||\vec{p}_i + \vec{p}_j||^2
$$

These features are computed for all particle pairs and capture the relative kinematics between particles. Instead of using raw values, the **logarithms** of these features (with appropriate handling of negative/zero values) are passed through a small embedding network to produce pairwise bias terms that are added to the attention scores:

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + B_{\mathrm{pair}}\right)V$$

where $B_{\mathrm{pair}} \in \mathbb{R}^{N \times N \times H}$ contains the learned pairwise biases for each attention head $H$.

##### Configuration Parameters

- `model: ParticleTransformer`: Specifies the model architecture.
- `embedding_dim: int`: Dimension of the embedding, by default `64`.
- `heads: int`: Number of attention heads, by default `8`.
- `dim_head: int`: Dimension of each attention head, by default `16`.
- `attn_dropout: float`: Dropout rate for attention weights, by default `0.0`.
- `ff_dropout: float`: Dropout rate for feedforward layers, by default `0.0`.
- `ff_hidden_mult: int`: Multiplier for the hidden dimension of the feedforward layers, by default `4`.
- `particle_blocks: int`: Number of particle transformer blocks (stage 1), by default `3`.
- `class_blocks: int`: Number of classification (cls) transformer blocks (stage 2), by default `2`.
- `add_particle_types: bool`: Whether to add particle type embeddings to jagged features, by default `false`.
- `first_attn_no_residual: bool`: Whether to remove the residual connection from the first attention layer, by default `false`.
- `particle_attention: dict[str, Any]`: Configuration for particle attention layers.
    - `embedding_layers: int | list[int]`: Number of hidden layers in the pairwise embedding network. If a list is provided, it specifies the number of units per layer. By default `2`.
    - `embedding_dim: int | None`: Number of hidden units per layer if `embedding_layers` is specified as an integer. If `null`, must provide a list. By default `48`.
    - `objects: list[str] | None`: List of object names to compute pairwise interactions for. If `null`, uses all objects. By default `null`. Possible object names are those defined in the dataset configuration (e.g., `jets`, `electrons`, `muons`).
    - `quantities: list[str] | None`: List of pairwise quantities to compute. If `null`, uses all four quantities (`delta_r`, `kt`, `z`, `m2`). By default `null`.

!!! Note "Computational Considerations"
    Pairwise feature computation scales as $O(N^2)$ where $N$ is the total number of particles per event. For events with many objects (e.g., 20+ jets), this can be memory-intensive.

## Compiling Models

All models can be compiled using PyTorch's `torch.compile` feature for improved performance during training and inference. Compilation can be enabled by setting the `compile` field to `true` in the model's `architecture_config`. Additional keyword arguments for `torch.compile` can be specified in the `compile_kwargs` field. See [PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.compile.html) for more details. The options are the same for all models described above:

- `compile: bool`: Whether to compile the model using `torch.compile` for improved performance, by default `false`.
- `compile_kwargs: dict[str | Any]`: Additional keyword arguments to pass to `torch.compile`, by default `null`.

## Scaled Dot-Product Attention

SDP (scaled dot product) attention backends can be used to optimize memory usage and speed of attention computation. Depending on the hardware and input sizes, different backends may provide better performance. It is recommended to experiment with different combinations to find the optimal configuration for a given use case.

Available options are:

- `MATH` is the PyTorch C++ attention implementation.
- `FLASH_ATTENTION` is the attention implementation from the flash attention paper.
- `EFFICIENT_ATTENTION` is the implementation from the facebook xformers library.
- `CUDNN_ATTENTION` is the implementation from the Nvidia CuDNN library.

SDP can be configured in any transformer model configurations under the `architecture_config.sdp_backend` field. For example:

- `sdp_backend: dict[str, bool] | None`: Specifies the backend for scaled dot-product attention, by default `null`.
    - `enable_math: bool`: Whether to enable math kernel, by default `false`.
    - `enable_flash: bool`: Whether to enable flash attention, by default `false`.
    - `enable_mem_efficient: bool`: Whether to enable memory efficient attention, by default `false`.
    - `enable_cudnn: bool`: Whether to enable cuDNN attention, by default `false`.

See [PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) for more details.

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
