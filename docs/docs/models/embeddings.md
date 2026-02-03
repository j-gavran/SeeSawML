
# Embeddings

In SeeSawML, embeddings are used to transform input physics variables into a higher-dimensional space, enabling the model to capture complex relationships within the data. There are two types of embeddings available: flat embeddings and jagged embeddings, each separated for continuous (numerical) and discrete (categorical) features. Categorical features are always embedded using a PyTorch [`nn.Embedding`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html) layer, while for numerical features there are three options:

1. Linear transformation of type `A*X + b` where `A` and `b` are learnable parameters.
2. Feature-wise linear transformation where each feature has its own linear layer.
3. Piecewise linear encoding (PLE) as described in [https://arxiv.org/abs/2203.05556](https://arxiv.org/abs/2203.05556), where multiple binning schemes are possible: learnable, uniform, or quantile-based.

<figure markdown="span">
  [![Embedding NN](../images/mlp_emb.png){ width="800" }](https://towardsdatascience.com/improving-tabtransformer-part-1-linear-numerical-embeddings-dbc3be3b5bb5/)
  <figcaption>Embeddings with both numerical and categorical features.</figcaption>
</figure>

Embeddings can be used with any model architecture in SeeSawML and they are part of the model. They are not limited to specific model types. They serve as a preprocessing step to transform input features before they are fed into the main model architecture. If `embedding_dim` is not specified explicitly, it defaults to the first hidden layer dimension of the model.

Embeddings can also be disabled altogether by setting `disable_embeddings: true` in the `architecture_config` for flat models only. In this case, the input features are used directly without any embedding transformation.

!!! Tip "Computing Quantile Bins for PLE"
    Quantile bins for piecewise linear encoding can be precomputed using the `calculate_quantiles` command before training the model. It can be enabled by setting the number of bins in `dataset_config.ple_bins`.

Embeddings configuration is specified under the `architecture_config.embeddings` field in the model configuration file.

## Configuration Options

- `reduction: str`: Reduction method for combining multiple feature embeddings. For flat embeddings: `mean`, `reshape`, `conv1d`, `attn` or `none`, by default `mean`. For jagged embeddings: `mean`, `reshape`, `conv2d` or `attn`, by default `mean`.

!!! Info "Reduction Methods"
    `reduction` specifies how to combine multiple embeddings per feature into a single embedding vector.
    Input to the embedding module is expected to have shape $(B, F)$ where $B$ is the batch size and $F$ is the number of features. The output of an embedding module is of shape $(B, F, E)$ where $E$ is the embedding dimension. If `reduction` is set to `mean`, the output will be averaged over the feature dimension resulting in shape $(B, E)$. If set to `reshape`, the output will be reshaped to $(B, F\cdot E)$. If set to `conv1d`, a 1D convolution will be applied over the feature dimension to combine the embeddings to get $(B, E)$. If set to `none`, the output will retain its shape $(B, F, E)$.

    For jagged features, the output of an embedding is $(B, P, F, E)$ and `reduction` is applied over the feature dimension $F$ to get $(B, P, E)$.

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

- `conv1d_embedding: bool`: Whether to use a 1D convolutional layer for projecting jagged feature embeddings per object. This option will disable all other embeddings and build a 1D convolutional layer with batch normalization and activation over the feature dimension for each object, by default `false`.

The `architecture_config` can also take the `post_embeddings` field to specify additional layers after the embedding layer.

- `post_embeddings: dict[str | Any] | None`: Additional layers to apply after the embedding layer, by default `null`.
    - `act: str`: Activation function to use.
    - `layernorm: bool`: Whether to apply layer normalization, by default `false`.
    - `dropout: float`: Dropout rate, by default `0.0`.
    - `batchnorm: bool`: Whether to apply batch normalization, by default `false`. Only for jagged embeddings.

!!! Note "Implementation Details"
    Embeddings are part of the model architecture and are trained jointly with the rest of the model parameters. They are defined in the [`flat_preprocessor`](../api/models/preprocessing.md#seesaw.models.flat_preprocessor) and [`jagged_preprocessor`](../api/models/preprocessing.md#seesaw.models.jagged_preprocessor) modules in the codebase. In both cases, the numerical and categorical features are handled separately and then concatenated to form the final embedding output.

    More specifically in the jagged case, the input to the embedding is of shape $K \times (B, P_k, F_k)$ where $K$ is the number of objects (e.g. jets, electrons, etc.), $B$ is the batch size, $P$ is the maximum number of particles (padded) and $F$ is the number of features per object. In this case $P$ and $F$ dimensions can vary for each object $k \in K$. The output of the embedding module is of shape $K \times (B, P_k, F_k, E)$ where $E$ is the feature embedding dimension per particle. After the per object embedding, a projection layer (`reduction`) is applied to get the final shape of $K \times (B, P_k, E)$ which is then concatenated over the object dimension to get a final sequence of shape $(B, \sum_{k}^K P_k, E)$ which is then fed into the model.
