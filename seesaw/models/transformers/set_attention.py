import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn

from seesaw.models.mlp import FeedForwardLayer
from seesaw.models.transformers.attention import Attend
from seesaw.models.transformers.ff_blocks import GeGLUNet


class SetNorm(nn.Module):
    def __init__(self, feature_dim: int, eps: float = 1e-5) -> None:
        """Equivariant layer normalization for sets.

        References
        ----------
        [1] - https://arxiv.org/abs/2206.11925
        [2] - https://github.com/rajesh-lab/deep_permutation_invariant

        """
        super().__init__()
        self.eps = eps

        self.weights = nn.Parameter(torch.empty(feature_dim))
        torch.nn.init.constant_(self.weights, 1.0)

        self.biases = nn.Parameter(torch.empty(feature_dim))
        torch.nn.init.constant_(self.biases, 0.0)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b, n, f = x.shape

        if mask is None:
            mask = torch.ones((b, n), dtype=torch.bool, device=x.device)

        means = (x * mask.unsqueeze(-1)).sum(dim=[1, 2]) / (mask.sum(dim=1) * f)
        means = means.reshape(b, 1)

        std = torch.sqrt(
            ((x - means.unsqueeze(-1)).square() * mask.unsqueeze(-1)).sum(dim=[1, 2]) / (mask.sum(dim=1) * f) + self.eps
        )
        std = std.reshape(b, 1)

        out = (x - means.unsqueeze(-1)) / std.unsqueeze(-1)

        out = out * self.weights + self.biases

        return out


class MaskedLayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return super().forward(x)


class SetAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        attn_dropout: float = 0.1,
        normalize_q: bool = True,
        use_setnorm: bool = True,
        use_flash: bool = False,
    ) -> None:
        super().__init__()
        self.heads = heads

        self.scale = dim_head**-0.5

        inner_dim = dim_head * heads

        self.normalize_q = normalize_q

        if self.normalize_q:
            self.q_norm = SetNorm(dim) if use_setnorm else MaskedLayerNorm(dim)

        self.kv_norm = SetNorm(dim) if use_setnorm else MaskedLayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.attend = Attend(heads, attn_dropout, use_flash)

        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        set_q_mask: torch.Tensor | None = None,
        set_kv_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if context is None:
            context = x

        if self.normalize_q:
            x = self.q_norm(x, mask=set_q_mask)

        context = self.kv_norm(context, mask=set_kv_mask)

        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        attn = self.attend(q, k, v, self.scale, attn_mask)

        out = rearrange(attn, "b h n d -> b n (h d)")

        return self.to_out(out)


class SetMultiheadAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        normalize_q: bool = True,
        use_setnorm: bool = True,
        attention_residual: bool = True,
        use_flash: bool = False,
    ) -> None:
        super().__init__()
        self.attention_residual = attention_residual

        self.attn = SetAttention(
            dim,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            normalize_q=normalize_q,
            use_setnorm=use_setnorm,
            use_flash=use_flash,
        )
        self.ff = GeGLUNet(dim, mult=mlp_dim // dim, dropout=ff_dropout, use_layernorm=False, output_dropout=False)

        self.norm = SetNorm(dim) if use_setnorm else MaskedLayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None,
        attn_mask: torch.Tensor | None = None,
        set_q_mask: torch.Tensor | None = None,
        set_kv_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.attn(x, context=context, attn_mask=attn_mask, set_q_mask=set_q_mask, set_kv_mask=set_kv_mask)

        if self.attention_residual:
            h = x + h

        h_norm = self.norm(h, mask=set_q_mask)
        out = h + self.ff(h_norm)
        return out


class SetAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        use_setnorm: bool = True,
        attention_residual: bool = True,
        use_flash: bool = False,
    ) -> None:
        super().__init__()
        self.attn = SetMultiheadAttentionBlock(
            dim,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            normalize_q=True,
            use_setnorm=use_setnorm,
            attention_residual=attention_residual,
            use_flash=use_flash,
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        set_q_mask: torch.Tensor | None = None,
        set_kv_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.attn(x, context=context, attn_mask=attn_mask, set_q_mask=set_q_mask, set_kv_mask=set_kv_mask)


class InducedSetAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        num_inducing_points: int = 1,
        use_setnorm: bool = True,
        use_flash: bool = False,
    ) -> None:
        super().__init__()
        self.inducing_points = nn.Parameter(torch.empty(1, num_inducing_points, dim))
        nn.init.xavier_uniform_(self.inducing_points)

        self.attn_1 = SetMultiheadAttentionBlock(
            dim,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            normalize_q=False,
            use_setnorm=use_setnorm,
            use_flash=use_flash,
        )
        self.attn_2 = SetMultiheadAttentionBlock(
            dim,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            normalize_q=True,
            use_setnorm=use_setnorm,
            use_flash=use_flash,
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        set_q_mask: torch.Tensor | None = None,
        set_kv_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inducing_points = repeat(self.inducing_points, "1 m d -> b m d", b=x.shape[0])

        h = self.attn_1(inducing_points, context=x, attn_mask=attn_mask, set_q_mask=set_q_mask, set_kv_mask=set_kv_mask)
        x = self.attn_2(x, context=h, attn_mask=attn_mask, set_q_mask=set_q_mask, set_kv_mask=set_kv_mask)

        return x


class PooledMultiheadAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        num_seeds: int = 1,
        use_setnorm: bool = True,
        use_flash: bool = False,
    ) -> None:
        super().__init__()
        self.seed_vectors = nn.Parameter(torch.empty(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.seed_vectors)

        self.attn = SetMultiheadAttentionBlock(
            dim,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            normalize_q=True,
            use_setnorm=use_setnorm,
            use_flash=use_flash,
        )

    def forward(
        self,
        z: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        set_q_mask: torch.Tensor | None = None,
        set_kv_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seed_vectors = repeat(self.seed_vectors, "1 k d -> b k d", b=z.shape[0])

        z = self.attn(seed_vectors, context=z, attn_mask=attn_mask, set_q_mask=set_q_mask, set_kv_mask=set_kv_mask)

        return z


class SetPredictorNet(nn.Module):
    def __init__(
        self,
        dim: int,
        num_seeds: int,
        inner_dim: int | None = None,
        dim_out: int = 1,
        act_out: str | None = None,
        depth: int = 3,
        act: str = "ReLU",
        pool_predict: bool = False,
        disable_reshape: bool = False,
        ff_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if inner_dim is None:
            inner_dim = dim

        self.pool_predict = pool_predict

        set_predictor: list[nn.Module] = []

        if not pool_predict and not disable_reshape:
            set_predictor.append(Rearrange("b k d -> b (k d)"))

        if num_seeds > 1 and not pool_predict:
            n_in = dim * num_seeds
        else:
            n_in = dim

        for i in range(depth):
            is_last = i == depth - 1
            set_predictor.append(
                FeedForwardLayer(
                    n_in=n_in if i == 0 else inner_dim,
                    n_out=dim_out if is_last else inner_dim,
                    act=act_out if is_last else act,
                    use_batchnorm=False,
                    dropout=0.0 if is_last else ff_dropout,
                )
            )

        self.set_predictor = nn.Sequential(*set_predictor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pool_predict:
            x = x.mean(dim=1)

        return self.set_predictor(x)


class SetTransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        encoder_depth: int,
        decoder_depth: int | None,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        num_inducing_points: int | None = None,
        num_seeds: int = 1,
        dim_out: int = 1,
        act_out: str | None = None,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        act_predict: str = "ReLU",
        depth_predict: int = 2,
        pool_predict: bool = False,
        disable_decoder_masking: bool = False,
        use_predict: bool = True,
        use_setnorm: bool = True,
        first_attn_no_residual: bool = False,
        use_flash: bool = False,
    ) -> None:
        """Set Transformer: Attention-based permutation invariant neural network for sets.

        References
        ----------
        [1] - https://arxiv.org/abs/1703.06114
        [2] - https://arxiv.org/abs/1810.00825
        [3] - https://github.com/TropComplique/set-transformer
        [4] - https://github.com/juho-lee/set_transformer

        """
        super().__init__()
        self.encoder = nn.ModuleList()

        if num_inducing_points is None:
            self.use_induced = False
        else:
            self.use_induced = True

        for i in range(encoder_depth):
            if not self.use_induced:
                self.encoder.append(
                    SetAttentionBlock(
                        dim=dim,
                        heads=heads,
                        dim_head=dim_head,
                        mlp_dim=mlp_dim,
                        attn_dropout=attn_dropout,
                        ff_dropout=ff_dropout,
                        use_setnorm=use_setnorm,
                        attention_residual=False if first_attn_no_residual and i == 0 else True,
                        use_flash=use_flash,
                    )
                )
            else:
                self.encoder.append(
                    InducedSetAttentionBlock(
                        dim=dim,
                        heads=heads,
                        dim_head=dim_head,
                        mlp_dim=mlp_dim,
                        attn_dropout=attn_dropout,
                        ff_dropout=ff_dropout,
                        num_inducing_points=num_inducing_points,  # type: ignore[arg-type]
                        use_setnorm=use_setnorm,
                        use_flash=use_flash,
                    )
                )

        self.norm = SetNorm(dim) if use_setnorm else MaskedLayerNorm(dim)

        self.pooling = PooledMultiheadAttention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            num_seeds=num_seeds,
            use_setnorm=use_setnorm,
            use_flash=use_flash,
        )

        self.disable_decoder_masking = disable_decoder_masking

        if decoder_depth is None or decoder_depth == 0 or num_seeds == 1:
            self.use_decoder = False
        else:
            self.use_decoder = True

        if self.use_decoder:
            self.decoder = nn.ModuleList()
            for _ in range(decoder_depth):  # type: ignore[arg-type]
                self.decoder.append(
                    SetAttentionBlock(
                        dim=dim,
                        heads=heads,
                        dim_head=dim_head,
                        attn_dropout=attn_dropout,
                        ff_dropout=ff_dropout,
                        mlp_dim=mlp_dim,
                        use_setnorm=use_setnorm,
                        use_flash=use_flash,
                    )
                )

        self.use_predict = use_predict

        if self.use_predict:
            self.predict = SetPredictorNet(
                dim=dim,
                num_seeds=num_seeds,
                inner_dim=mlp_dim,
                dim_out=dim_out,
                act_out=act_out,
                depth=depth_predict,
                act=act_predict,
                pool_predict=pool_predict,
                ff_dropout=ff_dropout,
            )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        set_q_mask: torch.Tensor | None = None,
        set_kv_mask: torch.Tensor | None = None,
        pooling_attn_mask: torch.Tensor | None = None,
        pooling_set_q_mask: torch.Tensor | None = None,
        pooling_set_kv_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for block in self.encoder:
            if self.use_induced:
                x = block(x, attn_mask=attn_mask, set_q_mask=set_q_mask, set_kv_mask=set_kv_mask)
            else:
                x = block(x, attn_mask=attn_mask, set_q_mask=set_q_mask, set_kv_mask=set_kv_mask)

        x = self.norm(x, mask=set_q_mask)
        x = self.pooling(x, attn_mask=pooling_attn_mask, set_q_mask=pooling_set_q_mask, set_kv_mask=pooling_set_kv_mask)

        if self.use_decoder:
            for block in self.decoder:
                if self.disable_decoder_masking:
                    x = block(x, attn_mask=None, set_q_mask=None, set_kv_mask=None)
                else:
                    x = block(x, attn_mask=attn_mask, set_q_mask=set_q_mask, set_kv_mask=set_kv_mask)

        if self.use_predict:
            x = self.predict(x)

        return x


class EventsSetDecoder(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        num_seeds: int = 1,
        dim_out: int = 1,
        act_out: str | None = None,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        act_predict: str = "ReLU",
        depth_predict: int = 2,
        pool_predict: bool = False,
        disable_decoder_masking: bool = False,
        use_setnorm: bool = True,
        use_flash: bool = False,
    ) -> None:
        super().__init__()
        self.pool_predict = pool_predict
        self.disable_decoder_masking = disable_decoder_masking

        self.decoder = nn.ModuleList()
        for _ in range(depth):
            self.decoder.append(
                SetAttentionBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    mlp_dim=mlp_dim,
                    attn_dropout=attn_dropout,
                    ff_dropout=ff_dropout,
                    use_setnorm=use_setnorm,
                    use_flash=use_flash,
                )
            )

        self.predict = SetPredictorNet(
            dim=dim,
            num_seeds=num_seeds,
            inner_dim=mlp_dim,
            dim_out=dim_out,
            act_out=act_out,
            depth=depth_predict,
            act=act_predict,
            pool_predict=pool_predict,
            ff_dropout=ff_dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        x_events: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        set_q_mask: torch.Tensor | None = None,
        set_kv_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for block in self.decoder:
            if self.disable_decoder_masking:
                x = block(x, context=x_events, attn_mask=None, set_q_mask=None, set_kv_mask=None)
            else:
                x = block(x, context=x_events, attn_mask=attn_mask, set_q_mask=set_q_mask, set_kv_mask=set_kv_mask)

        x = self.predict(x)

        return x
