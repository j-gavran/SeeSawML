from functools import partial
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.nn.attention import SDPBackend

from seesaw.models.transformers.ff_blocks import GeGLUNet


class Attend(nn.Module):
    def __init__(
        self, heads: int = 8, dropout_p: float = 0.1, use_flash: bool = False, sdp_kwargs: dict[str, bool] | None = None
    ) -> None:
        super().__init__()
        self.heads = heads
        self.dropout_p = dropout_p
        self.use_flash = use_flash

        if self.use_flash:
            self._setup_flash_attention(sdp_kwargs)

        self.softmax = nn.Softmax(dim=-1)

        if self.dropout_p > 0.0:
            self.dropout = nn.Dropout(dropout_p)

    def _setup_flash_attention(self, sdp_kwargs: dict[str, bool] | None = None) -> None:
        if sdp_kwargs is None:
            use_sdp_kwargs = {
                "enable_flash": True,
                "enable_math": True,
                "enable_mem_efficient": True,
                "enable_cudnn": True,
            }
        else:
            use_sdp_kwargs = sdp_kwargs

        str_to_backend = {
            "enable_flash": SDPBackend.FLASH_ATTENTION,
            "enable_mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
            "enable_math": SDPBackend.MATH,
            "enable_cudnn": SDPBackend.CUDNN_ATTENTION,
        }

        sdpa_backends = [str_to_backend[enable_str] for enable_str, enable in use_sdp_kwargs.items() if enable]

        self.sdp_context_manager = partial(torch.nn.attention.sdpa_kernel, sdpa_backends)

    def flash_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dropout_p = self.dropout_p if self.training else 0.0

        if mask is not None:
            mask = ~mask

        with self.sdp_context_manager():
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p, scale=scale)

        return out

    def attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dots = torch.einsum("b h i d, b h j d -> b h i j", q, k) * scale

        if mask is not None:
            dots = dots.masked_fill(mask, float("-inf"))

        attn = torch.nan_to_num(self.softmax(dots))

        if self.dropout_p > 0.0:
            attn = self.dropout(attn)

        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)

        return out

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v))

        if self.use_flash:
            out = self.flash_attention(q, k, v, scale, mask)
        else:
            out = self.attention(q, k, v, scale, mask)

        return out


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.1,
        pre_norm: bool = True,
        use_flash: bool = False,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.pre_norm = pre_norm

        self.scale = dim_head**-0.5

        if self.pre_norm:
            self.norm = nn.LayerNorm(dim)

        inner_dim = dim_head * heads

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.attend = Attend(heads, dropout, use_flash)

        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.pre_norm:
            x = self.norm(x)

        if context is None:
            context = x

        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        attn = self.attend(q, k, v, self.scale, mask)

        out = rearrange(attn, "b h n d -> b n (h d)")

        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        attn_dropout: float = 0.1,
        normalize_q: bool = False,
        dim_out: int | None = None,
        use_flash: bool = False,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.normalize_q = normalize_q

        if self.normalize_q:
            self.q_norm = nn.LayerNorm(dim)

        self.kv_norm = nn.LayerNorm(dim)

        self.scale = dim_head**-0.5

        inner_dim = dim_head * heads

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.attend = Attend(heads, attn_dropout, use_flash)

        self.to_out = nn.Linear(inner_dim, dim if dim_out is None else dim_out, bias=False)

    def forward(self, x: torch.Tensor, context: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.normalize_q:
            x = self.q_norm(x)

        context = self.kv_norm(context)

        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        attn = self.attend(q, k, v, self.scale, attn_mask)

        out = rearrange(attn, "b h n d -> b n (h d)")

        return self.to_out(out)


class ClassAttention(Attention):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.pre_norm:
            x = self.norm(x)

        if context is not None:
            context = torch.cat((x, context), dim=1)
        else:
            context = x

        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        attn = self.attend(q, k, v, self.scale, mask)

        out = rearrange(attn, "b h n d -> b n (h d)")

        return self.to_out(out)


class AttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        class_attention: bool = False,
        attention_residual: bool = True,
        pre_norm: bool = True,
        use_flash: bool = False,
    ) -> None:
        super().__init__()
        self.attention_residual = attention_residual

        self.attn: nn.Module

        if class_attention:
            self.attn = ClassAttention(
                dim, heads=heads, dim_head=dim_head, dropout=attn_dropout, pre_norm=pre_norm, use_flash=use_flash
            )
        else:
            self.attn = Attention(
                dim, heads=heads, dim_head=dim_head, dropout=attn_dropout, pre_norm=pre_norm, use_flash=use_flash
            )

        self.ff = GeGLUNet(dim, mult=mlp_dim // dim, dropout=ff_dropout, output_dropout=False)

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.attention_residual:
            x = x + self.attn(x, context, mask)
        else:
            x = self.attn(x, context, mask)

        x = x + self.ff(x)
        return x


class AttentionEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        class_attention: bool = False,
        first_attn_no_residual: bool = False,
        first_attn_no_layernorm: bool = False,
        use_flash: bool = False,
    ) -> None:
        super().__init__()
        attn_blocks = []

        for i in range(depth):
            if first_attn_no_residual is True and i == 0:
                attention_residual = False
            else:
                attention_residual = True

            if first_attn_no_layernorm is True and i == 0:
                pre_norm = False
            else:
                pre_norm = True

            attn_blocks.append(
                AttentionBlock(
                    dim,
                    heads,
                    dim_head,
                    mlp_dim,
                    attn_dropout,
                    ff_dropout,
                    class_attention=class_attention,
                    attention_residual=attention_residual,
                    pre_norm=pre_norm,
                    use_flash=use_flash,
                )
            )

        self.layers = nn.ModuleList(attn_blocks)

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, context, mask)
        return x
