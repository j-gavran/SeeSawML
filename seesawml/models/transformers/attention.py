from functools import partial
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.nn.attention import SDPBackend

from seesawml.models.transformers.ff_blocks import GeGLUNet


class Attend(nn.Module):
    def __init__(self, heads: int = 8, dropout_p: float = 0.1, sdp_backend: dict[str, bool] | None = None) -> None:
        """Base attention module supporting both standard and scaled dot-product attention (SDPA).

        Parameters
        ----------
        heads : int, optional
            Number of attention heads, by default 8.
        dropout_p : float, optional
            Dropout probability for attention weights, by default 0.1.
        sdp_backend : dict[str, bool] | None, optional
            Dictionary specifying which SDP backends to enable. If None, standard attention is used, by default None.
            Valid keys are:
                - "enable_math"
                - "enable_flash"
                - "enable_mem_efficient"
                - "enable_cudnn"

        Note
        ----
        Forward method can optionally take mask and bias tensors to modify attention scores before softmax. Mask should
        be a boolean tensor where True values are masked (set to -inf). Bias should be an additive tensor (float) added
        to the attention scores.

        """
        super().__init__()
        self.heads = heads
        self.dropout_p = dropout_p

        if sdp_backend is not None:
            self.use_sdp = True
            self._setup_sdp_attention(sdp_backend)
        else:
            self.use_sdp = False

        self.softmax = nn.Softmax(dim=-1)

        if self.dropout_p > 0.0:
            self.dropout = nn.Dropout(dropout_p)

    def _setup_sdp_attention(self, sdp_backend: dict[str, bool]) -> None:
        str_to_backend = {
            "enable_math": SDPBackend.MATH,
            "enable_flash": SDPBackend.FLASH_ATTENTION,
            "enable_mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
            "enable_cudnn": SDPBackend.CUDNN_ATTENTION,
        }

        used_sdp_backend: dict[str, bool] = {}

        for key, is_used in sdp_backend.items():
            if key not in str_to_backend:
                raise ValueError(f"Invalid key '{key}' in sdp_backend. Valid keys are: {list(str_to_backend.keys())}.")

            used_sdp_backend[key] = is_used

        for k in str_to_backend.keys():
            if k not in used_sdp_backend:
                used_sdp_backend[k] = False

        if not any(used_sdp_backend.values()):
            raise ValueError("At least one SDP backend must be enabled in sdp_backend!")

        sdpa_backends = [str_to_backend[enable_str] for enable_str, enable in used_sdp_backend.items() if enable]

        self.sdp_context_manager = partial(torch.nn.attention.sdpa_kernel, backends=sdpa_backends)

    def sdp_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float,
        mask: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dropout_p = self.dropout_p if self.training else 0.0

        additive_mask: torch.Tensor | None = None

        if mask is not None:
            additive_mask = torch.zeros_like(mask, dtype=q.dtype)
            additive_mask = additive_mask.masked_fill(mask, float("-inf"))

        if bias is not None:
            additive_mask = bias if additive_mask is None else additive_mask + bias

        with self.sdp_context_manager():
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=additive_mask, dropout_p=dropout_p, scale=scale)

        return out

    def attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float,
        mask: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dots = torch.einsum("b h i d, b h j d -> b h i j", q, k) * scale

        if mask is not None:
            dots = dots.masked_fill(mask, float("-inf"))

        if bias is not None:
            dots = dots + bias

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
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v))

        if self.use_sdp:
            out = self.sdp_attention(q, k, v, scale, mask, bias)
        else:
            out = self.attention(q, k, v, scale, mask, bias)

        return out


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.1,
        pre_norm: bool = True,
        sdp_backend: dict[str, bool] | None = None,
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

        self.attend = Attend(heads, dropout, sdp_backend)

        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.pre_norm:
            x = self.norm(x)

        if context is None:
            context = x

        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        attn = self.attend(q, k, v, self.scale, mask, bias)

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
        sdp_backend: dict[str, bool] | None = None,
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

        self.attend = Attend(heads, attn_dropout, sdp_backend)

        self.to_out = nn.Linear(inner_dim, dim if dim_out is None else dim_out, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        mask: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.normalize_q:
            x = self.q_norm(x)

        context = self.kv_norm(context)

        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        attn = self.attend(q, k, v, self.scale, mask, bias)

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
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.pre_norm:
            x = self.norm(x)

        if context is not None:
            context = torch.cat((x, context), dim=1)
        else:
            context = x

        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        attn = self.attend(q, k, v, self.scale, mask, bias)

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
        sdp_backend: dict[str, bool] | None = None,
    ) -> None:
        super().__init__()
        self.attention_residual = attention_residual

        self.attn: nn.Module

        if class_attention:
            self.attn = ClassAttention(
                dim,
                heads=heads,
                dim_head=dim_head,
                dropout=attn_dropout,
                pre_norm=pre_norm,
                sdp_backend=sdp_backend,
            )
        else:
            self.attn = Attention(
                dim,
                heads=heads,
                dim_head=dim_head,
                dropout=attn_dropout,
                pre_norm=pre_norm,
                sdp_backend=sdp_backend,
            )

        self.ff = GeGLUNet(dim, mult=mlp_dim // dim, dropout=ff_dropout, output_dropout=False)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.attention_residual:
            x = x + self.attn(x, context, mask, bias)
        else:
            x = self.attn(x, context, mask, bias)

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
        sdp_backend: dict[str, bool] | None = None,
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
                    sdp_backend=sdp_backend,
                )
            )

        self.layers = nn.ModuleList(attn_blocks)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, context, mask, bias)
        return x


def build_adjacency_attention_mask(invalid: torch.Tensor) -> torch.Tensor:
    """Computes adjacency attention mask from valid (i, j) mask.

    Position i is batch, j is object. If object does no exist (padded), then invalid j is True.

    The mask is passed to the pre-softmax attention scores, where True values are masked (set to -inf).

    Parameters
    ----------
    invalid : torch.Tensor
        Valid mask of shape (i, j), where True indicates padded (not present) entries.

    Returns
    -------
    torch.Tensor
        Adjacency attention mask of shape (b, 1, i, j).
    """
    mask = invalid[:, :, None] | invalid[:, None, :]
    return rearrange(mask, "b i j -> b 1 i j")
