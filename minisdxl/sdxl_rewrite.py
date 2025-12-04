# minsdxl_unet_v2025.py
# Apache-2.0 style (derived conceptually from Diffusers SDXL UNet patterns; rewritten here)
# v2025: SDPA/FlashAttention, optional QK-Norm, safer numerics, and **native LoRA adapters**
#
# Goals:
# - Works as a drop-in UNet-like module for SDXL-style conditioning (sample, timestep, encoder_hidden_states, added_cond_kwargs)
# - Adds LoRA support WITHOUT relying on diffusers/peft internals
# - Uses torch.scaled_dot_product_attention when available (FlashAttention-like on A100)
# - Keeps default behavior conservative (no “black image” gates by default)

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Utilities / outputs
# -----------------------------


@dataclass
class UNet2DConditionOutput:
    sample: torch.Tensor


def _exists(x) -> bool:
    return x is not None


# -----------------------------
# LoRA-enabled Linear
# -----------------------------


class LinearWithLoRA(nn.Module):
    """
    nn.Linear-compatible module with optional multi-adapter LoRA.
    - Base weights: weight/bias (compatible with state_dict loading for Linear modules)
    - LoRA adapters stored per adapter_name: A (down), B (up), alpha, scale.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

        # adapter_name -> {"A":..., "B":..., "alpha":float, "scale":float}
        self._lora: Dict[str, Dict[str, Any]] = {}
        self._active_adapters: Tuple[str, ...] = tuple()
        self._global_scale: float = 1.0

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    @torch.no_grad()
    def enable_lora(
        self,
        adapter_name: str = "default",
        rank: int = 8,
        alpha: float = 16.0,
        scale: float = 1.0,
    ):
        if rank <= 0:
            return
        if adapter_name in self._lora:
            # already exists; just update scale
            self._lora[adapter_name]["scale"] = float(scale)
            return

        A = nn.Parameter(torch.empty(rank, self.in_features))
        B = nn.Parameter(torch.empty(self.out_features, rank))
        # Common init: down ~ Kaiming, up = zeros
        nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        nn.init.zeros_(B)

        self._lora[adapter_name] = {
            "A": A,
            "B": B,
            "alpha": float(alpha),
            "rank": int(rank),
            "scale": float(scale),
        }
        # register parameters so they appear in state_dict
        self.register_parameter(f"lora_A__{adapter_name}", A)
        self.register_parameter(f"lora_B__{adapter_name}", B)

        if adapter_name not in self._active_adapters:
            self._active_adapters = (*self._active_adapters, adapter_name)

    @torch.no_grad()
    def set_adapters(self, adapter_names: Tuple[str, ...], global_scale: float = 1.0):
        self._active_adapters = tuple(a for a in adapter_names if a in self._lora)
        self._global_scale = float(global_scale)

    @torch.no_grad()
    def set_lora_scale(self, adapter_name: str, scale: float):
        if adapter_name in self._lora:
            self._lora[adapter_name]["scale"] = float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if not self._active_adapters:
            return out

        # Sum active adapters
        for name in self._active_adapters:
            cfg = self._lora[name]
            A: torch.Tensor = cfg["A"]
            B: torch.Tensor = cfg["B"]
            rank: int = cfg["rank"]
            alpha: float = cfg["alpha"]
            scale: float = cfg["scale"]
            # (alpha / r) scaling is standard
            w = (alpha / float(rank)) * scale * self._global_scale
            # x @ A^T -> (.., r) then @ B^T -> (.., out)
            out = out + (x.matmul(A.t()).matmul(B.t()) * w)
        return out


def _replace_linear(module: nn.Module) -> None:
    """
    In-place replaces nn.Linear layers with LinearWithLoRA (copying weights/bias).
    Use this if you want to retrofit LoRA into an existing model.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            new = LinearWithLoRA(
                child.in_features, child.out_features, bias=child.bias is not None
            )
            with torch.no_grad():
                new.weight.copy_(child.weight)
                if child.bias is not None and new.bias is not None:
                    new.bias.copy_(child.bias)
            setattr(module, name, new)
        else:
            _replace_linear(child)


# -----------------------------
# Time embeddings (SDXL-style)
# -----------------------------


class Timesteps(nn.Module):
    def __init__(self, num_channels: int = 320):
        super().__init__()
        self.num_channels = int(num_channels)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        # timesteps: (B,) or (N,)
        half_dim = self.num_channels // 2
        # exp(-log(10000) * i / (half_dim-0))
        exponent = -math.log(10000.0) * torch.arange(
            half_dim, dtype=torch.float32, device=timesteps.device
        )
        exponent = exponent / (half_dim - 0.0)

        emb = torch.exp(exponent)[None, :] * timesteps.float()[:, None]
        return torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)


class TimestepEmbedding(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_features, out_features, bias=True)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(out_features, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        return x


# -----------------------------
# ResNet block (conservative, SDXL-ish)
# -----------------------------


class ResnetBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_shortcut: bool = True,
        temb_channels: int = 1280,
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels, eps=1e-5, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_emb_proj = nn.Linear(temb_channels, out_channels, bias=True)

        self.norm2 = nn.GroupNorm(32, out_channels, eps=1e-5, affine=True)
        self.dropout = nn.Dropout(p=0.0, inplace=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.nonlinearity = nn.SiLU()

        self.conv_shortcut = None
        if conv_shortcut and in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.nonlinearity(h)
        h = self.conv1(h)

        # time embedding add
        temb = self.nonlinearity(temb)
        temb = self.time_emb_proj(temb)[:, :, None, None]
        h = h + temb

        h = self.norm2(h)
        h = self.nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        return x + h


# -----------------------------
# Attention (SDPA / FlashAttention-like) + optional QK-Norm + LoRA-ready linears
# -----------------------------


class Attention(nn.Module):
    def __init__(
        self,
        inner_dim: int,
        cross_attention_dim: Optional[int] = None,
        num_heads: Optional[int] = None,
        dropout: float = 0.0,
        use_sdpa: bool = True,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.inner_dim = int(inner_dim)
        if num_heads is None:
            head_dim = 64
            num_heads = inner_dim // head_dim
        self.num_heads = int(num_heads)
        self.head_dim = inner_dim // self.num_heads
        assert self.head_dim * self.num_heads == inner_dim, (
            "inner_dim must be divisible by num_heads"
        )

        self.scale = self.head_dim**-0.5
        self.use_sdpa = bool(use_sdpa)
        self.qk_norm = bool(qk_norm)

        if cross_attention_dim is None:
            cross_attention_dim = inner_dim

        # LoRA-ready linears (LinearWithLoRA)
        self.to_q = LinearWithLoRA(inner_dim, inner_dim, bias=False)
        self.to_k = LinearWithLoRA(cross_attention_dim, inner_dim, bias=False)
        self.to_v = LinearWithLoRA(cross_attention_dim, inner_dim, bias=False)

        self.to_out = nn.ModuleList(
            [
                LinearWithLoRA(inner_dim, inner_dim, bias=True),
                nn.Dropout(dropout, inplace=False),
            ]
        )

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, C) -> (B, H, T, D)
        b, t, c = x.shape
        x = x.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        return x

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, H, T, D) -> (B, T, C)
        b, h, t, d = x.shape
        return x.transpose(1, 2).contiguous().view(b, t, h * d)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # hidden_states: (B, T, C)
        context = (
            encoder_hidden_states if _exists(encoder_hidden_states) else hidden_states
        )

        q = self.to_q(hidden_states)
        k = self.to_k(context)
        v = self.to_v(context)

        q = self._reshape_heads(q)
        k = self._reshape_heads(k)
        v = self._reshape_heads(v)

        if self.qk_norm:
            # Normalize last dim to stabilize attention (helps prevent mush/blur in some regimes)
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)

        if self.use_sdpa and hasattr(F, "scaled_dot_product_attention"):
            # PyTorch 2.x SDPA (FlashAttention on supported GPUs)
            # (B, H, T, D)
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=False
            )
        else:
            # Fallback
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)

        out = self._merge_heads(out)
        out = self.to_out[0](out)
        out = self.to_out[1](out)
        return out


# -----------------------------
# FeedForward (GEGLU) with LoRA-ready linears
# -----------------------------


class GEGLU(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # output split -> x1 * gelu(x2)
        self.proj = LinearWithLoRA(in_features, out_features * 2, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.proj(x)
        x1, x2 = x_proj.chunk(2, dim=-1)
        return x1 * F.gelu(x2)


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        inner = hidden_size * 4
        self.net = nn.ModuleList(
            [
                GEGLU(hidden_size, hidden_size),  # GEGLU expands internally
                nn.Dropout(p=0.0, inplace=False),
                LinearWithLoRA(inner, hidden_size, bias=True),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GEGLU produces hidden_size*4 output because proj is (hidden -> hidden*8), then gelu-gate -> hidden*4
        x = self.net[0](x)
        x = self.net[1](x)
        x = self.net[2](x)
        return x


# -----------------------------
# Transformer block (conservative defaults; optional residual gating)
# -----------------------------


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: int = 2048,
        use_sdpa: bool = True,
        qk_norm: bool = False,
        zero_init_residual: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-5, elementwise_affine=True)
        self.attn1 = Attention(
            hidden_size, None, None, 0.0, use_sdpa=use_sdpa, qk_norm=qk_norm
        )

        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-5, elementwise_affine=True)
        self.attn2 = Attention(
            hidden_size,
            cross_attention_dim,
            None,
            0.0,
            use_sdpa=use_sdpa,
            qk_norm=qk_norm,
        )

        self.norm3 = nn.LayerNorm(hidden_size, eps=1e-5, elementwise_affine=True)
        self.ff = FeedForward(hidden_size)

        # Residual gates (OFF by default to avoid “black image” failure modes)
        init = 0.0 if zero_init_residual else 1.0
        self.g_attn1 = nn.Parameter(torch.tensor(init))
        self.g_attn2 = nn.Parameter(torch.tensor(init))
        self.g_ff = nn.Parameter(torch.tensor(init))

    def forward(
        self, x: torch.Tensor, encoder_hidden_states: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # self-attn
        h = self.norm1(x)
        h = self.attn1(h)
        x = x + self.g_attn1 * h

        # cross-attn
        h = self.norm2(x)
        h = self.attn2(h, encoder_hidden_states=encoder_hidden_states)
        x = x + self.g_attn2 * h

        # ff
        h = self.norm3(x)
        h = self.ff(h)
        x = x + self.g_ff * h
        return x


# -----------------------------
# 2D Transformer wrapper
# -----------------------------


class Transformer2DModel(nn.Module):
    def __init__(
        self,
        channels: int,
        n_layers: int,
        cross_attention_dim: int = 2048,
        use_sdpa: bool = True,
        qk_norm: bool = False,
        zero_init_residual: bool = False,
    ):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels, eps=1e-6, affine=True)

        self.proj_in = LinearWithLoRA(channels, channels, bias=True)
        self.blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    hidden_size=channels,
                    cross_attention_dim=cross_attention_dim,
                    use_sdpa=use_sdpa,
                    qk_norm=qk_norm,
                    zero_init_residual=zero_init_residual,
                )
                for _ in range(n_layers)
            ]
        )
        self.proj_out = LinearWithLoRA(channels, channels, bias=True)

    def forward(
        self, x: torch.Tensor, encoder_hidden_states: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        b, c, h, w = x.shape
        residual = x
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)  # (B, T, C)
        x = self.proj_in(x)

        for blk in self.blocks:
            x = blk(x, encoder_hidden_states=encoder_hidden_states)

        x = self.proj_out(x)
        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return x + residual


# -----------------------------
# Down/Up sampling blocks
# -----------------------------


class Downsample2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class DownBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, temb_ch: int = 1280):
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(
                    in_ch, out_ch, conv_shortcut=False, temb_channels=temb_ch
                ),
                ResnetBlock2D(
                    out_ch, out_ch, conv_shortcut=False, temb_channels=temb_ch
                ),
            ]
        )
        self.downsample = Downsample2D(out_ch)

    def forward(
        self, x: torch.Tensor, temb: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        skips = []
        for r in self.resnets:
            x = r(x, temb)
            skips.append(x)
        x = self.downsample(x)
        skips.append(x)
        return x, tuple(skips)


class CrossAttnDownBlock2D(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        n_layers: int,
        temb_ch: int = 1280,
        has_downsample: bool = True,
        use_sdpa: bool = True,
        qk_norm: bool = False,
        zero_init_residual: bool = False,
    ):
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(in_ch, out_ch, conv_shortcut=True, temb_channels=temb_ch),
                ResnetBlock2D(
                    out_ch, out_ch, conv_shortcut=False, temb_channels=temb_ch
                ),
            ]
        )
        self.attns = nn.ModuleList(
            [
                Transformer2DModel(
                    out_ch,
                    n_layers,
                    use_sdpa=use_sdpa,
                    qk_norm=qk_norm,
                    zero_init_residual=zero_init_residual,
                ),
                Transformer2DModel(
                    out_ch,
                    n_layers,
                    use_sdpa=use_sdpa,
                    qk_norm=qk_norm,
                    zero_init_residual=zero_init_residual,
                ),
            ]
        )
        self.downsample = Downsample2D(out_ch) if has_downsample else None

    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        skips = []
        for r, a in zip(self.resnets, self.attns):
            x = r(x, temb)
            x = a(x, encoder_hidden_states=encoder_hidden_states)
            skips.append(x)

        if self.downsample is not None:
            x = self.downsample(x)
            skips.append(x)

        return x, tuple(skips)


class CrossAttnUpBlock2D(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        prev_out_ch: int,
        n_layers: int,
        temb_ch: int = 1280,
        use_sdpa: bool = True,
        qk_norm: bool = False,
        zero_init_residual: bool = False,
    ):
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(
                    prev_out_ch + out_ch,
                    out_ch,
                    conv_shortcut=True,
                    temb_channels=temb_ch,
                ),
                ResnetBlock2D(
                    out_ch + out_ch, out_ch, conv_shortcut=True, temb_channels=temb_ch
                ),
                ResnetBlock2D(
                    out_ch + in_ch, out_ch, conv_shortcut=True, temb_channels=temb_ch
                ),
            ]
        )
        self.attns = nn.ModuleList(
            [
                Transformer2DModel(
                    out_ch,
                    n_layers,
                    use_sdpa=use_sdpa,
                    qk_norm=qk_norm,
                    zero_init_residual=zero_init_residual,
                ),
                Transformer2DModel(
                    out_ch,
                    n_layers,
                    use_sdpa=use_sdpa,
                    qk_norm=qk_norm,
                    zero_init_residual=zero_init_residual,
                ),
                Transformer2DModel(
                    out_ch,
                    n_layers,
                    use_sdpa=use_sdpa,
                    qk_norm=qk_norm,
                    zero_init_residual=zero_init_residual,
                ),
            ]
        )
        self.upsample = Upsample2D(out_ch)

    def forward(
        self,
        x: torch.Tensor,
        skips: Tuple[torch.Tensor, ...],
        temb: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        # pop 3 skip tensors
        for r, a in zip(self.resnets, self.attns):
            s = skips[-1]
            skips = skips[:-1]
            x = torch.cat([x, s], dim=1)
            x = r(x, temb)
            x = a(x, encoder_hidden_states=encoder_hidden_states)

        x = self.upsample(x)
        return x, skips


class UpBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, prev_out_ch: int, temb_ch: int = 1280):
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(
                    prev_out_ch + out_ch,
                    out_ch,
                    conv_shortcut=True,
                    temb_channels=temb_ch,
                ),
                ResnetBlock2D(
                    out_ch + out_ch, out_ch, conv_shortcut=True, temb_channels=temb_ch
                ),
                ResnetBlock2D(
                    out_ch + in_ch, out_ch, conv_shortcut=True, temb_channels=temb_ch
                ),
            ]
        )

    def forward(
        self, x: torch.Tensor, skips: Tuple[torch.Tensor, ...], temb: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        for r in self.resnets:
            s = skips[-1]
            skips = skips[:-1]
            x = torch.cat([x, s], dim=1)
            x = r(x, temb)
        return x, skips


class UNetMidBlock2DCrossAttn(nn.Module):
    def __init__(
        self,
        channels: int,
        n_layers: int = 10,
        temb_ch: int = 1280,
        use_sdpa: bool = True,
        qk_norm: bool = False,
        zero_init_residual: bool = False,
    ):
        super().__init__()
        self.res1 = ResnetBlock2D(
            channels, channels, conv_shortcut=False, temb_channels=temb_ch
        )
        self.attn = Transformer2DModel(
            channels,
            n_layers,
            use_sdpa=use_sdpa,
            qk_norm=qk_norm,
            zero_init_residual=zero_init_residual,
        )
        self.res2 = ResnetBlock2D(
            channels, channels, conv_shortcut=False, temb_channels=temb_ch
        )

    def forward(
        self, x: torch.Tensor, temb: torch.Tensor, encoder_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        x = self.res1(x, temb)
        x = self.attn(x, encoder_hidden_states=encoder_hidden_states)
        x = self.res2(x, temb)
        return x


# -----------------------------
# UNet v2025 (minSDXL-style)
# -----------------------------


@dataclass
class UNetV2025Config:
    in_channels: int = 4
    out_channels: int = 4
    base_channels: int = 320
    sample_size: int = 128  # 1024/8
    temb_channels: int = 1280
    add_time_embed_dim: int = 256
    cross_attention_dim: int = 2048

    # Optimizations
    use_sdpa: bool = True
    qk_norm: bool = True
    zero_init_residual: bool = False  # keep False for inference stability


class UNet2DConditionModelV2025(nn.Module):
    """
    minSDXL-UNet-v2025:
    - SDPA (FlashAttention-like) attention path
    - Optional QK-Norm
    - Native LoRA support on all LinearWithLoRA modules
    """

    def __init__(self, cfg: UNetV2025Config = UNetV2025Config()):
        super().__init__()
        self.cfg = cfg

        # keep a minimal "config-like" object to satisfy some pipeline code
        # (won't perfectly match diffusers UNet config, but enough for many wrappers)
        self.config = type("config", (), {})()
        self.config.in_channels = cfg.in_channels
        self.config.addition_time_embed_dim = cfg.add_time_embed_dim
        self.config.sample_size = cfg.sample_size

        self.conv_in = nn.Conv2d(
            cfg.in_channels, cfg.base_channels, kernel_size=3, padding=1
        )

        self.time_proj = Timesteps(cfg.base_channels)
        self.time_embedding = TimestepEmbedding(
            in_features=cfg.base_channels, out_features=cfg.temb_channels
        )

        # SDXL "added conditioning" embedding: pooled text embeds (1280) + time_id embeds (6*256=1536) => 2816
        self.add_time_proj = Timesteps(cfg.add_time_embed_dim)  # per time_id element
        self.add_embedding = TimestepEmbedding(
            in_features=2816, out_features=cfg.temb_channels
        )

        # Blocks (kept close to your pasted minSDXL skeleton)
        self.down_blocks = nn.ModuleList(
            [
                DownBlock2D(in_ch=320, out_ch=320, temb_ch=cfg.temb_channels),
                CrossAttnDownBlock2D(
                    in_ch=320,
                    out_ch=640,
                    n_layers=2,
                    temb_ch=cfg.temb_channels,
                    has_downsample=True,
                    use_sdpa=cfg.use_sdpa,
                    qk_norm=cfg.qk_norm,
                    zero_init_residual=cfg.zero_init_residual,
                ),
                CrossAttnDownBlock2D(
                    in_ch=640,
                    out_ch=1280,
                    n_layers=10,
                    temb_ch=cfg.temb_channels,
                    has_downsample=False,
                    use_sdpa=cfg.use_sdpa,
                    qk_norm=cfg.qk_norm,
                    zero_init_residual=cfg.zero_init_residual,
                ),
            ]
        )

        self.mid_block = UNetMidBlock2DCrossAttn(
            channels=1280,
            n_layers=10,
            temb_ch=cfg.temb_channels,
            use_sdpa=cfg.use_sdpa,
            qk_norm=cfg.qk_norm,
            zero_init_residual=cfg.zero_init_residual,
        )

        self.up_blocks = nn.ModuleList(
            [
                CrossAttnUpBlock2D(
                    in_ch=640,
                    out_ch=1280,
                    prev_out_ch=1280,
                    n_layers=10,
                    temb_ch=cfg.temb_channels,
                    use_sdpa=cfg.use_sdpa,
                    qk_norm=cfg.qk_norm,
                    zero_init_residual=cfg.zero_init_residual,
                ),
                CrossAttnUpBlock2D(
                    in_ch=320,
                    out_ch=640,
                    prev_out_ch=1280,
                    n_layers=2,
                    temb_ch=cfg.temb_channels,
                    use_sdpa=cfg.use_sdpa,
                    qk_norm=cfg.qk_norm,
                    zero_init_residual=cfg.zero_init_residual,
                ),
                UpBlock2D(
                    in_ch=320, out_ch=320, prev_out_ch=640, temb_ch=cfg.temb_channels
                ),
            ]
        )

        self.conv_norm_out = nn.GroupNorm(32, 320, eps=1e-5, affine=True)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(320, cfg.out_channels, kernel_size=3, padding=1)

        # Build a robust map for diffusers-style LoRA key matching: "lora_unet_{module_name_with_underscores}"
        self._build_lora_key_map()

    # -----------------------------
    # LoRA control APIs
    # -----------------------------

    def iter_lora_linears(self):
        for _, m in self.named_modules():
            if isinstance(m, LinearWithLoRA):
                yield m

    @torch.no_grad()
    def enable_lora(
        self,
        rank: int = 8,
        alpha: float = 16.0,
        adapter_name: str = "default",
        scale: float = 1.0,
    ):
        for m in self.iter_lora_linears():
            m.enable_lora(
                adapter_name=adapter_name, rank=rank, alpha=alpha, scale=scale
            )

    @torch.no_grad()
    def set_adapters(self, adapter_names: Tuple[str, ...], global_scale: float = 1.0):
        for m in self.iter_lora_linears():
            m.set_adapters(adapter_names, global_scale=global_scale)

    @torch.no_grad()
    def set_lora_scale(self, adapter_name: str, scale: float):
        for m in self.iter_lora_linears():
            m.set_lora_scale(adapter_name, scale)

    def _build_lora_key_map(self):
        # maps "down_blocks_1_attns_0_blocks_0_attn2_to_q" -> module object + attribute (for LinearWithLoRA)
        self._lora_key_to_module: Dict[str, LinearWithLoRA] = {}
        for name, m in self.named_modules():
            if isinstance(m, LinearWithLoRA):
                # diffusers LoRA uses "to_out.0" => "to_out_0"
                key = name.replace(".", "_")
                self._lora_key_to_module[key] = m

    @torch.no_grad()
    def load_lora_safetensors(
        self,
        lora_path: str,
        adapter_name: str = "default",
        scale: float = 1.0,
        strict: bool = False,
    ):
        """
        Loads SDXL UNet LoRA weights from a .safetensors file.

        Supports common "diffusers/kohya" key patterns:
          - lora_unet_{module_path_with_underscores}.lora_down.weight
          - lora_unet_{module_path_with_underscores}.lora_up.weight
          - optional ...alpha

        Note: This loader targets UNet only (ignores text encoder LoRA keys).
        """
        try:
            from safetensors.torch import load_file
        except Exception as e:
            raise RuntimeError(
                "Please `pip install safetensors` to load LoRA safetensors."
            ) from e

        state = load_file(lora_path)
        # collect pairs
        pairs: Dict[str, Dict[str, torch.Tensor]] = {}

        for k, v in state.items():
            if not k.startswith("lora_unet_"):
                continue
            base = k[len("lora_unet_") :]
            if base.endswith(".lora_down.weight"):
                key = base[: -len(".lora_down.weight")]
                pairs.setdefault(key, {})["down"] = v
            elif base.endswith(".lora_up.weight"):
                key = base[: -len(".lora_up.weight")]
                pairs.setdefault(key, {})["up"] = v
            elif base.endswith(".alpha"):
                key = base[: -len(".alpha")]
                pairs.setdefault(key, {})["alpha"] = v

        loaded = 0
        missed = 0

        for key, d in pairs.items():
            mod = self._lora_key_to_module.get(key, None)
            if mod is None:
                missed += 1
                continue
            if "down" not in d or "up" not in d:
                if strict:
                    raise RuntimeError(f"LoRA key '{key}' missing up/down weights.")
                missed += 1
                continue

            down = d["down"]
            up = d["up"]
            rank = int(down.shape[0])
            alpha = float(
                d.get("alpha", torch.tensor(rank, dtype=torch.float32)).item()
            )

            mod.enable_lora(
                adapter_name=adapter_name, rank=rank, alpha=alpha, scale=scale
            )

            # Our A is (rank, in), B is (out, rank)
            A: torch.Tensor = mod._lora[adapter_name]["A"]
            B: torch.Tensor = mod._lora[adapter_name]["B"]

            if A.shape != down.shape or B.shape != up.shape:
                if strict:
                    raise RuntimeError(
                        f"Shape mismatch for '{key}': down {down.shape} vs A {A.shape}, up {up.shape} vs B {B.shape}"
                    )
                missed += 1
                continue

            A.copy_(down)
            B.copy_(up)
            loaded += 1

        # Activate this adapter everywhere it exists
        self.set_adapters((adapter_name,), global_scale=1.0)

        if strict and missed > 0:
            raise RuntimeError(
                f"LoRA load strict failed: loaded={loaded}, missed={missed}"
            )
        return {"loaded": loaded, "missed": missed}

    # -----------------------------
    # Forward
    # -----------------------------

    def forward(
        self,
        sample: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> UNet2DConditionOutput:
        if added_cond_kwargs is None:
            added_cond_kwargs = {}

        # timesteps: scalar or (B,)
        if timesteps.ndim == 0:
            timesteps = timesteps[None]
        if timesteps.shape[0] == 1 and sample.shape[0] > 1:
            timesteps = timesteps.expand(sample.shape[0])

        # time embedding
        t_emb = self.time_proj(timesteps).to(dtype=sample.dtype)
        emb = self.time_embedding(t_emb)

        # SDXL added conditioning
        text_embeds = added_cond_kwargs.get("text_embeds", None)
        time_ids = added_cond_kwargs.get("time_ids", None)
        if _exists(text_embeds) and _exists(time_ids):
            # time_ids: (B, 6) typically
            # embed each id with Timesteps(256) -> (B*6, 256) -> (B, 1536)
            flat = time_ids.flatten()
            t2 = self.add_time_proj(flat).to(dtype=sample.dtype)
            t2 = t2.reshape(text_embeds.shape[0], -1)

            add = torch.cat([text_embeds.to(dtype=sample.dtype), t2], dim=-1)
            aug = self.add_embedding(add)
            emb = emb + aug

        # stem
        x = self.conv_in(sample)

        # down
        s0 = x
        x, d0 = self.down_blocks[0](x, temb=emb)  # d0: (s1,s2,s3)
        x, d1 = self.down_blocks[1](
            x, temb=emb, encoder_hidden_states=encoder_hidden_states
        )  # (s4,s5,s6)
        x, d2 = self.down_blocks[2](
            x, temb=emb, encoder_hidden_states=encoder_hidden_states
        )  # (s7,s8)

        # mid
        x = self.mid_block(x, temb=emb, encoder_hidden_states=encoder_hidden_states)

        # up
        # Follow your original skip wiring pattern (keeps consistency with your skeleton)
        x, _ = self.up_blocks[0](
            x,
            skips=(d1[-1], d2[0], d2[1]),
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
        )
        x, _ = self.up_blocks[1](
            x,
            skips=(d0[-1], d1[0], d1[1]),
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
        )
        x, _ = self.up_blocks[2](x, skips=(s0, d0[0], d0[1]), temb=emb)

        # out
        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)

        return UNet2DConditionOutput(sample=x)


# -----------------------------
# Convenience: retrofit LoRA into old modules
# -----------------------------


def convert_all_linear_to_lora(module: nn.Module) -> None:
    """
    If you already have a (non-v2025) model with nn.Linear, call this to convert to LinearWithLoRA.
    """
    _replace_linear(module)
