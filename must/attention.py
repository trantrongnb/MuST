"""Step 3: span-aligned text-to-frame cross-attention.

One multi-head cross-attention block lets each text query read the ``T`` CLIP
frame features.  Text is the Query; frames plus a fixed sinusoidal temporal
encoding are Key and Value.

Nothing in a *plain* block stops the query "nock arrow" -- the first of four
phases -- from putting most of its attention on the last frame, where the arrow
has already been released.  With ``K = 1`` support video that is easy to get
wrong and the model has no way to know it is wrong.

But the queries are not arbitrary sentences.  The LLM was asked for ``N``
*consecutive* phases, so Step 2 hands every query a temporal centre ``c`` and
half-width ``u`` (see :func:`must.subtext.span_geometry`).  Frames outside that
interval are penalised on the attention logits::

    b[m, t] = -gate[level(m)] * ( relu(|p_t - c_m| - u_m) / decay )^2

Three properties matter:

1. The bias is flat (exactly zero) inside the span, so a query is free to look
   anywhere it plausibly covers, and **the widest query covers the whole clip
   and stays unbiased by construction** -- not by a special case in the code.
2. ``gate -> 0`` recovers standard cross-attention exactly, so the model can
   reject the prior.  The learned gate is itself a reportable result: how much
   temporal prior each granularity actually asked for.
3. The prior exists *only because of* Step 2.  Deriving ``b`` needs both a
   centre and a width per query, and only nested spans give both.

``gate`` holds one learned scalar per span length and ``decay`` one shared
scalar, so the whole prior costs exactly ``N + 1`` parameters.  ``b`` is a
constant ``[M, T]`` matrix given the geometry, so its compute cost is nil; it is
handed to ``nn.MultiheadAttention`` through ``attn_mask`` as a float tensor,
which PyTorch adds to the attention logits.
"""

import math
import random
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

SPAN_PRIOR_MODES = ("off", "box", "random")


def inverse_softplus(value: float) -> float:
    """Raw value whose softplus equals ``value``.

    ``gate`` and ``decay`` are stored raw and passed through ``softplus`` to keep
    them positive, so ``gate`` approaches zero asymptotically rather than
    reaching it.
    """
    return math.log(math.expm1(value))


def shuffle_centers_within_level(
    centers: torch.Tensor,
    levels: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    """Control condition: keep every span's width, permute only its position.

    Each query then carries exactly the same amount of attention constraint as
    in the aligned model, but pointed at the wrong part of the timeline.  If the
    aligned prior cannot beat this control, the gain is generic regularisation
    rather than temporal alignment, and the component does not stand.
    """
    generator = random.Random(seed)
    shuffled = centers.clone()
    for level in levels.unique().tolist():
        index = (levels == level).nonzero(as_tuple=True)[0]
        if index.numel() < 2:
            continue  # a single span at this level cannot be permuted
        order = list(range(index.numel()))
        for _ in range(16):
            generator.shuffle(order)
            if any(position != slot for slot, position in enumerate(order)):
                break
        shuffled[index] = centers[index][torch.tensor(order, dtype=torch.long)]
    return shuffled


class TemporalPositionalEncoding(nn.Module):
    """Fixed sinusoidal encoding so attention can tell frame order apart."""

    def __init__(self, dim: int, max_len: int = 32, scale: float = 0.1):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / dim)
        )
        encoding = torch.zeros(max_len, dim)
        encoding[:, 0::2] = torch.sin(position * div_term) * scale
        encoding[:, 1::2] = torch.cos(position * div_term) * scale
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.size(1) > self.encoding.size(1):
            raise ValueError(
                f"Sequence length {frames.size(1)} exceeds positional limit "
                f"{self.encoding.size(1)}"
            )
        return frames + self.encoding[:, : frames.size(1)].to(frames.dtype)


class TextToFrameAttention(nn.Module):
    """Text queries read video frames, optionally biased by the span prior."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float,
        max_frames: int,
        geometry: Optional[List[Tuple[float, float, int]]] = None,
        subtext_count: int = 0,
        span_prior: str = "off",
        span_prior_seed: int = 1234,
        gate_init: float = 0.5,
        decay_init: float = 0.25,
    ):
        super().__init__()
        self.dim = dim
        self.span_prior = span_prior
        if span_prior not in SPAN_PRIOR_MODES:
            raise ValueError(f"Unknown span_prior '{span_prior}'")

        if span_prior != "off":
            if not geometry or subtext_count < 1:
                raise ValueError("span_prior requires span geometry and subtext_count")
            centers = torch.tensor([item[0] for item in geometry], dtype=torch.float32)
            half_widths = torch.tensor([item[1] for item in geometry], dtype=torch.float32)
            # levels index the per-length gate: a span of length L -> gate[L - 1]
            levels = torch.tensor([item[2] - 1 for item in geometry], dtype=torch.long)
            if span_prior == "random":
                centers = shuffle_centers_within_level(centers, levels, span_prior_seed)
            # Geometry is frozen: only gate and decay are learned.
            self.register_buffer("span_center", centers)
            self.register_buffer("span_half_width", half_widths)
            self.register_buffer("span_level", levels)
            self.prior_gate_raw = nn.Parameter(
                torch.full((subtext_count,), inverse_softplus(gate_init))
            )
            self.prior_decay_raw = nn.Parameter(
                torch.tensor(inverse_softplus(decay_init))
            )

        self.position = TemporalPositionalEncoding(dim, max_frames)
        self.query_norm = nn.LayerNorm(dim)
        self.frame_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def span_bias(self, frame_count: int, device) -> Optional[torch.Tensor]:
        """Additive attention bias ``[M, T]``; ``None`` when the prior is off.

        One ``[M, T]`` matrix covers the whole episode: ``nn.MultiheadAttention``
        broadcasts it over batch and heads.
        """
        if self.span_prior == "off":
            return None
        positions = (
            torch.arange(frame_count, device=device, dtype=torch.float32) + 0.5
        ) / frame_count
        # How far each frame falls outside the span, zero when it is inside.
        outside = (
            (positions.unsqueeze(0) - self.span_center.unsqueeze(1)).abs()
            - self.span_half_width.unsqueeze(1)
        ).clamp(min=0.0)
        gate = F.softplus(self.prior_gate_raw)[self.span_level].unsqueeze(1)
        decay = F.softplus(self.prior_decay_raw).clamp(min=1e-4)
        return -gate * (outside / decay).pow(2)

    def attend(
        self,
        frames: torch.Tensor,
        queries: torch.Tensor,
        return_attention: bool = False,
    ):
        """``frames [B,T,D]``, ``queries [B,M,D]`` -> readout ``[B,M,D]``."""
        if frames.ndim != 3 or queries.ndim != 3:
            raise ValueError(
                f"Expected frames [B,T,D] and queries [B,M,D], got "
                f"{tuple(frames.shape)} and {tuple(queries.shape)}"
            )
        if frames.size(0) != queries.size(0):
            raise ValueError(
                f"Batch mismatch: frames {frames.size(0)} vs queries {queries.size(0)}"
            )

        keys = self.frame_norm(self.position(frames))
        bias = self.span_bias(frames.size(1), frames.device)
        if bias is not None and bias.size(0) != queries.size(1):
            raise ValueError(
                f"Span prior covers {bias.size(0)} queries but got {queries.size(1)}"
            )
        attended, weights = self.attention(
            query=self.query_norm(queries),
            key=keys,
            value=keys,
            attn_mask=bias,
            need_weights=return_attention,
            average_attn_weights=True,
        )
        attended = self.dropout(attended)
        if return_attention:
            return attended, weights
        return attended

    def refine(self, patterns: torch.Tensor) -> torch.Tensor:
        return patterns + self.ffn(self.ffn_norm(patterns))

    def forward(self, frames: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        return self.refine(queries + self.attend(frames, queries))
