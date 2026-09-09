"""Motion-aware temporal context and zero-gated adapters.

CLIP encodes each frame independently, so its features carry appearance only:
they cannot separate an action from its time-reversed version.  MuST's span
prior decides *where along the clip* each text query looks, but it cannot invent
motion information that the frame features never had.  The two are orthogonal --
the prior fixes attention, this module fixes the features attention reads.

Every module here is a **zero-gated residual**::

    h = u + tanh(g) * Branch(u),        g initialised to 0

so at iteration 0 the frame features are exactly the raw CLIP features.  That
matters for MuST because the ``M`` text queries are encoded once by frozen CLIP
into a buffer: perturbing the visual features at step 0 would mean the frozen
text queries start out matched against features that CLIP never produced.

.. warning::

   Zero-initialise the **gate only**, never the branch as well.  With both at
   zero the gradient w.r.t. each is identically zero -- a dead saddle the module
   never leaves.  This is measured, not hypothesised: in an earlier version of
   this model the motion gate sat at exactly ``0.0000`` for 8000 iterations on
   SSv2 and HMDB, i.e. the model had no motion cue at all.
   :class:`TemporalMotionTransformer` is safe because ``up`` is randomly
   initialised; :class:`TemporalContext` seeds its kernel with a
   central-difference operator for the same reason.
"""

import math

import torch
import torch.nn as nn

TEMPORAL_CONTEXT_MODES = ("off", "transformer", "conv")


class ZeroGatedResidual(nn.Module):
    """``x + tanh(g) * MLP(LN(x))`` with ``g`` initialised to zero."""

    def __init__(self, dim: int, bottleneck: int = 128, dropout: float = 0.1):
        super().__init__()
        if bottleneck <= 0:
            raise ValueError("bottleneck must be positive")
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, dim),
            nn.Dropout(dropout),
        )
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + torch.tanh(self.gate) * self.block(values)


class TemporalContext(nn.Module):
    """Depthwise temporal convolution over the frame axis, zero-gated.

    The cheap ablation branch: ~1.5K parameters and a receptive field of three
    frames.  Kept so the paper can report how much of any gain needs a full
    transformer rather than a local motion cue.
    """

    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        self.conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
            bias=False,
        )
        # The kernel must NOT be zero-initialised -- see the module warning.
        # Seed a central-difference operator (an explicit motion prior) and keep
        # ONLY the gate at zero: the features still start as raw CLIP, but the
        # gate now receives a non-zero gradient and can wake up.
        with torch.no_grad():
            self.conv.weight.zero_()
            centre = kernel_size // 2
            for offset in range(1, centre + 1):
                self.conv.weight[:, 0, centre - offset] = -0.5 / offset
                self.conv.weight[:, 0, centre + offset] = 0.5 / offset
            self.conv.weight.add_(torch.randn_like(self.conv.weight) * 0.02)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """``[..., T, D] -> [..., T, D]``"""
        leading = values.shape[:-2]
        frames, dim = values.shape[-2], values.shape[-1]
        flat = values.reshape(-1, frames, dim).transpose(1, 2)
        delta = self.conv(flat).transpose(1, 2).reshape(*leading, frames, dim)
        return values + torch.tanh(self.gate) * delta


class TemporalMotionTransformer(nn.Module):
    """Zero-gated temporal attention over appearance and explicit frame changes.

    A bottleneck Transformer reads the CLIP frame embeddings together with their
    first-order differences, then writes a motion-aware residual back into CLIP
    space::

        z_t   = W_down LN(u_t)
        d_t   = z_t - z_{t-1},  d_1 = 0
        z~    = Enc(z + phi(d) + PE)
        h_t   = u_t + tanh(g) * W_up z~_t

    Three properties matter.  (i) The difference stream supplies motion directly
    as a token feature instead of forcing attention to infer it from appearance.
    (ii) Temporal self-attention relates distant frames, where a ``k=3``
    convolution only ever sees its immediate neighbours -- needed when a phase is
    long or interrupted.  (iii) The output gate starts at zero, so ``h_t = u_t``
    at iteration 0, while ``up`` is randomly initialised so the gate still gets
    gradient.
    """

    def __init__(
        self,
        dim: int,
        bottleneck: int = 128,
        heads: int = 4,
        layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        if bottleneck <= 0:
            raise ValueError("bottleneck must be positive")
        if heads <= 0 or bottleneck % heads != 0:
            raise ValueError("bottleneck must be divisible by heads")
        if layers <= 0:
            raise ValueError("layers must be positive")

        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=bottleneck,
            nhead=heads,
            dim_feedforward=4 * bottleneck,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
            enable_nested_tensor=False,
        )
        self.motion_mixer = nn.Sequential(
            nn.LayerNorm(bottleneck),
            nn.Linear(bottleneck, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, bottleneck),
        )
        self.up = nn.Linear(bottleneck, dim)
        self.gate = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _position(length: int, dim: int, device, dtype) -> torch.Tensor:
        half = (dim + 1) // 2
        scale = -math.log(10000.0) / max(half - 1, 1)
        frequency = torch.exp(
            torch.arange(half, device=device, dtype=torch.float32) * scale
        )
        angles = torch.arange(
            length, device=device, dtype=torch.float32
        ).unsqueeze(-1) * frequency.unsqueeze(0)
        position = torch.zeros(length, dim, device=device, dtype=torch.float32)
        position[:, 0::2] = angles.sin()[:, : position[:, 0::2].shape[-1]]
        position[:, 1::2] = angles.cos()[:, : position[:, 1::2].shape[-1]]
        return position.to(dtype=dtype)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """``[..., T, D] -> [..., T, D]``"""
        leading = values.shape[:-2]
        frames, dim = values.shape[-2], values.shape[-1]
        flat = values.reshape(-1, frames, dim)
        tokens = self.down(self.norm(flat))

        motion = torch.zeros_like(tokens)
        if frames > 1:
            motion[:, 1:] = tokens[:, 1:] - tokens[:, :-1]
        motion = self.motion_mixer(motion)
        position = self._position(frames, tokens.size(-1), tokens.device, tokens.dtype)
        context = self.encoder(tokens + motion + position.unsqueeze(0))
        delta = self.up(context).reshape(*leading, frames, dim)
        return values + torch.tanh(self.gate) * delta


def build_temporal_context(
    mode: str,
    dim: int,
    bottleneck: int = 128,
    heads: int = 4,
    layers: int = 1,
    dropout: float = 0.2,
    kernel_size: int = 3,
):
    """Return the temporal context module for ``mode``, or ``None`` when off."""
    if mode not in TEMPORAL_CONTEXT_MODES:
        raise ValueError(
            f"Unknown temporal_context {mode!r}; expected one of "
            f"{list(TEMPORAL_CONTEXT_MODES)}"
        )
    if mode == "off":
        return None
    if mode == "conv":
        return TemporalContext(dim, kernel_size=kernel_size)
    return TemporalMotionTransformer(
        dim,
        bottleneck=bottleneck,
        heads=heads,
        layers=layers,
        dropout=dropout,
    )
