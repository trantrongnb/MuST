"""LoRA for the CLIP visual tower.

Why LoRA rather than unfreezing the last N transformer blocks
-------------------------------------------------------------
``--clip_train_last_layers 1`` makes 7.48M parameters trainable -- more than
three times the whole MuST head (2.10M) -- to adapt exactly one block.  In the
5-way 1-shot regime the binding constraint is overfitting, not capacity, so
adding parameters is the wrong direction.

"Unfreeze the last N blocks" is also a coarse choice: a block is either fully
trainable or fully frozen, so adaptation is narrow and deep.  LoRA has the right
*shape* for this regime -- it reaches every block through a rank-``r``
bottleneck, so adaptation is broad and shallow.  At ``r = 8`` on the query and
value projections it adds ~295K parameters, an order below unfreezing a single
block.

The zero-initialised ``B`` matters for MuST specifically.  The text queries are
frozen CLIP text embeddings, and the span prior is calibrated against frame
features that are raw CLIP cosines.  With ``B = 0`` the wrapped layer is
bit-exact with the frozen original at step 0, so training starts from precisely
the model that the frozen text buffer was built for.

Practical note: LoRA needs a much larger learning rate than full fine-tuning --
~1e-4 rather than the 2e-6 used for ``--clip_learning_rate``.  Reusing the
fine-tuning learning rate is the most common way to make LoRA silently do
nothing, which is what the ``lora_b`` diagnostic printed by train.py is for.
"""

import math
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn

# Which projections inside a CLIP vision block to adapt.
LORA_TARGET_SETS: Dict[str, Tuple[str, ...]] = {
    "qv": ("q_proj", "v_proj"),
    "qkv": ("q_proj", "k_proj", "v_proj"),
    "qkvo": ("q_proj", "k_proj", "v_proj", "out_proj"),
    "qv_mlp": ("q_proj", "v_proj", "fc1", "fc2"),
}


class LoRALinear(nn.Module):
    """Frozen ``nn.Linear`` plus a trainable low-rank update.

        y = W0 x + (alpha / r) * B A dropout(x)

    ``lora_B`` is zero-initialised, so ``y == W0 x`` exactly at step 0.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.base = base
        self.base.requires_grad_(False)
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(self.rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        update = self.lora_dropout(values) @ self.lora_A.t() @ self.lora_B.t()
        return self.base(values) + self.scaling * update

    def extra_repr(self) -> str:
        return f"rank={self.rank}, scaling={self.scaling:g}"


def inject_lora(
    vision_model: nn.Module,
    rank: int,
    alpha: float = 16.0,
    dropout: float = 0.05,
    targets: str = "qv",
    last_layers: int = 0,
) -> int:
    """Replace the target projections with :class:`LoRALinear`.

    Returns how many linear layers were wrapped.

    Args:
        vision_model: ``CLIPModel.vision_model``.  Never pass the TEXT tower:
            MuST encodes the ``M`` prompts per class once into a frozen buffer,
            so adapting the text tower would make that buffer stale after the
            first step and would force a full catalog re-encode every iteration.
        last_layers: 0 adapts every block; N > 0 adapts only the last N.
    """
    if rank <= 0:
        return 0
    if targets not in LORA_TARGET_SETS:
        raise ValueError(
            f"Unknown LoRA target set {targets!r}; expected one of "
            f"{sorted(LORA_TARGET_SETS)}"
        )
    blocks = vision_model.encoder.layers
    if last_layers > 0:
        if last_layers > len(blocks):
            raise ValueError(
                f"CLIP visual has {len(blocks)} blocks, got "
                f"lora_last_layers={last_layers}"
            )
        blocks = blocks[-last_layers:]

    names = LORA_TARGET_SETS[targets]
    wrapped = 0
    for block in blocks:
        for holder in (block.self_attn, block.mlp):
            for name in names:
                child = getattr(holder, name, None)
                if isinstance(child, nn.Linear):
                    setattr(holder, name, LoRALinear(child, rank, alpha, dropout))
                    wrapped += 1
    if wrapped == 0:
        raise RuntimeError(
            f"LoRA target set {targets!r} matched no nn.Linear in the vision tower"
        )
    return wrapped


def lora_parameters(module: nn.Module) -> Iterable[Tuple[str, nn.Parameter]]:
    """Named parameters belonging to LoRA (matched by the ``lora_`` name prefix)."""
    for name, parameter in module.named_parameters():
        if is_lora_parameter(name):
            yield name, parameter


def is_lora_parameter(name: str) -> bool:
    return "lora_" in name


def lora_b_magnitude(module: nn.Module) -> float:
    """Mean |B| over every LoRA block -- the ``lora_b`` diagnostic.

    ``B`` starts at exactly zero by design, so a value still at zero after a few
    thousand iterations means the LoRA branch is inactive and every claim that
    rests on it is void.  Almost always a learning-rate mistake.
    """
    total = 0.0
    count = 0
    for name, parameter in module.named_parameters():
        if "lora_B" in name:
            total += parameter.detach().abs().sum().item()
            count += parameter.numel()
    return total / count if count else 0.0
