"""Frozen CLIP ViT-B/16 wrapper.

The text tower is always frozen: the ``M`` text queries per class are encoded
once at construction time and stored as a buffer, so MuST never back-propagates
into it.  Adapting it would make that buffer stale after the first step.

Three visual adaptation modes are supported:

* fully frozen (``train_visual_layers=0``, ``lora_rank=0``);
* ``train_visual_layers=N`` unfreezes the last ``N`` transformer blocks -- coarse
  and expensive (7.48M parameters for a single block);
* ``lora_rank=r`` keeps every weight frozen and adapts through rank-``r``
  bottlenecks instead (~295K at ``r=8``).  See :mod:`must.lora` for why this is
  the right shape when the binding constraint is overfitting rather than
  capacity.

The two can be combined but usually should not be; ``train.py`` warns when both
are on.
"""

import os
from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPEncoder(nn.Module):
    def __init__(
        self,
        model_path: str,
        frame_batch_size: int = 64,
        train_visual_layers: int = 0,
        gradient_checkpointing: bool = False,
        lora_rank: int = 0,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        lora_targets: str = "qv",
        lora_last_layers: int = 0,
    ):
        super().__init__()
        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"CLIP checkpoint directory does not exist: {model_path}"
            )

        from transformers import CLIPModel, CLIPTokenizer

        self.model_path = os.path.abspath(model_path)
        self.frame_batch_size = int(frame_batch_size)
        self.train_visual_layers = max(0, int(train_visual_layers))
        self.lora_rank = max(0, int(lora_rank))
        # `train_visual` controls train-mode and whether encode_images enables
        # grad. LoRA needs that, but must NOT unfreeze any full block -- doing so
        # would make the whole 86M-parameter tower trainable again and defeat the
        # point of wrapping the projections.
        self.train_visual = self.train_visual_layers > 0 or self.lora_rank > 0

        self.model = CLIPModel.from_pretrained(
            self.model_path,
            local_files_only=True,
            use_safetensors=False,
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
        )
        self.output_dim = int(self.model.config.projection_dim)

        self.model.requires_grad_(False)

        # LoRA goes in before any unfreezing, so that LoRALinear freezing its own
        # base cannot fight with a later requires_grad_(True).
        self.lora_layers = 0
        if self.lora_rank > 0:
            from .lora import inject_lora

            # Visual tower only: see inject_lora's docstring.
            self.lora_layers = inject_lora(
                self.model.vision_model,
                rank=self.lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                targets=lora_targets,
                last_layers=lora_last_layers,
            )

        if self.train_visual_layers > 0:
            layers = self.model.vision_model.encoder.layers
            if self.train_visual_layers > len(layers):
                raise ValueError(
                    f"CLIP visual has {len(layers)} layers, got "
                    f"train_visual_layers={self.train_visual_layers}"
                )
            for layer in layers[-self.train_visual_layers :]:
                layer.requires_grad_(True)
            self.model.vision_model.post_layernorm.requires_grad_(True)
            self.model.visual_projection.requires_grad_(True)

        if self.train_visual:
            if gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
        else:
            self.model.eval()

    def lora_parameter_count(self) -> int:
        return sum(p.numel() for n, p in self.named_parameters() if "lora_" in n)

    def train(self, mode: bool = True):
        """Keep the frozen towers in eval mode whatever the caller asks for."""
        if self.train_visual:
            super().train(mode)
            self.model.train(mode)
            self.model.text_model.eval()
        else:
            super().train(False)
            self.model.eval()
        return self

    def _tensor_from_output(self, output, field: str, projection=None) -> torch.Tensor:
        """Normalise the shapes ``get_*_features`` returns across transformers
        versions into a ``[B, output_dim]`` tensor."""
        if torch.is_tensor(output):
            return output
        value = getattr(output, field, None)
        if torch.is_tensor(value):
            return value
        pooled = getattr(output, "pooler_output", None)
        if torch.is_tensor(pooled):
            # Newer versions already return the projected embedding here; older
            # ones return the pre-projection pooled state.
            if pooled.size(-1) == self.output_dim:
                return pooled
            if projection is None:
                raise TypeError(
                    f"Pooled CLIP output has width {pooled.size(-1)}, expected "
                    f"{self.output_dim}, and no projection head was given"
                )
            return projection(pooled)
        raise TypeError(f"Unsupported CLIP output type: {type(output).__name__}")

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """``[B*T, 3, H, W] -> [B*T, D]``.  Chunked to bound activation memory."""
        grad_enabled = self.train_visual and torch.is_grad_enabled()
        with torch.set_grad_enabled(grad_enabled):
            outputs = []
            for chunk in images.split(self.frame_batch_size, dim=0):
                features = self.model.get_image_features(pixel_values=chunk)
                features = self._tensor_from_output(
                    features, "image_embeds", self.model.visual_projection
                )
                if not grad_enabled:
                    features = features.float()
                outputs.append(features)
            return torch.cat(outputs, dim=0)

    @torch.no_grad()
    def encode_texts(
        self,
        texts: Iterable[str],
        batch_size: int = 64,
        max_length: int = 77,
    ) -> torch.Tensor:
        """L2-normalised text embeddings on CPU, computed once at start-up."""
        text_list: List[str] = list(texts)
        if not text_list:
            return torch.empty(0, self.output_dim)

        device = next(self.model.parameters()).device
        outputs = []
        for start in range(0, len(text_list), batch_size):
            batch = text_list[start : start + batch_size]
            tokens = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            features = self.model.get_text_features(**tokens)
            features = self._tensor_from_output(
                features, "text_embeds", self.model.text_projection
            )
            outputs.append(F.normalize(features.float(), dim=-1).cpu())
        return torch.cat(outputs, dim=0)

    def count_truncated(self, texts: Iterable[str], max_length: int = 77) -> int:
        """How many prompts exceed the CLIP context window and get cut.

        Truncation is silent and destroys the pyramid: the long spans get cut
        back to the same prefix as the short ones, so the coarse granularities
        stop being distinguishable.
        """
        truncated = 0
        for text in texts:
            length = len(self.tokenizer(text, truncation=False)["input_ids"])
            if length > max_length:
                truncated += 1
        return truncated
