"""MuST: Multi-granularity Sub-Text queries for few-shot action recognition.

Four steps, in order:

1. **Sub-text catalog** (offline).  An LLM decomposes each class into ``N``
   atomic phases.  Static JSON, not part of training.
2. **Multi-granularity composition.**  ``subtext.build_class_prompts`` turns
   those ``N`` sub-texts into ``M`` prompts spanning ``N`` temporal
   granularities, encoded once by the frozen CLIP text encoder and stored as a
   buffer.  No parameters, no gradients.
3. **Span-aligned cross-attention.**  One block; each text query reads the ``T``
   CLIP frame features, biased toward the frames its span actually covers.
4. **Prototype matching.**  Support patterns are averaged over shots; a query
   video is assigned the class whose prototype has the highest mean cosine
   similarity over the ``M`` granularities.  Plain cross-entropy.

Everything not needed for that story is deliberately absent: no support
adaptation, no global branch, no auxiliary losses, and no exclusive/negative
branch unless the ablation switch asks for it.

Attribute names (``clip``, ``attention``, ``text_queries``, ``logit_scale``) are
load-bearing: they are the checkpoint's state-dict keys.  Renaming them breaks
every existing checkpoint.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import TextToFrameAttention
from .backbone import CLIPEncoder
from .subtext import build_class_prompts, load_catalog, pattern_count, span_geometry
from .temporal import ZeroGatedResidual, build_temporal_context


class MuST(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.seq_len = int(args.seq_len)
        self.granularity = args.granularity
        self.use_exclusive = bool(getattr(args, "use_exclusive", False))
        self.exclusive_weight = float(getattr(args, "exclusive_weight", 0.25))

        self.class_names, catalog = load_catalog(args.subtext_path)
        self.subtext_count = len(catalog[0])
        self.pattern_num = pattern_count(self.subtext_count, self.granularity)

        self.clip = CLIPEncoder(
            args.clip_model_path,
            frame_batch_size=args.frame_batch_size,
            train_visual_layers=args.clip_train_last_layers,
            gradient_checkpointing=args.clip_gradient_checkpointing,
            lora_rank=int(getattr(args, "lora_rank", 0)),
            lora_alpha=float(getattr(args, "lora_alpha", 16.0)),
            lora_dropout=float(getattr(args, "lora_dropout", 0.05)),
            lora_targets=getattr(args, "lora_targets", "qv"),
            lora_last_layers=int(getattr(args, "lora_last_layers", 0)),
        )
        self.dim = self.clip.output_dim

        # --- Motion-aware frame features, between CLIP and the attention -----
        # Both are zero-gated, so at iteration 0 the frame features are exactly
        # raw CLIP -- which is what the frozen text query buffer was built for.
        self.temporal_context = getattr(args, "temporal_context", "off")
        self.temporal = build_temporal_context(
            self.temporal_context,
            self.dim,
            bottleneck=int(getattr(args, "temporal_bottleneck", 128)),
            heads=int(getattr(args, "temporal_heads", 4)),
            layers=int(getattr(args, "temporal_layers", 1)),
            dropout=float(getattr(args, "temporal_dropout", 0.2)),
        )
        self.visual_adapter = (
            ZeroGatedResidual(
                self.dim,
                bottleneck=int(getattr(args, "adapter_bottleneck", 128)),
                dropout=args.dropout,
            )
            if bool(getattr(args, "visual_adapter", False))
            else None
        )

        # --- Step 2: the text queries, encoded once and frozen ---------------
        prompts = build_class_prompts(
            self.class_names,
            catalog,
            granularity=self.granularity,
            global_prompt_template=args.global_prompt_template,
            prepend_class_name=bool(getattr(args, "prepend_class_name", False)),
        )
        self.prompts = prompts
        flat_prompts = [text for items in prompts for text in items]
        truncated = self.clip.count_truncated(flat_prompts, args.max_text_length)
        if truncated:
            print(
                f"WARNING: {truncated}/{len(flat_prompts)} composed prompts exceed "
                f"{args.max_text_length} CLIP tokens and will be truncated. "
                f"Run scripts/check_subtext_length.py and shorten the sub-texts.",
                flush=True,
            )
        embeddings = self.clip.encode_texts(
            flat_prompts,
            max_length=args.max_text_length,
        ).reshape(len(self.class_names), self.pattern_num, self.dim)
        self.register_buffer("text_queries", embeddings)

        # --- Step 3: the one trainable block ---------------------------------
        self.span_prior = getattr(args, "span_prior", "off")
        self.attention = TextToFrameAttention(
            dim=self.dim,
            num_heads=args.num_heads,
            dropout=args.dropout,
            max_frames=max(32, self.seq_len),
            geometry=span_geometry(self.subtext_count, self.granularity),
            subtext_count=self.subtext_count,
            span_prior=self.span_prior,
            span_prior_seed=int(getattr(args, "seed", 1234)),
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    # ------------------------------------------------------------------ parts

    def _encode_frames(self, images: torch.Tensor) -> torch.Tensor:
        """``[B*T, 3, H, W] -> [B, T, D]``, L2-normalised per frame.

        CLIP -> temporal context -> adapter -> L2. Both middle stages are
        identity at initialisation and absent entirely by default, so the
        default path is the plain normalised CLIP feature.
        """
        if images.size(0) % self.seq_len != 0:
            raise ValueError(
                f"Image count {images.size(0)} is not divisible by "
                f"seq_len={self.seq_len}"
            )
        features = self.clip.encode_images(images).float()
        features = features.reshape(-1, self.seq_len, self.dim)
        if self.temporal is not None:
            features = self.temporal(features)
        if self.visual_adapter is not None:
            features = self.visual_adapter(features)
        return F.normalize(features, dim=-1)

    def gate_values(self):
        """Learned residual gates, for the training-time diagnostics.

        Both start at exactly zero by design, so a gate still at zero after a
        few thousand iterations means that branch never activated and any claim
        resting on it is void.
        """
        gates = {}
        if self.temporal is not None:
            gates["motion"] = torch.tanh(self.temporal.gate).item()
        if self.visual_adapter is not None:
            gates["adapter"] = torch.tanh(self.visual_adapter.gate).item()
        return gates

    def _patterns(self, frames: torch.Tensor, queries: torch.Tensor):
        """Instance patterns, plus exclusive patterns when the ablation asks."""
        attended = self.attention.attend(frames, queries)
        positive = self.attention.refine(queries + attended)
        if not self.use_exclusive:
            return positive, None
        # queries - attended: what the prompt describes but the video does not show.
        negative = self.attention.refine(queries - attended)
        return positive, negative

    @staticmethod
    def _class_mean(values: torch.Tensor, labels: torch.Tensor, way: int):
        """Average the ``K`` shots of each class into one prototype."""
        outputs = []
        for class_index in range(way):
            selected = values[labels == class_index]
            if selected.numel() == 0:
                raise ValueError(f"Episode has no support for class {class_index}")
            outputs.append(selected.mean(dim=0))
        return torch.stack(outputs, dim=0)

    # --------------------------------------------------------------- episode

    def forward(
        self,
        support_images,
        support_labels,
        target_images,
        class_ids,
        target_labels=None,
    ):
        del target_labels
        class_ids = class_ids.long()
        support_labels = support_labels.long()
        way = class_ids.numel()
        if way <= 1:
            raise ValueError("MuST requires at least two episode classes")

        support_frames = self._encode_frames(support_images)
        target_frames = self._encode_frames(target_images)
        queries = self.text_queries[class_ids]  # [W, M, D]

        # --- Support side: one pattern set per class, averaged over K shots ---
        support_queries = queries[support_labels]  # [W*K, M, D]
        support_positive, support_negative = self._patterns(
            support_frames, support_queries
        )
        prototype_positive = self._class_mean(support_positive, support_labels, way)
        prototype_negative = (
            self._class_mean(support_negative, support_labels, way)
            if support_negative is not None
            else None
        )

        # --- Query side: every query video is read once per candidate class ---
        query_count = target_frames.size(0)
        repeated_frames = target_frames.repeat_interleave(way, dim=0)
        repeated_queries = queries.unsqueeze(0).expand(query_count, -1, -1, -1)
        repeated_queries = repeated_queries.reshape(
            query_count * way, self.pattern_num, self.dim
        )
        query_positive, query_negative = self._patterns(repeated_frames, repeated_queries)
        query_positive = query_positive.reshape(
            query_count, way, self.pattern_num, self.dim
        )
        if query_negative is not None:
            query_negative = query_negative.reshape(
                query_count, way, self.pattern_num, self.dim
            )

        # --- Step 4: mean cosine similarity over the M granularities ---------
        similarity = F.cosine_similarity(
            query_positive,
            prototype_positive.unsqueeze(0),
            dim=-1,
        ).mean(dim=-1)
        if self.use_exclusive:
            cross = 0.5 * (
                F.cosine_similarity(
                    query_negative, prototype_positive.unsqueeze(0), dim=-1
                ).mean(dim=-1)
                + F.cosine_similarity(
                    query_positive, prototype_negative.unsqueeze(0), dim=-1
                ).mean(dim=-1)
            )
            similarity = similarity - self.exclusive_weight * cross

        logits = similarity * self.logit_scale.exp().clamp(max=100.0)
        return {"logits": logits, "similarity": similarity}

    def loss(self, task_dict, model_dict):
        labels = task_dict["target_labels"].long()
        return {"L_ce": F.cross_entropy(model_dict["logits"], labels)}

    # ------------------------------------------------------------ checkpoints

    def checkpoint_state_dict(self):
        """Store trainable modules only; frozen CLIP is reloaded from disk."""
        trainable_clip = {
            name
            for name, parameter in self.named_parameters()
            if name.startswith("clip.model.") and parameter.requires_grad
        }
        return {
            key: value
            for key, value in self.state_dict().items()
            if not key.startswith("clip.model.") or key in trainable_clip
        }

    def load_checkpoint_state_dict(self, state_dict):
        incompatible = self.load_state_dict(state_dict, strict=False)
        invalid_missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith("clip.model.")
        ]
        if invalid_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                f"Invalid MuST checkpoint. Missing={invalid_missing}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
