"""Command-line arguments.

The two proposed components are the two switches ``--granularity`` (how many
temporal levels the pyramid keeps) and ``--span_prior`` (whether attention is
biased toward the frames each query covers).  Everything else is standard
episodic-training plumbing, and every ablation row keeps it fixed.
"""

import argparse
import os
import sys
from pathlib import Path

from .lora import LORA_TARGET_SETS
from .subtext import GRANULARITY_MODES
from .temporal import TEMPORAL_CONTEXT_MODES

# .../MuST -- the project root, one level above this package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Where the extracted frame datasets live.  Override with MUST_DATA_ROOT (which
# scripts/env.sh sets from DATA_ROOT), or per-run with --dataset.
DATA_ROOT = Path(
    os.environ.get("MUST_DATA_ROOT", str(PROJECT_ROOT / "data" / "datasets"))
)


def _add_shared_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--dataset",
        default=str((DATA_ROOT / "hmdb_root").resolve()),
        help="Root of the extracted frames: <root>/<class_name>/<video_id>/<frame>.jpg",
    )
    parser.add_argument(
        "--split_path",
        default=str((PROJECT_ROOT / "splits" / "hmdb").resolve()),
        help="Directory holding trainlist.txt / vallist.txt / testlist.txt.",
    )
    parser.add_argument(
        "--subtext_path",
        default=str(
            (PROJECT_ROOT / "data" / "sub_texts" / "hmdb51_class_subtexts.json").resolve()
        ),
        help="JSON catalog {class_name: [sub-text, ...]} with N sub-texts per class.",
    )
    parser.add_argument(
        "--clip_model_path",
        default=str((PROJECT_ROOT / "pretrained" / "clip-vit-base-patch16").resolve()),
    )

    # --- Contribution 1: the multi-granularity pyramid -----------------------
    parser.add_argument(
        "--granularity",
        choices=sorted(GRANULARITY_MODES),
        default="pyramid",
        help=(
            "Which contiguous sub-text spans become text queries. "
            "pyramid (proposed) = all lengths 1..N; atomic = single sub-texts; "
            "full = one concatenation of all N; global = class-name prompt only."
        ),
    )
    parser.add_argument(
        "--prepend_class_name",
        action="store_true",
        help="Prefix every composed prompt with the readable class name.",
    )
    parser.add_argument(
        "--global_prompt_template",
        default="a video of a person performing {}",
        help="Prompt used by --granularity global.",
    )
    parser.add_argument("--max_text_length", type=int, default=77)

    # --- Contribution 2: the span-aligned attention prior --------------------
    parser.add_argument(
        "--span_prior",
        choices=["off", "box", "random"],
        default="box",
        help=(
            "Temporal prior on the cross-attention. box (proposed) biases each "
            "query toward the frames its span covers; off is plain attention; "
            "random keeps every span width but permutes the centres, the control "
            "that separates temporal alignment from generic regularisation."
        ),
    )

    # --- Episode setup -------------------------------------------------------
    parser.add_argument("--way", type=int, default=5)
    parser.add_argument("--eval_way", type=int, default=5)
    parser.add_argument("--shot", type=int, default=1)
    parser.add_argument("--query_per_class", "-qpc", type=int, default=6)
    parser.add_argument("--query_per_class_test", "-qpct", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=224)

    # --- Model ---------------------------------------------------------------
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--use_exclusive",
        action="store_true",
        help="Ablation: also match the (query - attended) exclusive patterns.",
    )
    parser.add_argument("--exclusive_weight", type=float, default=0.25)
    parser.add_argument("--frame_batch_size", type=int, default=64)
    parser.add_argument(
        "--clip_train_last_layers",
        type=int,
        default=0,
        help=(
            "Fine-tune the last N CLIP visual blocks. 0 (default) keeps every "
            "CLIP weight frozen and adapts through LoRA instead; set 1 or 4 for "
            "the backbone-adaptation ablation."
        ),
    )
    parser.add_argument(
        "--clip_gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "On by default because LoRA makes every visual block trainable, so "
            "activations are retained for all twelve; without checkpointing this "
            "OOMs on a 32 GB card where --clip_train_last_layers 1 fits."
        ),
    )

    # --- LoRA on the CLIP visual tower ---------------------------------------
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=8,
        help=(
            "Default backbone adaptation. r=8 on q,v adds ~295K parameters "
            "against 7.48M for --clip_train_last_layers 1, and reaches every "
            "block through a rank-r bottleneck instead of making one block fully "
            "trainable, which is the right shape when the binding constraint is "
            "overfitting rather than capacity. 0 disables LoRA."
        ),
    )
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_targets",
        choices=sorted(LORA_TARGET_SETS),
        default="qv",
        help="Which projections inside each CLIP vision block to adapt.",
    )
    parser.add_argument(
        "--lora_last_layers",
        type=int,
        default=0,
        help="0 adapts every CLIP visual block; N>0 adapts only the last N.",
    )

    # --- Motion-aware temporal context ---------------------------------------
    parser.add_argument(
        "--temporal_context",
        choices=list(TEMPORAL_CONTEXT_MODES),
        default="transformer",
        help=(
            "Frame-feature front-end. transformer (default): bottleneck "
            "Transformer over appearance plus first-order frame differences "
            "(~363K). conv: depthwise k=3 ablation branch (~1.5K). off: none. "
            "All are zero-gated, so they are identity at initialisation. "
            "Orthogonal to --span_prior: the prior decides WHERE each query "
            "looks, this decides what it reads."
        ),
    )
    parser.add_argument("--temporal_bottleneck", type=int, default=128)
    parser.add_argument("--temporal_heads", type=int, default=4)
    parser.add_argument("--temporal_layers", type=int, default=1)
    parser.add_argument("--temporal_dropout", type=float, default=0.2)
    parser.add_argument(
        "--visual_adapter",
        action="store_true",
        help="Add a second zero-gated residual MLP on the frame features.",
    )
    parser.add_argument("--adapter_bottleneck", type=int, default=128)

    # --- Runtime -------------------------------------------------------------
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--disable_horizontal_flip",
        action="store_true",
        help="Turn off random flip for direction-sensitive datasets such as SSv2.",
    )
    parser.add_argument(
        "--preload_frames",
        action="store_true",
        help="Read all frames into RAM before training.",
    )
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)


def _resolve_common(args):
    for name in ("dataset", "split_path", "subtext_path", "clip_model_path"):
        setattr(args, name, os.path.abspath(os.path.expanduser(getattr(args, name))))
    if args.num_gpus not in (0, 1):
        raise ValueError("This implementation supports one GPU or CPU")
    return args


def _dataset_tag(dataset_path: str) -> str:
    """Short name used to lay out work/<tag>/<K>-shot/<granularity>_<prior>/."""
    lowered = dataset_path.lower()
    for key, tag in (
        ("ucf", "ucf"),
        ("kinetics", "kinetics"),
        ("ssv2_small", "ssv2_small"),
        ("hmdb", "hmdb"),
    ):
        if key in lowered:
            return tag
    return Path(dataset_path).stem


def _backbone_tag(args) -> str:
    """How the visual tower is adapted, as a path segment.

    Part of the checkpoint directory so that the backbone-adaptation ablation
    rows cannot land on the same path and overwrite each other -- they differ in
    neither granularity nor span prior.
    """
    parts = []
    if int(getattr(args, "lora_rank", 0)) > 0:
        parts.append(f"lora{args.lora_rank}")
    if int(getattr(args, "clip_train_last_layers", 0)) > 0:
        parts.append(f"clip{args.clip_train_last_layers}")
    if getattr(args, "temporal_context", "off") != "off":
        parts.append(args.temporal_context)
    return "_".join(parts) if parts else "frozen"


def parse_train_args(argv=None):
    parser = argparse.ArgumentParser(description="Train MuST")
    _add_shared_arguments(parser)
    parser.add_argument("--tasks_per_batch", type=int, default=4)
    parser.add_argument("--training_iterations", type=int, default=10000)
    parser.add_argument(
        "--val_split",
        choices=["val", "test"],
        default="val",
        help=(
            "Which split the periodic in-training evaluation runs on, and "
            "therefore which split checkpoint_best.pt is selected by. 'val' "
            "(default) keeps the test split untouched until eval.py, so the "
            "reported accuracy is genuinely held out. 'test' matches the "
            "protocol used by much of the few-shot action recognition "
            "literature, but makes the resulting number a model-selection "
            "score rather than a held-out one; those runs land under a "
            "_seltest directory and the log records the choice."
        ),
    )
    parser.add_argument("--num_val_tasks", type=int, default=1000)
    parser.add_argument("--num_test_tasks", type=int, default=10000)
    parser.add_argument("--print_interval", type=int, default=100)
    parser.add_argument("--val_interval", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument(
        "--clip_learning_rate",
        type=float,
        default=None,
        help="Separate learning rate for the unfrozen CLIP visual blocks.",
    )
    parser.add_argument(
        "--lora_lr",
        type=float,
        default=1e-4,
        help=(
            "Learning rate for the LoRA group. LoRA needs about two orders more "
            "than full fine-tuning (1e-4 vs --clip_learning_rate 2e-6); reusing "
            "the fine-tuning rate is the most common way to make LoRA silently "
            "learn nothing. Watch the lora_b diagnostic."
        ),
    )
    parser.add_argument(
        "--warmup_iterations",
        type=int,
        default=1000,
        help=(
            "Linear learning-rate warm-up over the first N iterations. Applies to "
            "every parameter group, so it does not confound the backbone "
            "adaptation ablation."
        ),
    )
    parser.add_argument(
        "--scalar_lr_mult",
        type=float,
        default=10.0,
        help=(
            "Learning-rate multiplier for scalar and 1-D parameters (gates, "
            "span prior, logit scale), which are also exempted from weight "
            "decay. They converge far more slowly than the matrices."
        ),
    )
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pretrained_checkpoint", "-pc", default=None)
    args = _resolve_common(parser.parse_args(argv))

    if args.checkpoint_dir is None:
        args.checkpoint_dir = str(
            PROJECT_ROOT
            / "work"
            / _dataset_tag(args.dataset)
            / f"{args.shot}-shot"
            / (
                f"{args.granularity}_{args.span_prior}_{_backbone_tag(args)}"
                # Selecting on test is a different protocol, not a different
                # hyper-parameter: keep its runs on a separate path so the two
                # can never be confused for one another.
                + ("_seltest" if getattr(args, "val_split", "val") == "test" else "")
            )
        )
    args.checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    return args


def parse_eval_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate MuST")
    _add_shared_arguments(parser)
    parser.add_argument("--pretrained_checkpoint", "-pc", required=True)
    parser.add_argument("--num_test_tasks", type=int, default=10000)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--config_from_checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Take the model-shaping arguments (granularity, span prior, LoRA, "
            "temporal context, ...) from the checkpoint's own stored args, so "
            "evaluation cannot silently disagree with training. Anything passed "
            "explicitly on the command line still wins."
        ),
    )
    args = _resolve_common(parser.parse_args(argv))
    args.pretrained_checkpoint = os.path.abspath(args.pretrained_checkpoint)
    if args.config_from_checkpoint:
        adopt_checkpoint_config(args, parser, argv)
    if args.output is None:
        args.output = os.path.join(
            os.path.dirname(args.pretrained_checkpoint),
            f"eval_{args.eval_way}way_{args.shot}shot.txt",
        )
    return args


# Arguments that change the module tree or the text-query buffer. Evaluating with
# any of these different from training either fails to load the checkpoint or,
# worse, silently measures a different model, so they are read back from the
# checkpoint rather than retyped.
MODEL_SHAPING_ARGS = (
    "granularity",
    "prepend_class_name",
    "global_prompt_template",
    "max_text_length",
    "span_prior",
    "seq_len",
    "num_heads",
    "dropout",
    "use_exclusive",
    "exclusive_weight",
    "clip_train_last_layers",
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    "lora_targets",
    "lora_last_layers",
    "temporal_context",
    "temporal_bottleneck",
    "temporal_heads",
    "temporal_layers",
    "temporal_dropout",
    "visual_adapter",
    "adapter_bottleneck",
    "seed",
)


def _explicit_flags(parser, argv):
    """Which destinations the user actually typed, so they keep priority."""
    if argv is None:
        argv = sys.argv[1:]
    typed = set()
    for action in parser._actions:
        for option in action.option_strings:
            for token in argv:
                if token == option or token.startswith(option + "="):
                    typed.add(action.dest)
    return typed


def adopt_checkpoint_config(args, parser, argv):
    """Copy the model-shaping arguments out of the checkpoint into ``args``."""
    import torch

    try:
        stored = torch.load(
            args.pretrained_checkpoint, map_location="cpu", weights_only=False
        ).get("args")
    except Exception as error:  # noqa: BLE001
        print(
            f"WARNING: could not read the config stored in "
            f"{args.pretrained_checkpoint} ({type(error).__name__}: {error}); "
            f"using the command-line arguments as given.",
            flush=True,
        )
        return
    if not isinstance(stored, dict):
        return

    typed = _explicit_flags(parser, argv)
    adopted = []
    for name in MODEL_SHAPING_ARGS:
        if name not in stored or name in typed or not hasattr(args, name):
            continue
        if getattr(args, name) != stored[name]:
            adopted.append(f"{name}={stored[name]!r} (was {getattr(args, name)!r})")
        setattr(args, name, stored[name])
    if adopted:
        print(
            "Adopted from the checkpoint: " + ", ".join(adopted),
            flush=True,
        )
