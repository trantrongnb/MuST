"""Smoke test for the MuST model. Loads real CLIP weights, uses random frames.

Runs on a tiny synthetic catalog so text encoding stays fast; uses the GPU when
one is visible.
"""

import argparse
import json
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from must.model import MuST  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 6
FAILURES = []


def _tiny_catalog(subtext_count: int = 4):
    """Six classes with ``subtext_count`` atomic sub-texts each, in a temp file."""
    catalog = {
        f"class_{index:02d}": [
            f"phase {step} of action {index}" for step in range(subtext_count)
        ]
        for index in range(NUM_CLASSES)
    }
    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump(catalog, handle)
    handle.close()
    return handle.name


CATALOG_PATH = _tiny_catalog()


def check(condition, message):
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        FAILURES.append(message)


def make_args(granularity="pyramid", **overrides):
    args = argparse.Namespace(
        subtext_path=CATALOG_PATH,
        clip_model_path=os.path.join(ROOT, "pretrained", "clip-vit-base-patch16"),
        granularity=granularity,
        prepend_class_name=False,
        global_prompt_template="a video of a person performing {}",
        max_text_length=77,
        seq_len=4,
        num_heads=8,
        dropout=0.1,
        use_exclusive=False,
        exclusive_weight=0.25,
        frame_batch_size=32,
        clip_train_last_layers=0,
        clip_gradient_checkpointing=False,
        span_prior="box",
        seed=1234,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def run_episode(model, way=3, shot=2, queries=4, seq_len=4):
    support = torch.randn(way * shot * seq_len, 3, 224, 224, device=DEVICE)
    target = torch.randn(queries * seq_len, 3, 224, 224, device=DEVICE)
    support_labels = torch.arange(way, device=DEVICE).repeat_interleave(shot)
    class_ids = torch.arange(way, device=DEVICE)
    return model(support, support_labels, target, class_ids)


def build(granularity="pyramid", **overrides):
    return MuST(make_args(granularity, **overrides)).to(DEVICE)


def test_shapes_and_granularity():
    print("test_shapes_and_granularity")
    for granularity, expected_m in (
        ("pyramid", 10),
        ("atomic", 4),
        ("full", 1),
        ("global", 1),
    ):
        model = build(granularity).eval()
        check(
            model.pattern_num == expected_m,
            f"{granularity}: pattern_num == {expected_m}",
        )
        check(
            tuple(model.text_queries.shape) == (NUM_CLASSES, expected_m, 512),
            f"{granularity}: text query buffer is [{NUM_CLASSES}, {expected_m}, 512]",
        )
        with torch.no_grad():
            out = run_episode(model)
        check(
            tuple(out["logits"].shape) == (4, 3),
            f"{granularity}: logits are [num_query, way]",
        )
        check(torch.isfinite(out["logits"]).all(), f"{granularity}: logits are finite")


def test_backward_and_trainable_params():
    print("test_backward_and_trainable_params")
    model = build().train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    total = sum(p.numel() for p in trainable)
    check(total > 0, f"model has trainable parameters ({total:,})")
    check(
        total < 5_000_000,
        f"trainable parameter count stays small ({total:,} < 5M)",
    )
    check(
        not model.text_queries.requires_grad,
        "text queries are a frozen buffer",
    )
    check(
        all(not p.requires_grad for p in model.clip.parameters()),
        "CLIP is fully frozen when clip_train_last_layers=0",
    )

    out = run_episode(model)
    labels = torch.tensor([0, 1, 2, 0], device=DEVICE)
    loss = model.loss({"target_labels": labels}, out)["L_ce"]
    check(torch.isfinite(loss), f"cross-entropy loss is finite ({loss.item():.4f})")
    loss.backward()
    grads = [p for p in trainable if p.grad is not None and p.grad.abs().sum() > 0]
    check(len(grads) > 0, f"gradients reach {len(grads)}/{len(trainable)} tensors")


def test_exclusive_ablation():
    print("test_exclusive_ablation")
    model = build(use_exclusive=True).eval()
    with torch.no_grad():
        out = run_episode(model)
    check(tuple(out["logits"].shape) == (4, 3), "exclusive branch keeps the logit shape")
    check(torch.isfinite(out["logits"]).all(), "exclusive logits are finite")


def test_checkpoint_roundtrip():
    print("test_checkpoint_roundtrip")
    model = build().eval()
    state = model.checkpoint_state_dict()
    check(
        not any(key.startswith("clip.model.") for key in state),
        "frozen CLIP weights are excluded from the checkpoint",
    )
    check("text_queries" in state, "text query buffer is saved with the checkpoint")

    torch.manual_seed(0)
    with torch.no_grad():
        before = run_episode(model)["logits"]
    restored = build().eval()
    restored.load_checkpoint_state_dict(state)
    torch.manual_seed(0)
    with torch.no_grad():
        after = run_episode(restored)["logits"]
    check(torch.allclose(before, after, atol=1e-5), "reloaded model reproduces logits")


def test_support_shot_invariance():
    print("test_support_shot_invariance")
    # A 1-shot episode must not depend on the order of the query videos.
    model = build().eval()
    torch.manual_seed(1)
    support = torch.randn(3 * 4, 3, 224, 224, device=DEVICE)
    target = torch.randn(2 * 4, 3, 224, 224, device=DEVICE)
    labels = torch.arange(3, device=DEVICE)
    class_ids = torch.arange(3, device=DEVICE)
    with torch.no_grad():
        straight = model(support, labels, target, class_ids)["logits"]
        swapped = model(
            support,
            labels,
            torch.cat([target[4:], target[:4]]),
            class_ids,
        )["logits"]
    check(
        torch.allclose(straight, swapped.flip(0), atol=1e-4),
        "query videos are scored independently of their order",
    )


def test_span_prior_properties():
    print("test_span_prior_properties")
    model = build(span_prior="box").eval()
    bias = model.attention.span_bias(4, DEVICE)
    check(tuple(bias.shape) == (10, 4), "bias is [M, T]")
    check((bias <= 0).all(), "bias only ever penalises, never rewards")
    check(
        torch.allclose(bias[-1], torch.zeros_like(bias[-1])),
        "widest span is unbiased on every frame, by construction",
    )
    # Zero region must widen as spans grow: count unpenalised frames per level.
    free = (bias == 0).sum(dim=1)
    levels = model.attention.span_level
    per_level = [free[levels == l].float().mean().item() for l in range(4)]
    check(
        per_level == sorted(per_level),
        f"unpenalised frames grow with span length {per_level}",
    )
    off = build(span_prior="off").eval()
    check(off.attention.span_bias(4, DEVICE) is None, "span_prior=off yields no bias")


def test_zero_gate_recovers_plain_attention():
    print("test_zero_gate_recovers_plain_attention")
    # The claim that gate -> 0 restores standard cross-attention must hold
    # exactly, otherwise the ablation is not comparing like with like.
    model = build(span_prior="box").eval()
    with torch.no_grad():
        model.attention.prior_gate_raw.fill_(-30.0)  # softplus(-30) ~ 0
        bias = model.attention.span_bias(4, DEVICE)
        check(bias.abs().max().item() < 1e-6, "gate->0 drives the bias to zero")
        torch.manual_seed(0)
        biased = run_episode(model)["logits"]
    plain = build(span_prior="off").eval()
    plain.load_state_dict(model.state_dict(), strict=False)
    with torch.no_grad():
        torch.manual_seed(0)
        reference = run_episode(plain)["logits"]
    check(
        torch.allclose(biased, reference, atol=1e-4),
        "zero gate reproduces the span_prior=off logits",
    )


def test_random_control_keeps_widths():
    print("test_random_control_keeps_widths")
    aligned = build(span_prior="box").eval()
    control = build(span_prior="random").eval()
    check(
        torch.allclose(aligned.attention.span_half_width, control.attention.span_half_width),
        "random control preserves every span width",
    )
    check(
        torch.equal(aligned.attention.span_level, control.attention.span_level),
        "random control preserves the level assignment",
    )
    check(
        not torch.allclose(aligned.attention.span_center, control.attention.span_center),
        "random control actually moves the span centers",
    )
    check(
        sorted(aligned.attention.span_center.tolist())
        == sorted(control.attention.span_center.tolist()),
        "the multiset of centers is unchanged, only the assignment differs",
    )


def test_prior_parameters_are_learned():
    print("test_prior_parameters_are_learned")
    model = build(span_prior="box").train()
    baseline = sum(p.numel() for p in build(span_prior="off").parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    check(
        total - baseline == model.subtext_count + 1,
        f"prior adds exactly N+1={model.subtext_count + 1} parameters (got {total - baseline})",
    )
    out = run_episode(model)
    loss = model.loss({"target_labels": torch.tensor([0, 1, 2, 0], device=DEVICE)}, out)["L_ce"]
    loss.backward()
    gate = model.attention.prior_gate_raw
    decay = model.attention.prior_decay_raw
    check(gate.grad is not None and gate.grad.abs().sum() > 0, "gradient reaches the gate")
    check(decay.grad is not None and decay.grad.abs().sum() > 0, "gradient reaches the decay")


def test_prior_works_for_other_catalog_sizes():
    print("test_prior_works_for_other_catalog_sizes")
    for subtext_count, expected_m in ((3, 6), (5, 15)):
        path = _tiny_catalog(subtext_count)
        model = MuST(make_args(subtext_path=path)).to(DEVICE).eval()
        check(model.pattern_num == expected_m, f"N={subtext_count} gives M={expected_m}")
        check(
            tuple(model.attention.prior_gate_raw.shape) == (subtext_count,),
            f"N={subtext_count}: one gate per span length",
        )
        bias = model.attention.span_bias(8, DEVICE)
        check(
            torch.allclose(bias[-1], torch.zeros_like(bias[-1])),
            f"N={subtext_count}: widest span stays unbiased",
        )
        with torch.no_grad():
            out = run_episode(model)
        check(torch.isfinite(out["logits"]).all(), f"N={subtext_count}: logits are finite")
        os.unlink(path)


def main():
    print(f"device: {DEVICE}\n")
    for test in (
        test_shapes_and_granularity,
        test_backward_and_trainable_params,
        test_exclusive_ablation,
        test_checkpoint_roundtrip,
        test_support_shot_invariance,
        test_span_prior_properties,
        test_zero_gate_recovers_plain_attention,
        test_random_control_keeps_widths,
        test_prior_parameters_are_learned,
        test_prior_works_for_other_catalog_sizes,
    ):
        test()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed")
        raise SystemExit(1)
    print("all model checks passed")
    os.unlink(CATALOG_PATH)


if __name__ == "__main__":
    main()
