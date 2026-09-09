"""Tests for the LoRA branch and the motion-aware temporal context.

Both components are zero-initialised on purpose, so most of what needs checking
is that they are *exactly* identity at step 0 and that they can nevertheless
still learn -- the dead-saddle failure these designs exist to avoid.
"""

import argparse
import json
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from must.lora import LoRALinear, inject_lora, lora_b_magnitude  # noqa: E402
from must.model import MuST  # noqa: E402
from must.temporal import (  # noqa: E402
    TemporalContext,
    TemporalMotionTransformer,
    ZeroGatedResidual,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 6
FAILURES = []


def check(condition, message):
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        FAILURES.append(message)


def _tiny_catalog(subtext_count: int = 4):
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


def make_args(**overrides):
    args = argparse.Namespace(
        subtext_path=CATALOG_PATH,
        clip_model_path=os.path.join(ROOT, "pretrained", "clip-vit-base-patch16"),
        granularity="pyramid",
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
        lora_rank=0,
        lora_alpha=16.0,
        lora_dropout=0.05,
        lora_targets="qv",
        lora_last_layers=0,
        temporal_context="off",
        temporal_bottleneck=128,
        temporal_heads=4,
        temporal_layers=1,
        temporal_dropout=0.2,
        visual_adapter=False,
        adapter_bottleneck=128,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def build(**overrides):
    return MuST(make_args(**overrides)).to(DEVICE)


def run_episode(model, way=3, shot=2, queries=4, seq_len=4):
    support = torch.randn(way * shot * seq_len, 3, 224, 224, device=DEVICE)
    target = torch.randn(queries * seq_len, 3, 224, 224, device=DEVICE)
    support_labels = torch.arange(way, device=DEVICE).repeat_interleave(shot)
    class_ids = torch.arange(way, device=DEVICE)
    return model(support, support_labels, target, class_ids)


# --------------------------------------------------------------------- LoRA


def test_lora_linear_is_identity_at_init():
    print("test_lora_linear_is_identity_at_init")
    base = torch.nn.Linear(16, 32)
    wrapped = LoRALinear(base, rank=8, alpha=16.0, dropout=0.0)
    values = torch.randn(5, 16)
    check(
        torch.equal(wrapped(values), base(values)),
        "B=0 makes the wrapped layer bit-exact with the frozen original",
    )
    check(
        not wrapped.base.weight.requires_grad,
        "the base weight stays frozen",
    )
    check(
        wrapped.lora_A.requires_grad and wrapped.lora_B.requires_grad,
        "only the low-rank factors are trainable",
    )
    check(abs(wrapped.scaling - 2.0) < 1e-9, "scaling is alpha/r")


def test_lora_wraps_and_costs_what_it_claims():
    print("test_lora_wraps_and_costs_what_it_claims")
    model = build(lora_rank=8).eval()
    check(model.clip.lora_layers == 24, "qv over 12 blocks wraps 24 linear layers")
    count = model.clip.lora_parameter_count()
    check(
        250_000 < count < 350_000,
        f"LoRA r=8 costs ~295K parameters (got {count:,})",
    )
    frozen = build(lora_rank=0, clip_train_last_layers=1).eval()
    clip_count = sum(
        p.numel()
        for n, p in frozen.named_parameters()
        if n.startswith("clip.model.") and p.requires_grad
    )
    check(
        count < clip_count / 10,
        f"LoRA is an order below unfreezing one block ({count:,} vs {clip_count:,})",
    )
    check(
        all(
            not p.requires_grad
            for n, p in model.named_parameters()
            if n.startswith("clip.model.") and "lora_" not in n
        ),
        "every original CLIP weight stays frozen when LoRA is on",
    )


def test_lora_is_identity_at_init_end_to_end():
    print("test_lora_is_identity_at_init_end_to_end")
    torch.manual_seed(0)
    with_lora = build(lora_rank=8).eval()
    without = build(lora_rank=0).eval()
    without.load_checkpoint_state_dict(
        {k: v for k, v in with_lora.checkpoint_state_dict().items() if "lora_" not in k}
    )
    torch.manual_seed(0)
    with torch.no_grad():
        a = run_episode(with_lora)["logits"]
    torch.manual_seed(0)
    with torch.no_grad():
        b = run_episode(without)["logits"]
    check(
        torch.allclose(a, b, atol=1e-5),
        "at initialisation LoRA reproduces the frozen-CLIP logits exactly",
    )
    check(
        lora_b_magnitude(with_lora) == 0.0,
        "the lora_b diagnostic reads exactly 0 before any training",
    )


def test_lora_receives_gradient():
    print("test_lora_receives_gradient")
    model = build(lora_rank=8).train()
    out = run_episode(model)
    loss = model.loss({"target_labels": torch.tensor([0, 1, 2, 0], device=DEVICE)}, out)
    loss["L_ce"].backward()
    b_grads = [
        p.grad
        for n, p in model.named_parameters()
        if "lora_B" in n and p.grad is not None
    ]
    check(len(b_grads) > 0, f"gradient reaches {len(b_grads)} lora_B tensors")
    check(
        any(g.abs().sum().item() > 0 for g in b_grads),
        "lora_B gradient is non-zero, so the branch can leave B=0",
    )


def test_lora_checkpoint_roundtrip():
    print("test_lora_checkpoint_roundtrip")
    model = build(lora_rank=8).eval()
    state = model.checkpoint_state_dict()
    lora_keys = [k for k in state if "lora_" in k]
    check(len(lora_keys) == 48, f"both factors of 24 layers are saved ({len(lora_keys)})")
    check(
        not any(
            k.startswith("clip.model.") and "lora_" not in k for k in state
        ),
        "frozen CLIP weights are still excluded from the checkpoint",
    )
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(0.0, 0.02)
    check(lora_b_magnitude(model) > 0, "lora_b becomes non-zero once B moves")

    state = model.checkpoint_state_dict()
    torch.manual_seed(0)
    with torch.no_grad():
        before = run_episode(model)["logits"]
    restored = build(lora_rank=8).eval()
    restored.load_checkpoint_state_dict(state)
    torch.manual_seed(0)
    with torch.no_grad():
        after = run_episode(restored)["logits"]
    check(
        torch.allclose(before, after, atol=1e-5),
        "a trained LoRA branch survives the checkpoint round-trip",
    )


# ---------------------------------------------------------------- temporal


def test_zero_gated_branches_are_identity():
    print("test_zero_gated_branches_are_identity")
    values = torch.randn(2, 8, 512)
    for name, module in (
        ("motion transformer", TemporalMotionTransformer(512)),
        ("depthwise conv", TemporalContext(512)),
        ("visual adapter", ZeroGatedResidual(512)),
    ):
        module.eval()
        with torch.no_grad():
            check(
                torch.allclose(module(values), values, atol=1e-6),
                f"{name} is identity at initialisation",
            )
            check(
                float(torch.tanh(module.gate)) == 0.0,
                f"{name} gate starts at exactly 0",
            )


def test_no_dead_saddle():
    print("test_no_dead_saddle")
    # The failure this design exists to avoid: with both the gate AND the branch
    # zero-initialised, d(loss)/d(gate) and d(loss)/d(branch) are both
    # identically zero and the module never activates.
    for name, module in (
        ("motion transformer", TemporalMotionTransformer(64, bottleneck=32, heads=4)),
        ("depthwise conv", TemporalContext(64)),
    ):
        module.train()
        values = torch.randn(2, 8, 64)
        module(values).pow(2).sum().backward()
        check(
            module.gate.grad is not None and module.gate.grad.abs().item() > 0,
            f"{name}: the gate receives non-zero gradient despite starting at 0",
        )
        branch = module.up.weight if hasattr(module, "up") else module.conv.weight
        check(
            branch.abs().sum().item() > 0,
            f"{name}: the branch itself is NOT zero-initialised",
        )


def test_motion_transformer_uses_frame_order():
    print("test_motion_transformer_uses_frame_order")
    # The whole point of the difference stream: a reversed clip must produce a
    # different residual, which raw CLIP features alone cannot do.
    module = TemporalMotionTransformer(64, bottleneck=32, heads=4).eval()
    with torch.no_grad():
        module.gate.fill_(1.0)
        values = torch.randn(1, 8, 64)
        forward = module(values)
        backward = module(values.flip(1)).flip(1)
    check(
        not torch.allclose(forward, backward, atol=1e-4),
        "reversing the clip changes the motion residual",
    )


def test_temporal_context_in_the_model():
    print("test_temporal_context_in_the_model")
    for mode, low, high in (("transformer", 300_000, 420_000), ("conv", 500, 3_000)):
        model = build(temporal_context=mode).eval()
        count = sum(p.numel() for p in model.temporal.parameters())
        check(low < count < high, f"{mode}: costs {count:,} parameters")
        check(
            "motion" in model.gate_values()
            and model.gate_values()["motion"] == 0.0,
            f"{mode}: the motion diagnostic starts at 0",
        )
        with torch.no_grad():
            out = run_episode(model)
        check(torch.isfinite(out["logits"]).all(), f"{mode}: logits are finite")

    off = build(temporal_context="off").eval()
    check(off.temporal is None and off.gate_values() == {}, "off builds no module")


def test_temporal_context_is_identity_at_init_end_to_end():
    print("test_temporal_context_is_identity_at_init_end_to_end")
    # Cold-start protection: the M text queries are encoded once by frozen CLIP,
    # so at step 0 the frame features must still be the raw CLIP features.
    torch.manual_seed(0)
    with_motion = build(temporal_context="transformer", visual_adapter=True).eval()
    plain = build(temporal_context="off").eval()
    plain.load_checkpoint_state_dict(
        {
            k: v
            for k, v in with_motion.checkpoint_state_dict().items()
            if not k.startswith(("temporal.", "visual_adapter."))
        }
    )
    torch.manual_seed(0)
    with torch.no_grad():
        a = run_episode(with_motion)["logits"]
    torch.manual_seed(0)
    with torch.no_grad():
        b = run_episode(plain)["logits"]
    check(
        torch.allclose(a, b, atol=1e-5),
        "motion transformer + adapter reproduce the plain logits at step 0",
    )


def test_motion_gate_receives_gradient_in_the_model():
    print("test_motion_gate_receives_gradient_in_the_model")
    model = build(temporal_context="transformer").train()
    out = run_episode(model)
    model.loss({"target_labels": torch.tensor([0, 1, 2, 0], device=DEVICE)}, out)[
        "L_ce"
    ].backward()
    gate = model.temporal.gate
    check(
        gate.grad is not None and gate.grad.abs().item() > 0,
        "the motion gate receives gradient through the full episode loss",
    )


def test_combined_budget():
    print("test_combined_budget")
    full = build(lora_rank=8, temporal_context="transformer")
    baseline = build(clip_train_last_layers=1)
    full_count = sum(p.numel() for p in full.parameters() if p.requires_grad)
    base_count = sum(p.numel() for p in baseline.parameters() if p.requires_grad)
    check(
        full_count < base_count,
        f"LoRA + motion ({full_count:,}) is cheaper than last-1 fine-tuning "
        f"({base_count:,})",
    )


def main():
    print(f"device: {DEVICE}\n")
    for test in (
        test_lora_linear_is_identity_at_init,
        test_lora_wraps_and_costs_what_it_claims,
        test_lora_is_identity_at_init_end_to_end,
        test_lora_receives_gradient,
        test_lora_checkpoint_roundtrip,
        test_zero_gated_branches_are_identity,
        test_no_dead_saddle,
        test_motion_transformer_uses_frame_order,
        test_temporal_context_in_the_model,
        test_temporal_context_is_identity_at_init_end_to_end,
        test_motion_gate_receives_gradient_in_the_model,
        test_combined_budget,
    ):
        test()
    print()
    os.unlink(CATALOG_PATH)
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed")
        raise SystemExit(1)
    print("all LoRA and temporal-context checks passed")


if __name__ == "__main__":
    main()
