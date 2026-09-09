#!/usr/bin/env python3
"""Figures and read-out for the span prior.

  (a) the [M, T] attention bias matrix, whose unpenalised region widens with
      span length and goes completely flat on the last row -- the sub-text
      pyramid showing up inside attention;
  (b) the learned per-level gate read back from a trained checkpoint, i.e. how
      much temporal prior each granularity actually asked for.

Always plot from a TRAINED checkpoint. Without -pc this prints the values at
initialisation, where every gate is 0.5 by construction -- four equal bars mean
you are looking at the initialisation, not a result.

    python3 scripts/plot_span_prior.py -pc work/hmdb/1-shot/pyramid_box/checkpoint_best.pt
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from must.subtext import load_catalog, span_geometry  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def bias_matrix(geometry, frame_count, gate, decay):
    """The same formula as must.attention.TextToFrameAttention.span_bias."""
    positions = (torch.arange(frame_count, dtype=torch.float32) + 0.5) / frame_count
    centers = torch.tensor([item[0] for item in geometry])
    widths = torch.tensor([item[1] for item in geometry])
    levels = torch.tensor([item[2] - 1 for item in geometry], dtype=torch.long)
    outside = (
        (positions.unsqueeze(0) - centers.unsqueeze(1)).abs() - widths.unsqueeze(1)
    ).clamp(min=0.0)
    return -gate[levels].unsqueeze(1) * (outside / decay).pow(2), levels


def main():
    parser = argparse.ArgumentParser(description="Span prior figures")
    parser.add_argument(
        "--subtext_path",
        default=os.path.join(
            PROJECT_ROOT, "data", "sub_texts", "hmdb51_class_subtexts.json"
        ),
    )
    parser.add_argument("--granularity", default="pyramid")
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--pretrained_checkpoint", "-pc", default=None)
    parser.add_argument("--output", default=None, help="Save a PNG as well as printing.")
    args = parser.parse_args()

    _, catalog = load_catalog(args.subtext_path)
    subtext_count = len(catalog[0])
    geometry = span_geometry(subtext_count, args.granularity)

    if args.pretrained_checkpoint:
        state = torch.load(args.pretrained_checkpoint, map_location="cpu")["model_state_dict"]
        gate = F.softplus(state["attention.prior_gate_raw"]).float()
        decay = F.softplus(state["attention.prior_decay_raw"]).float()
        source = os.path.basename(args.pretrained_checkpoint)
    else:
        gate = F.softplus(torch.full((subtext_count,), math.log(math.expm1(0.5))))
        decay = F.softplus(torch.tensor(math.log(math.expm1(0.25))))
        source = "initialisation (no checkpoint given)"

    bias, _ = bias_matrix(geometry, args.seq_len, gate, decay)
    print(f"Source: {source}")
    print(f"N={subtext_count}, M={len(geometry)}, T={args.seq_len}, decay={decay:.4f}\n")

    print("Learned gate per span length:")
    for level in range(subtext_count):
        bar = "#" * int(round(gate[level].item() * 20))
        note = "  (prior is identically 0 here)" if level == subtext_count - 1 else ""
        print(f"  length {level + 1}: {gate[level]:.4f}  {bar}{note}")

    print(f"\nBias matrix [M={len(geometry)}, T={args.seq_len}]:")
    print(f"{'span':<14}" + "".join(f"{'f' + str(t + 1):>8}" for t in range(args.seq_len)))
    for row, (_, _, length) in zip(bias, geometry):
        print(f"{'length ' + str(length):<14}" + "".join(f"{v:>8.2f}" for v in row))

    if args.output:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, 2, figsize=(11, 4), width_ratios=[2, 1])
        image = axes[0].imshow(bias.numpy(), aspect="auto", cmap="viridis")
        axes[0].set_xlabel("frame")
        axes[0].set_ylabel("text query (short span -> long span)")
        axes[0].set_title("Attention bias $b[m,t]$")
        axes[0].set_xticks(range(args.seq_len), [str(t + 1) for t in range(args.seq_len)])
        axes[0].set_yticks(
            range(len(geometry)), [f"len {g[2]}" for g in geometry], fontsize=7
        )
        figure.colorbar(image, ax=axes[0])

        axes[1].bar(range(1, subtext_count + 1), gate.numpy(), color="#4c72b0")
        axes[1].set_xlabel("span length")
        axes[1].set_ylabel("learned gate $\\lambda$")
        axes[1].set_title("Prior strength per granularity")
        axes[1].set_xticks(range(1, subtext_count + 1))

        figure.tight_layout()
        figure.savefig(args.output, dpi=200)
        print(f"\nSaved figure to {args.output}")


if __name__ == "__main__":
    main()
