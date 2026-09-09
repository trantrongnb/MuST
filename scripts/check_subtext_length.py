#!/usr/bin/env python3
"""Report CLIP token lengths of the composed prompts.

CLIP truncates at 77 tokens.  When the atomic sub-texts are long sentences, the
length-3 and length-4 spans of the pyramid get cut back to the same prefix and
collapse onto each other -- the coarse granularities silently disappear and the
method quietly degrades to its own ablation.  Run this before training on a new
sub-text catalog; it exits non-zero and names the offenders.

    python3 scripts/check_subtext_length.py --subtext_path data/sub_texts/x.json
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from must.subtext import build_class_prompts, load_catalog, span_lengths  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="CLIP token length report")
    parser.add_argument(
        "--subtext_path",
        default=os.path.join(
            PROJECT_ROOT, "data", "sub_texts", "hmdb51_class_subtexts.json"
        ),
    )
    parser.add_argument(
        "--clip_model_path",
        default=os.path.join(PROJECT_ROOT, "pretrained", "clip-vit-base-patch16"),
    )
    parser.add_argument("--granularity", default="pyramid")
    parser.add_argument("--max_text_length", type=int, default=77)
    parser.add_argument("--show_worst", type=int, default=5)
    args = parser.parse_args()

    from transformers import CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(args.clip_model_path, local_files_only=True)
    class_names, catalog = load_catalog(args.subtext_path)
    subtext_count = len(catalog[0])
    prompts = build_class_prompts(class_names, catalog, args.granularity)
    lengths = span_lengths(subtext_count, args.granularity)

    print(f"Catalog: {args.subtext_path}")
    print(f"Classes: {len(class_names)}, sub-texts per class: {subtext_count}")
    print(f"Granularity: {args.granularity} -> M = {len(prompts[0])}\n")

    by_span = {}
    worst = []
    for name, items in zip(class_names, prompts):
        for span_length, text in zip(lengths, items):
            token_count = len(tokenizer(text, truncation=False)["input_ids"])
            by_span.setdefault(span_length, []).append(token_count)
            worst.append((token_count, span_length, name, text))

    print(f"{'span':>5}  {'count':>6}  {'mean':>6}  {'max':>5}  {'truncated':>10}")
    total_truncated = 0
    for span_length in sorted(by_span):
        counts = by_span[span_length]
        truncated = sum(1 for value in counts if value > args.max_text_length)
        total_truncated += truncated
        print(
            f"{span_length:>5}  {len(counts):>6}  {sum(counts) / len(counts):>6.1f}  "
            f"{max(counts):>5}  {truncated:>10}"
        )

    total = sum(len(values) for values in by_span.values())
    print(f"\nTruncated prompts: {total_truncated}/{total}")
    if total_truncated:
        print(
            "\nWARNING: truncated prompts lose their tail sub-texts, so the coarse\n"
            "granularities stop being distinguishable. Shorten the atomic sub-texts\n"
            "to a few words each."
        )
        worst.sort(reverse=True)
        print(f"\nLongest {args.show_worst} prompts:")
        for token_count, span_length, name, text in worst[: args.show_worst]:
            preview = text if len(text) <= 110 else text[:107] + "..."
            print(f"  [{token_count:>4} tokens, span={span_length}] {name}: {preview}")
        raise SystemExit(1)
    print("All composed prompts fit inside the CLIP context window.")


if __name__ == "__main__":
    main()
