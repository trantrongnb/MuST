#!/usr/bin/env python3
"""Fetch the CLIP ViT-B/16 weights MuST expects.

The weights are ~600 MB, so they are not in the repository. This downloads them
once into ``pretrained/clip-vit-base-patch16/``, which is where
``--clip_model_path`` points by default.

    python3 scripts/download_clip.py

Both towers stay frozen during training; only the LoRA branch adapts the visual
one, so this is the only checkpoint the model ever needs.
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIR = os.path.join(PROJECT_ROOT, "pretrained", "clip-vit-base-patch16")
REPO_ID = "openai/clip-vit-base-patch16"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_DIR)
    parser.add_argument("--repo_id", default=REPO_ID)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the directory already holds a model.",
    )
    args = parser.parse_args()

    weights = ("pytorch_model.bin", "model.safetensors")
    if not args.force and any(
        os.path.isfile(os.path.join(args.output, name)) for name in weights
    ):
        print(f"CLIP weights already present in {args.output}; nothing to do.")
        return 0

    try:
        from transformers import CLIPModel, CLIPTokenizer, CLIPImageProcessor
    except ImportError:
        print(
            "transformers is not installed. Run:\n"
            "    python3 -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    os.makedirs(args.output, exist_ok=True)
    print(f"Downloading {args.repo_id} -> {args.output}")
    # backbone.py loads with use_safetensors=False, so save the .bin format.
    CLIPModel.from_pretrained(args.repo_id).save_pretrained(
        args.output, safe_serialization=False
    )
    CLIPTokenizer.from_pretrained(args.repo_id).save_pretrained(args.output)
    CLIPImageProcessor.from_pretrained(args.repo_id).save_pretrained(args.output)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
