#!/usr/bin/env python3
"""Test-split evaluation.

Appends one tab-separated line per run to ``eval_<way>way_<shot>shot.txt`` next
to the checkpoint, so repeated evaluations accumulate rather than overwrite.
"""

import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from must.dataset import VideoDataset  # noqa: E402
from must.model import MuST  # noqa: E402
from must.options import parse_eval_args  # noqa: E402
from must.utils import (  # noqa: E402
    aggregate_accuracy,
    confidence_interval_95,
    prepare_task,
    set_seed,
)


@torch.no_grad()
def main():
    args = parse_eval_args()
    print(f"Sub-text catalog: {args.subtext_path}", flush=True)
    print(f"Granularity: {args.granularity}, span_prior: {args.span_prior}", flush=True)
    set_seed(args.seed)
    use_cuda = torch.cuda.is_available() and args.num_gpus > 0
    device = torch.device("cuda" if use_cuda else "cpu")
    amp_enabled = bool(args.amp and use_cuda)

    checkpoint = torch.load(args.pretrained_checkpoint, map_location="cpu")
    model = MuST(args).to(device)
    model.load_checkpoint_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = VideoDataset(args)
    dataset.set_split("test")
    loader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers)

    accuracies = []
    correct = 0
    total = 0
    start = time.time()
    report_interval = max(1, args.num_test_tasks // 20)
    for iteration, raw_task in enumerate(loader):
        if iteration >= args.num_test_tasks:
            break
        task = prepare_task(raw_task, device)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp_enabled
        ):
            outputs = model(
                task["support_set"],
                task["support_labels"],
                task["target_set"],
                task["batch_class_list"],
            )
        predictions = outputs["logits"].argmax(dim=-1)
        labels = task["target_labels"].long()
        correct += int((predictions == labels).sum().item())
        total += labels.numel()
        accuracies.append(aggregate_accuracy(outputs["logits"], labels).item())

        if (iteration + 1) % report_interval == 0:
            print(
                f"Episode {iteration + 1}/{args.num_test_tasks}: "
                f"running accuracy={100.0 * correct / max(total, 1):.2f}%",
                flush=True,
            )

    mean, confidence = confidence_interval_95(accuracies)
    elapsed = time.time() - start
    result = (
        f"{os.path.basename(args.pretrained_checkpoint)}\t"
        f"granularity={args.granularity} M={model.pattern_num} "
        f"span_prior={args.span_prior}\t"
        f"{args.eval_way}-way {args.shot}-shot: "
        f"{mean * 100.0:.2f}+/-{confidence * 100.0:.2f}\t"
        f"{correct}/{total}\t{elapsed:.1f}s\n"
    )
    print(result, end="")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "a", encoding="utf-8") as handle:
        handle.write(result)


if __name__ == "__main__":
    main()
