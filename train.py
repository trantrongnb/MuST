#!/usr/bin/env python3
"""Episodic training loop for MuST.

Gradients are accumulated over ``--tasks_per_batch`` episodes before a step, so
one "batch" is several 5-way episodes.  Validation runs every ``--val_interval``
iterations and keeps the best checkpoint by validation accuracy.
"""

import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from must.dataset import VideoDataset  # noqa: E402
from must.lora import is_lora_parameter, lora_b_magnitude  # noqa: E402
from must.model import MuST  # noqa: E402
from must.options import parse_train_args  # noqa: E402
from must.utils import (  # noqa: E402
    aggregate_accuracy,
    atomic_torch_save,
    confidence_interval_95,
    current_learning_rate,
    prepare_task,
    print_and_log,
    set_seed,
)


class Learner:
    def __init__(self):
        self.args = parse_train_args()
        set_seed(self.args.seed)
        os.makedirs(self.args.checkpoint_dir, exist_ok=True)
        self.log_file = open(
            os.path.join(self.args.checkpoint_dir, "log.txt"),
            "a",
            buffering=1,
            encoding="utf-8",
        )
        print_and_log(self.log_file, f"Options: {self.args}")
        print_and_log(self.log_file, f"Checkpoint directory: {self.args.checkpoint_dir}")

        use_cuda = torch.cuda.is_available() and self.args.num_gpus > 0
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.amp_enabled = bool(self.args.amp and use_cuda)
        print_and_log(self.log_file, f"Device: {self.device}, AMP: {self.amp_enabled}")

        self.model = MuST(self.args).to(self.device)
        self.dataset = VideoDataset(self.args)
        self.loader = DataLoader(
            self.dataset,
            batch_size=1,
            num_workers=self.args.num_workers,
        )

        print_and_log(
            self.log_file,
            f"Granularity: {self.args.granularity} -> "
            f"M={self.model.pattern_num} text queries from "
            f"N={self.model.subtext_count} sub-texts per class | "
            f"span_prior={self.args.span_prior}",
        )
        self._build_optimizer()

        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        self.start_iteration = 0
        self.best_val_accuracy = 0.0

        if self.args.resume:
            self._load_checkpoint(
                os.path.join(self.args.checkpoint_dir, "checkpoint_final.pt"),
                resume_optimizer=True,
            )
        elif self.args.pretrained_checkpoint:
            self._load_checkpoint(
                self.args.pretrained_checkpoint,
                resume_optimizer=False,
            )

    def _build_optimizer(self):
        """Four parameter groups, each with its own base learning rate.

        head      the attention block and any zero-gated branches
        scalars   gates, span prior, logit scale -- no weight decay, LR x mult,
                  because 1-D parameters converge far more slowly than matrices
        lora      ~1e-4, two orders above the fine-tuning rate; reusing
                  clip_learning_rate here is the classic way to make LoRA do
                  nothing at all
        clip      the unfrozen CLIP blocks, at clip_learning_rate
        """
        head, head_no_decay, lora_trainable, clip_trainable = [], [], [], []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if is_lora_parameter(name):
                lora_trainable.append(parameter)
            elif name.startswith("clip.model."):
                clip_trainable.append(parameter)
            elif parameter.ndim <= 1:
                head_no_decay.append(parameter)
            else:
                head.append(parameter)

        trainable = head + head_no_decay + lora_trainable + clip_trainable
        lora_count = sum(p.numel() for p in lora_trainable)
        clip_count = sum(p.numel() for p in clip_trainable)
        total = sum(p.numel() for p in trainable)
        print_and_log(
            self.log_file,
            f"Trainable parameters: {total:,} (MuST head: "
            f"{total - lora_count - clip_count:,}, LoRA: {lora_count:,}, "
            f"CLIP visual: {clip_count:,})",
        )
        if lora_trainable:
            print_and_log(
                self.log_file,
                f"LoRA: rank={self.args.lora_rank} alpha={self.args.lora_alpha:g} "
                f"targets={self.args.lora_targets} dropout={self.args.lora_dropout:g}, "
                f"wrapped {self.model.clip.lora_layers} linear layers, "
                f"lr={self.args.lora_lr:g}",
            )
        if clip_trainable and lora_trainable:
            print_and_log(
                self.log_file,
                "WARNING: LoRA and --clip_train_last_layers are both on. LoRA is "
                "meant to REPLACE unfreezing blocks; running both defeats the "
                "parameter argument for it.",
            )
        if self.model.temporal is not None:
            print_and_log(
                self.log_file,
                f"Temporal context: {self.args.temporal_context} "
                f"({sum(p.numel() for p in self.model.temporal.parameters()):,} "
                f"parameters, gate starts at 0)",
            )

        parameter_groups = []
        if head:
            parameter_groups.append({"params": head, "base_lr": self.args.learning_rate})
        if head_no_decay:
            parameter_groups.append(
                {
                    "params": head_no_decay,
                    "base_lr": self.args.learning_rate * self.args.scalar_lr_mult,
                    "weight_decay": 0.0,
                }
            )
            print_and_log(
                self.log_file,
                f"No weight decay on {len(head_no_decay)} scalar/1-D tensors, "
                f"LR x{self.args.scalar_lr_mult:g}",
            )
        if lora_trainable:
            parameter_groups.append(
                {"params": lora_trainable, "base_lr": self.args.lora_lr}
            )
        if clip_trainable:
            clip_lr = (
                self.args.clip_learning_rate
                if self.args.clip_learning_rate is not None
                else self.args.learning_rate
            )
            parameter_groups.append({"params": clip_trainable, "base_lr": clip_lr})
            print_and_log(self.log_file, f"CLIP visual learning rate: {clip_lr}")
        if self.args.warmup_iterations > 0:
            print_and_log(
                self.log_file,
                f"LR warm-up: {self.args.warmup_iterations} iterations",
            )

        if self.args.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(
                parameter_groups,
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
            )
        else:
            self.optimizer = torch.optim.SGD(
                parameter_groups,
                lr=self.args.learning_rate,
                momentum=0.9,
                nesterov=True,
                weight_decay=self.args.weight_decay,
            )

    def _set_learning_rate(self, iteration: int):
        learning_rate = current_learning_rate(
            self.args.learning_rate,
            iteration,
            self.args.training_iterations,
            self.args.warmup_iterations,
        )
        for group in self.optimizer.param_groups:
            base_lr = group.get("base_lr", self.args.learning_rate)
            group["lr"] = current_learning_rate(
                base_lr,
                iteration,
                self.args.training_iterations,
                self.args.warmup_iterations,
            )
        return learning_rate

    def _diagnostics(self) -> str:
        """Gates that start at zero by design, so a lasting zero means dead.

        `motion` is tanh(g) of the temporal context; `lora_b` is mean |B|. Both
        are exactly 0 at initialisation, so a value still at 0 after a few
        thousand iterations means that branch never activated -- almost always a
        learning-rate mistake -- and any result attributed to it is void.
        """
        values = self.model.gate_values()
        if self.args.lora_rank > 0:
            values["lora_b"] = lora_b_magnitude(self.model)
        if not values:
            return ""
        return ", " + ", ".join(f"{k}: {v:.4f}" for k, v in values.items())

    def _forward(self, task):
        return self.model(
            task["support_set"],
            task["support_labels"],
            task["target_set"],
            task["batch_class_list"],
            task.get("target_labels"),
        )

    def train_task(self, raw_task):
        self.model.train()
        task = prepare_task(raw_task, self.device)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.amp_enabled,
        ):
            outputs = self._forward(task)
            losses = self.model.loss(task, outputs)
            total_loss = sum(losses.values()) / self.args.tasks_per_batch

        if not torch.isfinite(total_loss):
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(f"Non-finite loss: {total_loss.item()}")
        self.scaler.scale(total_loss).backward()
        accuracy = aggregate_accuracy(outputs["logits"], task["target_labels"])
        return (
            {key: value.detach().item() for key, value in losses.items()},
            accuracy.detach().item(),
        )

    @torch.no_grad()
    def evaluate(self, split: str = "val", num_tasks: int = None, log_progress: bool = True):
        previous_split = self.dataset.split
        self.dataset.set_split(split)
        self.model.eval()
        num_tasks = num_tasks or self.args.num_val_tasks
        accuracies = []
        report_interval = max(1, num_tasks // 10)

        for iteration, raw_task in enumerate(self.loader):
            if iteration >= num_tasks:
                break
            task = prepare_task(raw_task, self.device)
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.amp_enabled,
            ):
                outputs = self._forward(task)
            accuracies.append(
                aggregate_accuracy(outputs["logits"], task["target_labels"]).item()
            )

            completed = iteration + 1
            if log_progress and (completed % report_interval == 0 or completed == num_tasks):
                running_accuracy = 100.0 * sum(accuracies) / max(len(accuracies), 1)
                print_and_log(
                    self.log_file,
                    f"Eval[{split}] [{completed}/{num_tasks}], "
                    f"running Acc: {running_accuracy:.2f}%",
                )

        self.dataset.set_split(previous_split)
        mean, confidence = confidence_interval_95(accuracies)
        return mean * 100.0, confidence * 100.0

    def _save_checkpoint(self, iteration: int, filename: str):
        payload = {
            "iteration": iteration,
            "best_val_accuracy": self.best_val_accuracy,
            "model_state_dict": self.model.checkpoint_state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "args": vars(self.args),
        }
        atomic_torch_save(payload, os.path.join(self.args.checkpoint_dir, filename))

    def _load_checkpoint(self, path: str, resume_optimizer: bool):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        checkpoint = torch.load(path, map_location="cpu")
        self.model.load_checkpoint_state_dict(checkpoint["model_state_dict"])
        if resume_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if checkpoint.get("scaler_state_dict"):
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            self.start_iteration = int(checkpoint.get("iteration", 0))
            self.best_val_accuracy = float(checkpoint.get("best_val_accuracy", 0.0))
        print_and_log(self.log_file, f"Loaded checkpoint: {path}")

    def run(self):
        self.optimizer.zero_grad(set_to_none=True)
        iteration = self.start_iteration
        start_time = time.time()
        loss_history = {}
        accuracy_history = []

        for raw_task in self.loader:
            if iteration >= self.args.training_iterations:
                break
            iteration += 1
            learning_rate = self._set_learning_rate(iteration)
            losses, accuracy = self.train_task(raw_task)
            accuracy_history.append(accuracy)
            for key, value in losses.items():
                loss_history.setdefault(key, []).append(value)

            should_step = (
                iteration % self.args.tasks_per_batch == 0
                or iteration == self.args.training_iterations
            )
            if should_step:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            if iteration % self.args.print_interval == 0:
                elapsed = time.time() - start_time
                estimated_total = elapsed * self.args.training_iterations / iteration
                loss_text = ", ".join(
                    f"{key}: {sum(values) / len(values):.4f}"
                    for key, values in loss_history.items()
                )
                print_and_log(
                    self.log_file,
                    f"Task [{iteration}/{self.args.training_iterations}], "
                    f"LR: {learning_rate:.6g}, Acc: "
                    f"{sum(accuracy_history) / len(accuracy_history):.3f}, "
                    f"{loss_text}{self._diagnostics()}, "
                    f"ETA: {(estimated_total - elapsed) / 3600:.1f}h",
                )
                loss_history = {}
                accuracy_history = []

            if iteration % self.args.val_interval == 0:
                split = self.args.val_split
                print_and_log(
                    self.log_file,
                    f"Starting Eval[{split}] at iteration {iteration} "
                    f"for {self.args.num_val_tasks} episodes",
                )
                val_accuracy, confidence = self.evaluate(
                    split, self.args.num_val_tasks, log_progress=True
                )
                print_and_log(
                    self.log_file,
                    f"Validation[{split}] @ {iteration}: "
                    f"{val_accuracy:.2f} +/- {confidence:.2f}",
                )
                if val_accuracy > self.best_val_accuracy:
                    self.best_val_accuracy = val_accuracy
                    self._save_checkpoint(iteration, "checkpoint_best.pt")
                    print_and_log(self.log_file, "Saved checkpoint_best.pt")

                self._save_checkpoint(iteration, "checkpoint_final.pt")

        self._save_checkpoint(iteration, "checkpoint_final.pt")
        with open(
            os.path.join(self.args.checkpoint_dir, "config.json"), "w", encoding="utf-8"
        ) as handle:
            json.dump(vars(self.args), handle, indent=2)
        print_and_log(self.log_file, "Training complete")
        self.log_file.close()


def main():
    Learner().run()


if __name__ == "__main__":
    main()
