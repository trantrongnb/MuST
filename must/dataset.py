"""Episodic frame loader.

One ``__getitem__`` call returns one ``W``-way ``K``-shot episode.  Class ids
follow the *sorted* sub-text catalog, so a class always maps to the same text
queries regardless of which episode it lands in -- that is what lets the text
buffer be indexed by ``batch_class_list``.
"""

import io
import json
import os
import random
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torchvision import transforms

from videotransforms.video_transforms import (
    CenterCrop,
    ColorJitter,
    Compose,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
)

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class Split:
    """Videos of one split, grouped by class id."""

    def __init__(self):
        self.videos_by_class: Dict[int, List[List[str]]] = defaultdict(list)

    def add_video(self, paths: List[str], class_id: int):
        self.videos_by_class[class_id].append(paths)

    def get_video(self, class_id: int, index: int) -> List[str]:
        return self.videos_by_class[class_id][index]

    def num_videos(self, class_id: int) -> int:
        return len(self.videos_by_class[class_id])

    def classes(self) -> List[int]:
        return sorted(self.videos_by_class)

    def __len__(self):
        return sum(len(items) for items in self.videos_by_class.values())


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.data_dir = Path(args.dataset).resolve()
        self.seq_len = int(args.seq_len)
        self.img_size = int(args.img_size)
        self.way = int(args.way)
        self.eval_way = int(args.eval_way)
        self.shot = int(args.shot)
        self.query_per_class = int(args.query_per_class)
        self.query_per_class_test = int(args.query_per_class_test)
        self.split = "train"
        self.frame_cache = None

        self.zip = self.data_dir.is_file() and self.data_dir.suffix.lower() == ".zip"
        if not self.zip and not self.data_dir.is_dir():
            raise FileNotFoundError(f"Dataset path does not exist: {self.data_dir}")

        with open(args.subtext_path, "r", encoding="utf-8") as handle:
            text_catalog = json.load(handle)
        self.class_names = sorted(text_catalog)
        self.class_to_id = {name: index for index, name in enumerate(self.class_names)}

        self.annotation_path = Path(args.split_path).resolve()
        self.split_membership = self._load_split_membership()
        self.train_split = Split()
        self.val_split = Split()
        self.test_split = Split()
        self.splits = {
            "train": self.train_split,
            "val": self.val_split,
            "test": self.test_split,
        }

        self.tensor_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(CLIP_MEAN, CLIP_STD)]
        )
        self._setup_transforms()
        self._index_dataset()
        self._validate_splits()
        if bool(getattr(args, "preload_frames", False)):
            self._preload_frame_bytes()

        print(f"Loaded dataset: {self.data_dir}")
        print(
            "Episodes: train={} videos/{} classes, val={}/{}, test={}/{}".format(
                len(self.train_split),
                len(self.train_split.classes()),
                len(self.val_split),
                len(self.val_split.classes()),
                len(self.test_split),
                len(self.test_split.classes()),
            ),
            flush=True,
        )

    # ----------------------------------------------------------- transforms

    def _setup_transforms(self):
        if self.img_size != 224:
            raise ValueError("CLIP ViT-B/16 currently requires --img_size 224")
        train_transforms = [Resize(256)]
        if not bool(getattr(self.args, "disable_horizontal_flip", False)):
            train_transforms.append(RandomHorizontalFlip())
        train_transforms.extend(
            [
                RandomCrop(self.img_size),
                ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            ]
        )
        test_transforms = [Resize(256), CenterCrop(self.img_size)]
        self.video_transform = {
            "train": Compose(train_transforms),
            "val": Compose(test_transforms),
            "test": Compose(test_transforms),
        }

    # --------------------------------------------------------------- indexing

    @staticmethod
    def _split_entry_key(line: str) -> str:
        """Normalise one split-file line to ``class_name/video_id``."""
        value = line.strip()
        if not value:
            return ""

        # Entries are usually "class name/video_id"; some formats append a
        # numeric label, so only strip the last field when it is clearly one.
        # Plain split() would break class names containing spaces, e.g.
        # "playing monopoly/video_id".
        parts = value.rsplit(maxsplit=1)
        if len(parts) == 2 and parts[1].lstrip("-").isdigit() and "/" in parts[0]:
            value = parts[0]

        path_value = Path(value)
        path_parts = path_value.parts
        if len(path_parts) >= 2:
            class_name = path_parts[-2]
            video_name = Path(path_parts[-1]).stem
            return f"{class_name}/{video_name}"
        return path_value.stem

    def _load_split_membership(self) -> Dict[str, str]:
        membership: Dict[str, str] = {}
        for split_name in ("train", "val", "test"):
            split_file = self.annotation_path / f"{split_name}list.txt"
            if not split_file.is_file():
                raise FileNotFoundError(f"Missing split file: {split_file}")
            for line in split_file.read_text(encoding="utf-8").splitlines():
                key = self._split_entry_key(line)
                if not key:
                    continue
                previous = membership.setdefault(key, split_name)
                if previous != split_name:
                    raise ValueError(
                        f"Video {key} occurs in both {previous} and {split_name} splits"
                    )
        return membership

    def _find_class_directories(self) -> List[Tuple[str, str]]:
        """Class directories, either directly under the root or one level down."""
        class_directories: List[Tuple[str, str]] = []
        data_root = str(self.data_dir)
        for child_name in sorted(os.listdir(data_root)):
            child_path = os.path.join(data_root, child_name)
            if not os.path.isdir(child_path):
                continue
            if child_name in self.class_to_id:
                class_directories.append((child_name, child_path))
                continue
            for class_name in sorted(os.listdir(child_path)):
                if class_name not in self.class_to_id:
                    continue
                class_path = os.path.join(child_path, class_name)
                if os.path.isdir(class_path):
                    class_directories.append((class_name, class_path))
        return class_directories

    def _index_dataset(self):
        if self.zip:
            self._index_zip_dataset()
            return

        class_directories = self._find_class_directories()
        if not class_directories:
            raise RuntimeError(
                f"No class directories matching the sub-text catalog were found in "
                f"{self.data_dir}"
            )

        for class_name, class_dir in class_directories:
            class_id = self.class_to_id[class_name]
            for video_name in sorted(os.listdir(class_dir)):
                split_name = self.split_membership.get(
                    f"{class_name}/{video_name}",
                    self.split_membership.get(video_name),
                )
                if split_name is None:
                    continue
                video_dir = os.path.join(class_dir, video_name)
                # Fast path: one listdir per video, no stat call per frame.
                try:
                    frame_names = os.listdir(video_dir)
                except NotADirectoryError:
                    continue
                frames = sorted(
                    os.path.join(video_dir, name)
                    for name in frame_names
                    if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
                )
                if len(frames) >= self.seq_len:
                    self.splits[split_name].add_video(frames, class_id)

    def _index_zip_dataset(self):
        """Read an uncompressed frame archive once, to avoid random HDD I/O."""
        self.zip_memory = self.data_dir.read_bytes()
        self.zip_file = zipfile.ZipFile(io.BytesIO(self.zip_memory))
        videos = defaultdict(list)
        for name in self.zip_file.namelist():
            if Path(name).suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            parts = name.rstrip("/").split("/")
            if len(parts) < 3:
                continue
            class_name, video_name = parts[-3], parts[-2]
            if class_name in self.class_to_id:
                key = f"{class_name}/{video_name}"
                if key in self.split_membership or video_name in self.split_membership:
                    videos[(class_name, video_name)].append(name)

        for (class_name, video_name), frames in videos.items():
            if len(frames) < self.seq_len:
                continue
            split_name = self.split_membership.get(
                f"{class_name}/{video_name}",
                self.split_membership[video_name],
            )
            self.splits[split_name].add_video(
                sorted(frames), self.class_to_id[class_name]
            )

    def _iter_frame_paths(self):
        seen = set()
        for split in self.splits.values():
            for videos in split.videos_by_class.values():
                for paths in videos:
                    for path in paths:
                        if path not in seen:
                            seen.add(path)
                            yield path

    def _preload_frame_bytes(self):
        if self.zip:
            print("Frame preload skipped: zip archive is already held in RAM.", flush=True)
            return
        paths = list(self._iter_frame_paths())
        self.frame_cache = {}
        total = len(paths)
        print(f"Preloading {total} frames into RAM...", flush=True)
        for index, path in enumerate(paths, 1):
            with open(path, "rb") as handle:
                self.frame_cache[path] = handle.read()
            if index % 20000 == 0 or index == total:
                print(f"  preloaded {index}/{total} frames", flush=True)

    def _validate_splits(self):
        """Fail at start-up, not 3 hours in, when a split cannot form episodes."""
        required = {
            "train": (self.way, self.shot + self.query_per_class),
            "val": (self.eval_way, self.shot + self.query_per_class_test),
            "test": (self.eval_way, self.shot + self.query_per_class_test),
        }
        for split_name, (way, samples_per_class) in required.items():
            split = self.splits[split_name]
            eligible = [
                class_id
                for class_id in split.classes()
                if split.num_videos(class_id) >= samples_per_class
            ]
            if len(eligible) < way:
                raise ValueError(
                    f"Split {split_name} has only {len(eligible)} eligible classes for "
                    f"{way}-way with {samples_per_class} videos per class"
                )

    # ---------------------------------------------------------------- sampling

    def __len__(self):
        # Episodes are drawn on demand; the loop is bounded by --training_iterations.
        return 1_000_000

    def set_split(self, split: str):
        if split not in self.splits:
            raise ValueError(f"Unknown split: {split}")
        self.split = split

    def _read_frame(self, path: str) -> Image.Image:
        if self.frame_cache is not None:
            with Image.open(io.BytesIO(self.frame_cache[path])) as image:
                image.load()
                return image.convert("RGB")
        if self.zip:
            with self.zip_file.open(path, "r") as handle:
                with Image.open(handle) as image:
                    image.load()
                    return image.convert("RGB")
        with Image.open(path) as image:
            image.load()
            return image.convert("RGB")

    def _sample_video(self, class_id: int, video_index: int) -> torch.Tensor:
        """``T`` frames, one per uniform segment: random in train, centre otherwise."""
        paths = self.splits[self.split].get_video(class_id, video_index)
        frame_count = len(paths)
        interval = frame_count // self.seq_len
        if self.split == "train":
            indices = [
                random.randint(index * interval, (index + 1) * interval - 1)
                for index in range(self.seq_len)
            ]
        else:
            indices = [
                (index * interval + (index + 1) * interval - 1) // 2
                for index in range(self.seq_len)
            ]
        images = [self._read_frame(paths[index]) for index in indices]
        transformed = self.video_transform[self.split](images)
        return torch.stack([self.tensor_transform(image) for image in transformed])

    def __getitem__(self, index):
        del index
        split = self.splits[self.split]
        way = self.way if self.split == "train" else self.eval_way
        query_count = (
            self.query_per_class if self.split == "train" else self.query_per_class_test
        )
        required_count = self.shot + query_count
        eligible_classes = [
            class_id
            for class_id in split.classes()
            if split.num_videos(class_id) >= required_count
        ]
        episode_classes = random.sample(eligible_classes, way)

        support_set = []
        support_labels = []
        target_set = []
        target_labels = []
        real_support_labels = []
        real_target_labels = []

        for episode_label, class_id in enumerate(episode_classes):
            selected = random.sample(range(split.num_videos(class_id)), required_count)
            for video_index in selected[: self.shot]:
                support_set.append(self._sample_video(class_id, video_index))
                support_labels.append(episode_label)
                real_support_labels.append(class_id)
            for video_index in selected[self.shot :]:
                target_set.append(self._sample_video(class_id, video_index))
                target_labels.append(episode_label)
                real_target_labels.append(class_id)

        return {
            "support_set": torch.cat(support_set, dim=0),
            "support_labels": torch.tensor(support_labels, dtype=torch.long),
            "real_support_labels": torch.tensor(real_support_labels, dtype=torch.long),
            "target_set": torch.cat(target_set, dim=0),
            "target_labels": torch.tensor(target_labels, dtype=torch.long),
            "real_target_labels": torch.tensor(real_target_labels, dtype=torch.long),
            "batch_class_list": torch.tensor(episode_classes, dtype=torch.long),
        }
