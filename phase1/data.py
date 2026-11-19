"""Shared frame pairing, preprocessing, and sequence-level P3 splits."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import json
from pathlib import Path
import re

import numpy as np
from scipy.io import loadmat
from scipy.io.matlab import MatReadError
import torch
from torch.utils.data import Dataset


SHAPE = (3, 114, 10)
SUBJECTS = tuple(f"S{i:02}" for i in range(1, 41))
ACTIONS = tuple(f"A{i:02}" for i in range(1, 28))
FRAME = re.compile(r"frame(\d+)\.mat")


def read_csi(path):
    raw = loadmat(path, variable_names=("CSIamp",))["CSIamp"]
    if raw.shape != SHAPE or not np.isrealobj(raw):
        raise ValueError(f"Unexpected CSIamp shape or dtype in {path}: {raw.shape}, {raw.dtype}")
    data = raw.astype(np.float64)
    valid = np.isfinite(data)
    if not valid.any():
        raise ValueError(f"No finite CSI samples in {path}")
    # Match DT-Pose slice-wise repair. A wholly invalid slice uses the block mean.
    invalid = not valid.all()
    empty_slice = False
    if invalid:
        block_mean = data[valid].mean()
        for i in range(SHAPE[-1]):
            plane, mask = data[:, :, i], valid[:, :, i]
            empty_slice |= not mask.any()
            plane[~mask] = plane[mask].mean() if mask.any() else block_mean
    low, high = data.min(), data.max()
    constant = high == low
    data = np.zeros_like(data) if constant else (data - low) / (high - low)
    return data.astype(np.float32), (invalid, empty_slice, constant)


def partitions(setting, seed):
    """Reproduce MM-Fi/DT-Pose outer splits, then reserve inner validation."""
    result = {}
    subjects = np.array(SUBJECTS)
    for action_index, action in enumerate(ACTIONS):
        if setting == "s1":
            order = np.random.RandomState(seed + action_index).permutation(subjects)
            outer_train, test = order[:32], order[32:]
            inner = np.random.RandomState(seed + 10_000 + action_index).permutation(outer_train)
            val = inner[:3]
        elif setting == "s2":
            test = subjects[4::5]
            outer_train = np.array([s for s in subjects if s not in test])
            val = np.random.RandomState(seed + 10_000).choice(outer_train, size=4, replace=False)
        else:
            outer_train, test = subjects[:30], subjects[30:]
            rng = np.random.RandomState(seed + 10_000)
            val = [rng.choice(outer_train[i:i+10]) for i in (0, 10, 20)]
        result.update(((s, action), "test") for s in test)
        result.update(((s, action), "validation") for s in val)
        result.update(((s, action), "train") for s in outer_train if s not in val)
    return result


def audit_recording(root, directory, split):
    key = directory.relative_to(root).as_posix()
    labels = np.load(directory / "ground_truth.npy", allow_pickle=False)
    if labels.ndim != 3 or labels.shape[1:] != (17, 3) or not np.isrealobj(labels):
        raise ValueError(f"{key}: unexpected pose shape or dtype {labels.shape}, {labels.dtype}")
    counts = Counter()
    frames = {}
    seen = set()
    for file in sorted((directory / "wifi-csi").glob("*.mat")):
        match = FRAME.fullmatch(file.name)
        if match is None:
            continue
        number = int(match.group(1))
        if number in seen:
            raise ValueError(f"{key}: duplicate frame number {number}")
        seen.add(number)
        # Both official loaders pair frame001.mat with row 0.
        if not 1 <= number <= len(labels):
            counts["out_of_range"] += 1
            continue
        if not np.isfinite(labels[number - 1]).all():
            counts["invalid_pose"] += 1
            continue
        if not file.stat().st_size:
            counts["empty_csi"] += 1
            continue
        try:
            _, flags = read_csi(file)
        except (OSError, ValueError, KeyError, TypeError, NotImplementedError, EOFError, MatReadError):
            counts["unreadable_csi"] += 1
            continue
        counts.update(name for name, flag in zip(("repaired", "empty_slice", "constant"), flags) if flag)
        frames[number] = file.name
    counts["missing_csi"] = len(set(range(1, len(labels) + 1)) - seen)
    counts["usable"] = len(frames)
    return {"recording": key, "split": split, "frames": sorted(frames.items()),
            "label_rows": len(labels), "audit": dict(counts)}


def build_manifest(root, setting, seed, pelvis_index, units, allow_partial=False, workers=4):
    groups = partitions(setting, seed)
    directories = {}
    for path in sorted(root.glob("E0[1-4]/S[0-9][0-9]/A[0-9][0-9]")):
        key = (path.parent.name, path.name)
        if key in groups and (path / "ground_truth.npy").is_file() and (path / "wifi-csi").is_dir():
            if key in directories:
                raise ValueError(f"Duplicate subject/action directory: {path}")
            directories[key] = path
    if not directories:
        raise ValueError(f"No recordings found under {root}")
    if len(directories) != len(groups) and not allow_partial:
        raise ValueError(f"Found {len(directories)}/{len(groups)} recordings under {root}. "
                         "Finish the download or use --allow-partial for a smoke test.")
    tasks = [(root, path, groups[key]) for key, path in sorted(directories.items())]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = []
        for record in pool.map(lambda task: audit_recording(*task), tasks):
            records.append(record)
            if len(records) % 100 == 0:
                print(f"Audited {len(records)}/{len(tasks)} recordings", flush=True)
    return {"version": 1, "protocol": "P3", "setting": setting, "split_seed": seed,
            "pelvis_index": pelvis_index, "pose_units": units,
            "partial": len(records) != len(groups) or any(not record["frames"] for record in records),
            "recordings": records}


@lru_cache(maxsize=96)
def pose_labels(path):
    return np.load(path, allow_pickle=False)


class Frames(Dataset):
    def __init__(self, root, manifest, split, limit=None):
        self.root = Path(root)
        self.pelvis_index = manifest["pelvis_index"]
        self.metres = 1.0 if manifest["pose_units"] == "m" else 0.001
        self.rows = [(entry["recording"], number, filename)
                     for entry in manifest["recordings"] if entry["split"] == split
                     for number, filename in entry["frames"]][:limit]
        if not self.rows:
            raise ValueError(f"Manifest has no usable {split} frames")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        recording, number, filename = self.rows[index]
        directory = self.root / recording
        csi, _ = read_csi(directory / "wifi-csi" / filename)
        pose = pose_labels(directory / "ground_truth.npy")[number - 1].astype(np.float32)
        pose = (pose - pose[self.pelvis_index]) * self.metres
        return torch.from_numpy(csi), torch.from_numpy(pose), f"{recording}/{number:03d}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--setting", choices=("s1", "s2", "s3"), default="s1")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--pelvis-index", type=int, choices=range(17), required=True)
    parser.add_argument("--pose-units", choices=("m", "mm"), required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1.")
    if args.output.exists():
        parser.error("--output already exists. Reuse that manifest or choose a new path.")
    manifest = build_manifest(args.data, args.setting, args.split_seed, args.pelvis_index,
                              args.pose_units, args.allow_partial, args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, separators=(",", ":"))
    totals = Counter()
    for record in manifest["recordings"]:
        totals[record["split"]] += len(record["frames"])
    print(f"{args.output} ({'partial' if manifest['partial'] else 'complete'}): {dict(totals)}")


if __name__ == "__main__":
    main()
