"""Check every expected CSI file and pose-label array before submitting a job."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/p3-s1.json"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    missing = []
    invalid_labels = []
    checked = 0
    for recording in manifest["recordings"]:
        directory = args.data / recording["recording"]
        label_file = directory / "ground_truth.npy"
        expected = [label_file] + [directory / "wifi-csi" / filename
                                   for _, filename in recording["frames"]]
        for path in expected:
            checked += 1
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(str(path))
        if label_file.is_file() and label_file.stat().st_size:
            try:
                labels = np.load(label_file, mmap_mode="r", allow_pickle=False)
                if labels.shape != (recording["label_rows"], 17, 3):
                    invalid_labels.append(str(label_file))
            except (OSError, ValueError, EOFError):
                invalid_labels.append(str(label_file))
    print(f"Files checked: {checked:,}")
    print(f"Missing or empty files: {len(missing):,}")
    print(f"Invalid label arrays: {len(invalid_labels):,}")
    for path in (missing + invalid_labels)[:10]:
        print(path)
    raise SystemExit(bool(missing or invalid_labels))


if __name__ == "__main__":
    main()
