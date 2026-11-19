# Wi-Fi CSI 3D Human Pose Estimation with Diffusion

This NUS final-year project investigates conditional diffusion for 17-joint 3D
pose estimation from MM-Fi Wi-Fi CSI. Phase 1 compares the full DT-Pose
encoder/pose-decoder architecture with the same encoder followed by a CSI-adapted
GRC and ReLU pose head. Phase 2 diffusion refinement is proposed work.

## Atlas batch jobs

This branch includes PBS submission, continuation and testing for NUS Atlas.
Follow [atlas/README.md](atlas/README.md) for the complete workflow.

## Setup

Use Python 3.11 or 3.12. Download this repository or clone it, then run these
commands from its root directory.

```bash
python -m venv .venv
```

Activate the environment using `.venv\Scripts\Activate.ps1` in Windows PowerShell
or `source .venv/bin/activate` on Linux/macOS. Then install dependencies and
restore the original split manifest.

```bash
python -m pip install -r requirements.txt
python scripts/restore_manifest.py
```

CPU execution works with the default installation. For GPU training, install a
CUDA-enabled PyTorch build suitable for your GPU and driver using the official
installation selector. Training selects CUDA when available with `--device auto`.
[PyTorch installation](https://pytorch.org/get-started/locally/)

## Run Phase 1

Place the downloaded dataset under `data/MMFi_Dataset`, with `E01` through `E04`
inside it, or supply its existing location through `--data`.
[Dataset setup](data/README.md)

```bash
python scripts/check_dataset.py --data data/MMFi_Dataset
python phase1/pretrain_encoder.py --data data/MMFi_Dataset --manifest data/p3-s1.json --output runs/shared_encoder.pt --epochs 30 --batch-size 8
python phase1/train_baseline.py --data data/MMFi_Dataset --manifest data/p3-s1.json --encoder-checkpoint runs/shared_encoder.pt --output runs/dt_pose --epochs 30 --batch-size 16
python phase1/train_grc.py --data data/MMFi_Dataset --manifest data/p3-s1.json --encoder-checkpoint runs/shared_encoder.pt --output runs/csi_grc --epochs 30 --batch-size 16
```

Both pose models freeze the same pretrained encoder. Each saves `best.pt` by
lowest validation MPJPE and `last.pt` after every completed epoch. Both new files
include model weights, optimizer and random states. Encoder checkpoints select
best by pretraining loss. Component guides explain evaluation and continuation.

| Component | Guide |
| --- | --- |
| Shared DT-Pose encoder pretraining | [Encoder README](phase1/encoder/README.md) |
| Full DT-Pose encoder and pose decoder | [DT-Pose README](phase1/dt_pose/README.md) |
| DT-Pose encoder, CSI GRC and pose head | [Phase 1 GRC README](phase1/csi_grc/README.md) |

The supplied P3-S1 manifest has 232,551 training, 24,057 validation and 64,152
test frames. Preprocessing, joint order, pelvis centring and splits are shared.
Evaluation reports MPJPE and PA-MPJPE in millimetres, with lower values better.

The `atlas` branch adds PBS submission and continuation scripts for NUS Atlas.
[Atlas branch](https://github.com/caipng02/Wi-Fi-CSI-Based-3D-Human-Pose-Estimation-with-Diffusion/tree/atlas)

## Project pipeline and references

The proposed complete system is CSI encoder → joint GRC features and coarse pose
→ conditional diffusion with skeletal and temporal guidance over N timestamps.
P3-S1 supports development, with P3-S2 and P3-S3 planned for further evaluation.

This repository contains the native PyTorch architecture ports used for the FYP
experiment. Source attribution and implementation details are recorded in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
