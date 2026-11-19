# Full DT-Pose baseline

This is the DT-Pose encoder and pose-decoder architecture port used in the FYP.
The experiment freezes the shared pretrained encoder and trains the complete
pose decoder. Its 285 CSI patch tokens are mean-pooled, combined with 17 joint
prompts, and processed by three graph convolutions and three joint Transformer
blocks. The ReLU pose head maps each joint `256 → 64 → 3`.

Run from the repository root after setup and encoder pretraining.

## Train

```bash
python phase1/train_baseline.py --data data/MMFi_Dataset --manifest data/p3-s1.json --encoder-checkpoint runs/shared_encoder.pt --output runs/dt_pose --epochs 30 --batch-size 16 --device auto
```

`best.pt` is selected by lowest validation MPJPE. `last.pt` contains the latest
completed epoch. Each new checkpoint includes the full pose model, encoder,
AdamW optimizer, random/data-order states, epoch and validation metrics.
`history.jsonl` stores training MSE and validation MPJPE/PA-MPJPE per epoch.

## Continue from latest or best

```bash
python phase1/train_baseline.py --data data/MMFi_Dataset --manifest data/p3-s1.json --resume runs/dt_pose/last.pt --output runs/dt_pose_next --epochs 40 --batch-size 16 --device auto
```

Use `best.pt` as `--resume` to restore the selected epoch and its optimizer.
Use a fresh output folder, the same manifest, batch size, learning rate and seed.
The frozen-encoder setting is restored automatically. If interrupted during an
epoch, resume from the last completed epoch. `--epochs` is the final epoch number.

## Evaluate and export

```bash
python phase1/train_baseline.py --data data/MMFi_Dataset --manifest data/p3-s1.json --checkpoint runs/dt_pose/best.pt --output runs/dt_pose_test --split test --export --batch-size 16 --device auto
```

`metrics.json` reports aggregate and per-joint MPJPE and PA-MPJPE in millimetres.
The NPZ files under `predictions/` contain sample IDs, `pose_m`, `target_m`, and
`joint_features` of shape `batch × 17 × 256`. Existing model-only checkpoints
remain compatible for evaluation. A legacy resume requires the matching
`--encoder-checkpoint` to preserve encoder freezing and initializes fresh AdamW
state when optimizer state is absent.
