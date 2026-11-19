# Phase 1 CSI-adapted GRC model

The shared DT-Pose encoder outputs 285 CSI tokens of width 256. GRC prepends
17 learned joint tokens and applies two four-head Transformer blocks over the
302-token sequence. It keeps the joint outputs, applies LayerNorm and a
`256 → 64` projection, then predicts coordinates with a shared
`64 → 32 → 3` ReLU head. Transformer feed-forward layers use GELU.
The model returns pelvis-centred coarse poses and 64 features for every joint.

Run from the repository root after setup and shared encoder pretraining.

## Train

```bash
python phase1/train_grc.py --data data/MMFi_Dataset --manifest data/p3-s1.json --encoder-checkpoint runs/shared_encoder.pt --output runs/csi_grc --epochs 30 --batch-size 16 --device auto
```

Both pose models use the same frozen encoder, preprocessing, split manifest,
coordinate MSE and AdamW settings. This compares their complete pose decoders.
`best.pt` uses lowest validation MPJPE. `last.pt` uses the latest completed epoch.
Both new files save the full model, optimizer, random/data-order states and
validation metadata. Latest also carries the complete best checkpoint, allowing
the next session to recover both files after an interrupted save.

## Continue from latest or best

```bash
python phase1/train_grc.py --data data/MMFi_Dataset --manifest data/p3-s1.json --resume runs/csi_grc/last.pt --output runs/csi_grc_next --epochs 40 --batch-size 16 --device auto
```

To continue from the best epoch, choose `best.pt` as `--resume`. Its optimizer
state belongs to that selected epoch. Use a fresh output folder for each session.
`--epochs 40` means train through epoch 40. The encoder stays frozen when a new
checkpoint is restored.

For the existing Atlas epoch-2 checkpoint, supply its original shared encoder.
The model weights load at epoch 2 and training starts at epoch 3. The old file
has no optimizer state, so the first continuation uses fresh AdamW state.

```bash
python phase1/train_grc.py --data data/MMFi_Dataset --manifest data/p3-s1.json --resume path/to/atlas/grc/best.pt --encoder-checkpoint path/to/atlas/shared_encoder.pt --output runs/csi_grc_from_epoch2 --epochs 7 --batch-size 16 --device auto
```

If the inherited legacy best remains best, its historical optimizer state is
still unavailable. Every newly completed epoch has a full resumable `last.pt`,
and each new validation improvement creates a full resumable `best.pt`.

## Evaluate and export

```bash
python phase1/train_grc.py --data data/MMFi_Dataset --manifest data/p3-s1.json --checkpoint runs/csi_grc/best.pt --output runs/csi_grc_test --split test --export --batch-size 16 --device auto
```

`metrics.json` contains aggregate and per-joint MPJPE and PA-MPJPE in millimetres.
NPZ exports contain sample IDs, `pose_m`, `target_m`, and `joint_features` of shape
`batch × 17 × 64`. Choose `--split train` or `--split validation` to prepare
separate coarse-pose/feature exports for future Phase 2 work.
