# Shared DT-Pose encoder

Run commands from the repository root after following the main setup guide.
The encoder maps a `3 × 114 × 10` CSI block through 2 × 2 patch embedding into
285 tokens of width 256. A class token and four Transformer blocks complete
the encoder. The pose models use the 285 patch outputs after LayerNorm.

Pretraining uses masked CSI reconstruction, adjacent-frame contrastive learning
and uniformity regularization. Adjacent pairs come from consecutive frames in
the same training recording. The original architecture and objective are in
`phase1/pretrain_encoder.py` and `phase1/models.py`.

## Train

```bash
python phase1/pretrain_encoder.py --data data/MMFi_Dataset --manifest data/p3-s1.json --output runs/shared_encoder.pt --epochs 30 --batch-size 8 --device auto
```

`runs/shared_encoder.pt` is the latest full checkpoint and supplies encoder
weights to both pose models. Its companion folder `runs/shared_encoder/` saves
`best.pt`, `last.pt`, `config.json` and `history.jsonl` after each completed epoch.
Best means lowest average pretraining objective on training pairs. Pretraining
has no supervised pose head, so MPJPE is evaluated after pose-decoder training.

Both best and latest checkpoints save the encoder, reconstruction decoder,
projector, AdamW optimizer, random/data-order states, completed epoch and losses.
The architecture, masking ratio, loss formula and optimizer hyperparameters match
the existing FYP implementation.

## Continue from latest or best

```bash
python phase1/pretrain_encoder.py --data data/MMFi_Dataset --manifest data/p3-s1.json --resume runs/shared_encoder/last.pt --output runs/shared_encoder_next.pt --epochs 40 --batch-size 8 --device auto
```

Choose `best.pt` in the same command to continue from the best epoch and its
optimizer state. `--epochs` is the final epoch number. A saved epoch 30 with
`--epochs 40` trains epochs 31–40. Choose a fresh output name for each session.

Older encoder-only exports remain usable for `--encoder-checkpoint` in pose
training. Continuing pretraining requires a new full checkpoint because the
older exports omit the reconstruction/projector and optimizer states.
Cross-device numerical differences are possible when moving between CPU and GPU.
