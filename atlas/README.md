# Run Phase 1 on NUS Atlas

This branch adds PBS files to the portable experiment. Download the `atlas`
branch ZIP on your PC and upload its extracted repository to Atlas through
MobaXterm SFTP. If GitHub access is available on Atlas, cloning this branch also
works. Run the following commands from the uploaded repository root.

The supplied `environment.sh` loads `PyTorch/2.1.2-foss-2023a`, the CPU module
used in the existing Atlas experiment. Check availability with `module avail`.
Update this file if your Atlas module differs. Alternatively, set
`ATLAS_ENV_FILE` to your own activation script.

```bash
source atlas/environment.sh
python scripts/restore_manifest.py
python scripts/check_dataset.py --data "/hpctmp/$USER/fyp/MMFi_Dataset"
```

If a Windows editor changed line endings, restore Linux line endings before
submission with `sed -i 's/\r$//' atlas/*.sh atlas/*.pbs`.

## Submit the complete workflow

```bash
bash atlas/submit.sh all "/hpctmp/$USER/fyp/MMFi_Dataset"
```

This submits shared encoder pretraining, both pose models and their tests.
The default final epoch is 30. Baseline and GRC wait for successful pretraining.
Each test waits for its own training job to exit successfully. The queue is
`parallel`, with 12 CPU cores and 32 GB RAM. Pretraining requests 72 hours,
pose training 48 hours and testing 12 hours. These are walltime limits.

The script prints the run folder and job IDs. `pbs_jobs.tsv` stores those IDs.
Logs are named `<stage>.<job-ID>.log` in the run folder. Pose checkpoints are
under `baseline/` and `grc/`. Encoder checkpoints are under `shared_encoder/`,
with `shared_encoder.pt` also pointing to the latest saved encoder state.

```bash
qstat -u "$USER"
qstat -xf JOB_ID
tail -n 15 -f /absolute/path/to/run/grc.JOB_ID.log
```

## Continue one component

Use the exact manifest and matching shared encoder from the original run.
The new session creates a fresh run folder and carries the previous best model
forward. Set `END_EPOCH` to the final target epoch number.

```bash
END_EPOCH=10 RESUME_CHECKPOINT=/absolute/path/to/old/run/grc/last.pt ENCODER_CHECKPOINT=/absolute/path/to/old/run/shared_encoder.pt bash atlas/submit.sh grc "/hpctmp/$USER/fyp/MMFi_Dataset"
```

A saved epoch 5 trains epochs 6–10. Replace `last.pt` with a new full `best.pt`
to resume that epoch and optimizer. Older Atlas model-only checkpoints start
with a fresh optimizer. The provided encoder must match the saved model.

For encoder pretraining continuation, use its full training checkpoint.

```bash
END_EPOCH=40 RESUME_CHECKPOINT=/absolute/path/to/old/run/shared_encoder/last.pt bash atlas/submit.sh pretrain "/hpctmp/$USER/fyp/MMFi_Dataset"
```

## Test an existing checkpoint independently

```bash
POSE_CHECKPOINT=/absolute/path/to/old/run/grc/best.pt bash atlas/submit.sh test_grc "/hpctmp/$USER/fyp/MMFi_Dataset"
```

Use `test_baseline` for the DT-Pose baseline. Results go to the new run's
`test_grc/` or `test_baseline/` folder and include metrics and NPZ predictions.
This command works after a training timeout because it has no training-job
dependency. A timeout prevents the original `afterok` test job from starting.

`QUEUE`, `SELECT`, `PRETRAIN_WALLTIME`, `TRAIN_WALLTIME` and `TEST_WALLTIME`
can override the defaults within the queue limits. `DEVICE=cpu` is the default.
Select CUDA only with a matching GPU allocation and CUDA-enabled environment.
Save completed run folders to your own long-term storage, including checkpoints,
configs, histories, logs and the exact split manifest.
