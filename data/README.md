# MM-Fi Wi-Fi CSI and pose labels

Obtain MM-Fi through the authors' dataset access links, or use the dataset copy
already downloaded for the FYP. The authors maintain the download links in the
official toolbox README. This experiment needs the Wi-Fi CSI and 3D pose labels.
[MM-Fi toolbox and dataset links](https://github.com/ybhbingo/MMFi_dataset)

The default directory layout is

```text
data/
  p3-s1.json.gz
  p3-s1.json              # restored by scripts/restore_manifest.py
  MMFi_Dataset/
    E01/
      S01/
        A01/
          ground_truth.npy
          wifi-csi/
            frame001.mat
            ...
    E02/
    E03/
    E04/
```

Existing data can stay elsewhere. Pass its absolute root containing `E01` to
each command's `--data` option.

Run `python scripts/restore_manifest.py` from the repository root. The gzip file
restores the exact original manifest bytes and verifies their SHA256. This keeps
existing Atlas checkpoints compatible. Do this before dataset checking or training.

```bash
python scripts/check_dataset.py --data data/MMFi_Dataset
```

The manifest assigns whole recordings to P3-S1 training, validation and test.
It contains 783/232,551 training recordings/frames, 81/24,057 validation, and
216/64,152 test. `frame001.mat` pairs with label row 0. Each CSI amplitude block
is `3 × 114 × 10`. The ten internal samples belong to one pose timestamp.

Shared preprocessing repairs non-finite amplitudes with the finite mean within
each internal time slice. An entirely invalid slice uses the block mean.
Each block is min-max normalized, with constant blocks mapped to zero.
The manifest records unusable samples excluded by the original audit.
Targets and predictions are pelvis-centred using joint 0, stored in metres and
evaluated in millimetres. `phase1/data.py` contains the unchanged implementation.

To audit a separate dataset copy and create a new experiment manifest, choose
a new output path. Existing checkpoints require their original manifest.

```bash
python phase1/data.py --data data/MMFi_Dataset --output data/new-p3-s1.json --setting s1 --split-seed 0 --pelvis-index 0 --pose-units m
```
