# Wi-Fi-CSI-Based-3D-Human-Pose-Estimation-with-Diffusion

## Project overview

This final-year project investigates whether conditional diffusion can improve **3D human pose estimation from Wi-Fi channel state information (CSI)**. It adapts [mmDiff](https://arxiv.org/abs/2403.16198), originally designed for mmWave radar, to Wi-Fi CSI amplitude data from the [MM-Fi dataset](https://arxiv.org/abs/2305.10345). The system takes a short window of observed CSI and estimates a 17-joint 3D skeleton at each corresponding timestamp.

## Proposed pipeline

1. **Build a CSI window.** Group consecutive CSI blocks from the same subject–action recording. Each block corresponds to one pose timestamp. The initial design uses five blocks per window.
2. **Estimate coarse poses.** The [DT-Pose](https://arxiv.org/abs/2501.09411) encoder extracts feature tokens from each CSI block. An adapted Global Radar Context (GRC) module converts those tokens into joint-specific features, and an MLP pose head predicts one coarse 3D skeleton per block.
3. **Refine the poses jointly.** The coarse poses initialize conditional diffusion at inference. A graph-based denoiser uses the joint features, estimated bone lengths, and guidance from a proposed reliability-guided temporal module to refine all poses in the window.

## Planned evaluation

Compare the diffusion model with deterministic Wi-Fi CSI baselines on MM-Fi using pelvis-aligned MPJPE as the primary metric and PA-MPJPE as a secondary metric. Use P3-S1 for development, P3-S2 for the primary cross-subject evaluation, and P3-S3 for secondary cross-environment evaluation.
