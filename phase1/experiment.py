"""Shared training, metrics, checkpointing, and prediction export."""

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler

from data import Frames
from models import Phase1


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def device_for(name):
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available()
                        else "cpu" if name == "auto" else name)


def loader(dataset, batch, workers, seed, shuffle=False, device="cpu"):
    sampler = RandomSampler(dataset, generator=torch.Generator().manual_seed(seed)) if shuffle else None
    return DataLoader(dataset, batch_size=batch, sampler=sampler, num_workers=workers,
                      persistent_workers=workers > 0, pin_memory=torch.device(device).type == "cuda",
                      generator=torch.Generator().manual_seed(seed))


def save_checkpoint(value, path):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def random_states(training, validation, device):
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        "sampler": training.sampler.generator.get_state(),
        "training_loader": training.generator.get_state(),
        "validation_loader": validation.generator.get_state() if validation is not None else None,
    }


def restore_random_states(state, training, validation, device):
    random.setstate(state["python"])
    algorithm, keys, position, has_gauss, cached_gaussian = state["numpy"]
    np.random.set_state((algorithm, np.array(keys, dtype=np.uint32),
                         position, has_gauss, cached_gaussian))
    torch.set_rng_state(state["torch"].cpu())
    if device.type == "cuda" and state["cuda"]:
        if len(state["cuda"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
        else:
            torch.cuda.set_rng_state(state["cuda"][0].cpu(), device)
    training.sampler.generator.set_state(state["sampler"].cpu())
    training.generator.set_state(state["training_loader"].cpu())
    if validation is not None:
        validation.generator.set_state(state["validation_loader"].cpu())


def cpu_copy(value):
    """Keep historical optimizer and model tensors independent of later updates."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    return value


def saved_best(state):
    """Read current or earlier checkpoint layouts without inventing optimizer state."""
    if "best_checkpoint" in state:
        return state["best_checkpoint"]
    if "best_model" in state and state["best_epoch"] != state["epoch"]:
        return {"model": state["best_model"], "epoch": state["best_epoch"],
                "validation": state["best_validation"], "variant": state["variant"],
                "manifest_sha256": state["manifest_sha256"], "smoke": state["smoke"],
                "encoder_frozen": state["encoder_frozen"]}
    return {key: value for key, value in state.items()
            if key not in ("best_checkpoint", "best_model", "best_epoch", "best_validation")}


def procrustes(pred, target):
    """Per-frame optimal rotation, translation and uniform scale, no reflection."""
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)
    x = pred - pred.mean(1, keepdims=True)
    y_mean = target.mean(1, keepdims=True)
    y = target - y_mean
    u, singular, vt = np.linalg.svd(x.transpose(0, 2, 1) @ y)
    sign = np.where(np.linalg.det(u @ vt) < 0, -1.0, 1.0)
    u[:, :, -1] *= sign[:, None]
    singular[:, -1] *= sign
    variance = np.square(x).sum(axis=(1, 2))
    scale = np.divide(singular.sum(1), variance, out=np.zeros_like(variance),
                      where=variance > np.finfo(np.float64).eps)
    return scale[:, None, None] * (x @ (u @ vt)) + y_mean


@torch.inference_mode()
def evaluate(model, batches, device, export=None):
    model.eval()
    per_joint = np.zeros(17)
    pa_per_joint = np.zeros(17)
    frames = 0
    if export:
        export.mkdir(parents=True, exist_ok=False)
    for i, (csi, target, ids) in enumerate(batches):
        pose, features = model(csi.to(device, non_blocking=True))
        pose = pose.float().cpu().numpy()
        target = target.numpy()
        if not np.isfinite(pose).all():
            raise FloatingPointError("Non-finite predicted pose")
        per_joint += np.linalg.norm(pose - target, axis=-1).sum(0)
        pa_per_joint += np.linalg.norm(procrustes(pose, target) - target, axis=-1).sum(0)
        frames += len(ids)
        if export:
            np.savez(export / f"batch_{i:05d}.npz", sample_id=np.array(ids),
                     pose_m=pose, target_m=target, joint_features=features.float().cpu().numpy())
    joint_mm = per_joint * 1000 / frames
    pa_joint_mm = pa_per_joint * 1000 / frames
    return {"frames": frames, "mpjpe_mm": float(joint_mm.mean()),
            "pa_mpjpe_mm": float(pa_joint_mm.mean()), "per_joint_mpjpe_mm": joint_mm.tolist(),
            "per_joint_pa_mpjpe_mm": pa_joint_mm.tolist()}


def load_encoder(model, checkpoint, manifest_hash, allow_smoke, dev):
    state = torch.load(checkpoint, weights_only=True, map_location=dev)
    if state.get("manifest_sha256") != manifest_hash:
        raise ValueError("Encoder checkpoint must come from the training portion of this manifest. "
                         "Use pretrain_encoder.py with the same manifest.")
    if state.get("partial") and not allow_smoke:
        raise ValueError("A limited pretraining run may only be used with --smoke.")
    model.encoder.load_state_dict(state["encoder"], strict=True)
    model.encoder.requires_grad_(False)


def run(variant):
    parser = argparse.ArgumentParser(description=f"Train or evaluate Phase 1 ({variant}).")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path, help="Freeze a shared train-only encoder")
    parser.add_argument("--checkpoint", type=Path, help="Evaluate this trained Phase 1 model")
    parser.add_argument("--resume", type=Path,
                        help="Continue training from last.pt or an older best.pt; use a new output folder")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--export", action="store_true", help="Export poses and features during evaluation")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Final epoch number, including epochs completed before --resume")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--smoke", type=int, metavar="FRAMES",
                        help="Limit each partition to this many frames. Results are smoke tests.")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or args.lr <= 0:
        parser.error("epochs, batch size and LR must be positive. workers must be nonnegative.")
    if args.smoke is not None and args.smoke < 1:
        parser.error("--smoke must be positive.")
    if (args.export or args.split != "validation") and not args.checkpoint:
        parser.error("--split and --export apply to --checkpoint evaluation.")
    if args.checkpoint and args.encoder_checkpoint:
        parser.error("--checkpoint already includes encoder weights.")
    if args.checkpoint and args.resume:
        parser.error("Use --checkpoint for evaluation or --resume for training.")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest["partial"] and args.smoke is None:
        parser.error("Partial manifests require --smoke.")
    manifest_hash = sha256(args.manifest)
    seed_everything(args.seed)
    dev = device_for(args.device)
    model = Phase1(variant, manifest["pelvis_index"]).to(dev)
    state = None
    if args.checkpoint or args.resume:
        state = torch.load(args.checkpoint or args.resume, weights_only=True, map_location="cpu")
        if state["variant"] != variant or state["manifest_sha256"] != manifest_hash:
            raise ValueError("Checkpoint variant or manifest does not match this run")
        if state["smoke"] and not args.smoke:
            raise ValueError("A smoke-test checkpoint requires --smoke.")
        if args.resume and args.epochs <= state["epoch"]:
            parser.error("--epochs must exceed the saved epoch when resuming.")
        if args.resume and "encoder_frozen" not in state and not args.encoder_checkpoint:
            parser.error("For an older checkpoint, also pass --encoder-checkpoint to preserve "
                         "the frozen-encoder experiment.")
        model.load_state_dict(state["model"], strict=True)
    args.output.mkdir(parents=True, exist_ok=False)

    def batches(split, shuffle=False):
        return loader(Frames(args.data, manifest, split, args.smoke), args.batch_size,
                      args.workers, args.seed, shuffle, dev)

    config = vars(args) | {"variant": variant, "manifest_sha256": manifest_hash,
                           "torch_version": str(torch.__version__), "device_used": str(dev)}
    if args.checkpoint:
        metrics = evaluate(model, batches(args.split), dev,
                           args.output / "predictions" if args.export else None)
        (args.output / "metrics.json").write_text(json.dumps(
            {"split": args.split, "smoke": bool(args.smoke), **metrics}, indent=2), encoding="utf-8")
        print(f"{args.split}: MPJPE {metrics['mpjpe_mm']:.3f} mm, "
              f"PA-MPJPE {metrics['pa_mpjpe_mm']:.3f} mm ({metrics['frames']} frames)")
    else:
        if args.encoder_checkpoint:
            if args.resume:
                encoder_state = torch.load(args.encoder_checkpoint, weights_only=True, map_location="cpu")
                if any(not torch.equal(value.cpu(), encoder_state["encoder"][key])
                       for key, value in model.encoder.state_dict().items()):
                    raise ValueError("Resume checkpoint and shared encoder have different weights.")
            load_encoder(model, args.encoder_checkpoint, manifest_hash, bool(args.smoke), dev)
            config["encoder_sha256"] = sha256(args.encoder_checkpoint)
        if args.resume and "encoder_frozen" in state:
            model.encoder.requires_grad_(not state["encoder_frozen"])
        encoder_frozen = not any(p.requires_grad for p in model.encoder.parameters())
        config["encoder_frozen"] = encoder_frozen
        # The encoder is created before either decoder, so matched seeds also
        # give both variants the same starting encoder in a scratch comparison.
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                      lr=args.lr, weight_decay=0.01)
        training, validation = batches("train", True), batches("validation")
        best = float("inf")
        best_checkpoint = None
        history = []
        start_epoch = 1
        if args.resume:
            start_epoch = state["epoch"] + 1
            config["resume_epoch"] = state["epoch"]
            history = state.get("history", [])
            best_checkpoint = saved_best(state)
            best_checkpoint["encoder_frozen"] = encoder_frozen
            best = best_checkpoint["validation"]["mpjpe_mm"]
            if "optimizer" in state:
                previous = state["training_settings"]
                for name in ("batch_size", "seed", "lr", "smoke"):
                    if previous[name] != getattr(args, name):
                        raise ValueError(f"Resume --{name.replace('_', '-')} must match "
                                         f"the saved value {previous[name]!r}.")
                optimizer.load_state_dict(state["optimizer"])
                restore_random_states(state["random_states"], training, validation, dev)
                config["optimizer_restored"] = True
                print(f"Restored epoch {state['epoch']} weights, optimizer and random states.", flush=True)
            else:
                config["optimizer_restored"] = False
                print(f"Loaded epoch {state['epoch']} weights. This older checkpoint has no optimizer "
                      "or random states; continuing with a fresh optimizer.", flush=True)
            print(f"Continuing at epoch {start_epoch}; training through epoch {args.epochs}. "
                  f"Best validation MPJPE so far: {best:.3f} mm "
                  f"(epoch {best_checkpoint['epoch']}).", flush=True)
            save_checkpoint(best_checkpoint, args.output / "best.pt")
            del state
        (args.output / "config.json").write_text(json.dumps(config, default=str, indent=2), encoding="utf-8")
        use_amp = dev.type == "cuda" and torch.cuda.is_bf16_supported()
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            if encoder_frozen:
                model.encoder.eval()
            total = 0.0
            for csi, target, _ in training:
                csi = csi.to(dev, non_blocking=True)
                target = target.to(dev, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext():
                    pose, _ = model(csi)
                loss = F.mse_loss(pose.float(), target)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                total += loss.detach().item() * len(csi)
            metrics = evaluate(model, validation, dev)
            row = {"epoch": epoch, "train_mse_m2": total / len(training.dataset), **metrics}
            history.append(row)
            current = {"checkpoint_version": 3, "checkpoint_type": "pose",
                       "model": cpu_copy(model.state_dict()), "epoch": epoch, "variant": variant,
                       "manifest_sha256": manifest_hash, "smoke": bool(args.smoke),
                       "validation": metrics, "encoder_frozen": encoder_frozen,
                       "optimizer": cpu_copy(optimizer.state_dict()), "history": history.copy(),
                       "random_states": random_states(training, validation, dev),
                       "training_settings": {name: getattr(args, name)
                                             for name in ("batch_size", "seed", "lr", "smoke")}}
            improved = metrics["mpjpe_mm"] < best
            if improved:
                best = metrics["mpjpe_mm"]
                best_checkpoint = current
            # last.pt carries the complete best checkpoint as well, so resuming
            # can recover both files after a session ends during a save.
            save_checkpoint(current | {"best_checkpoint": best_checkpoint}, args.output / "last.pt")
            if improved:
                save_checkpoint(best_checkpoint, args.output / "best.pt")
            history_temporary = args.output / "history.jsonl.tmp"
            history_temporary.write_text("".join(json.dumps(item) + "\n" for item in history),
                                         encoding="utf-8")
            history_temporary.replace(args.output / "history.jsonl")
            print(f"{variant} epoch {epoch}: loss {row['train_mse_m2']:.6f}, "
                  f"validation MPJPE {metrics['mpjpe_mm']:.3f} mm, "
                  f"PA-MPJPE {metrics['pa_mpjpe_mm']:.3f} mm", flush=True)
    if args.checkpoint:
        (args.output / "config.json").write_text(json.dumps(config, default=str, indent=2), encoding="utf-8")
