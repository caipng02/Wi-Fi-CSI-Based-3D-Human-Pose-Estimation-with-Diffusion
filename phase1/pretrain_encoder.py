"""Pretrain one shared DT-Pose encoder on training recordings only."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from data import Frames
from experiment import (cpu_copy, device_for, loader, random_states, restore_random_states,
                        save_checkpoint, seed_everything, sha256)
from models import DTEncoder, PATCHES, WIDTH, ViTBlock


class AdjacentPairs(Dataset):
    def __init__(self, frames):
        self.frames = frames
        self.indices = []
        # Frames are sorted by recording and numeric filename in the manifest.
        for i, (first, second) in enumerate(zip(frames.rows, frames.rows[1:])):
            if first[0] == second[0] and first[1] + 1 == second[1]:
                self.indices.append(i)
        if not self.indices:
            raise ValueError("Pretraining requires consecutive valid CSI frames in training recordings.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        start = self.indices[index]
        return self.frames[start][0], self.frames[start + 1][0]


class DTPretrain(nn.Module):
    """DT-Pose objectives ported to the shared, native PyTorch encoder."""

    def __init__(self, mask_ratio):
        super().__init__()
        self.encoder = DTEncoder()
        self.mask_ratio = mask_ratio
        self.visible = int(PATCHES * (1 - mask_ratio))
        self.mask_token = nn.Parameter(torch.empty(1, 1, WIDTH))
        self.positions = nn.Parameter(torch.empty(1, PATCHES + 1, WIDTH))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.positions, std=0.02)
        self.decoder = nn.Sequential(ViTBlock(4 * WIDTH), ViTBlock(4 * WIDTH), nn.Linear(WIDTH, 12))
        self.projector = nn.Sequential(nn.Linear(WIDTH, WIDTH), nn.BatchNorm1d(WIDTH),
                                       nn.ReLU(), nn.Linear(WIDTH, WIDTH),
                                       nn.BatchNorm1d(WIDTH), nn.ReLU())

    def forward(self, first, second, epoch):
        count = first.size(0)
        csi = torch.cat((first, second))
        patches = self.encoder.patches(csi)
        order = torch.rand(patches.shape[:2], device=csi.device).argsort(1)
        keep = order[:, :self.visible]
        visible = patches.gather(1, keep.unsqueeze(-1).expand(-1, -1, WIDTH))
        encoded = self.encoder.encode(visible)
        shuffled = torch.cat((encoded[:, 1:],
                              self.mask_token.expand(len(csi), PATCHES - self.visible, -1)), dim=1)
        restored = shuffled.gather(1, order.argsort(1).unsqueeze(-1).expand(-1, -1, WIDTH))
        reconstructed = self.decoder(torch.cat((encoded[:, :1], restored), dim=1) + self.positions)[:, 1:]
        # Match the released MAE reconstruction head layout: channel, row, column.
        target = F.unfold(csi, kernel_size=2, stride=2).transpose(1, 2)
        hidden = order.argsort(1) >= self.visible
        squared_error = (reconstructed[:count] - target[:count]).square()
        reconstruction = (squared_error * hidden[:count, :, None]).mean() / self.mask_ratio
        representation = self.projector(encoded[:, 1:].mean(1))
        logits = (representation[:count] @ representation[count:].T) / 0.5
        positives = torch.arange(count, device=csi.device)
        contrastive = F.cross_entropy(logits, positives)
        diagonal = torch.eye(count, device=csi.device, dtype=torch.bool)
        cls = F.normalize(encoded[:count, 0], dim=-1)
        correlations = cls @ cls.T
        uniformity = correlations.masked_fill(diagonal, 0).square().mean()
        contrastive_weight = min(0.0001 + 0.0099 * epoch / 400, 0.01)
        return reconstruction + 0.01 * uniformity + contrastive_weight * contrastive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path,
                        help="Best/latest folder; defaults to the output filename without .pt")
    parser.add_argument("--resume", type=Path, help="Resume a full encoder best.pt or last.pt")
    parser.add_argument("--epochs", type=int, default=30, help="Final epoch number after any resume")
    parser.add_argument("--batch-size", type=int, default=8, help="Number of adjacent pairs per batch")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--smoke", type=int, metavar="FRAMES")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or args.lr <= 0:
        parser.error("epochs, batch size and LR must be positive. workers must be nonnegative.")
    if args.smoke is not None and args.smoke < 2:
        parser.error("--smoke must be at least 2.")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest["partial"] and args.smoke is None:
        parser.error("Partial manifests require --smoke.")
    if args.output.suffix != ".pt":
        parser.error("--output must be an encoder checkpoint filename ending in .pt.")
    manifest_hash = sha256(args.manifest)
    checkpoint_dir = args.checkpoint_dir or args.output.with_suffix("")
    if args.output.exists() or checkpoint_dir.exists():
        parser.error("Choose a new --output filename and checkpoint directory for this session.")
    seed_everything(args.seed)
    device = device_for(args.device)
    pairs = AdjacentPairs(Frames(args.data, manifest, "train", args.smoke))
    batches = loader(pairs, args.batch_size, args.workers, args.seed, shuffle=True, device=device)
    model = DTPretrain(mask_ratio=0.8).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    history = []
    best_checkpoint = None
    best_loss = float("inf")
    start_epoch = 1
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        if state.get("checkpoint_type") != "encoder" or "optimizer" not in state:
            parser.error("Resuming pretraining requires a full encoder training checkpoint. "
                         "Older encoder-only exports remain usable with --encoder-checkpoint "
                         "when training a pose decoder.")
        if state["manifest_sha256"] != manifest_hash:
            parser.error("The checkpoint and manifest do not match.")
        for name in ("batch_size", "seed", "lr", "smoke"):
            if state["training_settings"][name] != getattr(args, name):
                parser.error(f"Resume --{name.replace('_', '-')} must match "
                             f"{state['training_settings'][name]!r}.")
        if args.epochs <= state["epoch"]:
            parser.error("--epochs must exceed the saved epoch.")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        history = state["loss"].copy()
        start_epoch = state["epoch"] + 1
        best_checkpoint = state.get("best_checkpoint", state)
        best_loss = best_checkpoint["loss"][-1]
        restore_random_states(state["random_states"], batches, None, device)
        print(f"Restored encoder epoch {state['epoch']}, optimizer and random states. "
              f"Continuing at epoch {start_epoch}.", flush=True)
        del state
    checkpoint_dir.mkdir(parents=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    settings = {key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()}
    (checkpoint_dir / "config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    if best_checkpoint is not None:
        save_checkpoint(best_checkpoint, checkpoint_dir / "best.pt")
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total = 0.0
        for first, second in batches:
            optimizer.zero_grad(set_to_none=True)
            loss = model(first.to(device), second.to(device), epoch - 1)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite pretraining loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total += loss.item() * len(first)
        history.append(total / len(pairs))
        model_state = cpu_copy(model.state_dict())
        current = {"checkpoint_version": 3, "checkpoint_type": "encoder", "epoch": epoch,
                   "model": model_state,
                   "encoder": {name: model_state["encoder." + name]
                               for name in model.encoder.state_dict()},
                   "optimizer": cpu_copy(optimizer.state_dict()),
                   "manifest_sha256": manifest_hash, "partial": bool(args.smoke),
                   "pairs": len(pairs), "loss": history.copy(), "settings": settings,
                   "random_states": random_states(batches, None, device),
                   "training_settings": {name: getattr(args, name)
                                         for name in ("batch_size", "seed", "lr", "smoke")}}
        improved = history[-1] < best_loss
        if improved:
            best_loss = history[-1]
            best_checkpoint = current
        latest = current | {"best_checkpoint": best_checkpoint}
        save_checkpoint(latest, checkpoint_dir / "last.pt")
        if improved:
            save_checkpoint(best_checkpoint, checkpoint_dir / "best.pt")
        # Keep the original --output interface usable by existing pose jobs.
        save_checkpoint(latest, args.output)
        temporary = checkpoint_dir / "history.jsonl.tmp"
        temporary.write_text("".join(json.dumps({"epoch": i + 1, "train_loss": value}) + "\n"
                                     for i, value in enumerate(history)), encoding="utf-8")
        temporary.replace(checkpoint_dir / "history.jsonl")
        print(f"Pretrain epoch {epoch}: loss {history[-1]:.6f}", flush=True)


if __name__ == "__main__":
    main()
