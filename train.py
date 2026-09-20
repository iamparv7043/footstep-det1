"""
train.py

Trains the heavier FootstepCNN from scratch and produces a full
diagnostic report after training:
  1. Loss vs epoch plot        -> checkpoints/loss_curve.png
  2. Accuracy vs epoch plot    -> checkpoints/accuracy_curve.png
  3. Epochs actually trained (vs early stopping)
  4. GPU used (or CPU)
  5. Parameter count + breakdown by layer type
  6. Loss function used
  7. Dropout rate
  8. Batch size
  9. Average inference time per sample (measured on the test set)

All of this is also written to checkpoints/training_report.txt and
printed to the console at the end of the run.

Usage:
    python train.py --dataset_root dataset --epochs 40
"""

import argparse
import time
import copy
import platform

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")  # no display needed, just save PNGs
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from dataset import (
    build_segment_index, split_by_source_file, FootstepDataset,
    class_counts, LABEL_NAMES,
)
from preprocessing.features import DEFAULT_PARAMS
from models.cnn import FootstepCNN


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", type=str, default="dataset")
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--duration_sec", type=float, default=1.0)
    p.add_argument("--hop_sec", type=float, default=0.5)
    p.add_argument("--n_fft", type=int, default=1024)
    p.add_argument("--hop_length", type=int, default=320)
    p.add_argument("--win_length", type=int, default=1024)
    p.add_argument("--n_mels", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4,
                    help="parallel worker processes for audio preprocessing. "
                         "0 = single-threaded (slow, CPU-bound). 4-8 is a good "
                         "starting point on most machines; set to your CPU core "
                         "count if unsure.")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def make_loaders(args, mel_params):
    print("Indexing dataset...")
    segments = build_segment_index(args.dataset_root, args.sr, args.duration_sec, args.hop_sec)
    if len(segments) == 0:
        raise RuntimeError(f"No segments found under '{args.dataset_root}'.")

    train_segs, val_segs, test_segs = split_by_source_file(segments, seed=args.seed)

    print(f"Total segments: {len(segments)}")
    print(f"  Train: {len(train_segs)}  {class_counts(train_segs)}")
    print(f"  Val:   {len(val_segs)}  {class_counts(val_segs)}")
    print(f"  Test:  {len(test_segs)}  {class_counts(test_segs)}")

    train_ds = FootstepDataset(train_segs, sr=args.sr, duration_sec=args.duration_sec,
                                mel_params=mel_params, augment=True)
    val_ds = FootstepDataset(val_segs, sr=args.sr, duration_sec=args.duration_sec,
                              mel_params=mel_params, augment=False)
    test_ds = FootstepDataset(test_segs, sr=args.sr, duration_sec=args.duration_sec,
                               mel_params=mel_params, augment=False)

    # num_workers > 0 parallelizes the CPU-bound audio decode + Log-Mel
    # computation across processes, so the GPU isn't left waiting on
    # single-threaded librosa work between batches. persistent_workers
    # avoids re-spawning worker processes every epoch. pin_memory speeds
    # up the host->GPU transfer when CUDA is available.
    use_cuda = torch.cuda.is_available()
    loader_kwargs = dict(
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=use_cuda,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, class_counts(train_segs)


def compute_class_weights(counts: dict, device) -> torch.Tensor:
    total = counts[0] + counts[1]
    w0 = total / (2.0 * max(counts[0], 1))
    w1 = total / (2.0 * max(counts[1], 1))
    return torch.tensor([w0, w1], dtype=torch.float32, device=device)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for x, aux, y in loader:
            x, aux, y = x.to(device), aux.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(x, aux)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(y.cpu().numpy().tolist())

    avg_loss = total_loss / max(len(loader.dataset), 1)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", pos_label=1, zero_division=0
    )
    return avg_loss, acc, precision, recall, f1


def measure_inference_time(model, loader, device, n_batches: int = 20) -> float:
    """Average per-sample inference time in milliseconds, CPU or GPU."""
    model.eval()
    times = []
    with torch.no_grad():
        for i, (x, aux, y) in enumerate(loader):
            if i >= n_batches:
                break
            x, aux = x.to(device), aux.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x, aux)
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            times.append(dt / x.size(0))
    return float(np.mean(times)) * 1000 if times else float("nan")


def get_gpu_info(device: torch.device) -> str:
    if device.type == "cuda":
        return f"{torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})"
    return f"CPU ({platform.processor() or platform.machine()})"


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} -- {get_gpu_info(device)}")

    mel_params = dict(sample_rate=args.sr, n_fft=args.n_fft, hop_length=args.hop_length,
                       win_length=args.win_length, n_mels=args.n_mels)

    train_loader, val_loader, test_loader, train_counts = make_loaders(args, mel_params)

    model = FootstepCNN(n_mels=args.n_mels, dropout=args.dropout, use_aux_features=True).to(device)
    param_breakdown = model.param_breakdown()
    total_params = model.count_params()
    print(f"Model parameters: {total_params:,}")
    print(f"Parameter breakdown: {param_breakdown}")

    class_weights = compute_class_weights(train_counts, device)
    print(f"Class weights (no_footstep, footstep): {class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [],
               "train_f1": [], "val_f1": []}

    best_f1 = -1.0
    best_state = None
    best_epoch = 0
    epochs_without_improve = 0
    epochs_run = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc, train_p, train_r, train_f1 = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc, val_p, val_r, val_f1 = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_f1)
        dt = time.time() - t0
        epochs_run = epoch

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)

        print(f"Epoch {epoch:02d}/{args.epochs} ({dt:.1f}s) | "
              f"train_loss={train_loss:.4f} acc={train_acc:.3f} f1={train_f1:.3f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.3f} P={val_p:.3f} R={val_r:.3f} F1={val_f1:.3f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= args.patience:
                print(f"Early stopping at epoch {epoch} (no val F1 improvement for {args.patience} epochs).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc, test_p, test_r, test_f1 = run_epoch(
        model, test_loader, criterion, optimizer, device, train=False)
    print(f"\nFinal TEST metrics: acc={test_acc:.3f} P={test_p:.3f} R={test_r:.3f} F1={test_f1:.3f}")

    inference_ms = measure_inference_time(model, test_loader, device)

    # --- Plots ---
    import os
    os.makedirs("checkpoints", exist_ok=True)

    epochs_x = list(range(1, len(history["train_loss"]) + 1))

    plt.figure(figsize=(7, 5))
    plt.plot(epochs_x, history["train_loss"], label="Train Loss")
    plt.plot(epochs_x, history["val_loss"], label="Val Loss")
    plt.axvline(best_epoch, color="gray", linestyle="--", alpha=0.5, label=f"Best epoch ({best_epoch})")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss vs Epoch")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("checkpoints/loss_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(epochs_x, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs_x, history["val_acc"], label="Val Accuracy")
    plt.axvline(best_epoch, color="gray", linestyle="--", alpha=0.5, label=f"Best epoch ({best_epoch})")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy vs Epoch")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("checkpoints/accuracy_curve.png", dpi=150)
    plt.close()

    # --- Checkpoint (self-contained: preprocessing params + arch bundled in) ---
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_arch": {"n_mels": args.n_mels, "use_aux_features": True, "dropout": args.dropout},
        "class_names": LABEL_NAMES,
        "sample_rate": args.sr,
        "duration_sec": args.duration_sec,
        "mel_params": mel_params,
        "best_val_f1": best_f1,
    }
    torch.save(checkpoint, args.checkpoint)

    # --- Training report ---
    report_lines = [
        "=" * 60,
        "TRAINING REPORT",
        "=" * 60,
        f"1. Epochs trained: {epochs_run} / {args.epochs} max (best model from epoch {best_epoch})",
        f"2. Device / GPU: {get_gpu_info(device)}",
        f"3. Total parameters: {total_params:,}",
        f"   Parameter breakdown by layer type: {param_breakdown}",
        f"4. Loss function: {criterion.__class__.__name__} (class-weighted: {class_weights.tolist()})",
        f"5. Dropout rate: {args.dropout}",
        f"6. Batch size: {args.batch_size}",
        f"7. Average inference time: {inference_ms:.3f} ms/sample",
        f"8. Optimizer: AdamW (lr={args.lr})",
        f"9. Final test metrics: acc={test_acc:.4f} precision={test_p:.4f} recall={test_r:.4f} f1={test_f1:.4f}",
        f"10. Best validation F1: {best_f1:.4f}",
        "",
        "Plots saved to: checkpoints/loss_curve.png, checkpoints/accuracy_curve.png",
        "=" * 60,
    ]
    report_text = "\n".join(report_lines)
    with open("checkpoints/training_report.txt", "w") as f:
        f.write(report_text + "\n")

    print("\n" + report_text)
    print(f"\nSaved best model (val F1={best_f1:.3f}) to {args.checkpoint}")


if __name__ == "__main__":
    main()
