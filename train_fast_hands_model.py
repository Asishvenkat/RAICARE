"""
Fast RA training on XR_HAND using a lightweight transfer-learning model.

Designed for much faster training on CPU while still targeting high validation accuracy.
Run:
    python train_fast_hands_model.py
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms


# ---------------------------
# Fast configuration
# ---------------------------
SEED = 42
IMG_SIZE = 160  # Smaller image size for much faster training
BATCH_SIZE = 64
NUM_WORKERS = 0  # Best default for Windows
EPOCHS = 20
EARLY_STOPPING_PATIENCE = 4
UNFREEZE_AT_EPOCH = 3
LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_BACKBONE = 2e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.03

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "MURA-v1.1"
TRAIN_CSV = DATA_DIR / "train_labeled_studies.csv"
VALID_CSV = DATA_DIR / "valid_labeled_studies.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

BEST_MODEL_PATH = MODELS_DIR / "fast_hands_best.pth"
FINAL_MODEL_PATH = MODELS_DIR / "fast_hands_final.pth"
HISTORY_PATH = MODELS_DIR / "fast_hands_history.json"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class HandsDataset(Dataset):
    """Image-level XR_HAND dataset derived from study-level CSV labels."""

    def __init__(self, csv_file: Path, transform: transforms.Compose | None = None):
        self.transform = transform
        self.base_dir = csv_file.parent.parent
        self.image_paths: list[Path] = []
        self.labels: list[int] = []

        with open(csv_file, "r", encoding="utf-8") as f:
            for line in f:
                row = line.strip()
                if not row:
                    continue
                study_path, label = row.split(",")
                if "XR_HAND" not in study_path:
                    continue

                study_dir = self.base_dir / study_path
                if not study_dir.exists():
                    continue

                for img_path in sorted(study_dir.glob("*.png")):
                    self.image_paths.append(img_path)
                    self.labels.append(int(label))

        if not self.image_paths:
            raise ValueError(f"No XR_HAND images found in {csv_file}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[idx]


class FastRAClassifier(nn.Module):
    """EfficientNet-B0 backbone for strong speed/accuracy trade-off."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        self.backbone = models.efficientnet_b0(weights=weights)

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def freeze_backbone(model: FastRAClassifier) -> None:
    for name, param in model.backbone.named_parameters():
        if not name.startswith("classifier"):
            param.requires_grad = False


def unfreeze_backbone(model: FastRAClassifier) -> None:
    for param in model.backbone.parameters():
        param.requires_grad = True


def build_weighted_sampler(labels: list[int]) -> WeightedRandomSampler:
    class_counts = np.bincount(labels)
    class_weights = 1.0 / np.clip(class_counts, a_min=1, a_max=None)
    sample_weights = [class_weights[y] for y in labels]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def latest_epoch_checkpoint() -> tuple[Path | None, int]:
    files = sorted(MODELS_DIR.glob("fast_hands_epoch*.pth"))
    if not files:
        return None, 0
    latest = files[-1]
    try:
        epoch = int(latest.stem.replace("fast_hands_epoch", ""))
    except ValueError:
        return None, 0
    return latest, epoch


@dataclass
class Stats:
    loss: float
    acc: float


def run_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.amp.GradScaler,
) -> Stats:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch_idx, (images, labels) in enumerate(loader, start=1):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        preds = logits.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_count += labels.size(0)

        if batch_idx % 20 == 0:
            print(f"  [train] batch {batch_idx}/{len(loader)} | loss {loss.item():.4f}")

    return Stats(
        loss=total_loss / total_count,
        acc=100.0 * total_correct / total_count,
    )


def run_valid_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> Stats:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)

            preds = logits.argmax(dim=1)
            total_loss += loss.item() * labels.size(0)
            total_correct += (preds == labels).sum().item()
            total_count += labels.size(0)

    return Stats(
        loss=total_loss / total_count,
        acc=100.0 * total_correct / total_count,
    )


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    epoch: int,
    best_valid_acc: float,
    path: Path,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch,
            "best_valid_acc": best_valid_acc,
            "architecture": "efficientnet_b0",
            "img_size": IMG_SIZE,
        },
        path,
    )


def main() -> None:
    set_seed(SEED)

    print("=" * 72)
    print("Fast RA Training on XR_HAND (EfficientNet-B0)")
    print("=" * 72)
    print(f"Device: {DEVICE}")
    print(f"Train CSV: {TRAIN_CSV}")
    print(f"Valid CSV: {VALID_CSV}")

    train_tf = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.97, 1.03)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    valid_tf = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_ds = HandsDataset(TRAIN_CSV, transform=train_tf)
    valid_ds = HandsDataset(VALID_CSV, transform=valid_tf)

    train_neg = sum(1 for y in train_ds.labels if y == 0)
    train_pos = sum(1 for y in train_ds.labels if y == 1)
    print(f"Train images: {len(train_ds)} (negative={train_neg}, positive={train_pos})")
    print(f"Valid images: {len(valid_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=build_weighted_sampler(train_ds.labels),
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    model = FastRAClassifier(num_classes=2).to(DEVICE)
    freeze_backbone(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    head_params = list(model.backbone.classifier.parameters())
    backbone_params = [p for n, p in model.backbone.named_parameters() if not n.startswith("classifier")]

    optimizer = optim.AdamW(
        [
            {"params": head_params, "lr": LEARNING_RATE_HEAD},
            {"params": backbone_params, "lr": LEARNING_RATE_BACKBONE},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    best_valid_acc = 0.0
    best_epoch = 0
    patience = 0
    start_epoch = 1

    history: dict[str, list[float] | int | str] = {
        "train_loss": [],
        "train_acc": [],
        "valid_loss": [],
        "valid_acc": [],
        "epoch_time_sec": [],
        "best_epoch": 0,
        "timestamp": str(np.datetime64("now")),
    }

    resume_path, resume_epoch = latest_epoch_checkpoint()
    if resume_path is not None:
        print(f"Resuming from {resume_path.name} (epoch {resume_epoch})")
        ckpt = torch.load(resume_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        best_valid_acc = ckpt.get("best_valid_acc", 0.0)
        start_epoch = resume_epoch + 1
        if resume_epoch >= UNFREEZE_AT_EPOCH:
            unfreeze_backbone(model)
        if HISTORY_PATH.exists():
            with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                history = json.load(f)
            best_epoch = int(history.get("best_epoch", 0))

    train_start = perf_counter()

    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_start = perf_counter()

        if epoch == UNFREEZE_AT_EPOCH:
            print(f"[Epoch {epoch}] Unfreezing backbone for fine-tuning")
            unfreeze_backbone(model)

        train_stats = run_train_epoch(model, train_loader, criterion, optimizer, scaler)
        valid_stats = run_valid_epoch(model, valid_loader, criterion)
        scheduler.step()

        epoch_sec = perf_counter() - epoch_start

        history["train_loss"].append(round(train_stats.loss, 6))
        history["train_acc"].append(round(train_stats.acc, 4))
        history["valid_loss"].append(round(valid_stats.loss, 6))
        history["valid_acc"].append(round(valid_stats.acc, 4))
        history["epoch_time_sec"].append(round(epoch_sec, 2))

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train {train_stats.acc:.2f}% ({train_stats.loss:.4f}) | "
            f"Valid {valid_stats.acc:.2f}% ({valid_stats.loss:.4f}) | "
            f"{epoch_sec:.1f}s"
        )

        if valid_stats.acc > best_valid_acc:
            best_valid_acc = valid_stats.acc
            best_epoch = epoch
            history["best_epoch"] = best_epoch
            patience = 0
            save_checkpoint(model, optimizer, scheduler, epoch, best_valid_acc, BEST_MODEL_PATH)
            print(f"  New best: {best_valid_acc:.2f}%")
        else:
            patience += 1

        epoch_ckpt = MODELS_DIR / f"fast_hands_epoch{epoch}.pth"
        save_checkpoint(model, optimizer, scheduler, epoch, best_valid_acc, epoch_ckpt)

        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if patience >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    total_minutes = (perf_counter() - train_start) / 60.0
    save_checkpoint(model, optimizer, scheduler, epoch, best_valid_acc, FINAL_MODEL_PATH)

    print("=" * 72)
    print("Training finished")
    print(f"Best validation accuracy: {best_valid_acc:.2f}% at epoch {best_epoch}")
    print(f"Best model: {BEST_MODEL_PATH}")
    print(f"Final model: {FINAL_MODEL_PATH}")
    print(f"History: {HISTORY_PATH}")
    print(f"Elapsed: {total_minutes:.2f} minutes")

    if best_valid_acc >= 95.0:
        print("Target achieved: >=95% validation accuracy.")
    else:
        print("Target not reached yet; try more epochs or GPU fine-tuning.")


if __name__ == "__main__":
    main()
