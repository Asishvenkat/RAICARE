"""
Train a DenseNet121 model on XR_HAND studies from MURA to detect RA.

Goal: maximize validation performance on hand-only data with robust transfer learning.
Run:
    python train_densenet121_hands.py
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
# Configuration
# ---------------------------

SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0          # 0 required on Windows (spawn-based multiprocessing)
EPOCHS = 30
EARLY_STOPPING_PATIENCE = 7
LEARNING_RATE_BACKBONE = 1e-4
LEARNING_RATE_CLASSIFIER = 5e-4
WEIGHT_DECAY = 1e-4
UNFREEZE_AT_EPOCH = 5
LABEL_SMOOTHING = 0.05

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "MURA-v1.1"
TRAIN_CSV = DATA_DIR / "train_labeled_studies.csv"
VALID_CSV = DATA_DIR / "valid_labeled_studies.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

BEST_MODEL_PATH = MODELS_DIR / "densenet121_hands_best.pth"
FINAL_MODEL_PATH = MODELS_DIR / "densenet121_hands_final.pth"
HISTORY_PATH = MODELS_DIR / "densenet121_hands_history.json"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class HandsStudyDataset(Dataset):
    """Expand XR_HAND study labels into image-level samples."""

    def __init__(self, csv_file: Path, transform: transforms.Compose | None = None):
        self.csv_file = csv_file
        self.transform = transform
        self.base_dir = csv_file.parent.parent
        self.image_paths: list[Path] = []
        self.labels: list[int] = []

        with open(csv_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                study_path, label = line.split(",")
                if "XR_HAND" not in study_path:
                    continue

                study_dir = self.base_dir / study_path
                if not study_dir.exists():
                    continue

                for image_file in sorted(study_dir.glob("*.png")):
                    self.image_paths.append(image_file)
                    self.labels.append(int(label))

        if len(self.image_paths) == 0:
            raise ValueError(f"No XR_HAND images found in {csv_file}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return image, label


class DenseNet121RA(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        weights = models.DenseNet121_Weights.IMAGENET1K_V1
        self.backbone = models.densenet121(weights=weights)

        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.35),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def find_latest_checkpoint() -> tuple[Path | None, int]:
    """Return the path and epoch number of the most recent epoch checkpoint, or (None, 0)."""
    checkpoints = sorted(MODELS_DIR.glob("densenet121_hands_epoch*.pth"))
    if not checkpoints:
        return None, 0
    latest = checkpoints[-1]
    # Extract epoch number from filename, e.g. densenet121_hands_epoch3.pth -> 3
    try:
        epoch_num = int(latest.stem.replace("densenet121_hands_epoch", ""))
    except ValueError:
        return None, 0
    return latest, epoch_num


def freeze_backbone_except_classifier(model: DenseNet121RA) -> None:
    for name, param in model.backbone.named_parameters():
        if not name.startswith("classifier"):
            param.requires_grad = False


def unfreeze_all_layers(model: DenseNet121RA) -> None:
    for param in model.backbone.parameters():
        param.requires_grad = True


@dataclass
class EpochStats:
    loss: float
    acc: float


def build_sampler(labels: list[int]) -> WeightedRandomSampler:
    class_counts = np.bincount(labels)
    class_weights = 1.0 / np.clip(class_counts, a_min=1, a_max=None)
    sample_weights = [class_weights[label] for label in labels]

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.amp.GradScaler,
) -> EpochStats:
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds = outputs.argmax(dim=1)
            running_loss += loss.item() * labels.size(0)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

            # Print progress every batch
            print(f"  [train] Batch: {running_total}/{len(loader.dataset)} | Loss: {loss.item():.4f}")

    return EpochStats(
        loss=running_loss / running_total,
        acc=100.0 * running_correct / running_total,
    )


def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> EpochStats:
    model.eval()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            preds = outputs.argmax(dim=1)
            running_loss += loss.item() * labels.size(0)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

            # Print progress every batch
            print(f"  [valid] Batch: {running_total}/{len(loader.dataset)} | Loss: {loss.item():.4f}")

    return EpochStats(
        loss=running_loss / running_total,
        acc=100.0 * running_correct / running_total,
    )


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    epoch: int,
    best_valid_acc: float,
    save_path: Path,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch,
            "best_valid_acc": best_valid_acc,
            "architecture": "densenet121",
            "img_size": IMG_SIZE,
        },
        save_path,
    )


def main() -> None:
    set_seed(SEED)

    print("=" * 72)
    print("DenseNet121 Training on XR_HAND (MURA)")
    print("=" * 72)
    print(f"Device: {DEVICE}")
    print(f"Train CSV: {TRAIN_CSV}")
    print(f"Valid CSV: {VALID_CSV}")

    train_transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(12),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    valid_transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = HandsStudyDataset(TRAIN_CSV, transform=train_transform)
    valid_dataset = HandsStudyDataset(VALID_CSV, transform=valid_transform)

    sampler = build_sampler(train_dataset.labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    train_neg = sum(1 for y in train_dataset.labels if y == 0)
    train_pos = sum(1 for y in train_dataset.labels if y == 1)

    print(f"Train images: {len(train_dataset)} (negative={train_neg}, positive={train_pos})")
    print(f"Valid images: {len(valid_dataset)}")

    model = DenseNet121RA(num_classes=2).to(DEVICE)
    freeze_backbone_except_classifier(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    classifier_params = list(model.backbone.classifier.parameters())
    backbone_params = [
        p for n, p in model.backbone.named_parameters() if not n.startswith("classifier")
    ]

    optimizer = optim.AdamW(
        [
            {"params": classifier_params, "lr": LEARNING_RATE_CLASSIFIER},
            {"params": backbone_params, "lr": LEARNING_RATE_BACKBONE},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    best_valid_acc = 0.0
    best_epoch = -1
    patience_counter = 0
    start_epoch = 1

    history: dict[str, list[float] | int | str] = {
        "train_loss": [],
        "train_acc": [],
        "valid_loss": [],
        "valid_acc": [],
        "epoch_time_sec": [],
        "best_epoch": -1,
        "timestamp": str(np.datetime64("now")),
    }

    # --- Auto-resume from latest epoch checkpoint ---
    resume_ckpt, resume_epoch = find_latest_checkpoint()
    if resume_ckpt is not None:
        print(f"\nResuming from checkpoint: {resume_ckpt.name} (epoch {resume_epoch})")
        ckpt = torch.load(resume_ckpt, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        best_valid_acc = ckpt.get("best_valid_acc", 0.0)
        start_epoch = resume_epoch + 1
        # If backbone was already unfrozen before the stop, keep it unfrozen
        if resume_epoch >= UNFREEZE_AT_EPOCH:
            unfreeze_all_layers(model)
            print("  Backbone restored to unfrozen state.")
        # Reload history if available
        if HISTORY_PATH.exists():
            with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                history = json.load(f)
            best_epoch = history.get("best_epoch", -1)
            print(f"  History loaded. Best so far: {best_valid_acc:.2f}% at epoch {best_epoch}")
    else:
        print("No checkpoint found — starting fresh.")

    training_start = perf_counter()

    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_start = perf_counter()

        if epoch == UNFREEZE_AT_EPOCH:
            print(f"\n[Epoch {epoch}] Unfreezing full DenseNet backbone for fine-tuning")
            unfreeze_all_layers(model)

        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        valid_stats = validate_one_epoch(model, valid_loader, criterion)
        scheduler.step()

        elapsed = perf_counter() - epoch_start

        history["train_loss"].append(round(train_stats.loss, 6))
        history["train_acc"].append(round(train_stats.acc, 4))
        history["valid_loss"].append(round(valid_stats.loss, 6))
        history["valid_acc"].append(round(valid_stats.acc, 4))
        history["epoch_time_sec"].append(round(elapsed, 2))

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_stats.loss:.4f} | Train Acc: {train_stats.acc:.2f}% | "
            f"Valid Loss: {valid_stats.loss:.4f} | Valid Acc: {valid_stats.acc:.2f}% | "
            f"Time: {elapsed:.1f}s"
        )

        if valid_stats.acc > best_valid_acc:
            best_valid_acc = valid_stats.acc
            best_epoch = epoch
            history["best_epoch"] = best_epoch
            patience_counter = 0
            save_checkpoint(model, optimizer, scheduler, epoch, best_valid_acc, BEST_MODEL_PATH)
            print(f"  -> New best checkpoint saved: {BEST_MODEL_PATH.name}")
        else:
            patience_counter += 1

        # Save checkpoint at end of every epoch
        epoch_ckpt_path = MODELS_DIR / f"densenet121_hands_epoch{epoch}.pth"
        save_checkpoint(model, optimizer, scheduler, epoch, best_valid_acc, epoch_ckpt_path)
        print(f"  -> Epoch checkpoint saved: {epoch_ckpt_path.name}")

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    total_elapsed = perf_counter() - training_start

    save_checkpoint(model, optimizer, scheduler, epoch, best_valid_acc, FINAL_MODEL_PATH)

    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 72)
    print("Training complete")
    print(f"Best validation accuracy: {best_valid_acc:.2f}% at epoch {best_epoch}")
    print(f"Best model: {BEST_MODEL_PATH}")
    print(f"Final model: {FINAL_MODEL_PATH}")
    print(f"History: {HISTORY_PATH}")
    print(f"Total training time: {total_elapsed / 60:.2f} minutes")

    if best_valid_acc >= 95.0:
        print("Target achieved: validation accuracy is >= 95%.")
    else:
        print(
            "Target not reached yet. Try increasing EPOCHS, tuning augmentation, "
            "or training with a stronger GPU for longer fine-tuning."
        )


if __name__ == "__main__":
    main()
