"""
Train a VGG19-based model on XR_HAND studies from MURA to detect RA.

Key behavior:
- Auto-resumes from the latest epoch checkpoint when training is interrupted.
- Stops early when validation accuracy reaches TARGET_VALID_ACC.
- Uses speed-oriented defaults for faster convergence.

Run:
    python train_vgg19_hands.py
"""

from __future__ import annotations

import argparse
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
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


# ---------------------------
# Default configuration
# ---------------------------
SEED = 42
DEFAULT_IMG_SIZE = 128
DEFAULT_EPOCHS = 12
DEFAULT_BATCH_SIZE = 48
NUM_WORKERS = 2
EARLY_STOPPING_PATIENCE = 5
MIN_EPOCHS_BEFORE_EARLY_STOP = 12
UNFREEZE_AT_EPOCH = 3
TARGET_VALID_ACC = 95.0
DEFAULT_MAX_TRAIN_HOURS = 1.9
LOG_INTERVAL = 25

LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_BACKBONE = 1e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.01

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "MURA-v1.1"
TRAIN_CSV = DATA_DIR / "train_labeled_studies.csv"
VALID_CSV = DATA_DIR / "valid_labeled_studies.csv"

MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

BEST_MODEL_PATH = MODELS_DIR / "vgg19_hands_best.pth"
FINAL_MODEL_PATH = MODELS_DIR / "vgg19_hands_final.pth"
HISTORY_PATH = MODELS_DIR / "vgg19_hands_history.json"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class HandsDataset(Dataset):
    """Image-level XR_HAND dataset derived from study-level labels."""

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

                for image_path in sorted(study_dir.glob("*.png")):
                    self.image_paths.append(image_path)
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


class VGG19HandsClassifier(nn.Module):
    """VGG19 features + lightweight classifier head for faster fine-tuning."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        weights = models.VGG19_Weights.IMAGENET1K_V1
        vgg = models.vgg19(weights=weights)

        self.features = vgg.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return self.classifier(x)


def freeze_backbone(model: VGG19HandsClassifier) -> None:
    for param in model.features.parameters():
        param.requires_grad = False


def unfreeze_backbone(model: VGG19HandsClassifier) -> None:
    for param in model.features.parameters():
        param.requires_grad = True


def unfreeze_last_block(model: VGG19HandsClassifier) -> None:
    # VGG19 block5 starts near index 28 in torchvision's features sequence.
    for param in model.features.parameters():
        param.requires_grad = False
    for layer in model.features[28:]:
        for param in layer.parameters():
            param.requires_grad = True


def build_class_weights(labels: list[int]) -> torch.Tensor:
    class_counts = np.bincount(labels, minlength=2)
    total = class_counts.sum()
    weights = total / (2.0 * np.clip(class_counts, a_min=1, a_max=None))
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def latest_epoch_checkpoint() -> tuple[Path | None, int]:
    checkpoints = sorted(MODELS_DIR.glob("vgg19_hands_epoch*.pth"))
    if not checkpoints:
        return None, 0

    latest = checkpoints[-1]
    try:
        epoch = int(latest.stem.replace("vgg19_hands_epoch", ""))
    except ValueError:
        return None, 0
    return latest, epoch


def maybe_trim_history(history: dict[str, list[float] | int | str], epoch: int) -> None:
    for key in ["train_loss", "train_acc", "valid_loss", "valid_acc", "epoch_time_sec"]:
        if key in history and isinstance(history[key], list):
            history[key] = history[key][:epoch]


@dataclass
class EpochStats:
    loss: float
    acc: float


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.amp.GradScaler,
    log_interval: int,
) -> EpochStats:
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0


    total_batches = len(loader)
    for batch_idx, (images, labels) in enumerate(loader, start=1):
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

        if log_interval > 0 and (batch_idx % log_interval == 0 or batch_idx == total_batches):
            print(f"    Batch {batch_idx}/{total_batches}")

    return EpochStats(
        loss=running_loss / running_total,
        acc=100.0 * running_correct / running_total,
    )


def validate_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> EpochStats:
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

    return EpochStats(
        loss=running_loss / running_total,
        acc=100.0 * running_correct / running_total,
    )


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_valid_acc: float,
    img_size: int,
    path: Path,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "epoch": epoch,
            "best_valid_acc": best_valid_acc,
            "architecture": "vgg19",
            "img_size": img_size,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VGG19 for RA detection on XR_HAND")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Total training epochs")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE, help="Image size")
    parser.add_argument(
        "--max-train-hours",
        type=float,
        default=DEFAULT_MAX_TRAIN_HOURS,
        help="Stop training after this many hours (checkpoint is still saved)",
    )
    parser.add_argument(
        "--full-finetune",
        action="store_true",
        help="Unfreeze VGG19 backbone for full fine-tuning (slower, mainly for GPU)",
    )
    parser.add_argument(
        "--unfreeze-epoch",
        type=int,
        default=UNFREEZE_AT_EPOCH,
        help="Epoch to start backbone fine-tuning (0 disables)",
    )
    parser.add_argument(
        "--unfreeze-mode",
        choices=["none", "last-block", "full"],
        default="last-block",
        help="How much of the backbone to unfreeze for fine-tuning",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable auto-resume from latest checkpoint",
    )
    parser.add_argument(
        "--reset-checkpoints",
        action="store_true",
        help="Delete existing VGG19 checkpoints/history before training",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=EARLY_STOPPING_PATIENCE,
        help="Stop after this many non-improving epochs (set <=0 to disable)",
    )
    parser.add_argument(
        "--min-epochs-before-early-stop",
        type=int,
        default=MIN_EPOCHS_BEFORE_EARLY_STOP,
        help="Do not apply early stopping before this epoch",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    epochs = args.epochs
    batch_size = args.batch_size
    img_size = args.img_size
    unfreeze_epoch = max(0, args.unfreeze_epoch)
    unfreeze_mode = "full" if args.full_finetune else args.unfreeze_mode
    max_train_hours = max(0.0, args.max_train_hours)
    max_train_seconds = max_train_hours * 3600.0
    early_stopping_patience = args.early_stopping_patience
    min_epochs_before_early_stop = max(1, args.min_epochs_before_early_stop)

    set_seed(SEED)

    if args.reset_checkpoints:
        for p in MODELS_DIR.glob("vgg19_hands_epoch*.pth"):
            p.unlink(missing_ok=True)
        BEST_MODEL_PATH.unlink(missing_ok=True)
        FINAL_MODEL_PATH.unlink(missing_ok=True)
        HISTORY_PATH.unlink(missing_ok=True)
        print("Removed previous VGG19 checkpoints and history.")

    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print("=" * 72)
    print("VGG19 Training on XR_HAND (MURA)")
    print("=" * 72)
    print(f"Device: {DEVICE}")
    print(f"Train CSV: {TRAIN_CSV}")
    print(f"Valid CSV: {VALID_CSV}")
    print(f"Config: epochs={epochs}, batch_size={batch_size}, img_size={img_size}")
    print(f"Fine-tune: mode={unfreeze_mode}, unfreeze_epoch={unfreeze_epoch}")
    if early_stopping_patience > 0:
        print(
            "Early stopping: "
            f"patience={early_stopping_patience}, "
            f"min_epoch={min_epochs_before_early_stop}"
        )
    else:
        print("Early stopping: disabled")
    if max_train_hours > 0:
        print(f"Max train time: {max_train_hours:.2f} hours")

    train_transform = transforms.Compose(
        [
            transforms.Resize((img_size + 20, img_size + 20)),
            transforms.RandomCrop((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(8),
            transforms.ColorJitter(brightness=0.12, contrast=0.12),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    valid_transform = transforms.Compose(
        [
            transforms.Resize((img_size + 20, img_size + 20)),
            transforms.CenterCrop((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = HandsDataset(TRAIN_CSV, transform=train_transform)
    valid_dataset = HandsDataset(VALID_CSV, transform=valid_transform)

    train_neg = sum(1 for y in train_dataset.labels if y == 0)
    train_pos = sum(1 for y in train_dataset.labels if y == 1)
    print(f"Train images: {len(train_dataset)} (negative={train_neg}, positive={train_pos})")
    print(f"Valid images: {len(valid_dataset)}")

    num_workers = 0 if DEVICE.type == "cpu" else NUM_WORKERS

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    model = VGG19HandsClassifier(num_classes=2).to(DEVICE)
    if DEVICE.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    freeze_backbone(model)

    class_weights = build_class_weights(train_dataset.labels)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    head_params = list(model.classifier.parameters())
    backbone_params = list(model.features.parameters())

    optimizer = optim.AdamW(
        [
            {"params": head_params, "lr": LEARNING_RATE_HEAD},
            {"params": backbone_params, "lr": LEARNING_RATE_BACKBONE},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
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

    if not args.no_resume:
        resume_path, resume_epoch = latest_epoch_checkpoint()
        if resume_path is not None:
            print(f"Resuming from checkpoint: {resume_path.name} (epoch {resume_epoch})")
            ckpt = torch.load(resume_path, map_location=DEVICE)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            if "scaler_state" in ckpt:
                scaler.load_state_dict(ckpt["scaler_state"])
            best_valid_acc = ckpt.get("best_valid_acc", 0.0)
            start_epoch = resume_epoch + 1

            if unfreeze_epoch > 0 and resume_epoch >= unfreeze_epoch:
                if unfreeze_mode == "full":
                    unfreeze_backbone(model)
                elif unfreeze_mode == "last-block":
                    unfreeze_last_block(model)

            if HISTORY_PATH.exists():
                with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                    history = json.load(f)
                maybe_trim_history(history, resume_epoch)
                best_epoch = int(history.get("best_epoch", 0))

    if start_epoch > epochs:
        print(
            f"Nothing to do: checkpoint epoch ({start_epoch - 1}) already >= requested epochs ({epochs})."
        )
        return

    train_start = perf_counter()

    for epoch in range(start_epoch, epochs + 1):
        elapsed_before_epoch = perf_counter() - train_start
        if max_train_seconds > 0 and elapsed_before_epoch >= max_train_seconds:
            print("Max training time reached before next epoch. Stopping.")
            break

        epoch_start = perf_counter()
        print(f"[Epoch {epoch}/{epochs}] Starting...")

        if unfreeze_epoch > 0 and epoch == unfreeze_epoch:
            if unfreeze_mode == "full":
                print(f"[Epoch {epoch}] Unfreezing full VGG19 backbone")
                unfreeze_backbone(model)
            elif unfreeze_mode == "last-block":
                print(f"[Epoch {epoch}] Unfreezing VGG19 last feature block")
                unfreeze_last_block(model)

        train_stats = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            log_interval=LOG_INTERVAL,
        )
        valid_stats = validate_one_epoch(model, valid_loader, criterion)
        scheduler.step()

        epoch_sec = perf_counter() - epoch_start

        history["train_loss"].append(round(train_stats.loss, 6))
        history["train_acc"].append(round(train_stats.acc, 4))
        history["valid_loss"].append(round(valid_stats.loss, 6))
        history["valid_acc"].append(round(valid_stats.acc, 4))
        history["epoch_time_sec"].append(round(epoch_sec, 2))

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train {train_stats.acc:.2f}% ({train_stats.loss:.4f}) | "
            f"Valid {valid_stats.acc:.2f}% ({valid_stats.loss:.4f}) | "
            f"{epoch_sec:.1f}s"
        )

        if valid_stats.acc > best_valid_acc:
            best_valid_acc = valid_stats.acc
            best_epoch = epoch
            history["best_epoch"] = best_epoch
            patience = 0
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_valid_acc,
                img_size,
                BEST_MODEL_PATH,
            )
            print(f"  New best: {best_valid_acc:.2f}%")
        else:
            patience += 1

        epoch_ckpt = MODELS_DIR / f"vgg19_hands_epoch{epoch}.pth"
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_valid_acc,
            img_size,
            epoch_ckpt,
        )

        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if best_valid_acc >= TARGET_VALID_ACC:
            print(f"Target reached ({TARGET_VALID_ACC:.1f}%+). Stopping early.")
            break

        if (
            early_stopping_patience > 0
            and epoch >= min_epochs_before_early_stop
            and patience >= early_stopping_patience
        ):
            print(
                "Early stopping triggered "
                f"(no improvement for {patience} epochs)."
            )
            break

        elapsed_after_epoch = perf_counter() - train_start
        if max_train_seconds > 0 and elapsed_after_epoch >= max_train_seconds:
            print("Max training time reached. Stopping and saving progress.")
            break

    total_minutes = (perf_counter() - train_start) / 60.0

    save_checkpoint(
        model,
        optimizer,
        scheduler,
        scaler,
        epoch,
        best_valid_acc,
        img_size,
        FINAL_MODEL_PATH,
    )

    print("=" * 72)
    print("Training finished")
    print(f"Best validation accuracy: {best_valid_acc:.2f}% at epoch {best_epoch}")
    print(f"Best model: {BEST_MODEL_PATH}")
    print(f"Final model: {FINAL_MODEL_PATH}")
    print(f"History: {HISTORY_PATH}")
    print(f"Elapsed: {total_minutes:.2f} minutes")

    if best_valid_acc >= TARGET_VALID_ACC:
        print(f"Target achieved: >={TARGET_VALID_ACC:.1f}% validation accuracy.")
    else:
        print("Target not reached yet; continue training or tune hyperparameters.")


if __name__ == "__main__":
    main()
