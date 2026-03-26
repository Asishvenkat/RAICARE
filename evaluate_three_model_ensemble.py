"""
Evaluate and combine 3 trained hand-Xray models for maximum validation accuracy.

Models used:
- ensemble_model_best.pth (ResNet50)
- densenet121_hands_best.pth (DenseNet121)
- fast_hands_best.pth (EfficientNet-B0)

Run:
    python evaluate_three_model_ensemble.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "MURA-v1.1"
VALID_CSV = DATA_DIR / "valid_labeled_studies.csv"
MODELS_DIR = BASE_DIR / "models"

RESULTS_PATH = MODELS_DIR / "three_model_ensemble_results.json"


# ---------------------------
# Dataset (XR_HAND only)
# ---------------------------
class HandsValidDataset:
    def __init__(self, csv_file: Path):
        self.image_paths: list[Path] = []
        self.labels: list[int] = []
        base_dir = csv_file.parent.parent

        with open(csv_file, "r", encoding="utf-8") as f:
            for row in f:
                row = row.strip()
                if not row:
                    continue

                study_path, label = row.split(",")
                if "XR_HAND" not in study_path:
                    continue

                study_dir = base_dir / study_path
                if not study_dir.exists():
                    continue

                for img_path in sorted(study_dir.glob("*.png")):
                    self.image_paths.append(img_path)
                    self.labels.append(int(label))

        if not self.image_paths:
            raise ValueError("No XR_HAND validation images found")


# ---------------------------
# Model definitions
# ---------------------------
class ResNet50RA(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(2048, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)


class DenseNet121RA(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.35),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)



# VGG19 model for ensemble
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


@dataclass
class EvalResult:
    acc: float
    precision: float
    recall: float
    f1: float
    tp: int
    tn: int
    fp: int
    fn: int


def load_checkpoint(model: nn.Module, ckpt_name: str) -> nn.Module:
    ckpt_path = MODELS_DIR / ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()
    print(f"Loaded {ckpt_name} (best_valid_acc={ckpt.get('best_valid_acc', 'n/a')})")
    return model


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> EvalResult:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    acc = 100.0 * (tp + tn) / len(y_true)
    precision = 100.0 * tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = 100.0 * tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return EvalResult(
        acc=acc,
        precision=precision,
        recall=recall,
        f1=f1,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
    )


def main() -> None:
    print("=" * 72)
    print("3-Model Ensemble Evaluation (XR_HAND validation)")
    print("=" * 72)
    print(f"Device: {DEVICE}")

    dataset = HandsValidDataset(VALID_CSV)
    y_true = np.array(dataset.labels, dtype=np.int64)
    print(f"Validation samples: {len(dataset.image_paths)}")

    # Model-specific transforms to match training setups.
    tf_224 = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


    resnet = load_checkpoint(ResNet50RA(), "ensemble_model_best.pth")
    densenet = load_checkpoint(DenseNet121RA(), "densenet121_hands_best.pth")
    vgg19 = load_checkpoint(VGG19HandsClassifier(), "vgg19_hands_best.pth")


    probs_resnet: list[float] = []
    probs_densenet: list[float] = []
    probs_vgg19: list[float] = []


    with torch.no_grad():
        for idx, img_path in enumerate(dataset.image_paths, start=1):
            image = Image.open(img_path).convert("RGB")

            x224 = tf_224(image).unsqueeze(0).to(DEVICE)

            p_res = torch.softmax(resnet(x224), dim=1)[0, 1].item()
            p_den = torch.softmax(densenet(x224), dim=1)[0, 1].item()
            p_vgg = torch.softmax(vgg19(x224), dim=1)[0, 1].item()

            probs_resnet.append(p_res)
            probs_densenet.append(p_den)
            probs_vgg19.append(p_vgg)

            if idx % 100 == 0:
                print(f"Processed {idx}/{len(dataset.image_paths)} images")


    p_res = np.array(probs_resnet, dtype=np.float64)
    p_den = np.array(probs_densenet, dtype=np.float64)
    p_vgg = np.array(probs_vgg19, dtype=np.float64)

    # Individual model metrics
    pred_res = (p_res >= 0.5).astype(np.int64)
    pred_den = (p_den >= 0.5).astype(np.int64)
    pred_vgg = (p_vgg >= 0.5).astype(np.int64)

    metrics_res = evaluate_predictions(y_true, pred_res)
    metrics_den = evaluate_predictions(y_true, pred_den)
    metrics_vgg = evaluate_predictions(y_true, pred_vgg)


    print("\nIndividual model accuracy:")
    print(f"- ResNet50:     {metrics_res.acc:.2f}%")
    print(f"- DenseNet121:  {metrics_den.acc:.2f}%")
    print(f"- VGG19:        {metrics_vgg.acc:.2f}%")


    # Weight search to maximize validation accuracy.
    best_acc = -1.0
    best_w = (1.0, 0.0, 0.0)
    best_pred = pred_res

    step = 0.05
    grid = np.arange(0.0, 1.0 + step, step)
    for w1 in grid:
        for w2 in grid:
            w3 = 1.0 - w1 - w2
            if w3 < 0.0:
                continue
            p_ens = w1 * p_res + w2 * p_den + w3 * p_vgg
            pred = (p_ens >= 0.5).astype(np.int64)
            acc = 100.0 * np.mean(pred == y_true)
            if acc > best_acc:
                best_acc = acc
                best_w = (float(w1), float(w2), float(w3))
                best_pred = pred


    metrics_ens = evaluate_predictions(y_true, best_pred)

    print("\nBest weighted ensemble:")
    print(
        "- weights (ResNet, DenseNet, VGG19): "
        f"({best_w[0]:.2f}, {best_w[1]:.2f}, {best_w[2]:.2f})"
    )
    print(f"- Accuracy:  {metrics_ens.acc:.2f}%")
    print(f"- Precision: {metrics_ens.precision:.2f}%")
    print(f"- Recall:    {metrics_ens.recall:.2f}%")
    print(f"- F1 score:  {metrics_ens.f1:.2f}%")
    print(
        f"- Confusion matrix: TP={metrics_ens.tp}, TN={metrics_ens.tn}, "
        f"FP={metrics_ens.fp}, FN={metrics_ens.fn}"
    )


    payload = {
        "device": str(DEVICE),
        "validation_samples": int(len(y_true)),
        "individual": {
            "resnet50": metrics_res.__dict__,
            "densenet121": metrics_den.__dict__,
            "vgg19": metrics_vgg.__dict__,
        },
        "ensemble": {
            "weights": {
                "resnet50": best_w[0],
                "densenet121": best_w[1],
                "vgg19": best_w[2],
            },
            "metrics": metrics_ens.__dict__,
        },
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved evaluation report: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
