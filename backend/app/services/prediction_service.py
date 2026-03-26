"""Prediction Service - 3-model ensemble RA detection inference."""

import json
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_DIR = Path(__file__).parent.parent.parent.parent / "models"
ENSEMBLE_RESULTS_PATH = MODELS_DIR / "three_model_ensemble_results.json"


class ResNet50RA(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)


class DenseNet121RA(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.backbone = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
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



class PredictionService:
    def __init__(self):
        self.resnet_model: nn.Module | None = None
        self.densenet_model: nn.Module | None = None
        self.vgg19_model: nn.Module | None = None

        # Default equal weights; overridden if tuned ensemble results exist.
        self.weights = {
            "resnet50": 1.0 / 3.0,
            "densenet121": 1.0 / 3.0,
            "vgg19": 1.0 / 3.0,
        }

        self.tf_224 = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )


        self._load_models()
        self._load_ensemble_weights()

    def _load_checkpoint_into_model(self, model: nn.Module, model_path: Path, model_name: str) -> nn.Module | None:
        if not model_path.exists():
            print(f"[WARN] {model_name} checkpoint not found: {model_path}")
            return None

        checkpoint = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state"])
        model.to(DEVICE)
        model.eval()
        print(f"[OK] {model_name} loaded")
        return model

    def _load_models(self) -> None:
        self.resnet_model = self._load_checkpoint_into_model(
            ResNet50RA(num_classes=2),
            MODELS_DIR / "ensemble_model_best.pth",
            "ResNet50",
        )
        self.densenet_model = self._load_checkpoint_into_model(
            DenseNet121RA(num_classes=2),
            MODELS_DIR / "densenet121_hands_best.pth",
            "DenseNet121",
        )
        self.vgg19_model = self._load_checkpoint_into_model(
            VGG19HandsClassifier(num_classes=2),
            MODELS_DIR / "vgg19_hands_best.pth",
            "VGG19",
        )

    def _load_ensemble_weights(self) -> None:
        if not ENSEMBLE_RESULTS_PATH.exists():
            print("[INFO] No tuned ensemble weights found; using equal weights")
            return

        try:
            with open(ENSEMBLE_RESULTS_PATH, "r", encoding="utf-8") as f:
                payload = json.load(f)
            weights = payload.get("ensemble", {}).get("weights", {})
            if set(weights.keys()) == {"resnet50", "densenet121", "vgg19"}:
                self.weights = {
                    "resnet50": float(weights["resnet50"]),
                    "densenet121": float(weights["densenet121"]),
                    "vgg19": float(weights["vgg19"]),
                }
                print(f"[OK] Loaded tuned ensemble weights: {self.weights}")
        except Exception as exc:
            print(f"[WARN] Failed to parse ensemble weights, using defaults: {exc}")

    def _predict_positive_prob(self, model: nn.Module, image_tensor: torch.Tensor) -> float:
        with torch.no_grad():
            logits = model(image_tensor)
            prob_pos = torch.softmax(logits, dim=1)[0, 1].item()
        return prob_pos

    def predict_image(self, image_file) -> Dict[str, Any]:
        image = Image.open(image_file.file).convert("RGB")

        available_models = []

        if self.resnet_model is not None:
            x224 = self.tf_224(image).unsqueeze(0).to(DEVICE)
            p_res = self._predict_positive_prob(self.resnet_model, x224)
            available_models.append(("resnet50", p_res))

        if self.densenet_model is not None:
            x224 = self.tf_224(image).unsqueeze(0).to(DEVICE)
            p_den = self._predict_positive_prob(self.densenet_model, x224)
            available_models.append(("densenet121", p_den))


        if self.vgg19_model is not None:
            x224 = self.tf_224(image).unsqueeze(0).to(DEVICE)
            p_vgg = self._predict_positive_prob(self.vgg19_model, x224)
            available_models.append(("vgg19", p_vgg))

        if not available_models:
            raise Exception("No prediction model is loaded")

        # Normalize active weights only across currently loaded models.
        active_weight_sum = sum(self.weights[name] for name, _ in available_models)
        ensemble_positive = sum((self.weights[name] / active_weight_sum) * prob for name, prob in available_models)

        pred = 1 if ensemble_positive >= 0.5 else 0
        confidence_positive = ensemble_positive * 100.0
        confidence_negative = (1.0 - ensemble_positive) * 100.0

        if pred == 0:
            severity_level = "none"
            result_percentage = 0.0
        else:
            result_percentage = float(confidence_positive)
            if confidence_positive < 60:
                severity_level = "mild"
            elif confidence_positive < 80:
                severity_level = "moderate"
            else:
                severity_level = "severe"

        per_model_probs = {name: round(prob * 100.0, 2) for name, prob in available_models}
        print(
            "[DEBUG] Ensemble probs(%): "
            f"{per_model_probs} | combined={confidence_positive:.2f}%"
        )

        return {
            "prediction": "Positive (RA Detected)" if pred == 1 else "Negative (No RA)",
            "result_percentage": float(round(result_percentage, 2)),
            "severity_level": severity_level,
            "confidence": float(round(max(confidence_positive, confidence_negative), 2)),
            "is_positive": pred == 1,
            "ensemble": {
                "weights": self.weights,
                "model_positive_probabilities": per_model_probs,
                "combined_positive_probability": float(round(confidence_positive, 2)),
            },
        }


# Global prediction service instance
prediction_service = PredictionService()