"""
Prediction Service - RA Detection Model Inference (ResNet50)
"""
import torch
import torch.nn as nn
from torchvision import transforms, models
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Dict, Any

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
MODELS_DIR = Path(__file__).parent.parent.parent.parent / "models"


class OptimizedModel(nn.Module):
    """Optimized ResNet50 model for RA detection"""

    def __init__(self, num_classes=2):
        super(OptimizedModel, self).__init__()

        # Use ResNet50 - same as training
        self.resnet = models.resnet50(pretrained=True)

        # Replace classifier
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)


class PredictionService:
    def __init__(self):
        # Primary model (used for predictions)
        self.resnet_model = None

        self._load_models()

    def _load_models(self):
        """Load ResNet50 model used for predictions"""

        # Load ResNet50
        resnet_path = MODELS_DIR / "ensemble_model_best.pth"
        if resnet_path.exists():
            self.resnet_model = OptimizedModel(num_classes=2).to(DEVICE)
            checkpoint = torch.load(resnet_path, map_location=DEVICE)
            self.resnet_model.load_state_dict(checkpoint['model_state'])
            self.resnet_model.eval()
            print("[OK] ResNet50 model loaded successfully")
        else:
            print(f"[ERROR] ResNet50 model not found at {resnet_path}")
            self.resnet_model = None

    def predict_image(self, image_file) -> Dict[str, Any]:
        """
        Make prediction on an uploaded image file using primary ResNet50 model

        Args:
            image_file: FastAPI UploadFile object

        Returns:
            Dict with prediction results from ResNet50 (primary model)
        """
        if self.resnet_model is None:
            raise Exception("ResNet50 Model not loaded")

        # Load and preprocess image
        image = Image.open(image_file.file).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        image_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            # Primary Model: ResNet50 (used for final prediction)
            outputs = self.resnet_model(image_tensor)

            # Get probabilities
            probs = torch.softmax(outputs, dim=1)

            # Get prediction (1 = RA Positive, 0 = RA Negative)
            pred = torch.argmax(outputs, dim=1).item()
            confidence_positive = probs[0, 1].item() * 100  # Confidence for positive class
            confidence_negative = probs[0, 0].item() * 100  # Confidence for negative class
            
            print(f"DEBUG MODEL OUTPUT - ResNet50 outputs: {outputs}")
            print(f"DEBUG MODEL OUTPUT - Probs: {probs}")
            print(f"DEBUG MODEL OUTPUT - pred={pred}, conf_pos={confidence_positive:.2f}, conf_neg={confidence_negative:.2f}")
            
        # Determine severity based on ResNet50 prediction (primary model)
        if pred == 0:  # Negative (No RA)
            severity_level = "none"
            result_percentage = 0.0  # No RA detected = 0% RA score
        else:  # Positive (RA Detected)
            result_percentage = float(confidence_positive)
            if confidence_positive < 60:
                severity_level = "mild"
            elif confidence_positive < 80:
                severity_level = "moderate"
            else:
                severity_level = "severe"

        print(f"DEBUG - Prediction: pred={pred}, confidence_pos={confidence_positive:.2f}, confidence_neg={confidence_negative:.2f}, result_pct={result_percentage:.2f}")

        return {
            "prediction": "Positive (RA Detected)" if pred == 1 else "Negative (No RA)",
            "result_percentage": float(round(result_percentage, 2)),
            "severity_level": severity_level,
            "confidence": float(round(max(confidence_positive, confidence_negative), 2)),
            "is_positive": pred == 1
        }


# Global prediction service instance
prediction_service = PredictionService()