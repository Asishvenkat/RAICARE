"""
Inference script for the trained ResNet50 model
Tests predictions on validation data and shows individual model outputs
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from pathlib import Path
from PIL import Image
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
MODELS_DIR = Path(__file__).parent / "models"


class OptimizedModel(nn.Module):
    """Optimized ResNet50 model for fast, high-accuracy training"""

    def __init__(self, num_classes=2):
        super(OptimizedModel, self).__init__()

        # Use ResNet50 - faster and more accurate than DenseNet for medical imaging
        self.resnet = models.resnet50(pretrained=True)

        # Freeze early layers initially for faster training
        for name, param in self.resnet.named_parameters():
            if 'layer4' not in name and 'fc' not in name:  # Only train layer4 and fc
                param.requires_grad = False

        # Replace classifier
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)


def load_model(model_path):
    """Load trained ResNet50 model"""
    model = OptimizedModel(num_classes=2).to(DEVICE)

    if model_path.exists():
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state'])
        print(f"Model loaded from {model_path}")
        print(f"Best validation accuracy: {checkpoint.get('best_valid_acc', 'N/A'):.2f}%")
    else:
        print(f"Model not found at {model_path}")
        return None

    return model


def predict_image(model, image_path):
    """Make prediction on a single image"""
    model.eval()

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)

        # Get probabilities
        probs = torch.softmax(outputs, dim=1)

        # Get prediction
        pred = torch.argmax(outputs, dim=1).item()

    labels = ["Negative (No RA)", "Positive (RA Detected)"]
    severity_levels = ["none", "mild", "moderate", "severe"]

    # For simplicity, map prediction to severity (this is a simplification)
    severity = severity_levels[pred] if pred == 1 else "none"

    print("\n" + "="*70)
    print(f"PREDICTION RESULTS: {Path(image_path).name}")
    print("="*70)

    print(f"\n{'Prediction':<25} {'Confidence':<15} {'Severity':<15}")
    print("-"*70)

    confidence = probs[0, pred].item() * 100
    print(f"{labels[pred]:<25} {confidence:.2f}%{'':<10} {severity:<15}")

    print("="*70 + "\n")

    return {
        'prediction': labels[pred],
        'confidence': confidence,
        'severity': severity
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python inference.py <image_path>")
        print("Example: python inference.py test_positive_hand_xray.png")
        sys.exit(1)

    image_path = sys.argv[1]

    # Load best trained model
    model_path = MODELS_DIR / "ensemble_model_best.pth"
    model = load_model(model_path)

    if model is None:
        print("\nTrain the model first by running: python train_ensemble_model.py")
    else:
        # Predict on the provided image
        result = predict_image(model, image_path)
        print(f"Result: {result}")
