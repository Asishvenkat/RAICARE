# RA Detection Model (Backend)

This backend uses a single active model for inference: ResNet50.

## Model Used
- **Architecture**: ResNet50 (ImageNet pretrained)
- **Checkpoint**: `models/ensemble_model_best.pth`
- **Classes**: 2 (RA Positive, RA Negative)
- **Device**: CUDA if available, otherwise CPU

## Preprocessing
- Resize to 224x224
- Convert to RGB
- Normalize with ImageNet mean/std

## Outputs
- `prediction`: Positive (RA Detected) or Negative (No RA)
- `result_percentage`: confidence for the positive class (0-100)
- `confidence`: max class confidence (0-100)
- `severity_level`:
  - `none` if negative
  - `mild` if positive and < 60%
  - `moderate` if positive and < 80%
  - `severe` if positive and >= 80%

## Notes
- Only ResNet50 is loaded for inference.
- Additional model code has been removed to keep the backend focused on the active model.
