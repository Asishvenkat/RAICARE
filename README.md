# Rheumatoid Arthritis X-ray Detection Web App

A full-stack web application for detecting Rheumatoid Arthritis from X-ray images using a trained ResNet50 model.

## Tech Stack
- Frontend: React + Tailwind CSS
- Backend: FastAPI (Python)
- Database: MongoDB (Beanie ODM)
- ML Model: ResNet50 (PyTorch, transfer learning)
- External Services: Cloudinary (image hosting), Google Gemini (chat assistant)

## Features
- Authenticated upload and prediction history
- X-ray upload (JPG, JPEG, PNG)
- RA prediction with confidence and severity level
- Cloudinary image storage
- AI chatbot recommendations based on severity

## Model Build and Training

### Dataset
- MURA v1.1 dataset
- Uses XR_HAND studies only
- Train/validation splits from `data/MURA-v1.1/train_labeled_studies.csv` and `data/MURA-v1.1/valid_labeled_studies.csv`

### Preprocessing and Augmentation
- Resize to 224x224
- RGB conversion
- Normalize with ImageNet mean/std
- Training augmentations:
  - Random horizontal flip (p=0.5)
  - Random rotation (15 degrees)
  - Color jitter (brightness/contrast/saturation)

### Model Architecture
- Base: ResNet50 pretrained on ImageNet
- Classifier head: Dropout(0.5) + Linear(2048 -> 2)
- Initial training freezes early layers (all except `layer4` and `fc`)
- Progressive unfreezing: `layer3` unfrozen after epoch 5

### Training Configuration
- Script: `train_ensemble_model.py`
- Framework: PyTorch
- Loss: CrossEntropyLoss
- Optimizer: Adam (lr=0.001)
- Scheduler: CosineAnnealingLR
- Epochs: 8
- Batch size: 8
- Device: CUDA if available, otherwise CPU

### Outputs and Artifacts
- Best checkpoint: `models/ensemble_model_best.pth`
- Final checkpoint: `models/ensemble_model_final.pth`
- Training history: `models/training_history.json`

### Metrics and Visualization
- Generate training curves:
  - Script: `generate_metrics_graphs.py`
  - Output: `models/training_metrics.png`, `models/training_vs_validation.png`

### Inference
- Script: `inference.py`
- Loads `models/ensemble_model_best.pth`
- Outputs class prediction, confidence, and severity label

### Backend Inference Pipeline
- Image preprocessing: resize 224x224, normalize with ImageNet mean/std
- Model: ResNet50 (single active model)
- Output mapping:
  - `result_percentage`: confidence for positive class (0-100)
  - `severity_level`:
    - `none` for negative
    - `mild` for < 60%
    - `moderate` for < 80%
    - `severe` for >= 80%

## Setup Instructions

### Prerequisites
- Python 3.8+
- Node.js 14+
- MongoDB instance
- Cloudinary account
- Gemini API key

### Quick Start (Pre-trained Models Included)
If you just want to run predictions without training:

1. **Get the Dataset:**
   ```bash
   python download_dataset.py
   # This downloads MURA v1.1 dataset to data/ directory
   ```

2. **The trained models are already included** in the repository at `models/ensemble_model_best.pth`

3. **Proceed to Backend Setup** below.

### Full Setup (If Models Not Available)
If you need to retrain the model:

1. Get MURA dataset manually from Stanford ML Group
2. Run training: `python train_ensemble_model.py`
3. Models will be saved to `models/` directory

### Backend Setup
1. Navigate to the backend directory:
   ```
   cd backend
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate  # On Windows
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the backend directory with:
   ```
   MONGODB_URL=your_mongodb_connection_string
   DATABASE_NAME=raicare_db
   SECRET_KEY=your_jwt_secret
   CLOUDINARY_CLOUD_NAME=your_cloudinary_name
   CLOUDINARY_API_KEY=your_cloudinary_key
   CLOUDINARY_API_SECRET=your_cloudinary_secret
   GEMINI_API_KEY=your_gemini_key
   ```

5. Run the backend:
   ```
   uvicorn main:app --reload
   ```

### Frontend Setup
1. Navigate to the frontend directory:
   ```
   cd frontend
   ```

2. Install dependencies:
   ```
   npm install
   ```

3. Start the development server:
   ```
   npm start
   ```

## API Endpoints
- `POST /prediction/upload` - Upload X-ray and save prediction
- `GET /prediction/history` - Get user's prediction history
- `GET /prediction/latest` - Get most recent prediction

## Usage
1. Open the frontend at `http://localhost:3000`
2. Upload an X-ray image (.jpg or .png)
3. View the prediction results and recommendations

## Disclaimer
This project is for research and educational use only. It is not a medical device and should not be used for clinical decisions.
