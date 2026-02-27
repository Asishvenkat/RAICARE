"""
DenseNet Training: Fast single model training
Trains on 50% of XR_HAND images from MURA dataset
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from pathlib import Path
import numpy as np
from PIL import Image
import json
from datetime import datetime
from time import perf_counter

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8  # Increased for faster training
EPOCHS = 8  # Reduced for faster training while maintaining accuracy
LEARNING_RATE = 0.001  # Higher LR for faster convergence
IMG_SIZE = 224
DATA_DIR = Path(__file__).parent / "data" / "MURA-v1.1"
TRAIN_CSV = DATA_DIR / "train_labeled_studies.csv"
VALID_CSV = DATA_DIR / "valid_labeled_studies.csv"
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

print(f"Using device: {DEVICE}")
print(f"Training CSV: {TRAIN_CSV}")
print(f"Validation CSV: {VALID_CSV}")


class XRayDataset(Dataset):
    """Custom dataset for MURA XR_HAND images"""
    
    def __init__(self, csv_file, transform=None):
        self.csv_file = Path(csv_file)
        self.transform = transform
        self.images = []
        self.labels = []
        self.base_dir = self.csv_file.parent.parent  # Go up to data dir
        
        # Read CSV and filter XR_HAND studies only
        print(f"Loading from {self.csv_file}...")
        with open(self.csv_file, 'r') as f:
            for line in f:
                study_path, label = line.strip().split(',')
                
                # Only process XR_HAND studies
                if 'XR_HAND' in study_path:
                    full_study_path = self.base_dir / study_path
                    
                    # Get all images in this study folder
                    if full_study_path.exists():
                        for img_file in full_study_path.glob("*.png"):
                            self.images.append(str(img_file))
                            self.labels.append(int(label))
        
        print(f"Loaded {len(self.images)} XR_HAND images")
        if len(self.images) == 0:
            raise ValueError(f"No XR_HAND images found in {self.csv_file}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image on error
            image = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


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


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    print("Training epoch...")
    start_time = perf_counter()
    for i, (images, labels) in enumerate(train_loader):
        if i == 0:
            print(f"Processing first batch at {datetime.now().strftime('%H:%M:%S')}...")
        batch_start = perf_counter()
        
        images, labels = images.to(device), labels.to(device)
        if i == 0:
            print(f"Data moved to device at {datetime.now().strftime('%H:%M:%S')}")
        
        # Forward pass
        outputs = model(images)
        if i == 0:
            print(f"Forward pass completed at {datetime.now().strftime('%H:%M:%S')}")
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_time = perf_counter() - batch_start
        if i == 0:
            print(f"First batch completed in {batch_time:.2f}s at {datetime.now().strftime('%H:%M:%S')}")
        
        # Metrics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Print progress every 10 batches (reduced for debugging)
        if (i + 1) % 10 == 0 or i == len(train_loader) - 1:
            current_acc = 100 * correct / total
            print(f"  Batch {i+1:3d}/{len(train_loader):3d} | Loss: {loss.item():.4f} | Acc: {current_acc:.2f}% | Time: {batch_time:.2f}s")
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def validate(model, valid_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    print("Validating...")
    with torch.no_grad():
        for i, (images, labels) in enumerate(valid_loader):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Print progress every 10 batches
            if (i + 1) % 10 == 0 or i == len(valid_loader) - 1:
                current_acc = 100 * correct / total
                print(f"  Batch {i+1:2d}/{len(valid_loader):2d} | Loss: {loss.item():.4f} | Acc: {current_acc:.2f}%")
    
    avg_loss = total_loss / len(valid_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def main():
    print("\n" + "="*60)
    print("DENSENET TRAINING: Fast single model")
    print("="*60)
    
    # Enhanced data augmentation for better accuracy
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    valid_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    print("\nLoading datasets...")
    try:
        train_dataset = XRayDataset(TRAIN_CSV, transform=train_transform)
        valid_dataset = XRayDataset(VALID_CSV, transform=valid_transform)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Valid samples: {len(valid_dataset)}")
    
    # Initialize model
    print("\nInitializing Optimized ResNet50 model...")
    model = OptimizedModel(num_classes=2).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Cosine annealing scheduler for faster convergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    # Training loop with progressive unfreezing and early stopping
    best_valid_acc = 0.0
    patience_counter = 0
    best_epoch = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'valid_loss': [],
        'valid_acc': [],
        'epoch_time_sec': [],
        'timestamp': datetime.now().isoformat()
    }
    
    print("\n" + "="*60)
    print("🚀 OPTIMIZED RESNET50 TRAINING (Fast + High Accuracy)")
    print("="*60)
    print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target: >80% validation accuracy in less time")
    total_start = perf_counter()
    
    for epoch in range(EPOCHS):
        epoch_start = perf_counter()
        
        # Progressive unfreezing: unfreeze more layers after epoch 5
        if epoch == 5:
            print("🔥 Unfreezing more layers for fine-tuning...")
            for name, param in model.resnet.named_parameters():
                if 'layer3' in name:  # Unfreeze layer3 as well
                    param.requires_grad = True
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                           optimizer, DEVICE)
        valid_loss, valid_acc = validate(model, valid_loader, criterion, DEVICE)
        epoch_time = perf_counter() - epoch_start
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['valid_loss'].append(valid_loss)
        history['valid_acc'].append(valid_acc)
        history['epoch_time_sec'].append(epoch_time)
        
        scheduler.step()  # Cosine annealing doesn't need validation loss
        
        print(f"Epoch {epoch+1:2d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.2f}% | "
              f"Time: {epoch_time:.1f}s")
        
        # Calculate and display time remaining
        if epoch >= 1:  # Start estimating after first epoch
            avg_epoch_time = sum(history['epoch_time_sec']) / len(history['epoch_time_sec'])
            remaining_epochs = EPOCHS - (epoch + 1)
            estimated_remaining = avg_epoch_time * remaining_epochs
            print(f"  ⏱️  Estimated time remaining: {estimated_remaining/60:.1f} minutes")
        
        # Save best model
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save({
                'model_state': model.state_dict(),
                'best_valid_acc': best_valid_acc,
                'epoch': epoch + 1
            }, MODELS_DIR / "ensemble_model_best.pth")
            print(f"  ✓ Best model saved (Valid Acc: {valid_acc:.2f}%)")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= 5 and epoch >= 8:  # Minimum 8 epochs
            print(f"🛑 Early stopping at epoch {epoch+1} (no improvement for {patience_counter} epochs)")
            break
    
    # Save final model and history
    torch.save({
        'model_state': model.state_dict(),
        'best_valid_acc': best_valid_acc,
        'final_epoch': best_epoch if best_epoch > 0 else EPOCHS
    }, MODELS_DIR / "ensemble_model_final.pth")
    
    with open(MODELS_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    total_time = perf_counter() - total_start
    print("\n" + "="*60)
    print("🚀 OPTIMIZED TRAINING COMPLETED")
    print("="*60)
    print(f"Best Validation Accuracy: {best_valid_acc:.2f}% (Epoch {best_epoch})")
    print(f"Total Training Time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"Final Model Saved: {MODELS_DIR / 'ensemble_model_best.pth'}")
    print("="*60)
    total_time = perf_counter() - total_start
    print(f"Total training time: {total_time/60:.1f} minutes")
    print(f"Best Validation Accuracy: {best_valid_acc:.2f}%")
    print(f"Models saved to: {MODELS_DIR}")
    print(f"  - ensemble_model_best.pth")
    print(f"  - ensemble_model_final.pth")
    print(f"  - training_history.json")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
