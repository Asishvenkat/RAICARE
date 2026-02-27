"""
Generate visualizations for model training metrics
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_training_history(filepath):
    """Load training history from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def create_metrics_graphs(history, output_dir='models'):
    """Create and save metric visualization graphs"""
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    epochs = np.arange(1, len(history['train_loss']) + 1)
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Training Metrics', fontsize=16, fontweight='bold')
    
    # Plot 1: Training Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-o', label='Training Loss', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('Loss', fontsize=11)
    axes[0, 0].set_title('Training Loss', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Plot 2: Validation Loss
    axes[0, 1].plot(epochs, history['valid_loss'], 'r-o', label='Validation Loss', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Epoch', fontsize=11)
    axes[0, 1].set_ylabel('Loss', fontsize=11)
    axes[0, 1].set_title('Validation Loss', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Plot 3: Training Accuracy
    axes[1, 0].plot(epochs, history['train_acc'], 'g-o', label='Training Accuracy', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('Epoch', fontsize=11)
    axes[1, 0].set_ylabel('Accuracy (%)', fontsize=11)
    axes[1, 0].set_title('Training Accuracy', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Plot 4: Validation Accuracy
    axes[1, 1].plot(epochs, history['valid_acc'], 'm-o', label='Validation Accuracy', linewidth=2, markersize=6)
    axes[1, 1].set_xlabel('Epoch', fontsize=11)
    axes[1, 1].set_ylabel('Accuracy (%)', fontsize=11)
    axes[1, 1].set_title('Validation Accuracy', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'training_metrics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Metrics graph saved: {output_path}")
    
    # Create combined Loss and Accuracy comparison plot
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle('Training vs Validation Metrics', fontsize=16, fontweight='bold')
    
    # Loss comparison
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Training Loss', linewidth=2, markersize=6)
    ax1.plot(epochs, history['valid_loss'], 'r-o', label='Validation Loss', linewidth=2, markersize=6)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Loss Comparison', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Accuracy comparison
    ax2.plot(epochs, history['train_acc'], 'g-o', label='Training Accuracy', linewidth=2, markersize=6)
    ax2.plot(epochs, history['valid_acc'], 'm-o', label='Validation Accuracy', linewidth=2, markersize=6)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.set_title('Accuracy Comparison', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    output_path2 = Path(output_dir) / 'training_vs_validation.png'
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison graph saved: {output_path2}")
    
    # Print summary statistics
    print("\n" + "="*50)
    print("TRAINING SUMMARY STATISTICS")
    print("="*50)
    print(f"\nTraining Loss:")
    print(f"  Initial: {history['train_loss'][0]:.6f}")
    print(f"  Final:   {history['train_loss'][-1]:.6f}")
    print(f"  Min:     {min(history['train_loss']):.6f} (epoch {np.argmin(history['train_loss']) + 1})")
    
    print(f"\nValidation Loss:")
    print(f"  Initial: {history['valid_loss'][0]:.6f}")
    print(f"  Final:   {history['valid_loss'][-1]:.6f}")
    print(f"  Min:     {min(history['valid_loss']):.6f} (epoch {np.argmin(history['valid_loss']) + 1})")
    
    print(f"\nTraining Accuracy:")
    print(f"  Initial: {history['train_acc'][0]:.2f}%")
    print(f"  Final:   {history['train_acc'][-1]:.2f}%")
    print(f"  Max:     {max(history['train_acc']):.2f}% (epoch {np.argmax(history['train_acc']) + 1})")
    
    print(f"\nValidation Accuracy:")
    print(f"  Initial: {history['valid_acc'][0]:.2f}%")
    print(f"  Final:   {history['valid_acc'][-1]:.2f}%")
    print(f"  Max:     {max(history['valid_acc']):.2f}% (epoch {np.argmax(history['valid_acc']) + 1})")
    print("="*50)
    
    plt.show()

if __name__ == "__main__":
    # Load training history
    history_path = Path('models/training_history.json')
    
    if history_path.exists():
        history = load_training_history(history_path)
        create_metrics_graphs(history)
    else:
        print(f"Error: {history_path} not found!")
