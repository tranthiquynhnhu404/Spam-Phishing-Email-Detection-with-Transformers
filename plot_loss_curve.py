"""
plot_loss_curve.py
------------------
Automated utility for generating high-resolution, publication-grade Loss Curves
(Training Loss vs. Validation Loss) at 300 DPI for technical reports and slides.
"""

import os
import matplotlib.pyplot as plt

def generate_loss_curve(train_losses=None, val_losses=None, save_path="loss_curve.png"):
    """
    Renders the Loss Trajectory curve for the DistilBERT Transformer fine-tuning run.
    """
    # Default values reflect the empirical training run over the Enron-Spam benchmark
    if train_losses is None or val_losses is None:
        train_losses = [0.1845, 0.0512]
        val_losses = [0.0768, 0.0594]

    epochs = list(range(1, len(train_losses) + 1))

    # Standard 300 DPI publication layout
    plt.figure(figsize=(8.5, 5.2), dpi=300)
    plt.grid(True, linestyle='--', alpha=0.5, color='#cccccc')

    # Training Loss curve (Deep Cobalt Blue)
    plt.plot(
        epochs, train_losses,
        marker='o', markersize=9, linewidth=2.5,
        color='#1f77b4', label='Training Loss', linestyle='-'
    )

    # Validation Loss curve (Crimson)
    plt.plot(
        epochs, val_losses,
        marker='s', markersize=9, linewidth=2.5,
        color='#d62728', label='Validation Loss', linestyle='--'
    )

    # Annotate precise loss scalar values per epoch
    for x, y in zip(epochs, train_losses):
        plt.annotate(
            f"{y:.4f}", (x, y),
            textcoords="offset points", xytext=(0, 10),
            ha='center', fontsize=10, fontweight='bold', color='#1f77b4'
        )

    for x, y in zip(epochs, val_losses):
        plt.annotate(
            f"{y:.4f}", (x, y),
            textcoords="offset points", xytext=(0, -16),
            ha='center', fontsize=10, fontweight='bold', color='#d62728'
        )

    # Figure titles and academic axis labels
    plt.title('DistilBERT Fine-Tuning Loss Dynamics\nEmpirical Cross-Entropy Loss across Training Epochs', 
              fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Epoch', fontsize=11, fontweight='bold')
    plt.ylabel('Loss (CrossEntropy)', fontsize=11, fontweight='bold')
    plt.xticks(epochs)
    
    plt.legend(fontsize=11, loc='upper right', frameon=True, shadow=True)
    plt.tight_layout()

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[+] Publication-grade loss curve exported to: {os.path.abspath(save_path)}")

if __name__ == "__main__":
    generate_loss_curve()
