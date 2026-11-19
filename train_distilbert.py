"""
train_distilbert.py
-------------------
Fine-tuning the DistilBERT Transformer architecture (distilbert-base-uncased)
for binary email classification (Spam vs. Ham) using an explicit PyTorch DataLoader loop.

Key Technical Specifications:
- PyTorch DataLoader and AdamW optimization with gradient clipping for transparent training dynamics.
- Automated hardware detection (NVIDIA CUDA GPU with graceful CPU fallback).
- Configurable hyperparameters via CLI: --epochs, --batch_size, --max_length, --lr, --sample_size.
- Model checkpointing: Best validation F1-score weights and tokenizer saved to models/distilbert/.
- Automated visualization: Generates high-resolution publication-ready Loss Curves (300 DPI).
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score

TRAIN_CSV = "data/train.csv"
VAL_CSV = "data/val.csv"
MODEL_DIR = "models/distilbert"
BASE_MODEL_NAME = "distilbert-base-uncased"

class EmailDataset(Dataset):
    """PyTorch Dataset wrapper for textual token sequences and target labels."""
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }

def evaluate_model(model, dataloader, device):
    """Evaluate current model performance over validation DataLoader."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * len(labels)

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return avg_loss, acc, f1

def predict_proba(texts, model, tokenizer, device, batch_size=16, max_length=256):
    """
    Inference utility computing normalized softmax posterior probabilities [P(Ham), P(Spam)].
    Processes in mini-batches under torch.no_grad() for bounded memory consumption.
    """
    model.eval()
    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch_texts = [str(t) for t in texts[i:i + batch_size]]
        encoding = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

    if len(all_probs) == 0:
        return np.empty((0, 2))
    return np.vstack(all_probs)

def train(args):
    if not os.path.exists(TRAIN_CSV):
        raise FileNotFoundError(
            f"Dataset file {TRAIN_CSV} not found. Please run 'python prepare_data.py' first!"
        )

    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Compute Hardware Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute target device: {device}")
    if device.type == "cuda":
        print(f"    - Device Name: {torch.cuda.get_device_name(0)}")
        print(f"    - Total VRAM:  {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    print("[*] Reading partitioned dataset files...")
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)

    if args.sample_size and args.sample_size < len(train_df):
        print(f"[*] Fast-prototyping mode: sub-sampling {args.sample_size:,} records from training split...")
        train_df = train_df.sample(n=args.sample_size, random_state=42).reset_index(drop=True)
        val_sample = min(len(val_df), max(500, args.sample_size // 5))
        val_df = val_df.sample(n=val_sample, random_state=42).reset_index(drop=True)

    print(f"[*] Training samples:   {len(train_df):,}")
    print(f"[*] Validation samples: {len(val_df):,}")

    print(f"[*] Initializing Tokenizer and Transformer backbone: {BASE_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_NAME, num_labels=2)
    model.to(device)

    # Initialize Dataset and DataLoader instances
    train_dataset = EmailDataset(train_df["text"], train_df["label"], tokenizer, max_length=args.max_length)
    val_dataset = EmailDataset(val_df["text"], val_df["label"], tokenizer, max_length=args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Optimizer and Loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    total_start_time = time.time()
    train_losses = []
    val_losses = []

    print(f"\n[*] Initiating fine-tuning over {args.epochs} epochs (Batch size: {args.batch_size}, Max length: {args.max_length})...\n")

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        step_count = 0

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            step_count += 1

            if step % 200 == 0 or step == len(train_loader):
                current_avg_loss = running_loss / step_count
                print(f"  [Epoch {epoch}/{args.epochs}] Step {step}/{len(train_loader)} - Batch Loss: {current_avg_loss:.4f}")

        # Post-epoch evaluation
        epoch_train_loss = running_loss / max(1, step_count)
        val_loss, val_acc, val_f1 = evaluate_model(model, val_loader, device)
        epoch_time = time.time() - epoch_start_time

        train_losses.append(epoch_train_loss)
        val_losses.append(val_loss)

        print(f"--> [Epoch {epoch}/{args.epochs}] Elapsed: {epoch_time:.1f}s | "
              f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        # Persist optimal checkpoint based on validation F1 metric
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            print(f"[*] Validation F1 improved to {best_val_f1:.4f}. Saving checkpoint to {MODEL_DIR}...")
            model.save_pretrained(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)

    total_time = time.time() - total_start_time
    print(f"\n[+] DistilBERT fine-tuning completed in {total_time/60:.2f} minutes.")
    print(f"[+] Peak Validation F1-Score: {best_val_f1:.4f}")
    print(f"[+] Model weights and tokenizer artifacts persisted at: {MODEL_DIR}")

    # Generate and save publication-grade Loss Curve
    plot_loss_curve(train_losses, val_losses, save_path="loss_curve.png")

def plot_loss_curve(train_losses, val_losses, save_path="loss_curve.png"):
    """
    Renders and exports an academic-grade Loss Curve (Training vs. Validation Loss)
    formatted for presentation slides and technical research reports (300 DPI).
    """
    try:
        import matplotlib.pyplot as plt
        
        epochs = list(range(1, len(train_losses) + 1))
        
        plt.figure(figsize=(9, 5.5), dpi=300)
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

        # Training Loss trajectory
        plt.plot(
            epochs, train_losses, 
            marker='o', markersize=8, linewidth=2.5, 
            color='#1f77b4', label='Training Loss', linestyle='-'
        )
        # Validation Loss trajectory
        plt.plot(
            epochs, val_losses, 
            marker='s', markersize=8, linewidth=2.5, 
            color='#d62728', label='Validation Loss', linestyle='--'
        )

        # Numerical callout annotations per epoch
        for x, y in zip(epochs, train_losses):
            plt.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, fontweight='bold', color='#1f77b4')
        for x, y in zip(epochs, val_losses):
            plt.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, -15), ha='center', fontsize=9, fontweight='bold', color='#d62728')

        plt.title('Loss Trajectory (DistilBERT Transformer Architecture)\nTraining Loss vs. Validation Loss across Epochs', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Epoch', fontsize=12, fontweight='bold')
        plt.ylabel('Loss (CrossEntropy)', fontsize=12, fontweight='bold')
        plt.xticks(epochs)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(fontsize=11, loc='upper right', frameon=True, shadow=True)
        plt.tight_layout()

        # Save high-resolution PNG
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"[+] Publication-grade loss curve saved successfully to: {save_path}")
    except Exception as e:
        print(f"[!] Warning: Unable to generate loss curve plot: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT Transformer for Binary Email Spam/Ham Classification")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs (default: 2)")
    parser.add_argument("--batch_size", type=int, default=16, help="Mini-batch size (default: 16)")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum token sequence length (default: 256)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Peak learning rate for AdamW optimizer (default: 2e-5)")
    parser.add_argument("--sample_size", type=int, default=None, help="Sub-sample record limit for rapid prototyping")
    return parser.parse_args()

if __name__ == "__main__":
    train(parse_args())
