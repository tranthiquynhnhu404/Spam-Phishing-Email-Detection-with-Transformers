"""
evaluate.py
-----------
Comparative Empirical Evaluation: Baseline Multinomial Naive Bayes vs. Fine-Tuned DistilBERT.
Evaluated independently over the held-out Enron-Spam test set (data/test.csv).

Quantifies four core statistical metrics:
1. Precision: Positive Predictive Value: TP / (TP + FP)
2. Recall: Sensitivity / True Positive Rate: TP / (TP + FN)
3. F1-Score: Harmonic mean of Precision and Recall
4. ROC-AUC: Area Under the Receiver Operating Characteristic Curve (computed via proba[:, 1])

Outputs tabular results to standard output and serializes metrics to models/metrics_comparison.json.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score

TEST_CSV = "data/test.csv"
NB_MODEL_PATH = "models/naive_bayes.joblib"
DISTILBERT_DIR = "models/distilbert"
METRICS_JSON_PATH = "models/metrics_comparison.json"

def predict_distilbert_proba(texts, model, tokenizer, device, batch_size=32, max_length=256):
    """Batched probability inference under DistilBERT Transformer."""
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

def evaluate_models():
    if not os.path.exists(TEST_CSV):
        raise FileNotFoundError(f"Test split not found at {TEST_CSV}. Please run 'prepare_data.py' first!")

    print(f"[*] Ingesting held-out test split from: {TEST_CSV}...")
    test_df = pd.read_csv(TEST_CSV)
    X_test = test_df["text"].fillna("").tolist()
    y_true = test_df["label"].to_numpy()

    print(f"[*] Test sample count: {len(y_true):,} "
          f"(Ham: {(y_true == 0).sum():,}, Spam: {(y_true == 1).sum():,})")

    results = {}

    # 1. Baseline: Multinomial Naive Bayes Evaluation
    if os.path.exists(NB_MODEL_PATH):
        print(f"\n[*] Loading and evaluating Naive Bayes baseline from {NB_MODEL_PATH}...")
        nb_pipeline = joblib.load(NB_MODEL_PATH)
        nb_proba = nb_pipeline.predict_proba(X_test)
        nb_preds = np.argmax(nb_proba, axis=1)

        nb_prec = float(precision_score(y_true, nb_preds, zero_division=0))
        nb_rec = float(recall_score(y_true, nb_preds, zero_division=0))
        nb_f1 = float(f1_score(y_true, nb_preds, zero_division=0))
        nb_auc = float(roc_auc_score(y_true, nb_proba[:, 1]))
        nb_acc = float(accuracy_score(y_true, nb_preds))

        results["Naive Bayes"] = {
            "accuracy": round(nb_acc, 4),
            "precision": round(nb_prec, 4),
            "recall": round(nb_rec, 4),
            "f1": round(nb_f1, 4),
            "roc_auc": round(nb_auc, 4)
        }
    else:
        print(f"[!] Warning: Artifact {NB_MODEL_PATH} missing. Execute 'train_nb.py' first.")

    # 2. Main Architecture: Fine-tuned DistilBERT Evaluation
    if os.path.exists(DISTILBERT_DIR):
        print(f"\n[*] Loading and evaluating DistilBERT architecture from {DISTILBERT_DIR}...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Inference compute device: {device}")

        tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_DIR)
        model.to(device)

        bert_proba = predict_distilbert_proba(X_test, model, tokenizer, device, batch_size=32)
        bert_preds = np.argmax(bert_proba, axis=1)

        bert_prec = float(precision_score(y_true, bert_preds, zero_division=0))
        bert_rec = float(recall_score(y_true, bert_preds, zero_division=0))
        bert_f1 = float(f1_score(y_true, bert_preds, zero_division=0))
        bert_auc = float(roc_auc_score(y_true, bert_proba[:, 1]))
        bert_acc = float(accuracy_score(y_true, bert_preds))

        results["DistilBERT"] = {
            "accuracy": round(bert_acc, 4),
            "precision": round(bert_prec, 4),
            "recall": round(bert_rec, 4),
            "f1": round(bert_f1, 4),
            "roc_auc": round(bert_auc, 4)
        }
    else:
        print(f"[!] Warning: Directory {DISTILBERT_DIR} missing. Execute 'train_distilbert.py' first.")

    if not results:
        print("[!] No trained model artifacts available for evaluation.")
        return results

    # Print Comparative Performance Table
    print("\n" + "=" * 72)
    print("        HEAD-TO-HEAD MODEL BENCHMARK ON HELD-OUT TEST CORPUS")
    print("=" * 72)
    print(f"{'Model Architecture':<22} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'ROC-AUC':<10}")
    print("-" * 72)
    for model_name, metrics in results.items():
        print(f"{model_name:<22} | {metrics['precision']:<10.4f} | {metrics['recall']:<10.4f} | "
              f"{metrics['f1']:<10.4f} | {metrics['roc_auc']:<10.4f}")
    print("=" * 72)

    # Persist benchmark metrics to JSON
    os.makedirs(os.path.dirname(METRICS_JSON_PATH), exist_ok=True)
    with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Empirical benchmark metrics recorded to: {METRICS_JSON_PATH}")

    return results

if __name__ == "__main__":
    evaluate_models()
