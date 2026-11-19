"""
train_nb.py
-----------
Baseline Model: Multinomial Naive Bayes paired with TF-IDF Vectorization.
Trained on the Enron-Spam corpus and serialized to models/naive_bayes.joblib.
"""

import os
import time
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score

TRAIN_CSV = "data/train.csv"
VAL_CSV = "data/val.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "naive_bayes.joblib")

def train_naive_bayes():
    if not os.path.exists(TRAIN_CSV):
        raise FileNotFoundError(
            f"Target dataset file {TRAIN_CSV} not found. Please run 'python prepare_data.py' first!"
        )
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print("[*] Loading training and validation partitions...")
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    
    X_train = train_df["text"].fillna("")
    y_train = train_df["label"]
    
    X_val = val_df["text"].fillna("")
    y_val = val_df["label"]
    
    print(f"[*] Training samples:   {len(X_train):,}")
    print(f"[*] Validation samples: {len(X_val):,}")
    
    # Construct scikit-learn Pipeline: TF-IDF (10,000 features, unigram + bigram) + MultinomialNB
    print("[*] Initializing feature extraction pipeline: TfidfVectorizer + MultinomialNB...")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words="english",
            lowercase=True
        )),
        ("nb", MultinomialNB(alpha=1.0))
    ])
    
    start_time = time.time()
    print("[*] Fitting Multinomial Naive Bayes model on training partition...")
    pipeline.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    print(f"[+] Fitting completed in {elapsed_time:.2f} seconds.")
    
    # Validation Evaluation
    print("[*] Evaluating baseline model on validation partition...")
    val_preds = pipeline.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    val_f1 = f1_score(y_val, val_preds)
    
    print(f"[+] Validation performance metrics:")
    print(f"    - Accuracy: {val_acc:.4f}")
    print(f"    - F1-Score: {val_f1:.4f}")
    print("\nDetailed Classification Report (Validation):")
    print(classification_report(y_val, val_preds, target_names=["Ham", "Spam"]))
    
    # Serialize model artifact
    joblib.dump(pipeline, MODEL_PATH)
    print(f"[+] Model pipeline serialized successfully to: {MODEL_PATH}")

if __name__ == "__main__":
    train_naive_bayes()
