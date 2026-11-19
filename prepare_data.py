"""
prepare_data.py
---------------
Automated ingestion, preprocessing, and stratified partitioning of the Enron-Spam corpus.

Pipeline stages:
1. Fetch Enron-Spam dataset from upstream repository or Hugging Face mirror.
2. Clean textual fields: handle missing values, concatenate Subject and Message headers into a unified text field.
3. Filter out degenerate samples (empty or whitespace-only records).
4. Encode binary ground-truth labels: Ham (negative class) = 0, Spam (positive class) = 1.
5. Perform stratified train/validation/test split with an 80/10/10 ratio to maintain class balance.
6. Persist structured datasets to disk: data/train.csv, data/val.csv, and data/test.csv.
"""

import os
import sys
import io
import zipfile
import urllib.request
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = "data"
RAW_CSV_PATH = os.path.join(DATA_DIR, "enron_spam_data.csv")
GITHUB_ZIP_URL = "https://github.com/MWiechmann/enron_spam_data/raw/master/enron_spam_data.zip"
HF_CSV_URL = "https://huggingface.co/datasets/SetFit/enron_spam/resolve/main/enron_spam_data.csv"

def download_dataset():
    """Download the Enron-Spam dataset archive if not already cached locally."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    if os.path.exists(RAW_CSV_PATH) and os.path.getsize(RAW_CSV_PATH) > 0:
        print(f"[+] Found cached raw corpus: {RAW_CSV_PATH}")
        return

    print("[*] Downloading Enron-Spam dataset archive from GitHub repository...")
    try:
        req = urllib.request.Request(
            GITHUB_ZIP_URL,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            zip_content = io.BytesIO(response.read())
        with zipfile.ZipFile(zip_content) as z:
            z.extract("enron_spam_data.csv", path=DATA_DIR)
        print(f"[+] Successfully extracted dataset to: {RAW_CSV_PATH}")
        return
    except Exception as e:
        print(f"[!] GitHub mirror download failed: {e}")
        print("[*] Attempting fallback download from Hugging Face Datasets Hub...")

    try:
        req = urllib.request.Request(
            HF_CSV_URL,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=60) as response, open(RAW_CSV_PATH, "wb") as f:
            f.write(response.read())
        print(f"[+] Fallback download successful: {RAW_CSV_PATH}")
    except Exception as e:
        raise RuntimeError(f"Unable to retrieve Enron-Spam dataset from upstream sources: {e}")

def preprocess_and_split():
    """Preprocess raw email text and generate reproducible stratified data splits."""
    download_dataset()
    
    print("[*] Loading raw corpus into pandas DataFrame...")
    df = pd.read_csv(RAW_CSV_PATH)
    print(f"[*] Total raw records: {len(df):,}")
    
    # Standardize column naming conventions
    column_mapping = {
        "Message ID": "message_id",
        "Subject": "subject",
        "Message": "message",
        "Spam/Ham": "label_text"
    }
    df = df.rename(columns=column_mapping)
    
    # Handle missing values and strip extraneous whitespace
    df["subject"] = df["subject"].fillna("").astype(str).str.strip()
    df["message"] = df["message"].fillna("").astype(str).str.strip()
    
    # Concatenate Subject and Message into a unified document representation
    df["text"] = (df["subject"] + " " + df["message"]).str.strip()
    
    # Filter out empty records
    initial_count = len(df)
    df = df[df["text"].str.len() > 0].copy()
    dropped_count = initial_count - len(df)
    print(f"[*] Filtered out {dropped_count} empty records. Valid samples: {len(df):,}")
    
    # Encode binary target labels: ham = 0, spam = 1
    df["label"] = (df["label_text"].str.lower() == "spam").astype(int)
    
    # Retain required schema attributes
    selected_cols = ["message_id", "subject", "message", "text", "label", "label_text"]
    final_df = df[selected_cols].reset_index(drop=True)
    
    # Stratified partitioning: 80% Train, 10% Validation, 10% Test
    print("[*] Performing stratified train/validation/test partitioning (80/10/10)...")
    train_df, temp_df = train_test_split(
        final_df,
        test_size=0.20,
        random_state=42,
        stratify=final_df["label"]
    )
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["label"]
    )
    
    # Export partition files to CSV
    train_path = os.path.join(DATA_DIR, "train.csv")
    val_path = os.path.join(DATA_DIR, "val.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")
    
    train_df.to_csv(train_path, index=False, encoding="utf-8")
    val_df.to_csv(val_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")
    
    print("[+] Dataset preparation and partitioning completed successfully!")
    print(f"    - Train split: {len(train_df):,} samples -> {train_path} "
          f"(Spam: {train_df['label'].sum():,}, Ham: {(train_df['label'] == 0).sum():,})")
    print(f"    - Val split:   {len(val_df):,} samples -> {val_path} "
          f"(Spam: {val_df['label'].sum():,}, Ham: {(val_df['label'] == 0).sum():,})")
    print(f"    - Test split:  {len(test_df):,} samples -> {test_path} "
          f"(Spam: {test_df['label'].sum():,}, Ham: {(test_df['label'] == 0).sum():,})")

if __name__ == "__main__":
    preprocess_and_split()
