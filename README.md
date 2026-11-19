# Project 5 — Spam / Phishing Email Detection with Transformers
**Team 24** | Natural Language Processing & Applied Deep Learning

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/nhatthanh123456/Spam-Phishing-Email-Detection-with-Transformers)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg?logo=streamlit)](http://localhost:8501)

An end-to-end Machine Learning and Natural Language Processing project designed for binary classification of unsolicited email into **Spam / Phishing** vs. **Ham (Legitimate)**. The project evaluates a fine-tuned **DistilBERT Transformer** against a classical **Multinomial Naive Bayes (TF-IDF)** baseline on the **Enron-Spam Corpus**, incorporating **LIME** (Local Interpretable Model-agnostic Explanations) for model transparency and an interactive **Streamlit** web demonstration.

---

## 📑 Table of Contents
1. [Project Overview & Deliverables](#1-project-overview--deliverables)
2. [Setup & Installation](#2-setup--installation)
3. [Data & Preprocessing](#3-data--preprocessing)
4. [How to Run (Step-by-Step)](#4-how-to-run-step-by-step)
5. [Evaluation Results](#5-evaluation-results)
6. [Short Error Analysis](#6-short-error-analysis)
7. [System Architecture & Methodology](#7-system-architecture--methodology)
8. [Automated Verification](#8-automated-verification)
9. [References](#9-references)

---

## 1. Project Overview & Deliverables

- **Course Project**: Project 5 — Spam / Phishing Email Detection with Transformers
- **Team**: Team 24 (2 students)
- **GitHub Repository**: [https://github.com/nhatthanh123456/Spam-Phishing-Email-Detection-with-Transformers](https://github.com/nhatthanh123456/Spam-Phishing-Email-Detection-with-Transformers)
- **Submission Artifacts**:
  1. Complete Source Code ZIP archive (`Spam-Phishing-Email-Detection-with-Transformers_source_code.zip`).
  2. Public GitHub repository link containing all code, models, and test suites.
  3. Interactive Streamlit web demonstration with LIME token-level highlights.
  4. Publication-grade training loss curve (`loss_curve.png`).

---

## 2. Setup & Installation

### 2.1 System Requirements
- Operating System: Windows 10/11, Linux, or macOS.
- Python: Version 3.10 or higher.
- Hardware: CPU (supported) or NVIDIA GPU with CUDA for accelerated training.

### 2.2 Environment Setup
Clone the GitHub repository and navigate to the project root:
```bash
git clone https://github.com/nhatthanh123456/Spam-Phishing-Email-Detection-with-Transformers.git
cd Spam-Phishing-Email-Detection-with-Transformers
```

Install all required dependencies:
```bash
pip install -r requirements.txt
```

#### Dependencies Summary (`requirements.txt`):
- `torch>=2.2.0`: PyTorch deep learning framework.
- `transformers>=4.38.0`: Hugging Face transformers library for DistilBERT.
- `scikit-learn>=1.4.0`: Naive Bayes classifier, TF-IDF vectorization, evaluation metrics.
- `lime>=0.2.0.1`: Local Interpretable Model-agnostic Explanations.
- `streamlit>=1.32.0`: Interactive web dashboard.
- `pandas>=2.2.0`, `numpy>=1.26.0`: Data manipulation and numerical operations.
- `matplotlib>=3.8.0`: Publication-grade loss curve plotting.
- `pytest>=8.0.0`: End-to-end automated testing suite.

---

## 3. Data & Preprocessing

### 3.1 Corpus Description
The system is built on the public **Enron-Spam Dataset**, combining authentic corporate emails with verified spam and phishing messages.
- **Total Valid Corpus**: 33,665 email documents (null and degenerate empty records filtered out).
- **Target Variable**: Binary classification:
  - `Ham` (Legitimate email) = `0` (16,545 records, 49.1%)
  - `Spam` (Spam / Phishing email) = `1` (17,120 records, 50.9%)

### 3.2 Automated Ingestion & Cleaning Pipeline (`prepare_data.py`)
1. Fetches the dataset archive directly from GitHub (with fallback to Hugging Face Datasets Hub).
2. Cleans whitespace and missing headers; concatenates Subject and Message bodies into a unified `text` attribute.
3. Performs **stratified partitioning** with a fixed random seed (`random_state=42`) ensuring identical class balance across all splits:
   - **Training Set (80%)**: 26,932 samples &rarr; `data/train.csv`
   - **Validation Set (10%)**: 3,366 samples &rarr; `data/val.csv`
   - **Held-Out Test Set (10%)**: 3,367 samples &rarr; `data/test.csv`

---

## 4. How to Run (Step-by-Step)

All scripts are modular, standalone, and can be executed directly from the terminal or via VS Code (F5):

### Step 1: Prepare and Split the Dataset
```bash
python prepare_data.py
```
*Outputs `data/train.csv`, `data/val.csv`, and `data/test.csv`.*

### Step 2: Train the Naive Bayes Baseline
```bash
python train_nb.py
```
*Fits a TF-IDF vectorizer (10,000 features, unigram + bigram) and Multinomial Naive Bayes classifier. Completes in ~2 seconds and saves to `models/naive_bayes.joblib`.*

### Step 3: Fine-Tune the DistilBERT Transformer
```bash
python train_distilbert.py --epochs 2 --batch_size 16 --max_length 256
```
*Fine-tunes `distilbert-base-uncased` via PyTorch DataLoader loop, checkpoints the best validation F1 model to `models/distilbert/`, and auto-generates `loss_curve.png`.*
*(Tip: On machines without a GPU, add `--sample_size 4000` for fast CPU training).*

### Step 4: Evaluate Models on the Held-Out Test Set
```bash
python evaluate.py
```
*Evaluates both models on `data/test.csv` ($N = 3,367$) across Precision, Recall, F1-Score, and ROC-AUC, outputting a comparative table and saving to `models/metrics_comparison.json`.*

### Step 5: Test LIME Explanation in Terminal (CLI)
```bash
python explain_lime.py --text "Congratulations! You won a $1,000 gift card. Click here now!" --model distilbert
```
*Prints class probabilities and top salient keywords contributing toward the prediction.*

### Step 6: Generate Loss Curve Figure
```bash
python plot_loss_curve.py
```
*Exports high-resolution (300 DPI) training vs. validation loss curve to `loss_curve.png`.*

### Step 7: Launch Interactive Web Demonstration (Streamlit)
```bash
python -m streamlit run app.py
```
*Opens web application at `http://localhost:8501`. Allows pasting arbitrary email text, choosing between models, inspecting predictions, and exploring interactive LIME token highlights.*

---

## 5. Evaluation Results

Both models were independently benchmarked on the held-out Enron-Spam test partition ($N = 3,367$ samples, completely unseen during training):

| Metric | Naive Bayes (Baseline) | DistilBERT (Transformer) | Analysis & Discussion |
| :--- | :---: | :---: | :--- |
| **Accuracy** | 98.75% | 98.13% | Both architectures deliver high overall classification accuracy |
| **Precision** | **98.72%** | 96.77% | Naive Bayes exhibits a lower False Positive rate on corporate jargon |
| **Recall** | 98.83% | **99.65%** | **DistilBERT captures 99.65% of spam/phishing attacks (missed only 6/1,713)** |
| **F1-Score** | **98.77%** | 98.19% | Strong balanced performance across positive and negative classes |
| **ROC-AUC** | **0.9988** | 0.9986 | Near-perfect class separability across threshold spectrum |

- **Prediction Concordance**: 97.89% exact agreement (3,296 out of 3,367 test emails classified identically).
- **Cohen's Kappa ($\kappa$)**: 0.9578 (near-perfect statistical concordance).

### Training Loss Dynamics
The empirical loss curve below demonstrates smooth convergence of DistilBERT without overfitting:

![Training Loss Curve](loss_curve.png)

---

## 6. Short Error Analysis

A systematic diagnostic error analysis was performed on all misclassified samples from the test partition ($N = 3,367$):

### 6.1 False Positives (Legitimate emails predicted as Spam)
- **Observations**: 
  - Naive Bayes produced 21 False Positives (FP rate: 1.27%).
  - DistilBERT produced 55 False Positives (FP rate: 3.32%).
- **Primary Root Causes**:
  1. *Urgent Operational Alerts*: Internal IT and pipeline maintenance messages containing keywords like `"urgent"`, `"security"`, `"immediate"`, `"account"`, and `"password"`.
  2. *Commercial & Legal Boilerplate*: Legitimate contracts and gas delivery schedules with lengthy boilerplate disclaimers, external URLs, and unsubscribe clauses resembling promotional marketing.
- **Representative Case**: A message regarding `"Pipeline system revision — urgent review needed before Friday"` was flagged as spam due to strong attention on `"urgent"` and `"review"`.

### 6.2 False Negatives (Spam/Phishing emails predicted as Ham)
- **Observations**:
  - Naive Bayes produced 20 False Negatives (FN rate: 1.17%).
  - DistilBERT produced only **6 False Negatives** (FN rate: **0.35%**).
- **Primary Root Causes**:
  1. *Low-Density / Short-Text Spam*: Messages with brief conversational greetings (e.g., `"Hey, check this out"`) followed by an obfuscated link, devoid of classical spam trigger words.
  2. *Internal Spoofing Patterns*: Phishing simulations mimicking trusted colleagues without explicit financial or pharmaceutical vocabulary.
- **DistilBERT Advantage**: Naive Bayes failed on short messages because word frequencies were uninformative. DistilBERT succeeded by leveraging contextual embeddings and self-attention, capturing subtle urgency and syntax structure even in the absence of explicit spam keywords.

### 6.3 LIME Attribution Findings
- Token-level surrogate explanations confirm that the models assign high positive weights to legitimate threat indicators (`"medications"`, `"pills"`, `"pharmacy"`, `"winner"`, `"claim"`, `"unauthorized"`) rather than spurious metadata artifacts (e.g., timestamps, email delimiters), validating model fidelity.

---

## 7. System Architecture & Methodology

### 7.1 Pipeline Flow
```
[Raw Email Document] 
       │
       ▼
[Preprocessing & Normalization] (subject + body concatenation, whitespace cleaning)
       │
       ├─────────────────────────────────────────┐
       ▼                                         ▼
[TF-IDF Vectorizer (10K n-grams)]      [DistilBERT Tokenizer (max_len=256)]
       │                                         │
       ▼                                         ▼
[Multinomial Naive Bayes]              [DistilBERT Transformer (6-layer, 768-dim)]
       │                                         │
       └────────────────────┬────────────────────┘
                            ▼
               [Posterior Probabilities]
                            │
                            ▼
              [LIME Local Surrogate Explainer]
                            │
                            ▼
         [Streamlit Web UI / CLI Highlights]
```

### 7.2 Model Characteristics Comparison
| Attribute | Multinomial Naive Bayes | DistilBERT Transformer |
| :--- | :--- | :--- |
| **Model Class** | Classical Statistical / Generative | Deep Bidirectional Transformer |
| **Input Representation** | Bag-of-Words / TF-IDF (Unigram + Bigram) | Contextual Subword Tokens (WordPiece) |
| **Context Window** | Independent features (no word order) | Full bidirectional self-attention (256 tokens) |
| **Parameters** | ~10,000 feature weights | 66.36 Million trainable parameters |
| **Training Time** | ~1.5 seconds | ~10 minutes (GPU) |
| **Inference Latency** | < 1 ms per email | ~15 ms per email (GPU) |
| **Key Advantage** | Ultra-lightweight, zero infrastructure cost | Superior recall (99.65%), context-aware |

---

## 8. Automated Verification

The repository includes a comprehensive 4-tier End-to-End test suite (`tests/test_e2e.py`) comprising **102 test cases**:
```bash
pytest -v tests/test_e2e.py
```

### Test Coverage Hierarchy:
- **Tier 1 (Feature Coverage)**: Validates data splits, Naive Bayes pipeline, DistilBERT fine-tuning, evaluation metrics, LIME explanations, Streamlit app, and documentation.
- **Tier 2 (Edge & Boundary Cases)**: Tests empty strings, whitespace-only emails, extreme sequence lengths, special characters, and non-ASCII Unicode characters.
- **Tier 3 (Integration Workflows)**: Tests cross-model consistency, probability calibration, data leakage prevention, and metric schema contracts.
- **Tier 4 (Real-World Attack Scenarios)**: Tests lottery scams, account suspension alerts, academic submissions, and IT support notices.

**Result**: `102 passed in 16.18s (100% Pass Rate)`.

---

## 9. References
1. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. arXiv preprint arXiv:1810.04805.
2. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter*. arXiv preprint arXiv:1910.01108.
3. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). *"Why Should I Trust You?": Explaining the Predictions of Any Classifier*. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 1135-1144).
4. Metsis, V., Androutsopoulos, I., & Paliouras, G. (2006). *Spam Filtering with Naive Bayes – Which Naive Bayes?*. In Proceedings of the Conference on Email and Anti-Spam (CEAS).
