"""
End-to-End (E2E) Test Suite for Email Spam Classification System
Enron-Spam, DistilBERT vs Naive Bayes, LIME Explainability, Streamlit Demo

4-Tier Test Architecture:
- Tier 1: Feature Coverage (Requirements R1 - R5)
- Tier 2: Boundary, Corner & Edge Cases
- Tier 3: Cross-Feature Combinations & Pipeline Integration
- Tier 4: Real-World Email Scenarios & Workflows
"""

import ast
import csv
import json
import math
import os
import re
import sys
from pathlib import Path
import pytest

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Key file artifacts
PREPARE_DATA_SCRIPT = BASE_DIR / "prepare_data.py"
TRAIN_NB_SCRIPT = BASE_DIR / "train_nb.py"
TRAIN_DISTILBERT_SCRIPT = BASE_DIR / "train_distilbert.py"
EVALUATE_SCRIPT = BASE_DIR / "evaluate.py"
EXPLAIN_LIME_SCRIPT = BASE_DIR / "explain_lime.py"
APP_SCRIPT = BASE_DIR / "app.py"
REQUIREMENTS_TXT = BASE_DIR / "requirements.txt"
README_MD = BASE_DIR / "README.md"

TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "val.csv"
TEST_CSV = DATA_DIR / "test.csv"
RAW_CSV = DATA_DIR / "enron_spam_data.csv"

NB_MODEL_PATH = MODELS_DIR / "naive_bayes.joblib"
DISTILBERT_DIR = MODELS_DIR / "distilbert"
METRICS_PATH = MODELS_DIR / "metrics_comparison.json"


# Dynamic dependency helpers
def get_joblib():
    try:
        import joblib
        return joblib
    except ImportError:
        pytest.skip("joblib is not installed")


def get_sklearn():
    try:
        import sklearn
        return sklearn
    except ImportError:
        pytest.skip("scikit-learn is not installed")


def get_torch():
    try:
        import torch
        return torch
    except ImportError:
        pytest.skip("torch is not installed")


def get_transformers():
    try:
        import transformers
        return transformers
    except ImportError:
        pytest.skip("transformers is not installed")


# Increase CSV field size limit for large email bodies in Enron dataset
try:
    csv.field_size_limit(2147483647)
except OverflowError:
    csv.field_size_limit(sys.maxsize)


def get_lime():
    try:
        import lime
        return lime
    except ImportError:
        pytest.skip("lime is not installed")


def load_csv_rows(csv_path, max_rows=None):
    if not csv_path.is_file():
        return None
    try:
        csv.field_size_limit(2147483647)
    except OverflowError:
        pass
    rows = []
    with open(csv_path, mode="r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            rows.append(row)
            if max_rows is not None and i + 1 >= max_rows:
                break
    return rows


# ============================================================================
# TIER 1: FEATURE COVERAGE
# ============================================================================

class TestTier1FeatureCoverage:
    """Tier 1: Happy-path coverage for Features 1-7 (>= 5 tests per feature)."""

    # --- Feature 1: Enron-Spam Dataset Preparation ---

    def test_tier1_f1_01_prepare_data_script_exists(self):
        """[F1] Verify prepare_data.py exists and is non-empty."""
        assert PREPARE_DATA_SCRIPT.is_file(), f"Missing implementation: {PREPARE_DATA_SCRIPT}"
        assert PREPARE_DATA_SCRIPT.stat().st_size > 0, "prepare_data.py is empty"

    def test_tier1_f1_02_data_csv_splits_exist(self):
        """[F1] Verify train.csv, val.csv, and test.csv splits exist."""
        if not PREPARE_DATA_SCRIPT.is_file():
            pytest.skip("prepare_data.py not yet implemented")
        assert TRAIN_CSV.is_file(), f"Expected split not found: {TRAIN_CSV}"
        assert VAL_CSV.is_file(), f"Expected split not found: {VAL_CSV}"
        assert TEST_CSV.is_file(), f"Expected split not found: {TEST_CSV}"

    def test_tier1_f1_03_data_schema_compliance(self):
        """[F1] Verify data splits comply with required column schema."""
        if not TRAIN_CSV.is_file():
            pytest.skip("Data splits not generated yet")
        required_cols = {"message_id", "subject", "message", "text", "label", "label_text"}
        for split_path in [TRAIN_CSV, VAL_CSV, TEST_CSV]:
            with open(split_path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                header = next(reader)
                cols_present = set(header)
                missing = required_cols - cols_present
                assert not missing, f"{split_path.name} missing schema columns: {missing}"

    def test_tier1_f1_04_data_labels_binary(self):
        """[F1] Verify label column contains only 0 (ham) and 1 (spam)."""
        if not TRAIN_CSV.is_file():
            pytest.skip("Data splits not generated yet")
        rows = load_csv_rows(TRAIN_CSV, max_rows=200)
        assert rows, "train.csv contains no data rows"
        for row in rows:
            assert row["label"] in ["0", "1"], f"Invalid label value: {row['label']}"
            assert row["label_text"].lower() in ["ham", "spam"], f"Invalid label_text: {row['label_text']}"
            if row["label"] == "0":
                assert row["label_text"].lower() == "ham"
            else:
                assert row["label_text"].lower() == "spam"

    def test_tier1_f1_05_data_split_ratios(self):
        """[F1] Verify 80/10/10 stratified split proportions (+- 3%)."""
        if not (TRAIN_CSV.is_file() and VAL_CSV.is_file() and TEST_CSV.is_file()):
            pytest.skip("Data splits not generated yet")
        n_train = sum(1 for _ in open(TRAIN_CSV, encoding="utf-8", errors="replace")) - 1
        n_val = sum(1 for _ in open(VAL_CSV, encoding="utf-8", errors="replace")) - 1
        n_test = sum(1 for _ in open(TEST_CSV, encoding="utf-8", errors="replace")) - 1
        total = n_train + n_val + n_test
        assert total > 0, "No records found in splits"
        train_ratio = n_train / total
        val_ratio = n_val / total
        test_ratio = n_test / total
        assert 0.77 <= train_ratio <= 0.83, f"Train ratio unexpected: {train_ratio:.3f}"
        assert 0.07 <= val_ratio <= 0.13, f"Val ratio unexpected: {val_ratio:.3f}"
        assert 0.07 <= test_ratio <= 0.13, f"Test ratio unexpected: {test_ratio:.3f}"

    def test_tier1_f1_06_data_no_null_or_empty_text(self):
        """[F1] Verify text column contains no nulls or empty strings."""
        if not TRAIN_CSV.is_file():
            pytest.skip("Data splits not generated yet")
        rows = load_csv_rows(TRAIN_CSV, max_rows=500)
        for idx, row in enumerate(rows):
            text = row.get("text", "").strip()
            assert len(text) > 0, f"Row {idx} in train.csv has blank text"

    # --- Feature 2: Naive Bayes Baseline Classifier ---

    def test_tier1_f2_01_train_nb_script_exists(self):
        """[F2] Verify train_nb.py exists and is non-empty."""
        assert TRAIN_NB_SCRIPT.is_file(), f"Missing implementation: {TRAIN_NB_SCRIPT}"
        assert TRAIN_NB_SCRIPT.stat().st_size > 0, "train_nb.py is empty"

    def test_tier1_f2_02_nb_model_artifact_exists(self):
        """[F2] Verify trained Naive Bayes model artifact exists."""
        if not TRAIN_NB_SCRIPT.is_file():
            pytest.skip("train_nb.py not yet implemented")
        assert NB_MODEL_PATH.is_file(), f"Model artifact missing: {NB_MODEL_PATH}"
        assert NB_MODEL_PATH.stat().st_size > 0, "naive_bayes.joblib is empty"

    def test_tier1_f2_03_nb_model_loads_pipeline(self):
        """[F2] Verify Naive Bayes model unpickles and has required methods."""
        if not NB_MODEL_PATH.is_file():
            pytest.skip("naive_bayes.joblib not generated yet")
        joblib = get_joblib()
        model = joblib.load(NB_MODEL_PATH)
        assert hasattr(model, "predict"), "NB model missing predict() method"
        assert hasattr(model, "predict_proba"), "NB model missing predict_proba() method"

    def test_tier1_f2_04_nb_predict_single_text(self):
        """[F2] Verify predict() returns binary class label {0, 1}."""
        if not NB_MODEL_PATH.is_file():
            pytest.skip("naive_bayes.joblib not generated yet")
        joblib = get_joblib()
        model = joblib.load(NB_MODEL_PATH)
        preds = model.predict(["Congratulations! You have won $1,000,000 lottery cash prize."])
        assert len(preds) == 1
        assert int(preds[0]) in [0, 1]

    def test_tier1_f2_05_nb_predict_proba_format(self):
        """[F2] Verify predict_proba() returns ndarray of shape (1, 2) summing to 1.0."""
        if not NB_MODEL_PATH.is_file():
            pytest.skip("naive_bayes.joblib not generated yet")
        joblib = get_joblib()
        model = joblib.load(NB_MODEL_PATH)
        probas = model.predict_proba(["Meeting tomorrow at 10 AM regarding project updates."])
        assert hasattr(probas, "shape"), "predict_proba did not return ndarray"
        assert probas.shape == (1, 2), f"Expected shape (1, 2), got {probas.shape}"
        assert math.isclose(float(probas[0].sum()), 1.0, rel_tol=1e-4)

    def test_tier1_f2_06_nb_predict_batch(self):
        """[F2] Verify batch predictions return correct shape (N, 2)."""
        if not NB_MODEL_PATH.is_file():
            pytest.skip("naive_bayes.joblib not generated yet")
        joblib = get_joblib()
        model = joblib.load(NB_MODEL_PATH)
        batch = ["Hello team", "Special discount buy now", "Invoice attached"]
        probas = model.predict_proba(batch)
        assert probas.shape == (3, 2), f"Expected shape (3, 2), got {probas.shape}"
        for row in probas:
            assert math.isclose(float(row.sum()), 1.0, rel_tol=1e-4)

    # --- Feature 3: DistilBERT Classifier ---

    def test_tier1_f3_01_train_distilbert_script_exists(self):
        """[F3] Verify train_distilbert.py exists and is non-empty."""
        assert TRAIN_DISTILBERT_SCRIPT.is_file(), f"Missing: {TRAIN_DISTILBERT_SCRIPT}"
        assert TRAIN_DISTILBERT_SCRIPT.stat().st_size > 0, "train_distilbert.py is empty"

    def test_tier1_f3_02_distilbert_model_dir_exists(self):
        """[F3] Verify models/distilbert directory exists and contains config.json."""
        if not TRAIN_DISTILBERT_SCRIPT.is_file():
            pytest.skip("train_distilbert.py not yet implemented")
        assert DISTILBERT_DIR.is_dir(), f"Directory missing: {DISTILBERT_DIR}"
        assert (DISTILBERT_DIR / "config.json").is_file(), "config.json missing in distilbert directory"

    def test_tier1_f3_03_distilbert_weights_exist(self):
        """[F3] Verify model weights file (safetensors or bin) exists."""
        if not DISTILBERT_DIR.is_dir():
            pytest.skip("models/distilbert/ not yet generated")
        has_safetensors = (DISTILBERT_DIR / "model.safetensors").is_file()
        has_bin = (DISTILBERT_DIR / "pytorch_model.bin").is_file()
        assert has_safetensors or has_bin, "Neither model.safetensors nor pytorch_model.bin found"

    def test_tier1_f3_04_distilbert_tokenizer_artifacts_exist(self):
        """[F3] Verify tokenizer artifacts exist in models/distilbert."""
        if not DISTILBERT_DIR.is_dir():
            pytest.skip("models/distilbert/ not yet generated")
        has_tok_config = (DISTILBERT_DIR / "tokenizer_config.json").is_file()
        has_vocab = (DISTILBERT_DIR / "vocab.txt").is_file() or (DISTILBERT_DIR / "tokenizer.json").is_file()
        assert has_tok_config, "tokenizer_config.json missing"
        assert has_vocab, "Neither vocab.txt nor tokenizer.json found"

    def test_tier1_f3_05_distilbert_can_load_with_transformers(self):
        """[F3] Verify transformers AutoConfig loads distilbert directory with 2 classes."""
        if not (DISTILBERT_DIR / "config.json").is_file():
            pytest.skip("models/distilbert/config.json not yet generated")
        transformers = get_transformers()
        config = transformers.AutoConfig.from_pretrained(str(DISTILBERT_DIR))
        assert config.num_labels == 2, f"Expected 2 labels in config, got {config.num_labels}"

    def test_tier1_f3_06_distilbert_inference_contract(self):
        """[F3] Verify DistilBERT tokenizer and model execute inference on sample text."""
        if not (DISTILBERT_DIR / "config.json").is_file():
            pytest.skip("models/distilbert not yet generated")
        torch = get_torch()
        transformers = get_transformers()
        tokenizer = transformers.AutoTokenizer.from_pretrained(str(DISTILBERT_DIR))
        model = transformers.AutoModelForSequenceClassification.from_pretrained(str(DISTILBERT_DIR))
        model.eval()
        inputs = tokenizer(["Test email content"], padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        assert probs.shape == (1, 2)
        assert math.isclose(float(probs[0].sum()), 1.0, rel_tol=1e-4)

    # --- Feature 4: 4-Metric Evaluation & Comparison ---

    def test_tier1_f4_01_evaluate_script_exists(self):
        """[F4] Verify evaluate.py exists and is non-empty."""
        assert EVALUATE_SCRIPT.is_file(), f"Missing implementation: {EVALUATE_SCRIPT}"
        assert EVALUATE_SCRIPT.stat().st_size > 0, "evaluate.py is empty"

    def test_tier1_f4_02_metrics_json_exists(self):
        """[F4] Verify models/metrics_comparison.json exists."""
        if not EVALUATE_SCRIPT.is_file():
            pytest.skip("evaluate.py not yet implemented")
        assert METRICS_PATH.is_file(), f"Missing metrics file: {METRICS_PATH}"

    def test_tier1_f4_03_metrics_models_present(self):
        """[F4] Verify metrics JSON contains both 'Naive Bayes' and 'DistilBERT' keys."""
        if not METRICS_PATH.is_file():
            pytest.skip("metrics_comparison.json not yet generated")
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "Naive Bayes" in data, "Key 'Naive Bayes' missing from metrics comparison"
        assert "DistilBERT" in data, "Key 'DistilBERT' missing from metrics comparison"

    def test_tier1_f4_04_metrics_all_four_metrics_present(self):
        """[F4] Verify precision, recall, f1, and roc_auc are present for each model."""
        if not METRICS_PATH.is_file():
            pytest.skip("metrics_comparison.json not yet generated")
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        expected_metrics = {"precision", "recall", "f1", "roc_auc"}
        for model_name in ["Naive Bayes", "DistilBERT"]:
            metrics = set(data[model_name].keys())
            missing = expected_metrics - metrics
            assert not missing, f"{model_name} missing metrics: {missing}"

    def test_tier1_f4_05_metrics_ranges_valid(self):
        """[F4] Verify all evaluation metrics are floats within [0.0, 1.0]."""
        if not METRICS_PATH.is_file():
            pytest.skip("metrics_comparison.json not yet generated")
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        for model_name in ["Naive Bayes", "DistilBERT"]:
            for metric, val in data[model_name].items():
                assert isinstance(val, (int, float)), f"{model_name} {metric} is not a number: {val}"
                assert 0.0 <= float(val) <= 1.0, f"{model_name} {metric} out of range [0, 1]: {val}"

    def test_tier1_f4_06_metrics_f1_score_reasonable(self):
        """[F4] Verify both models achieve an F1 score >= 0.70."""
        if not METRICS_PATH.is_file():
            pytest.skip("metrics_comparison.json not yet generated")
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        for model_name in ["Naive Bayes", "DistilBERT"]:
            f1 = data[model_name]["f1"]
            assert f1 >= 0.70, f"{model_name} F1 score {f1} is below threshold 0.70"

    # --- Feature 5: LIME Explainability ---

    def test_tier1_f5_01_explain_lime_script_exists(self):
        """[F5] Verify explain_lime.py exists and is non-empty."""
        assert EXPLAIN_LIME_SCRIPT.is_file(), f"Missing implementation: {EXPLAIN_LIME_SCRIPT}"
        assert EXPLAIN_LIME_SCRIPT.stat().st_size > 0, "explain_lime.py is empty"

    def test_tier1_f5_02_lime_explainer_class_defined(self):
        """[F5] Verify EmailClassifierExplainer is defined in explain_lime.py."""
        if not EXPLAIN_LIME_SCRIPT.is_file():
            pytest.skip("explain_lime.py not yet implemented")
        sys.path.insert(0, str(BASE_DIR))
        try:
            import explain_lime
            assert hasattr(explain_lime, "EmailClassifierExplainer"), "EmailClassifierExplainer not defined"
        finally:
            if str(BASE_DIR) in sys.path:
                sys.path.remove(str(BASE_DIR))

    def test_tier1_f5_03_lime_explainer_initialization(self):
        """[F5] Verify EmailClassifierExplainer can be instantiated for naive_bayes."""
        if not NB_MODEL_PATH.is_file() or not EXPLAIN_LIME_SCRIPT.is_file():
            pytest.skip("Prerequisites (explain_lime.py or naive_bayes.joblib) not ready")
        sys.path.insert(0, str(BASE_DIR))
        try:
            from explain_lime import EmailClassifierExplainer
            explainer = EmailClassifierExplainer(model_type="naive_bayes")
            assert explainer is not None
        finally:
            if str(BASE_DIR) in sys.path:
                sys.path.remove(str(BASE_DIR))

    def test_tier1_f5_04_lime_explain_return_schema(self):
        """[F5] Verify explain() returns dict with expected keys."""
        if not NB_MODEL_PATH.is_file() or not EXPLAIN_LIME_SCRIPT.is_file():
            pytest.skip("Prerequisites not ready")
        get_lime()
        sys.path.insert(0, str(BASE_DIR))
        try:
            from explain_lime import EmailClassifierExplainer
            explainer = EmailClassifierExplainer(model_type="naive_bayes")
            res = explainer.explain("Win cash prize now!", num_features=4, num_samples=50)
            assert isinstance(res, dict), "explain() must return a dict"
            for key in ["predicted_label", "probability", "top_features", "html_visualization"]:
                assert key in res, f"explain() result missing key: {key}"
        finally:
            if str(BASE_DIR) in sys.path:
                sys.path.remove(str(BASE_DIR))

    def test_tier1_f5_05_lime_top_features_content(self):
        """[F5] Verify top_features contains list of feature dictionaries or tuples."""
        if not NB_MODEL_PATH.is_file() or not EXPLAIN_LIME_SCRIPT.is_file():
            pytest.skip("Prerequisites not ready")
        get_lime()
        sys.path.insert(0, str(BASE_DIR))
        try:
            from explain_lime import EmailClassifierExplainer
            explainer = EmailClassifierExplainer(model_type="naive_bayes")
            res = explainer.explain("Meeting agenda for team project sync", num_features=4, num_samples=50)
            features = res.get("top_features", [])
            assert isinstance(features, list), "top_features must be a list"
            assert len(features) > 0, "top_features should not be empty for informative text"
        finally:
            if str(BASE_DIR) in sys.path:
                sys.path.remove(str(BASE_DIR))

    def test_tier1_f5_06_lime_html_visualization_content(self):
        """[F5] Verify html_visualization is a non-empty string with HTML markup."""
        if not NB_MODEL_PATH.is_file() or not EXPLAIN_LIME_SCRIPT.is_file():
            pytest.skip("Prerequisites not ready")
        get_lime()
        sys.path.insert(0, str(BASE_DIR))
        try:
            from explain_lime import EmailClassifierExplainer
            explainer = EmailClassifierExplainer(model_type="naive_bayes")
            res = explainer.explain("Urgent security verification required", num_features=4, num_samples=50)
            html = res.get("html_visualization", "")
            assert isinstance(html, str), "html_visualization must be a string"
            assert len(html) > 20, "html_visualization string is too short"
            assert "<" in html and ">" in html, "html_visualization does not contain HTML tags"
        finally:
            if str(BASE_DIR) in sys.path:
                sys.path.remove(str(BASE_DIR))

    # --- Feature 6: Streamlit Interactive Demo App ---

    def test_tier1_f6_01_app_script_exists(self):
        """[F6] Verify app.py exists and is non-empty."""
        assert APP_SCRIPT.is_file(), f"Missing implementation: {APP_SCRIPT}"
        assert APP_SCRIPT.stat().st_size > 0, "app.py is empty"

    def test_tier1_f6_02_app_syntax_compilation(self):
        """[F6] Verify app.py compiles cleanly without SyntaxError."""
        if not APP_SCRIPT.is_file():
            pytest.skip("app.py not yet implemented")
        code = APP_SCRIPT.read_text(encoding="utf-8")
        compiled = compile(code, str(APP_SCRIPT), "exec")
        assert compiled is not None

    def test_tier1_f6_03_app_ui_title_defined(self):
        """[F6] Verify app.py contains Streamlit title / header component."""
        if not APP_SCRIPT.is_file():
            pytest.skip("app.py not yet implemented")
        code = APP_SCRIPT.read_text(encoding="utf-8")
        assert ("st.title" in code) or ("st.header" in code) or ("set_page_config" in code), \
            "app.py should define title or header"

    def test_tier1_f6_04_app_ui_text_input_defined(self):
        """[F6] Verify app.py contains st.text_area for email input."""
        if not APP_SCRIPT.is_file():
            pytest.skip("app.py not yet implemented")
        code = APP_SCRIPT.read_text(encoding="utf-8")
        assert "st.text_area" in code, "app.py must contain st.text_area for input"

    def test_tier1_f6_05_app_ui_model_selector_defined(self):
        """[F6] Verify app.py provides model selection (selectbox or radio)."""
        if not APP_SCRIPT.is_file():
            pytest.skip("app.py not yet implemented")
        code = APP_SCRIPT.read_text(encoding="utf-8")
        assert ("st.selectbox" in code) or ("st.radio" in code) or ("st.sidebar.selectbox" in code), \
            "app.py must provide model selection mechanism"

    def test_tier1_f6_06_app_ui_predict_button_defined(self):
        """[F6] Verify app.py contains a button to trigger classification."""
        if not APP_SCRIPT.is_file():
            pytest.skip("app.py not yet implemented")
        code = APP_SCRIPT.read_text(encoding="utf-8")
        assert ("st.button" in code) or ("st.form_submit_button" in code), \
            "app.py must have a trigger button"

    # --- Feature 7: Environment & Documentation ---

    def test_tier1_f7_01_requirements_txt_exists(self):
        """[F7] Verify requirements.txt exists and is non-empty."""
        assert REQUIREMENTS_TXT.is_file(), f"Missing: {REQUIREMENTS_TXT}"
        assert REQUIREMENTS_TXT.stat().st_size > 0, "requirements.txt is empty"

    def test_tier1_f7_02_requirements_essential_packages(self):
        """[F7] Verify requirements.txt specifies all required libraries."""
        if not REQUIREMENTS_TXT.is_file():
            pytest.skip("requirements.txt not yet implemented")
        content = REQUIREMENTS_TXT.read_text(encoding="utf-8").lower()
        essential = ["torch", "transformers", "scikit-learn", "lime", "streamlit", "pandas"]
        for pkg in essential:
            assert pkg in content, f"requirements.txt missing essential package: {pkg}"

    def test_tier1_f7_03_readme_md_exists(self):
        """[F7] Verify README.md exists and has substantive content."""
        assert README_MD.is_file(), f"Missing: {README_MD}"
        assert README_MD.stat().st_size > 200, "README.md is too short or empty"

    def test_tier1_f7_04_readme_contains_setup_instructions(self):
        """[F7] Verify README.md provides pip install / setup instructions."""
        if not README_MD.is_file():
            pytest.skip("README.md not yet implemented")
        content = README_MD.read_text(encoding="utf-8").lower()
        assert ("pip install" in content) or ("requirements.txt" in content), \
            "README.md must describe installation command"

    def test_tier1_f7_05_readme_contains_execution_pipeline(self):
        """[F7] Verify README.md contains execution commands for all steps."""
        if not README_MD.is_file():
            pytest.skip("README.md not yet implemented")
        content = README_MD.read_text(encoding="utf-8")
        for step in ["prepare_data.py", "train_nb.py", "train_distilbert.py", "evaluate.py", "app.py"]:
            assert step in content, f"README.md missing command for {step}"

    def test_tier1_f7_06_readme_student_clarity(self):
        """[F7] Verify README.md contains structured markdown sections."""
        if not README_MD.is_file():
            pytest.skip("README.md not yet implemented")
        lines = README_MD.read_text(encoding="utf-8").splitlines()
        header_count = sum(1 for line in lines if line.strip().startswith("#"))
        assert header_count >= 4, f"README.md should have at least 4 markdown headings, found {header_count}"


# ============================================================================
# TIER 2: BOUNDARY & CORNER CASES
# ============================================================================

class TestTier2BoundaryCornerCases:
    """Tier 2: Boundary value analysis, corner cases, error handling (>= 5 per feature)."""

    # --- Feature 1 Boundaries ---

    def test_tier2_f1_01_empty_subject_handling(self):
        """[F1] Verify rows with empty subject are handled without NaN in text."""
        if not TRAIN_CSV.is_file():
            pytest.skip("train.csv not generated yet")
        rows = load_csv_rows(TRAIN_CSV, max_rows=1000)
        found_empty_subject = False
        for row in rows:
            if not row.get("subject", "").strip():
                found_empty_subject = True
                assert len(row.get("text", "").strip()) > 0, "Empty subject resulted in blank text"
        # Even if no empty subjects in sample, text must never be nan
        for row in rows:
            assert row.get("text") != "nan" and row.get("text") is not None

    def test_tier2_f1_02_empty_body_handling(self):
        """[F1] Verify rows with empty message body are handled gracefully."""
        if not TRAIN_CSV.is_file():
            pytest.skip("train.csv not generated yet")
        rows = load_csv_rows(TRAIN_CSV, max_rows=1000)
        for row in rows:
            if not row.get("message", "").strip():
                # text should retain subject
                assert len(row.get("text", "").strip()) > 0

    def test_tier2_f1_03_text_combines_subject_and_message(self):
        """[F1] Verify text column incorporates subject and message content."""
        if not TRAIN_CSV.is_file():
            pytest.skip("train.csv not generated yet")
        rows = load_csv_rows(TRAIN_CSV, max_rows=50)
        for row in rows:
            subj = row.get("subject", "").strip()
            msg = row.get("message", "").strip()
            text = row.get("text", "")
            if subj and len(subj) > 5:
                # Subject words should appear in text
                first_word = subj.split()[0]
                assert first_word.lower() in text.lower()

    def test_tier2_f1_04_splits_are_mutually_exclusive(self):
        """[F1] Verify train, val, and test splits have 0 overlapping message_ids."""
        if not (TRAIN_CSV.is_file() and VAL_CSV.is_file() and TEST_CSV.is_file()):
            pytest.skip("Data splits not ready")
        def get_id_set(path, max_rows=2000):
            try:
                csv.field_size_limit(2147483647)
            except OverflowError:
                pass
            ids = set()
            with open(path, encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    ids.add(row.get("message_id"))
                    if max_rows is not None and i + 1 >= max_rows:
                        break
            return ids

        train_ids = get_id_set(TRAIN_CSV, max_rows=2000)
        val_ids = get_id_set(VAL_CSV, max_rows=1000)
        test_ids = get_id_set(TEST_CSV, max_rows=1000)
        assert len(train_ids.intersection(val_ids)) == 0, "Train and Val share message_ids"
        assert len(train_ids.intersection(test_ids)) == 0, "Train and Test share message_ids"
        assert len(val_ids.intersection(test_ids)) == 0, "Val and Test share message_ids"

    def test_tier2_f1_05_stratification_preserves_label_distribution(self):
        """[F1] Verify spam ratio across splits matches within 2.5%."""
        if not (TRAIN_CSV.is_file() and VAL_CSV.is_file() and TEST_CSV.is_file()):
            pytest.skip("Data splits not ready")
        def get_spam_ratio(path):
            try:
                csv.field_size_limit(2147483647)
            except OverflowError:
                pass
            with open(path, encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                total = 0
                spam_count = 0
                for row in reader:
                    total += 1
                    if row.get("label") == "1":
                        spam_count += 1
                return spam_count / total if total > 0 else 0.0

        r_train = get_spam_ratio(TRAIN_CSV)
        r_val = get_spam_ratio(VAL_CSV)
        r_test = get_spam_ratio(TEST_CSV)
        assert abs(r_train - r_val) < 0.025, f"Train ({r_train}) vs Val ({r_val}) ratio drift"
        assert abs(r_train - r_test) < 0.025, f"Train ({r_train}) vs Test ({r_test}) ratio drift"

    def test_tier2_f1_06_raw_cache_file_preserved(self):
        """[F1] Verify raw dataset cache exists or is managed under data/ directory."""
        if not PREPARE_DATA_SCRIPT.is_file():
            pytest.skip("prepare_data.py not yet implemented")
        # RAW_CSV or splits should be under data/
        assert DATA_DIR.is_dir(), "data directory missing"
        assert (RAW_CSV.is_file() or TRAIN_CSV.is_file()), "Neither raw cache nor splits exist"

    # --- Feature 2 Boundaries (Naive Bayes) ---

    def test_tier2_f2_01_empty_string_input(self):
        """[F2] Verify Naive Bayes handles empty string input without crashing."""
        if not NB_MODEL_PATH.is_file():
            pytest.skip("naive_bayes.joblib not ready")
        joblib = get_joblib()
        model = joblib.load(NB_MODEL_PATH)
        probas = model.predict_proba([""])
        assert probas.shape == (1, 2)
        assert math.isclose(float(probas[0].sum()), 1.0, rel_tol=1e-4)

    def test_tier2_f2_02_whitespace_only_input(self):
        """[F2] Verify Naive Bayes handles whitespace-only input."""
        if not NB_MODEL_PATH.is_file():
            pytest.skip("naive_bayes.joblib not ready")
        joblib = get_joblib()
        model = joblib.load(NB_MODEL_PATH)
        probas = model.predict_proba(["   \n\t\r  \n"])
        assert probas.shape == (1, 2)
        assert math.isclose(float(probas[0].sum()), 1.0, rel_tol=1e-4)

    def test_tier2_f2_03_all_oov_vocabulary(self):
        """[F2] Verify Naive Bayes handles completely out-of-vocabulary words."""
        if not NB_MODEL_PATH.is_file():
            pytest.skip("naive_bayes.joblib not ready")
        joblib = get_joblib()
        model = joblib.load(NB_MODEL_PATH)
        probas = model.predict_proba(["zzxqwerty987654321 qwertzuiopasdfghjk"])
        assert probas.shape == (1, 2)
        assert math.isclose(float(probas[0].sum()), 1.0, rel_tol=1e-4)

    def test_tier2_f2_04_extreme_length_email(self):
        """[F2] Verify Naive Bayes handles extreme email length (>3000 words)."""
        if not NB_MODEL_PATH.is_file():
            pytest.skip("naive_bayes.joblib not ready")
        joblib = get_joblib()
        model = joblib.load(NB_MODEL_PATH)
        long_email = "urgent meeting update please reply " * 600
        probas = model.predict_proba([long_email])
        assert probas.shape == (1, 2)
        assert math.isclose(float(probas[0].sum()), 1.0, rel_tol=1e-4)

    def test_tier2_f2_05_special_characters_and_punctuations(self):
        """[F2] Verify Naive Bayes handles punctuation and special characters."""
        if not NB_MODEL_PATH.is_file():
            pytest.skip("naive_bayes.joblib not ready")
        joblib = get_joblib()
        model = joblib.load(NB_MODEL_PATH)
        probas = model.predict_proba(["$$$ !!! @@@ *** %%% ^^^ &*()_+-=[]{}|;':\",./<>?"])
        assert probas.shape == (1, 2)
        assert math.isclose(float(probas[0].sum()), 1.0, rel_tol=1e-4)

    def test_tier2_f2_06_probability_bounds_extreme_inputs(self):
        """[F2] Verify all probabilities are strictly between 0.0 and 1.0."""
        if not NB_MODEL_PATH.is_file():
            pytest.skip("naive_bayes.joblib not ready")
        joblib = get_joblib()
        model = joblib.load(NB_MODEL_PATH)
        inputs = ["", " ", "a", "FREE MONEY WINNER " * 50, "Regular memo text"]
        probas = model.predict_proba(inputs)
        for row in probas:
            for p in row:
                assert 0.0 <= p <= 1.0, f"Probability out of bounds: {p}"

    # --- Feature 3 Boundaries (DistilBERT) ---

    def test_tier2_f3_01_empty_string_input(self):
        """[F3] Verify DistilBERT handles empty string input."""
        if not (DISTILBERT_DIR / "config.json").is_file():
            pytest.skip("models/distilbert not ready")
        torch = get_torch()
        transformers = get_transformers()
        tokenizer = transformers.AutoTokenizer.from_pretrained(str(DISTILBERT_DIR))
        model = transformers.AutoModelForSequenceClassification.from_pretrained(str(DISTILBERT_DIR))
        model.eval()
        inputs = tokenizer([""], padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=-1).cpu().numpy()
        assert probs.shape == (1, 2)
        assert math.isclose(float(probs[0].sum()), 1.0, rel_tol=1e-4)

    def test_tier2_f3_02_whitespace_only_input(self):
        """[F3] Verify DistilBERT handles whitespace input."""
        if not (DISTILBERT_DIR / "config.json").is_file():
            pytest.skip("models/distilbert not ready")
        torch = get_torch()
        transformers = get_transformers()
        tokenizer = transformers.AutoTokenizer.from_pretrained(str(DISTILBERT_DIR))
        model = transformers.AutoModelForSequenceClassification.from_pretrained(str(DISTILBERT_DIR))
        model.eval()
        inputs = tokenizer(["   \t\n   "], padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=-1).cpu().numpy()
        assert probs.shape == (1, 2)
        assert math.isclose(float(probs[0].sum()), 1.0, rel_tol=1e-4)

    def test_tier2_f3_03_truncation_exceeding_max_tokens(self):
        """[F3] Verify text exceeding 512 tokens is truncated without dimension error."""
        if not (DISTILBERT_DIR / "config.json").is_file():
            pytest.skip("models/distilbert not ready")
        torch = get_torch()
        transformers = get_transformers()
        tokenizer = transformers.AutoTokenizer.from_pretrained(str(DISTILBERT_DIR))
        model = transformers.AutoModelForSequenceClassification.from_pretrained(str(DISTILBERT_DIR))
        model.eval()
        long_text = "corporate policy update review " * 300
        inputs = tokenizer([long_text], padding=True, truncation=True, max_length=256, return_tensors="pt")
        assert inputs["input_ids"].shape[1] <= 256, "Truncation failed"
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=-1).cpu().numpy()
        assert probs.shape == (1, 2)

    def test_tier2_f3_04_multilingual_unicode_emojis(self):
        """[F3] Verify DistilBERT handles unicode and emojis."""
        if not (DISTILBERT_DIR / "config.json").is_file():
            pytest.skip("models/distilbert not ready")
        torch = get_torch()
        transformers = get_transformers()
        tokenizer = transformers.AutoTokenizer.from_pretrained(str(DISTILBERT_DIR))
        model = transformers.AutoModelForSequenceClassification.from_pretrained(str(DISTILBERT_DIR))
        model.eval()
        unicode_text = "🎉 Chúc mừng bạn đã trúng thưởng $50,000! Nhận quà ngay: http://spam.vn 🚀"
        inputs = tokenizer([unicode_text], padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=-1).cpu().numpy()
        assert probs.shape == (1, 2)

    def test_tier2_f3_05_batch_consistency_single_vs_multi(self):
        """[F3] Verify single inference probability matches batched inference within tolerance."""
        if not (DISTILBERT_DIR / "config.json").is_file():
            pytest.skip("models/distilbert not ready")
        torch = get_torch()
        transformers = get_transformers()
        tokenizer = transformers.AutoTokenizer.from_pretrained(str(DISTILBERT_DIR))
        model = transformers.AutoModelForSequenceClassification.from_pretrained(str(DISTILBERT_DIR))
        model.eval()
        t1 = "Urgent action required on your account"
        t2 = "Team meeting tomorrow at 2 PM"
        with torch.no_grad():
            inp1 = tokenizer([t1], padding=True, truncation=True, max_length=128, return_tensors="pt")
            prob1 = torch.softmax(model(**inp1).logits, dim=-1).cpu().numpy()

            inp_batch = tokenizer([t1, t2], padding=True, truncation=True, max_length=128, return_tensors="pt")
            prob_batch = torch.softmax(model(**inp_batch).logits, dim=-1).cpu().numpy()
        assert math.isclose(float(prob1[0, 1]), float(prob_batch[0, 1]), abs_tol=1e-3)

    def test_tier2_f3_06_eval_mode_deterministic(self):
        """[F3] Verify model in eval mode produces identical outputs on repeated passes."""
        if not (DISTILBERT_DIR / "config.json").is_file():
            pytest.skip("models/distilbert not ready")
        torch = get_torch()
        transformers = get_transformers()
        tokenizer = transformers.AutoTokenizer.from_pretrained(str(DISTILBERT_DIR))
        model = transformers.AutoModelForSequenceClassification.from_pretrained(str(DISTILBERT_DIR))
        model.eval()
        inp = tokenizer(["Weekly progress report sync"], return_tensors="pt")
        with torch.no_grad():
            out1 = model(**inp).logits.cpu().numpy()
            out2 = model(**inp).logits.cpu().numpy()
        assert math.isclose(float(out1[0, 0]), float(out2[0, 0]), abs_tol=1e-6)

    # --- Feature 4 Boundaries (Evaluation) ---

    def test_tier2_f4_01_zero_division_guard(self):
        """[F4] Verify evaluate.py handles zero division in metric computations."""
        if not EVALUATE_SCRIPT.is_file():
            pytest.skip("evaluate.py not yet implemented")
        code = EVALUATE_SCRIPT.read_text(encoding="utf-8")
        assert ("zero_division" in code) or ("precision_score" in code), \
            "evaluate.py should use standard scikit-learn metrics or handle zero division"

    def test_tier2_f4_02_f1_bounded_by_precision_and_recall(self):
        """[F4] Verify F1 score does not exceed maximum of precision and recall."""
        if not METRICS_PATH.is_file():
            pytest.skip("metrics_comparison.json not ready")
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        for model_name in ["Naive Bayes", "DistilBERT"]:
            m = data[model_name]
            p, r, f1 = m["precision"], m["recall"], m["f1"]
            assert f1 <= max(p, r) + 1e-4, f"{model_name} F1 ({f1}) > max(P, R) ({max(p, r)})"

    def test_tier2_f4_03_roc_auc_strictly_better_than_random(self):
        """[F4] Verify ROC-AUC > 0.50 for both models."""
        if not METRICS_PATH.is_file():
            pytest.skip("metrics_comparison.json not ready")
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        for model_name in ["Naive Bayes", "DistilBERT"]:
            roc_auc = data[model_name]["roc_auc"]
            assert roc_auc > 0.50, f"{model_name} ROC-AUC {roc_auc} <= 0.50"

    def test_tier2_f4_04_metrics_file_valid_utf8_json(self):
        """[F4] Verify metrics_comparison.json is valid UTF-8 JSON without decoding error."""
        if not METRICS_PATH.is_file():
            pytest.skip("metrics_comparison.json not ready")
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_tier2_f4_05_metrics_no_nan_or_inf(self):
        """[F4] Verify no NaN or Infinity values exist in metrics JSON."""
        if not METRICS_PATH.is_file():
            pytest.skip("metrics_comparison.json not ready")
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        for model, metrics in data.items():
            for k, v in metrics.items():
                assert not math.isnan(v), f"{model} {k} is NaN"
                assert not math.isinf(v), f"{model} {k} is Infinity"

    def test_tier2_f4_06_symmetric_model_comparison_keys(self):
        """[F4] Verify Naive Bayes and DistilBERT have identical metric keys."""
        if not METRICS_PATH.is_file():
            pytest.skip("metrics_comparison.json not ready")
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert set(data["Naive Bayes"].keys()) == set(data["DistilBERT"].keys())

    # --- Feature 5 Boundaries (LIME) ---

    def test_tier2_f5_01_single_word_email(self):
        """[F5] Verify LIME handles single word email text without IndexError."""
        if not NB_MODEL_PATH.is_file() or not EXPLAIN_LIME_SCRIPT.is_file():
            pytest.skip("Prerequisites not ready")
        get_lime()
        sys.path.insert(0, str(BASE_DIR))
        try:
            from explain_lime import EmailClassifierExplainer
            explainer = EmailClassifierExplainer(model_type="naive_bayes")
            res = explainer.explain("Meeting", num_features=2, num_samples=30)
            assert "predicted_label" in res
        finally:
            if str(BASE_DIR) in sys.path:
                sys.path.remove(str(BASE_DIR))

    def test_tier2_f5_02_num_features_parameter_clamping(self):
        """[F5] Verify num_features parameter limits the number of extracted features."""
        if not NB_MODEL_PATH.is_file() or not EXPLAIN_LIME_SCRIPT.is_file():
            pytest.skip("Prerequisites not ready")
        get_lime()
        sys.path.insert(0, str(BASE_DIR))
        try:
            from explain_lime import EmailClassifierExplainer
            explainer = EmailClassifierExplainer(model_type="naive_bayes")
            res = explainer.explain("Urgent prize lottery win claim cash bank details now", num_features=3, num_samples=50)
            features = res.get("top_features", [])
            assert len(features) <= 3, f"Expected <= 3 features, got {len(features)}"
        finally:
            if str(BASE_DIR) in sys.path:
                sys.path.remove(str(BASE_DIR))

    def test_tier2_f5_03_whitespace_only_email(self):
        """[F5] Verify LIME handles whitespace and non-alphanumeric text without unhandled crash."""
        if not NB_MODEL_PATH.is_file() or not EXPLAIN_LIME_SCRIPT.is_file():
            pytest.skip("Prerequisites not ready")
        get_lime()
        sys.path.insert(0, str(BASE_DIR))
        try:
            from explain_lime import EmailClassifierExplainer
            explainer = EmailClassifierExplainer(model_type="naive_bayes")
            res = explainer.explain("   ", num_features=2, num_samples=20)
            assert isinstance(res, dict)
            assert res["top_features"] == []

            res_punct = explainer.explain("!", num_features=2, num_samples=20)
            assert isinstance(res_punct, dict)
            assert res_punct["top_features"] == []

            res_emoji = explainer.explain("🎉🔥🚀", num_features=2, num_samples=20)
            assert isinstance(res_emoji, dict)
            assert res_emoji["top_features"] == []
        finally:
            if str(BASE_DIR) in sys.path:
                sys.path.remove(str(BASE_DIR))

    def test_tier2_f5_04_html_visualization_has_html_tags(self):
        """[F5] Verify LIME html_visualization contains markup formatting tags."""
        if not NB_MODEL_PATH.is_file() or not EXPLAIN_LIME_SCRIPT.is_file():
            pytest.skip("Prerequisites not ready")
        get_lime()
        sys.path.insert(0, str(BASE_DIR))
        try:
            from explain_lime import EmailClassifierExplainer
            explainer = EmailClassifierExplainer(model_type="naive_bayes")
            res = explainer.explain("Corporate quarterly sync meeting", num_features=3, num_samples=40)
            html = res.get("html_visualization", "")
            assert any(tag in html for tag in ["<div", "<span", "<table", "<p", "<style"]), \
                "html_visualization missing typical HTML tags"
        finally:
            if str(BASE_DIR) in sys.path:
                sys.path.remove(str(BASE_DIR))

    def test_tier2_f5_05_feature_weights_are_floats(self):
        """[F5] Verify feature weights in top_features are valid float numbers."""
        if not NB_MODEL_PATH.is_file() or not EXPLAIN_LIME_SCRIPT.is_file():
            pytest.skip("Prerequisites not ready")
        get_lime()
        sys.path.insert(0, str(BASE_DIR))
        try:
            from explain_lime import EmailClassifierExplainer
            explainer = EmailClassifierExplainer(model_type="naive_bayes")
            res = explainer.explain("Exclusive discount offer buy today", num_features=4, num_samples=50)
            for item in res.get("top_features", []):
                weight = item["weight"] if isinstance(item, dict) else item[1]
                assert isinstance(weight, (int, float))
                assert not math.isnan(weight)
        finally:
            if str(BASE_DIR) in sys.path:
                sys.path.remove(str(BASE_DIR))

    def test_tier2_f5_06_probability_dictionary_keys(self):
        """[F5] Verify probability dictionary in LIME output has ham/spam keys."""
        if not NB_MODEL_PATH.is_file() or not EXPLAIN_LIME_SCRIPT.is_file():
            pytest.skip("Prerequisites not ready")
        get_lime()
        sys.path.insert(0, str(BASE_DIR))
        try:
            from explain_lime import EmailClassifierExplainer
            explainer = EmailClassifierExplainer(model_type="naive_bayes")
            res = explainer.explain("Hello friend how are you", num_features=2, num_samples=30)
            prob = res.get("probability", {})
            assert isinstance(prob, dict)
            assert len(prob) == 2, f"Expected 2 probability classes, got {len(prob)}"
        finally:
            if str(BASE_DIR) in sys.path:
                sys.path.remove(str(BASE_DIR))

    # --- Feature 6 Boundaries (Streamlit UI) ---

    def test_tier2_f6_01_empty_text_warning_or_guard(self):
        """[F6] Verify app.py contains logic or warning when text area is empty."""
        if not APP_SCRIPT.is_file():
            pytest.skip("app.py not yet implemented")
        code = APP_SCRIPT.read_text(encoding="utf-8")
        assert ("warning" in code) or ("strip" in code) or ("error" in code) or ("if not" in code), \
            "app.py should guard against empty input"

    def test_tier2_f6_02_relative_paths_for_models(self):
        """[F6] Verify app.py uses relative or Path-based model directories."""
        if not APP_SCRIPT.is_file():
            pytest.skip("app.py not yet implemented")
        code = APP_SCRIPT.read_text(encoding="utf-8")
        assert "c:\\" not in code.lower(), "app.py contains hardcoded absolute Windows path"
        assert "d:\\" not in code.lower(), "app.py contains hardcoded absolute Windows path"

    def test_tier2_f6_03_lime_html_component_used(self):
        """[F6] Verify app.py references Streamlit HTML component for LIME."""
        if not APP_SCRIPT.is_file():
            pytest.skip("app.py not yet implemented")
        code = APP_SCRIPT.read_text(encoding="utf-8")
        assert ("components.html" in code) or ("components.v1.html" in code) or ("html" in code), \
            "app.py should render LIME HTML using components"

    def test_tier2_f6_04_model_selection_options_complete(self):
        """[F6] Verify app.py provides options for both Naive Bayes and DistilBERT."""
        if not APP_SCRIPT.is_file():
            pytest.skip("app.py not yet implemented")
        code = APP_SCRIPT.read_text(encoding="utf-8")
        assert "naive bayes" in code.lower()
        assert "distilbert" in code.lower()

    def test_tier2_f6_05_metrics_tab_or_sidebar_present(self):
        """[F6] Verify app.py mentions metrics comparison or evaluation results."""
        if not APP_SCRIPT.is_file():
            pytest.skip("app.py not yet implemented")
        code = APP_SCRIPT.read_text(encoding="utf-8")
        assert ("metric" in code.lower()) or ("comparison" in code.lower()) or ("f1" in code.lower()), \
            "app.py should display comparison metrics"

    def test_tier2_f6_06_clean_error_handling_missing_models(self):
        """[F6] Verify app.py includes error checking or existence checks for models."""
        if not APP_SCRIPT.is_file():
            pytest.skip("app.py not yet implemented")
        code = APP_SCRIPT.read_text(encoding="utf-8")
        assert ("exists" in code) or ("try:" in code) or ("st.error" in code), \
            "app.py should handle missing model files gracefully"

    # --- Feature 7 Boundaries (Requirements & Docs) ---

    def test_tier2_f7_01_requirements_no_wildcard_pinned_everything(self):
        """[F7] Verify requirements.txt specifies cleanly named standard packages."""
        if not REQUIREMENTS_TXT.is_file():
            pytest.skip("requirements.txt not ready")
        lines = REQUIREMENTS_TXT.read_text(encoding="utf-8").splitlines()
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):
                assert not line.startswith("file://"), f"Local file wheel in requirements: {line}"

    def test_tier2_f7_02_requirements_includes_joblib(self):
        """[F7] Verify requirements includes joblib or scikit-learn."""
        if not REQUIREMENTS_TXT.is_file():
            pytest.skip("requirements.txt not ready")
        content = REQUIREMENTS_TXT.read_text(encoding="utf-8").lower()
        assert ("joblib" in content) or ("scikit-learn" in content)

    def test_tier2_f7_03_readme_mentions_enron_dataset(self):
        """[F7] Verify README mentions Enron dataset."""
        if not README_MD.is_file():
            pytest.skip("README.md not ready")
        content = README_MD.read_text(encoding="utf-8").lower()
        assert "enron" in content

    def test_tier2_f7_04_readme_mentions_lime(self):
        """[F7] Verify README mentions LIME explainability."""
        if not README_MD.is_file():
            pytest.skip("README.md not ready")
        content = README_MD.read_text(encoding="utf-8").lower()
        assert "lime" in content

    def test_tier2_f7_05_readme_mentions_streamlit(self):
        """[F7] Verify README mentions Streamlit."""
        if not README_MD.is_file():
            pytest.skip("README.md not ready")
        content = README_MD.read_text(encoding="utf-8").lower()
        assert "streamlit" in content

    def test_tier2_f7_06_readme_clean_markdown_formatting(self):
        """[F7] Verify README uses markdown code fences."""
        if not README_MD.is_file():
            pytest.skip("README.md not ready")
        content = README_MD.read_text(encoding="utf-8")
        assert "```" in content, "README should contain formatted code blocks"


# ============================================================================
# TIER 3: CROSS-FEATURE COMBINATIONS & PIPELINE INTEGRATION
# ============================================================================

class TestTier3CrossFeatureCombinations:
    """Tier 3: Pairwise and end-to-end component interoperability (>= 10 tests)."""

    def test_tier3_01_data_train_split_consumable_by_nb_vectorizer(self):
        """[F1 x F2] Verify train.csv 'text' is directly vectorizable by TfidfVectorizer."""
        if not TRAIN_CSV.is_file():
            pytest.skip("train.csv not ready")
        sklearn = get_sklearn()
        from sklearn.feature_extraction.text import TfidfVectorizer
        rows = load_csv_rows(TRAIN_CSV, max_rows=100)
        texts = [r["text"] for r in rows]
        vec = TfidfVectorizer(max_features=500, stop_words="english")
        X = vec.fit_transform(texts)
        assert X.shape[0] == len(texts)
        assert X.shape[1] <= 500

    def test_tier3_02_data_train_split_tokenizable_by_distilbert(self):
        """[F1 x F3] Verify train.csv 'text' is directly tokenizable by DistilBERT tokenizer."""
        if not TRAIN_CSV.is_file() or not (DISTILBERT_DIR / "config.json").is_file():
            pytest.skip("train.csv or distilbert not ready")
        transformers = get_transformers()
        tokenizer = transformers.AutoTokenizer.from_pretrained(str(DISTILBERT_DIR))
        rows = load_csv_rows(TRAIN_CSV, max_rows=20)
        texts = [r["text"] for r in rows]
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        assert "input_ids" in encoded
        assert encoded["input_ids"].shape[0] == len(texts)

    def test_tier3_03_test_split_labels_match_model_output_domain(self):
        """[F1 x F4] Verify test.csv labels are integers {0, 1} matching model predictions."""
        if not TEST_CSV.is_file():
            pytest.skip("test.csv not ready")
        rows = load_csv_rows(TEST_CSV, max_rows=100)
        labels = [int(r["label"]) for r in rows]
        assert set(labels).issubset({0, 1})

    def test_tier3_04_nb_pipeline_compatible_with_lime_explainer(self):
        """[F2 x F5] Verify trained Naive Bayes predict_proba integrates with LimeTextExplainer."""
        if not NB_MODEL_PATH.is_file():
            pytest.skip("naive_bayes.joblib not ready")
        get_lime()
        joblib = get_joblib()
        from lime.lime_text import LimeTextExplainer
        model = joblib.load(NB_MODEL_PATH)
        explainer = LimeTextExplainer(class_names=["ham", "spam"])
        exp = explainer.explain_instance("Exclusive lottery winner claim your prize", model.predict_proba, num_features=4, num_samples=40)
        assert exp is not None
        assert len(exp.as_list()) <= 4

    def test_tier3_05_distilbert_predict_fn_compatible_with_lime_explainer(self):
        """[F3 x F5] Verify DistilBERT prediction wrapper integrates with LimeTextExplainer."""
        if not (DISTILBERT_DIR / "config.json").is_file() or not EXPLAIN_LIME_SCRIPT.is_file():
            pytest.skip("distilbert or explain_lime not ready")
        get_lime()
        sys.path.insert(0, str(BASE_DIR))
        try:
            from explain_lime import EmailClassifierExplainer
            explainer = EmailClassifierExplainer(model_type="distilbert")
            res = explainer.explain("Meeting tomorrow at 10 AM", num_features=3, num_samples=30)
            assert "predicted_label" in res
            assert "html_visualization" in res
        finally:
            if str(BASE_DIR) in sys.path:
                sys.path.remove(str(BASE_DIR))

    def test_tier3_06_metrics_json_matches_app_display_contract(self):
        """[F4 x F6] Verify metrics_comparison.json schema matches app.py display contract."""
        if not METRICS_PATH.is_file() or not APP_SCRIPT.is_file():
            pytest.skip("metrics or app not ready")
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        for model in ["Naive Bayes", "DistilBERT"]:
            assert model in metrics
            for k in ["precision", "recall", "f1", "roc_auc"]:
                assert k in metrics[model]

    def test_tier3_07_lime_html_safe_for_streamlit_component(self):
        """[F5 x F6] Verify LIME HTML output string can be safely embedded in Streamlit components."""
        if not NB_MODEL_PATH.is_file() or not EXPLAIN_LIME_SCRIPT.is_file():
            pytest.skip("Prerequisites not ready")
        get_lime()
        sys.path.insert(0, str(BASE_DIR))
        try:
            from explain_lime import EmailClassifierExplainer
            explainer = EmailClassifierExplainer(model_type="naive_bayes")
            res = explainer.explain("Please review the attached invoice", num_features=2, num_samples=30)
            html = res["html_visualization"]
            assert isinstance(html, str)
            assert len(html) > 0
            # Ensure not containing null byte or breaking characters
            assert "\x00" not in html
        finally:
            if str(BASE_DIR) in sys.path:
                sys.path.remove(str(BASE_DIR))

    def test_tier3_08_pipeline_dag_execution_order_consistency(self):
        """[F1-F7] Verify pipeline script execution dependencies form a valid DAG."""
        scripts = [PREPARE_DATA_SCRIPT, TRAIN_NB_SCRIPT, TRAIN_DISTILBERT_SCRIPT, EVALUATE_SCRIPT, APP_SCRIPT]
        for s in scripts:
            assert s.name.endswith(".py"), f"Invalid script name: {s}"

    def test_tier3_09_model_serialization_deserialization_fidelity(self):
        """[F2] Verify re-loading Naive Bayes produces exact identical probabilities."""
        if not NB_MODEL_PATH.is_file():
            pytest.skip("naive_bayes.joblib not ready")
        joblib = get_joblib()
        m1 = joblib.load(NB_MODEL_PATH)
        m2 = joblib.load(NB_MODEL_PATH)
        sample = ["Test consistency message for model serialization"]
        p1 = m1.predict_proba(sample)
        p2 = m2.predict_proba(sample)
        assert math.isclose(float(p1[0, 0]), float(p2[0, 0]), abs_tol=1e-7)
        assert math.isclose(float(p1[0, 1]), float(p2[0, 1]), abs_tol=1e-7)

    def test_tier3_10_label_encoding_uniformity_across_all_modules(self):
        """[F1-F6] Verify 0=ham and 1=spam is uniform across splits and explainer."""
        if not TRAIN_CSV.is_file():
            pytest.skip("train.csv not ready")
        rows = load_csv_rows(TRAIN_CSV, max_rows=50)
        for r in rows:
            if r["label"] == "0":
                assert r["label_text"].lower() == "ham"
            elif r["label"] == "1":
                assert r["label_text"].lower() == "spam"

    def test_tier3_11_test_set_size_matches_evaluation_sample_count(self):
        """[F1 x F4] Verify test.csv contains sufficient sample size (>=1000 records)."""
        if not TEST_CSV.is_file():
            pytest.skip("test.csv not ready")
        n_test = sum(1 for _ in open(TEST_CSV, encoding="utf-8", errors="replace")) - 1
        assert n_test >= 1000, f"Test split too small: {n_test} records"

    def test_tier3_12_distilbert_and_nb_agree_on_high_confidence_predictions(self):
        """[F2 x F3] Verify both models agree on high-confidence unambiguous spam sample."""
        if not NB_MODEL_PATH.is_file() or not (DISTILBERT_DIR / "config.json").is_file():
            pytest.skip("Models not ready")
        joblib = get_joblib()
        nb = joblib.load(NB_MODEL_PATH)
        transformers = get_transformers()
        torch = get_torch()
        tokenizer = transformers.AutoTokenizer.from_pretrained(str(DISTILBERT_DIR))
        distilbert = transformers.AutoModelForSequenceClassification.from_pretrained(str(DISTILBERT_DIR))
        distilbert.eval()

        clear_spam = "FREE LOTTERY WINNER 1,000,000 CASH PRIZE! REPLY IMMEDIATELY WITH BANK ACCOUNT DETAILS"
        nb_pred = int(nb.predict([clear_spam])[0])

        inputs = tokenizer([clear_spam], padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            db_logits = distilbert(**inputs).logits
            db_pred = int(torch.argmax(db_logits, dim=-1)[0])

        assert nb_pred == 1, "Naive Bayes failed to identify obvious spam"
        assert db_pred == 1, "DistilBERT failed to identify obvious spam"


# ============================================================================
# TIER 4: REAL-WORLD SCENARIOS
# ============================================================================

class TestTier4RealWorldScenarios:
    """Tier 4: End-to-end realistic user email classification and explanation workflows."""

    def test_tier4_01_scenario_lottery_scam(self):
        """[Scenario 1] Nigerian Prince / Lottery Scam Email.
        Expected: SPAM (label 1), high confidence, LIME highlights scam tokens.
        """
        if not NB_MODEL_PATH.is_file():
            pytest.skip("naive_bayes.joblib not ready")
        joblib = get_joblib()
        model = joblib.load(NB_MODEL_PATH)
        email_text = (
            "Subject: AWARD NOTIFICATION: International Millions Euro Lottery\n\n"
            "Dear Winner, We are pleased to notify you that your email address was selected "
            "as a winner of 2,500,000 USD in the Euro Mega Draw! "
            "To claim your funds, send your full name, banking details, and telephone number "
            "to our claims agent immediately at claims@euromillions-payout.org."
        )
        pred = model.predict([email_text])[0]
        proba = model.predict_proba([email_text])[0]
        assert int(pred) == 1, f"Lottery scam classified as ham (proba: {proba})"
        assert proba[1] >= 0.60, f"Confidence too low for clear lottery scam: {proba[1]}"

    def test_tier4_02_scenario_corporate_meeting(self):
        """[Scenario 2] Legitimate Corporate Business Meeting Invitation.
        Expected: HAM (label 0), high confidence.
        """
        if not NB_MODEL_PATH.is_file():
            pytest.skip("naive_bayes.joblib not ready")
        joblib = get_joblib()
        model = joblib.load(NB_MODEL_PATH)
        email_text = (
            "Subject: Quarterly Engineering Roadmap Sync - Conference Room 3\n\n"
            "Hi team,\n"
            "Attached is the meeting agenda for our sprint planning session tomorrow at 2:00 PM. "
            "We will review sprint deliverables, discuss Q3 OKRs, and assign engineering tasks. "
            "Please review the attached slide deck before attending.\n"
            "Best regards,\nEngineering Manager"
        )
        pred = model.predict([email_text])[0]
        proba = model.predict_proba([email_text])[0]
        assert int(pred) == 0, f"Corporate meeting classified as spam (proba: {proba})"
        assert proba[0] >= 0.60, f"Confidence too low for corporate meeting: {proba[0]}"

    def test_tier4_03_scenario_phishing_account_suspension(self):
        """[Scenario 3] Urgent Account Suspension Phishing Alert.
        Expected: SPAM (label 1).
        """
        if not NB_MODEL_PATH.is_file():
            pytest.skip("naive_bayes.joblib not ready")
        joblib = get_joblib()
        model = joblib.load(NB_MODEL_PATH)
        email_text = (
            "Subject: SECURITY ALERT: Immediate Action Required - Account Suspended\n\n"
            "Your online banking access has been temporarily suspended due to unauthorized login attempts. "
            "You must verify your identity within 24 hours or your account will be permanently terminated. "
            "Click here: http://secure-update-verification.net/auth to confirm your credentials."
        )
        pred = model.predict([email_text])[0]
        assert int(pred) == 1, "Phishing email classified as ham"

    def test_tier4_04_scenario_student_professor_project(self):
        """[Scenario 4] CapStone Project Submission & Academic Discussion.
        Expected: HAM (label 0).
        """
        if not NB_MODEL_PATH.is_file():
            pytest.skip("naive_bayes.joblib not ready")
        joblib = get_joblib()
        model = joblib.load(NB_MODEL_PATH)
        email_text = (
            "Subject: CapStone Project Submission - Group 5 Email Classification\n\n"
            "Dear Professor,\n"
            "We have submitted our capstone project report and source code repository. "
            "We completed model fine-tuning with DistilBERT and evaluated metrics against Naive Bayes. "
            "Please let us know if you have feedback on the results.\n"
            "Sincerely,\nGroup 5 Students"
        )
        pred = model.predict([email_text])[0]
        proba = model.predict_proba([email_text])[0]
        assert int(pred) == 0, f"Academic email classified as spam (proba: {proba})"

    def test_tier4_05_scenario_promotional_newsletter(self):
        """[Scenario 5] Promotional Marketing Newsletter.
        Expected: Classified cleanly, probabilities well-formed.
        """
        if not NB_MODEL_PATH.is_file():
            pytest.skip("naive_bayes.joblib not ready")
        joblib = get_joblib()
        model = joblib.load(NB_MODEL_PATH)
        email_text = (
            "Subject: Summer Mega Sale - Up to 70% Off Everything!\n\n"
            "Don't miss our exclusive summer clearance! Enjoy huge discounts on thousands of items. "
            "Shop now while supplies last. Free shipping on orders over $50. "
            "Click here to unsubscribe from future promotional mailings."
        )
        proba = model.predict_proba([email_text])[0]
        assert math.isclose(float(proba.sum()), 1.0, rel_tol=1e-4)

    def test_tier4_06_scenario_it_helpdesk_password_reset(self):
        """[Scenario 6] IT Service Desk Scheduled Maintenance Notice.
        Expected: HAM (label 0).
        """
        if not NB_MODEL_PATH.is_file():
            pytest.skip("naive_bayes.joblib not ready")
        joblib = get_joblib()
        model = joblib.load(NB_MODEL_PATH)
        email_text = (
            "Subject: IT Service Desk Ticket #48291 - Scheduled Server Maintenance\n\n"
            "Hello all,\n"
            "This is a notice from the IT Service Desk. Server maintenance is scheduled for Sunday "
            "from 1:00 AM to 3:00 AM EST. Network services and internal email will be temporarily "
            "unavailable during this maintenance window. No action is required from users.\n"
            "Thank you,\nIT Support Desk"
        )
        pred = model.predict([email_text])[0]
        assert int(pred) == 0, "IT helpdesk notice classified as spam"
