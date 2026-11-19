"""
app.py
------
Interactive web interface for email spam and phishing classification.
Built with Streamlit, comparing fine-tuned DistilBERT against a Naive Bayes baseline
with interactive LIME (Local Interpretable Model-agnostic Explanations) feature attribution.
Styled with Times New Roman typography and balanced academic layout spacing.
"""

import os
import json
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from explain_lime import EmailClassifierExplainer

# Page setup
st.set_page_config(
    page_title="Spam & Phishing Email Classifier",
    page_icon="✉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global Times New Roman typography and balanced layout spacing
st.markdown("""
<style>
    /* Apply Times New Roman to typography while strictly preserving icon ligatures */
    html, body, h1, h2, h3, h4, h5, h6, p, label, textarea, input, button, table {
        font-family: 'Times New Roman', Times, serif !important;
    }

    .stMarkdown, .stMarkdown p, .stCaption, [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
        font-family: 'Times New Roman', Times, serif !important;
    }

    /* Ensure Streamlit system icons (expander arrows, dropdown arrows) keep their icon font */
    [data-testid="stIconMaterial"], 
    [data-testid*="Icon"], 
    [data-testid*="icon"],
    [class*="material-symbols"], 
    [class*="material-icons"] {
        font-family: 'Material Symbols Rounded', 'Material Icons' !important;
        letter-spacing: normal !important;
    }

    /* Page container breathing room */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1100px !important;
    }

    /* Headings with academic styling */
    h1 {
        font-size: 2.1rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em !important;
        margin-bottom: 0.25rem !important;
    }
    
    .page-subtitle {
        font-size: 1.05rem !important;
        font-style: italic !important;
        color: #888888;
        margin-bottom: 1.75rem !important;
    }

    /* Clean spacing between section blocks */
    .section-spacing {
        margin-top: 1.5rem !important;
        margin-bottom: 0.75rem !important;
    }

    /* Custom button styling */
    .stButton button {
        font-size: 1rem !important;
        font-weight: 600 !important;
        padding: 0.45rem 1.25rem !important;
        border-radius: 6px !important;
    }

    /* Metric values in Times New Roman */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.95rem !important;
        font-weight: 500 !important;
    }

    /* Textarea padding and border */
    textarea {
        font-size: 1rem !important;
        line-height: 1.5 !important;
    }
</style>
""", unsafe_allow_html=True)

# Reference samples
SAMPLE_EMAILS = {
    "-- Select an example email --": "",
    "Phishing: Lottery & Gift Card Scam": (
        "Subject: Congratulations! You won a $1,000 Walmart Gift Card!\n\n"
        "Dear customer, you have been selected as our lucky winner today. "
        "Claim your prize immediately by clicking the link below. "
        "Urgent: this limited offer expires within 24 hours. Free cash bonus included!"
    ),
    "Spam: Online Pharmacy Promotion": (
        "Subject: Best online pharmacy - Discount pills and medications\n\n"
        "Buy cheap generic prescription meds with no doctor visit required. "
        "Instant delivery, fast approval, 100% confidential. "
        "Special discount for bulk purchase. Order now for lowest price guarantee."
    ),
    "Legitimate: Project Sprint Meeting": (
        "Subject: Weekly Project Status Meeting Notes\n\n"
        "Hi team, attached are the minutes and action items from our project sync this morning. "
        "Please review the updated deliverables schedule and let me know if you have any questions "
        "before Thursday's deployment review. Thanks, Gary."
    ),
    "Legitimate: Gas Transmission Contract": (
        "Subject: Gas transmission contract revision draft\n\n"
        "Good morning Vince, please find attached the revised draft of the pipeline agreement. "
        "We incorporated the feedback from legal and operational teams. "
        "Let us schedule a brief call tomorrow afternoon to finalize the terms."
    )
}

METRICS_JSON_PATH = "models/metrics_comparison.json"
NB_MODEL_PATH = "models/naive_bayes.joblib"
DISTILBERT_DIR = "models/distilbert"

@st.cache_resource
def get_explainer(model_type: str):
    """Load and cache the classifier and LIME explainer instance."""
    return EmailClassifierExplainer(model_type=model_type)

def main():
    # Header Section
    st.title("Email Spam & Phishing Classifier")
    st.markdown('<div class="page-subtitle">DistilBERT Transformer vs. Multinomial Naive Bayes Baseline (Enron Dataset)</div>', unsafe_allow_html=True)

    # ---------------- Sidebar ----------------
    st.sidebar.markdown("### Model Configuration")

    available_models = []
    if os.path.exists(DISTILBERT_DIR):
        available_models.append("DistilBERT")
    if os.path.exists(NB_MODEL_PATH):
        available_models.append("Naive Bayes")

    if not available_models:
        st.sidebar.error("Model artifacts not found in models/. Please run training scripts first.")
        available_models = ["Naive Bayes", "DistilBERT"]

    selected_model_name = st.sidebar.selectbox(
        "Active Classifier",
        options=available_models,
        index=0,
        help="DistilBERT provides deep contextual representation; Naive Bayes runs as a fast baseline."
    )
    model_type = "distilbert" if selected_model_name == "DistilBERT" else "naive_bayes"

    with st.sidebar.expander("LIME Settings", expanded=False):
        num_features = st.slider("Top Keywords", min_value=3, max_value=12, value=6)
        num_samples = st.slider(
            "Perturbation Samples",
            min_value=100,
            max_value=500,
            value=250,
            step=50,
            help="Higher values give more stable word attributions."
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Benchmark Performance")

    if os.path.exists(METRICS_JSON_PATH):
        try:
            with open(METRICS_JSON_PATH, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)

            table_rows = []
            for m_name, vals in metrics_data.items():
                table_rows.append({
                    "Model": m_name,
                    "Precision": f"{vals.get('precision', 0):.2%}",
                    "Recall": f"{vals.get('recall', 0):.2%}",
                    "F1": f"{vals.get('f1', 0):.2%}",
                    "ROC-AUC": f"{vals.get('roc_auc', 0):.4f}"
                })
            st.sidebar.dataframe(
                pd.DataFrame(table_rows),
                use_container_width=True,
                hide_index=True
            )
        except Exception as e:
            st.sidebar.caption(f"Could not load metrics: {e}")
    else:
        st.sidebar.caption("Run evaluate.py to generate metrics_comparison.json.")

    if os.path.exists("loss_curve.png"):
        with st.sidebar.expander("Training Loss Curve", expanded=False):
            st.image("loss_curve.png", caption="DistilBERT Loss Dynamics (300 DPI)", width="stretch")

    # ---------------- Input Section ----------------
    with st.container(border=True):
        col_hdr, col_select = st.columns([1.6, 1])
        with col_hdr:
            st.markdown("#### Email Document Input")
        with col_select:
            preset = st.selectbox(
                "Load example email",
                options=list(SAMPLE_EMAILS.keys()),
                index=0,
                label_visibility="collapsed"
            )
            if preset and SAMPLE_EMAILS[preset]:
                st.session_state["email_content"] = SAMPLE_EMAILS[preset]

        current_val = st.session_state.get("email_content", "")
        email_text = st.text_area(
            "Paste email text (Subject + Body):",
            value=current_val,
            height=170,
            placeholder="Type or paste the subject and body of an email here to analyze...",
            label_visibility="collapsed"
        )

        btn_col1, btn_col2, _ = st.columns([1.3, 1, 4])
        with btn_col1:
            classify_clicked = st.button("Classify Email", type="primary", use_container_width=True)
        with btn_col2:
            if st.button("Clear Input", use_container_width=True):
                st.session_state["email_content"] = ""
                st.rerun()

    # ---------------- Results Section ----------------
    if classify_clicked:
        if not email_text.strip():
            st.warning("Please provide an email text to analyze.")
            return

        try:
            with st.spinner(f"Analyzing email using {selected_model_name}..."):
                explainer = get_explainer(model_type)
                result = explainer.explain(
                    text=email_text,
                    num_features=num_features,
                    num_samples=num_samples
                )

            pred_label = result["predicted_label"]
            prob_spam = result["probability"]["Spam"]
            prob_ham = result["probability"]["Ham"]
            is_spam = (pred_label.lower() == "spam")

            # 1. Clean, Unified Verdict Card
            with st.container(border=True):
                col_verdict, col_spam_p, col_ham_p = st.columns([2.2, 1, 1])

                with col_verdict:
                    if is_spam:
                        st.error(f"### SPAM / PHISHING DETECTED\n**Model:** {selected_model_name} &nbsp;|&nbsp; **Confidence:** {prob_spam * 100:.1f}%")
                    else:
                        st.success(f"### LEGITIMATE EMAIL (HAM)\n**Model:** {selected_model_name} &nbsp;|&nbsp; **Confidence:** {prob_ham * 100:.1f}%")

                with col_spam_p:
                    st.metric("Spam Probability", f"{prob_spam * 100:.1f}%")
                with col_ham_p:
                    st.metric("Ham Probability", f"{prob_ham * 100:.1f}%")

                st.progress(float(prob_spam))

            # 2. Key Attribution Words (LIME)
            with st.container(border=True):
                st.markdown("#### Keyword Attribution Analysis (LIME)")
                st.markdown(
                    "<span style='font-size: 0.95rem; color: #888888; font-style: italic;'>"
                    "Top salient keywords contributing to this classification. "
                    "Positive weights indicate evidence for Spam; negative weights indicate evidence for Legitimate (Ham)."
                    "</span>",
                    unsafe_allow_html=True
                )

                if result["top_features"]:
                    chart_df = pd.DataFrame([
                        {
                            "Keyword": f["word"],
                            "Weight": f"{f['weight']:+.4f}",
                            "Supports": f["target"]
                        }
                        for f in result["top_features"]
                    ])

                    col_chart, col_table = st.columns([1.5, 1])
                    with col_chart:
                        bar_data = pd.DataFrame(
                            {"LIME Weight": [f["weight"] for f in result["top_features"]]},
                            index=[f["word"] for f in result["top_features"]]
                        )
                        st.bar_chart(bar_data, horizontal=True)

                    with col_table:
                        st.dataframe(chart_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No significant attribution keywords found in this input.")

            # 3. Interactive Highlighted Text
            with st.expander("View Interactive Highlighted Email Text", expanded=True):
                st.caption("Words highlighted in orange/red contribute to Spam, while blue/green words support Ham:")
                components.html(
                    result["html_visualization"],
                    height=360,
                    scrolling=True
                )

        except Exception as err:
            st.error(f"Classification error: {err}")
            st.exception(err)

if __name__ == "__main__":
    import sys
    if not st.runtime.exists():
        from streamlit.web import cli as stcli
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
    else:
        main()
