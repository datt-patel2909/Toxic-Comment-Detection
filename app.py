"""
app.py
------
Streamlit Web App for Toxic Comment Detection.
A polished, dark-themed UI for live toxicity prediction.

Usage:
    streamlit run app.py
"""

import os
import streamlit as st
import joblib
import pandas as pd
from src.data_preprocessing import clean_text


# ── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Toxic Comment Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Hero Header */
    .hero-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    .hero-header h1 {
        color: white;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-header p {
        color: rgba(255,255,255,0.85);
        font-size: 1.05rem;
        margin: 0.5rem 0 0;
        font-weight: 300;
    }

    /* Result Cards */
    .result-card {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }
    .toxic-result {
        background: linear-gradient(135deg, #ff6b6b20, #ee5a2420);
        border-left: 5px solid #ff6b6b;
    }
    .safe-result {
        background: linear-gradient(135deg, #00b89420, #00cec920);
        border-left: 5px solid #00b894;
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #f8f9ff, #f0f2ff);
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    }
    .metric-card h3 {
        color: #6C5CE7;
        font-size: 1.8rem;
        margin: 0;
        font-weight: 700;
    }
    .metric-card p {
        color: #636e72;
        font-size: 0.85rem;
        margin: 0.3rem 0 0;
        font-weight: 500;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2d3436, #1e272e);
    }
    section[data-testid="stSidebar"] .stMarkdown {
        color: #dfe6e9;
    }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .info-box {
        background: linear-gradient(135deg, #dfe6e930, #74b9ff20);
        border-left: 4px solid #0984e3;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Load Models ────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Load all saved models and the vectorizer."""
    models_dir = 'models'
    vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.pkl')

    if not os.path.exists(vectorizer_path):
        return None, None

    vectorizer = joblib.load(vectorizer_path)

    available_models = {}
    model_files = {
        'Logistic Regression': 'logistic_regression.pkl',
        'Linear SVM': 'linear_svm.pkl',
        'Random Forest': 'random_forest.pkl',
        'Naive Bayes': 'naive_bayes.pkl',
    }

    for name, filename in model_files.items():
        path = os.path.join(models_dir, filename)
        if os.path.exists(path):
            available_models[name] = joblib.load(path)

    return available_models, vectorizer


# ── Main App ───────────────────────────────────────────────────────────────
def main():
    # Hero Header
    st.markdown("""
    <div class="hero-header">
        <h1>🛡️ Toxic Comment Detector</h1>
        <p>Machine Learning-powered hate speech & toxicity detection</p>
    </div>
    """, unsafe_allow_html=True)

    # Load models
    models, vectorizer = load_models()

    if models is None or len(models) == 0:
        st.error("⚠️ **No trained models found!** Please run `python main.py` first to train the models.")
        st.code("python main.py", language="bash")
        return

    # ── Sidebar ────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        selected_model = st.selectbox(
            "Choose Model",
            list(models.keys()),
            index=0,
            help="Select which ML model to use for prediction"
        )

        st.markdown("---")
        st.markdown("## 📖 About")
        st.markdown("""
        This app uses **NLP** and **Machine Learning** to detect toxic comments.

        **Pipeline:**
        1. Text cleaning & preprocessing
        2. TF-IDF vectorization
        3. Classification with ML models

        **Models trained on:**
        Jigsaw Toxic Comment Dataset (~160K comments)
        """)

        st.markdown("---")
        st.markdown("## 🎯 Try These Examples")
        examples = [
            "You are such a wonderful person, thank you for your help!",
            "This article needs more references and citations.",
            "I hate you and hope you die you worthless piece of garbage",
            "The weather today is absolutely beautiful.",
        ]
        for ex in examples:
            if st.button(f"📝 {ex[:45]}...", key=ex[:20], use_container_width=True):
                st.session_state['example_text'] = ex

    # ── Main Content ───────────────────────────────────────────────────
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📝 Enter a Comment")

        # Get default text
        default_text = st.session_state.get('example_text', '')

        user_input = st.text_area(
            "Type or paste a comment to analyze:",
            value=default_text,
            height=150,
            placeholder="Enter a comment here to check if it's toxic...",
            key="comment_input"
        )

        analyze_btn = st.button("🔍 Analyze Comment", type="primary", use_container_width=True)

    with col2:
        st.markdown("### ℹ️ Current Model")
        st.markdown(f"""
        <div class="metric-card">
            <h3>{selected_model}</h3>
            <p>Active Classifier</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-box">
            <strong>Models available:</strong> {len(models)}<br>
            <strong>Features:</strong> TF-IDF (50K)<br>
            <strong>Type:</strong> Binary Classification
        </div>
        """, unsafe_allow_html=True)

    # ── Prediction ─────────────────────────────────────────────────────
    if analyze_btn and user_input.strip():
        with st.spinner("Analyzing..."):
            model = models[selected_model]

            # Clean and vectorize
            cleaned = clean_text(user_input)
            X = vectorizer.transform([cleaned])

            # Predict
            prediction = model.predict(X)[0]
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)[0]
                toxic_prob = proba[1]
            else:
                toxic_prob = float(prediction)

            confidence = max(toxic_prob, 1 - toxic_prob)

        # Display results
        st.markdown("---")
        st.markdown("### 📊 Analysis Result")

        if prediction == 1:
            st.markdown(f"""
            <div class="result-card toxic-result">
                <h2 style="color: #ff6b6b; margin: 0;">⚠️ TOXIC</h2>
                <p style="font-size: 1.1rem; margin: 0.5rem 0 0;">
                    This comment has been classified as <strong>toxic</strong> with
                    <strong>{confidence:.1%}</strong> confidence.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card safe-result">
                <h2 style="color: #00b894; margin: 0;">✅ NOT TOXIC</h2>
                <p style="font-size: 1.1rem; margin: 0.5rem 0 0;">
                    This comment has been classified as <strong>safe</strong> with
                    <strong>{confidence:.1%}</strong> confidence.
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Confidence metrics
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.metric("Toxic Probability", f"{toxic_prob:.1%}")
        with mc2:
            st.metric("Confidence", f"{confidence:.1%}")
        with mc3:
            st.metric("Model Used", selected_model.split()[0])

        # Show cleaned text
        with st.expander("🔍 Preprocessed Text"):
            st.code(cleaned)

    elif analyze_btn:
        st.warning("Please enter a comment to analyze.")

    # ── Training Results ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📈 Training Results & Visualizations")

    outputs_dir = 'outputs'
    if os.path.exists(outputs_dir):
        img_files = {
            'Model Comparison': 'model_comparison.png',
            'ROC Curves': 'roc_curves.png',
            'Confusion Matrices': 'confusion_matrices.png',
            'Training Time': 'training_time.png',
        }

        tabs = st.tabs(list(img_files.keys()))
        for tab, (title, filename) in zip(tabs, img_files.items()):
            with tab:
                img_path = os.path.join(outputs_dir, filename)
                if os.path.exists(img_path):
                    st.image(img_path, caption=title, use_column_width=True)
                else:
                    st.info(f"Chart not generated yet. Run `python main.py` first.")
    else:
        st.info("📊 Run `python main.py` first to generate training results and visualizations.")


if __name__ == '__main__':
    main()
