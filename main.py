"""
main.py
-------
Main pipeline: Load Data → Preprocess → Train → Evaluate → Visualize

Usage:
    python main.py
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

from src.data_preprocessing import load_data, explore_data, preprocess_data
from src.model_training import train_all_models
from src.evaluation import evaluate_all_models, generate_all_plots, print_summary_table


def main():
    """Run the full ML pipeline."""
    print("\n" + "🔥" * 30)
    print("  TOXIC COMMENT DETECTION — ML Pipeline")
    print("🔥" * 30)

    # ── 1. Load Data ──────────────────────────────────────────────────
    data_path = os.path.join('data', 'train.csv')
    if not os.path.exists(data_path):
        print(f"\n❌ Dataset not found at '{data_path}'")
        print("   Please download the Jigsaw dataset from Kaggle and place 'train.csv' in the data/ folder.")
        print("   📥 https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data")
        sys.exit(1)

    df = load_data(data_path)

    # ── 2. Explore ────────────────────────────────────────────────────
    explore_data(df)

    # ── 3. Preprocess ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, vectorizer = preprocess_data(
        df, max_features=50000, test_size=0.2
    )

    # ── 4. Train All Models ───────────────────────────────────────────
    trained_models = train_all_models(X_train, y_train, save_dir='models')

    # ── 5. Evaluate ───────────────────────────────────────────────────
    results = evaluate_all_models(trained_models, X_test, y_test)

    # ── 6. Visualize ──────────────────────────────────────────────────
    generate_all_plots(results, y_test, save_dir='outputs')

    # ── 7. Summary ────────────────────────────────────────────────────
    print_summary_table(results)

    print("\n✅ Pipeline complete!")
    print("   📂 Models saved in   : models/")
    print("   📊 Plots saved in    : outputs/")
    print("   🌐 Run the web app   : streamlit run app.py")


if __name__ == '__main__':
    main()
