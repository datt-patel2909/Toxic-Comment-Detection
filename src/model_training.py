"""
model_training.py
-----------------
Train multiple ML classifiers for toxic comment detection.
"""

import os
import time
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV


# ── Model Definitions ──────────────────────────────────────────────────────

def get_models() -> dict:
    """Return a dictionary of model name → model instance."""
    return {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver='liblinear',
            class_weight='balanced',
            random_state=42
        ),
        'Linear SVM': CalibratedClassifierCV(
            LinearSVC(max_iter=2000, C=1.0, class_weight='balanced', random_state=42),
            cv=3
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=300,
            max_depth=80,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        ),
        'Naive Bayes': MultinomialNB(alpha=0.1),
    }


# ── Training ───────────────────────────────────────────────────────────────

def train_model(name: str, model, X_train, y_train) -> tuple:
    """Train a single model and return (model, training_time)."""
    print(f"\n🏋️  Training {name} ...")
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"   ✅ Done in {elapsed:.2f}s")
    return model, elapsed


def train_all_models(X_train, y_train, save_dir: str = 'models') -> dict:
    """
    Train all models and save them.

    Returns:
        dict of {model_name: {'model': fitted_model, 'time': seconds}}
    """
    os.makedirs(save_dir, exist_ok=True)
    models = get_models()
    results = {}

    print("\n" + "=" * 60)
    print("  🚀  MODEL TRAINING")
    print("=" * 60)

    for name, model in models.items():
        fitted_model, elapsed = train_model(name, model, X_train, y_train)

        # Save model
        filename = name.lower().replace(' ', '_') + '.pkl'
        filepath = os.path.join(save_dir, filename)
        joblib.dump(fitted_model, filepath)
        print(f"   💾 Saved → {filepath}")

        results[name] = {
            'model': fitted_model,
            'time': elapsed,
        }

    print("\n" + "=" * 60)
    print("  ✅  All models trained and saved!")
    print("=" * 60)

    return results


def load_model(model_name: str, models_dir: str = 'models'):
    """Load a saved model by name."""
    filename = model_name.lower().replace(' ', '_') + '.pkl'
    filepath = os.path.join(models_dir, filename)
    return joblib.load(filepath)


if __name__ == '__main__':
    print("Run main.py instead to train models with data.")
