"""
predict.py
----------
Predict toxicity for a single comment using a saved model.
"""

import os
import joblib
from src.data_preprocessing import clean_text


def load_predictor(model_name: str = 'logistic_regression',
                   models_dir: str = 'models'):
    """Load a saved model and vectorizer."""
    model_path = os.path.join(models_dir, f'{model_name}.pkl')
    vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.pkl')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"Vectorizer not found: {vectorizer_path}")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


def predict_toxicity(text: str, model=None, vectorizer=None,
                     model_name: str = 'logistic_regression',
                     models_dir: str = 'models') -> dict:
    """
    Predict whether a comment is toxic.

    Args:
        text: Raw comment text
        model: Pre-loaded model (optional)
        vectorizer: Pre-loaded vectorizer (optional)
        model_name: Name of saved model to load (if model not provided)
        models_dir: Directory containing saved models

    Returns:
        dict with 'label', 'toxic_probability', 'cleaned_text'
    """
    if model is None or vectorizer is None:
        model, vectorizer = load_predictor(model_name, models_dir)

    # Clean and vectorize
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])

    # Predict
    prediction = model.predict(X)[0]

    # Get probability
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)[0]
        toxic_prob = proba[1]
    else:
        toxic_prob = float(prediction)

    return {
        'label': 'TOXIC ⚠️' if prediction == 1 else 'NOT TOXIC ✅',
        'is_toxic': bool(prediction),
        'toxic_probability': float(toxic_prob),
        'confidence': float(max(toxic_prob, 1 - toxic_prob)),
        'cleaned_text': cleaned,
    }


if __name__ == '__main__':
    # Interactive demo
    print("\n🔍 Toxic Comment Detector")
    print("=" * 40)
    print("Type a comment to check. Type 'quit' to exit.\n")

    while True:
        text = input("📝 Enter comment: ").strip()
        if text.lower() in ('quit', 'exit', 'q'):
            break
        if not text:
            continue

        result = predict_toxicity(text)
        print(f"   → {result['label']}")
        print(f"   → Confidence: {result['confidence']:.1%}")
        print(f"   → Toxic probability: {result['toxic_probability']:.1%}")
        print()
