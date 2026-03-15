"""
data_preprocessing.py
---------------------
Handles loading, cleaning, and vectorizing the toxic comment dataset.
"""

import os
import re
import string
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# ── Constants ──────────────────────────────────────────────────────────────
STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

LABEL_COLUMNS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


# ── Text Cleaning ──────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Clean a single comment string."""
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove newlines and extra whitespace
    text = re.sub(r'\n', ' ', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove digits
    text = re.sub(r'\d+', '', text)

    # Tokenize, remove stopwords, lemmatize
    tokens = text.split()
    tokens = [LEMMATIZER.lemmatize(word) for word in tokens if word not in STOPWORDS and len(word) > 2]

    return ' '.join(tokens)


# ── Data Loading ───────────────────────────────────────────────────────────

def load_data(data_path: str) -> pd.DataFrame:
    """Load the CSV dataset and return a DataFrame."""
    print(f"📂 Loading data from {data_path} ...")
    df = pd.read_csv(data_path)
    print(f"   ✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


def explore_data(df: pd.DataFrame) -> dict:
    """Print basic stats and return a summary dict."""
    print("\n📊 Dataset Overview:")
    print(f"   Total comments : {len(df):,}")
    print(f"   Columns        : {list(df.columns)}")
    print(f"   Missing values : {df.isnull().sum().sum()}")

    print("\n📊 Label Distribution:")
    label_counts = {}
    for col in LABEL_COLUMNS:
        count = df[col].sum()
        pct = count / len(df) * 100
        label_counts[col] = int(count)
        print(f"   {col:15s}: {count:6,} ({pct:.2f}%)")

    clean_count = len(df) - df[LABEL_COLUMNS].any(axis=1).sum()
    print(f"\n   🟢 Clean comments : {clean_count:,} ({clean_count/len(df)*100:.2f}%)")
    print(f"   🔴 Toxic comments : {len(df) - clean_count:,} ({(len(df)-clean_count)/len(df)*100:.2f}%)")

    return label_counts


# ── Preprocessing Pipeline ─────────────────────────────────────────────────

def preprocess_data(df: pd.DataFrame, max_features: int = 50000,
                    test_size: float = 0.2, random_state: int = 42,
                    save_dir: str = 'models'):
    """
    Full preprocessing pipeline:
      1. Clean text
      2. TF-IDF vectorization
      3. Train/test split
      4. Save vectorizer

    Returns: X_train, X_test, y_train, y_test, vectorizer
    """
    print("\n🧹 Cleaning text ...")
    df['cleaned_text'] = df['comment_text'].apply(clean_text)

    # Remove empty rows after cleaning
    df = df[df['cleaned_text'].str.strip() != '']
    print(f"   ✅ {len(df):,} comments after cleaning")

    print(f"\n🔤 Vectorizing with TF-IDF (max_features={max_features:,}) ...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),       # unigrams + bigrams
        min_df=3,                 # ignore very rare terms
        max_df=0.9,               # ignore very common terms
        sublinear_tf=True,        # apply log normalization
    )

    X = vectorizer.fit_transform(df['cleaned_text'])
    y = df['toxic'].values       # primary binary label

    print(f"   ✅ Feature matrix shape: {X.shape}")

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"\n📦 Split: {X_train.shape[0]:,} train / {X_test.shape[0]:,} test")

    # Save vectorizer
    os.makedirs(save_dir, exist_ok=True)
    vectorizer_path = os.path.join(save_dir, 'tfidf_vectorizer.pkl')
    joblib.dump(vectorizer, vectorizer_path)
    print(f"   💾 Vectorizer saved → {vectorizer_path}")

    return X_train, X_test, y_train, y_test, vectorizer


if __name__ == '__main__':
    # Quick test
    data_path = os.path.join('data', 'train.csv')
    df = load_data(data_path)
    explore_data(df)
    X_train, X_test, y_train, y_test, vec = preprocess_data(df)
    print("\n✅ Preprocessing complete!")
