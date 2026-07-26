"""
verbal_classifier.py

Implements the "lightweight classifier for verbal question sub-typing"
from the proposal's technology stack table (section 5.4 / 6). This is a
genuinely trained ML model: TF-IDF text features + Logistic Regression,
trained on verbal_dataset.csv, to classify a raw question string into one
of: blood_relation, direction_sense, coding_decoding, syllogism.

This determines which solver in verbal_solver.py gets invoked.
"""

import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

_vectorizer = None
_classifier = None
_DEFAULT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verbal_dataset.csv")


def train_classifier(csv_path: str = None, test_size: float = 0.2):
    """
    Trains the sub-type classifier from the synthetic verbal dataset and
    caches it in module-level globals (so app.py only pays the training
    cost once per session, not on every rerun).
    Returns the held-out test accuracy so you can report it in your PPT.
    """
    global _vectorizer, _classifier

    csv_path = csv_path or _DEFAULT_CSV
    df = pd.read_csv(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(
        df["question"], df["subtype"], test_size=test_size, random_state=42, stratify=df["subtype"]
    )

    _vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
    X_train_vec = _vectorizer.fit_transform(X_train)
    X_test_vec = _vectorizer.transform(X_test)

    _classifier = LogisticRegression(max_iter=1000)
    _classifier.fit(X_train_vec, y_train)

    preds = _classifier.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    return acc


def get_classifier():
    """Lazy-trains on first use so app.py doesn't need to call train_classifier() explicitly."""
    if _classifier is None or _vectorizer is None:
        train_classifier()
    return _vectorizer, _classifier


def classify_subtype(question_text: str) -> str:
    """Returns one of: blood_relation, direction_sense, coding_decoding, syllogism."""
    vectorizer, classifier = get_classifier()
    vec = vectorizer.transform([question_text])
    return classifier.predict(vec)[0]


if __name__ == "__main__":
    acc = train_classifier()
    print(f"Held-out test accuracy: {acc:.4f}")

    # Quick sanity checks with fresh, hand-written questions (not from the dataset)
    tests = [
        "Rahul is Priya's father. Priya is Ankit's mother. How is Rahul related to Ankit?",
        "Sonia walks 4km north, then 6km east. How far is she from the start?",
        "If DOG is coded as EPH, how is CAT coded?",
        "All roses are flowers. All flowers are plants. Conclusion: All roses are plants. Is it valid?",
    ]
    for t in tests:
        print(classify_subtype(t), "<-", t[:60])
