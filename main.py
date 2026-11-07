"""
Yelp sentiment training pipeline — single-file version.

What this file does:
1. Configuration
2. Read Yelp JSONL data
3. Clean and deduplicate reviews
4. Optional English-language filtering
5. Stratified train/validation/test split
6. Word + character TF-IDF features
7. Logistic Regression hyperparameter selection on validation macro-F1
8. Final testing and evaluation metrics
9. Save the model bundle for app.py
10. Save reports, confusion matrix, and a ZIP of outputs

Expected input: review_300k.json (one JSON object per line), with "text" and "stars".
Run in Colab after uploading the JSONL file:
    !python train_all_in_one.py
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

# ============================ CONFIG ============================

REVIEW_PATH = Path("/content/review_300k.json")
ARTIFACTS_DIR = Path("/content/artifacts")
OUTPUTS_DIR = Path("/content/outputs")
OUTPUTS_ZIP = Path("/content/outputs.zip")

RANDOM_SEED = 42
MAX_ROWS = 300_000          # Set to None to read all rows in the file
MIN_WORDS = 10
ENGLISH_ONLY = True         # Uses langdetect if installed; otherwise skips language filtering
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15      # Fraction of the full cleaned dataset
LR_C_GRID = [0.1, 1.0, 10.0]

WORD_MAX_FEATURES = 100_000
CHAR_MAX_FEATURES = 150_000
WORD_MIN_DF = 3
CHAR_MIN_DF = 3

LABEL_NAMES = ["negative", "neutral", "positive"]
LABEL_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}

# Set to True to install missing packages automatically in Colab/runtime.
AUTO_INSTALL = True

# ======================== DEPENDENCIES =========================

def ensure_dependencies():
    packages = {
        "numpy": "numpy",
        "pandas": "pandas",
        "sklearn": "scikit-learn",
        "joblib": "joblib",
        "matplotlib": "matplotlib",
    }
    missing = []
    for module, package in packages.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)

    if ENGLISH_ONLY:
        try:
            import langdetect  # noqa: F401
        except ImportError:
            # Language filtering is optional if the package is unavailable.
            if AUTO_INSTALL:
                missing.append("langdetect")

    if missing and AUTO_INSTALL:
        print("Installing missing packages:", ", ".join(sorted(set(missing))))
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *sorted(set(missing))])

    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import sklearn  # noqa: F401
    import joblib  # noqa: F401
    import matplotlib  # noqa: F401


ensure_dependencies()

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = RANDOM_SEED
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

# ============================ SEED ==============================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ====================== PREPROCESSING ===========================

def clean_text(value) -> str:
    """Conservative text cleaning: normalize whitespace and remove empty text."""
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().split())


def stars_to_label(stars):
    """Map Yelp star ratings to 3 sentiment classes."""
    try:
        stars = float(stars)
    except (TypeError, ValueError):
        return None
    if stars <= 2:
        return 0  # negative
    if stars == 3:
        return 1  # neutral
    if stars >= 4:
        return 2  # positive
    return None


def is_english(text: str) -> bool:
    if not LANGDETECT_AVAILABLE:
        return True
    try:
        return detect(text) == "en"
    except Exception:
        # Very short or ambiguous text: don't discard solely due to detector error.
        return True


def load_and_preprocess_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Upload review_300k.json to Colab's /content directory, or change REVIEW_PATH."
        )

    rows = []
    print(f"Reading dataset: {path}")
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            if MAX_ROWS is not None and line_no > MAX_ROWS:
                break
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = clean_text(item.get("text", item.get("review", "")))
            label = stars_to_label(item.get("stars", item.get("rating")))
            if not text or label is None:
                continue
            if len(text.split()) < MIN_WORDS:
                continue
            rows.append((text, label))

    df = pd.DataFrame(rows, columns=["text", "label"])
    if df.empty:
        raise ValueError("No usable reviews found. Check that JSONL rows contain 'text' and 'stars'.")

    before = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"Removed {before - len(df):,} duplicate reviews.")

    if ENGLISH_ONLY:
        if LANGDETECT_AVAILABLE:
            print("Filtering for English reviews (langdetect)...")
            keep = df["text"].map(is_english)
            df = df.loc[keep].reset_index(drop=True)
        else:
            print("langdetect is unavailable; English-only filtering skipped.")

    if df.empty:
        raise ValueError("No reviews remain after preprocessing.")

    df["label_name"] = df["label"].map({0: "negative", 1: "neutral", 2: "positive"})
    print(f"Usable reviews: {len(df):,}")
    print("Class distribution:")
    print(df["label_name"].value_counts().reindex(LABEL_NAMES, fill_value=0).to_string())
    return df

# ======================== DATA SPLIT ============================

def split_data(df: pd.DataFrame):
    X = df["text"].to_numpy()
    y = df["label"].to_numpy()

    # First hold out test set, then take validation from the remaining portion.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    val_fraction_of_trainval = VALIDATION_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_fraction_of_trainval,
        random_state=RANDOM_SEED,
        stratify=y_trainval,
    )
    print(f"Split sizes — train: {len(X_train):,}, validation: {len(X_val):,}, test: {len(X_test):,}")
    return X_train, X_val, X_test, y_train, y_val, y_test

# ======================== MODEL BUILDING ========================

def build_vectorizer() -> FeatureUnion:
    return FeatureUnion([
        ("word_tfidf", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=WORD_MIN_DF,
            max_features=WORD_MAX_FEATURES,
            sublinear_tf=True,
            strip_accents="unicode",
            dtype=np.float32,
        )),
        ("char_tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=CHAR_MIN_DF,
            max_features=CHAR_MAX_FEATURES,
            sublinear_tf=True,
            strip_accents="unicode",
            dtype=np.float32,
        )),
    ])


def train_and_select(X_train, y_train, X_val, y_val):
    print("\nFitting TF-IDF vectorizer on training data...")
    vectorizer = build_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    print(f"TF-IDF matrix shape: train={X_train_vec.shape}, validation={X_val_vec.shape}")

    best_model = None
    best_c = None
    best_score = -1.0
    candidate_results = []

    for c_value in LR_C_GRID:
        print(f"\nTraining Logistic Regression with C={c_value}...")
        model = LogisticRegression(
            C=c_value,
            max_iter=1000,
            solver="liblinear",
            class_weight=None,
            random_state=RANDOM_SEED,
        )
        model.fit(X_train_vec, y_train)
        val_predictions = model.predict(X_val_vec)
        score = f1_score(y_val, val_predictions, average="macro", zero_division=0)
        candidate_results.append({"C": c_value, "validation_macro_f1": score})
        print(f"Validation macro-F1: {score:.4f}")

        if score > best_score:
            best_score = score
            best_c = c_value
            best_model = model

    print(f"\nSelected C={best_c} (validation macro-F1={best_score:.4f})")
    return vectorizer, best_model, best_c, candidate_results

# ============================ EVALUATION =========================

def evaluate_model(model, vectorizer, X_test, y_test):
    print("\nTransforming test set...")
    X_test_vec = vectorizer.transform(X_test)
    predictions = model.predict(X_test_vec)

    accuracy = accuracy_score(y_test, predictions)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="weighted", zero_division=0
    )

    metrics = {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
    }

    print("\n================ TEST METRICS ================")
    for key, value in metrics.items():
        print(f"{key:20s}: {value:.4f}")
    print("\nClassification report:")
    print(classification_report(
        y_test, predictions, labels=[0, 1, 2], target_names=LABEL_NAMES, zero_division=0
    ))

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    report_dict = classification_report(
        y_test, predictions, labels=[0, 1, 2], target_names=LABEL_NAMES,
        output_dict=True, zero_division=0
    )
    pd.DataFrame(report_dict).transpose().to_csv(OUTPUTS_DIR / "classification_report.csv")
    pd.DataFrame([metrics]).to_csv(OUTPUTS_DIR / "metrics.csv", index=False)

    cm = confusion_matrix(y_test, predictions, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Yelp Sentiment — Confusion Matrix")
    fig.colorbar(image, ax=ax)
    ax.set_xticks(range(3), LABEL_NAMES, rotation=25, ha="right")
    ax.set_yticks(range(3), LABEL_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    pd.DataFrame({
        "text": X_test,
        "true_label": [LABEL_NAMES[int(x)] for x in y_test],
        "predicted_label": [LABEL_NAMES[int(x)] for x in predictions],
    }).to_csv(OUTPUTS_DIR / "test_predictions.csv", index=False)

    return metrics

# ========================= SAVE ARTIFACTS ========================

def save_model(vectorizer, model, best_c, candidate_results, metrics):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    # app.py should load this exact bundle: {"vec": vectorizer, "model": model}
    joblib.dump({"vec": vectorizer, "model": model}, ARTIFACTS_DIR / "logreg.joblib", compress=3)

    metadata = {
        "model_type": "TF-IDF + Logistic Regression",
        "labels": LABEL_NAMES,
        "label_to_id": LABEL_TO_ID,
        "best_C": best_c,
        "validation_candidates": candidate_results,
        "test_metrics": metrics,
        "preprocessing": {
            "minimum_words": MIN_WORDS,
            "english_only_requested": ENGLISH_ONLY,
            "langdetect_available": LANGDETECT_AVAILABLE,
            "duplicate_texts_removed": True,
        },
        "vectorizer": {
            "word_ngram_range": [1, 2],
            "word_max_features": WORD_MAX_FEATURES,
            "char_ngram_range": [3, 5],
            "char_max_features": CHAR_MAX_FEATURES,
        },
    }
    with (ARTIFACTS_DIR / "model.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(f"\nSaved model: {ARTIFACTS_DIR / 'logreg.joblib'}")
    print(f"Saved metadata: {ARTIFACTS_DIR / 'model.json'}")


def package_outputs():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    if ARTIFACTS_DIR.exists():
        shutil.copytree(ARTIFACTS_DIR, OUTPUTS_DIR / "artifacts", dirs_exist_ok=True)
    if OUTPUTS_ZIP.exists():
        OUTPUTS_ZIP.unlink()
    with zipfile.ZipFile(OUTPUTS_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in OUTPUTS_DIR.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(OUTPUTS_DIR.parent))
    print(f"Outputs saved in: {OUTPUTS_DIR}")
    print(f"Outputs archive created at: {OUTPUTS_ZIP}")

# ============================== MAIN =============================

def main():
    print("Yelp Sentiment — TF-IDF + Logistic Regression")
    print(f"Random seed: {RANDOM_SEED}")
    if not LANGDETECT_AVAILABLE and ENGLISH_ONLY:
        print("NOTE: langdetect not available, so language filtering will be skipped.")
    df = load_and_preprocess_data(REVIEW_PATH)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    vectorizer, model, best_c, candidates = train_and_select(
        X_train, y_train, X_val, y_val
    )
    metrics = evaluate_model(model, vectorizer, X_test, y_test)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(candidates).to_csv(OUTPUTS_DIR / "validation_model_selection.csv", index=False)
    save_model(vectorizer, model, best_c, candidates, metrics)
    package_outputs()

    
if __name__ == "__main__":
  main()
