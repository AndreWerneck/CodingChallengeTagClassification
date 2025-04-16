# train.py
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import classification_report, hamming_loss
from itertools import product
from tqdm import tqdm
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

from utils import preprocess_text_fields
from config import RANDOM_SEED

def run_cross_validation(texts:pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Performs grid search cross-validation for TF-IDF + Logistic Regression."""
    tfidf_configs = [
        {"max_features": 20000, "min_df": 1, "ngram_range": (1, 1)},
        {"max_features": 20000, "min_df": 3, "ngram_range": (1, 2)},
        {"max_features": 20000, "min_df": 3, "ngram_range": (1, 3)},
        {"max_features": 10000, "min_df": 1, "ngram_range": (1, 1)},
        {"max_features": 10000, "min_df": 3, "ngram_range": (1, 2)},
        {"max_features": 10000, "min_df": 3, "ngram_range": (1, 3)},
        {"max_features": 50000, "min_df": 1, "ngram_range": (1, 1)},
        {"max_features": 50000, "min_df": 3, "ngram_range": (1, 2)},
    ]

    lr_param_grid = [
        {"C": 1.0, "class_weight": "balanced", "solver": "liblinear", "max_iter": 1000},
        {"C": 1.0, "class_weight": None, "solver": "liblinear", "max_iter": 1000},
        {"C": 1.0, "class_weight": "balanced", "solver": "liblinear", "max_iter": 1000, "penalty": "l1"},
        {"C": 0.5, "class_weight": "balanced", "solver": "liblinear", "max_iter": 1000, "penalty": "l1"},
        {"C": 1.0, "class_weight": "balanced", "max_iter": 1000},
        {"C": 1.0, "class_weight": None, "max_iter": 1000},
        {"C": 0.5, "class_weight": "balanced", "max_iter": 1000},
    ]

    kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    results = []

    print("\U0001F501 Starting grid search over TF-IDF + LR params...\n")

    for tfidf_params, lr_params in tqdm(list(product(tfidf_configs, lr_param_grid)), desc="Grid Search"):
        tfidf = TfidfVectorizer(**tfidf_params)
        X = tfidf.fit_transform(texts)

        f1s, hammings = [], []

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = OneVsRestClassifier(LogisticRegression(**lr_params))
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
            f1s.append(report["weighted avg"]["f1-score"])
            hammings.append(hamming_loss(y_val, y_pred))

        results.append({
            "tfidf_params": tfidf_params,
            "lr_params": lr_params,
            "f1_mean": np.mean(f1s),
            "f1_std": np.std(f1s),
            "hamming_mean": np.mean(hammings),
            "hamming_std": np.std(hammings),
        })

    results_df = pd.DataFrame(results).sort_values(by="f1_mean", ascending=False)
    print("\n\U0001F3C6 Best Config:\n", results_df.iloc[0])
    return results_df.iloc[0]

def train_model(
    csv_path: str,
    model_path: str,
    vectorizer_path: str,
    binarizer_path: str,
    do_cv: bool = False
) -> None:
    print("Loading dataset...")
    df = pd.read_csv(csv_path)

    print("Preprocessing...")
    df["model_input"] = df.apply(preprocess_text_fields,axis=1)
    df["new_tags"] = df["new_tags"].apply(eval)

    tag_set = set(tag for tags in df["new_tags"] for tag in tags)
    print(f"Total unique tags: {len(tag_set)}")

    X = df["model_input"]
    y_raw = df["new_tags"]

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
    
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
    # Save test set for later evaluation
    print("Saving test set to disk...")
    test_df = pd.DataFrame({
        "model_input": X_test,
        "new_tags": mlb.inverse_transform(y_test)  # Converts back to raw tag lists
    })
    test_df["new_tags"] = test_df["new_tags"].apply(lambda tags: list(tags))  # Ensure JSON serializable
    os.makedirs("data", exist_ok=True)
    test_df.to_csv("data/test_data.csv")

    # Train with CV or default config
    if do_cv:
        best = run_cross_validation(X_train, y_train)
        tfidf = TfidfVectorizer(**best["tfidf_params"])
        model = OneVsRestClassifier(LogisticRegression(**best["lr_params"]))
    else:
        tfidf = TfidfVectorizer(max_features=20000, min_df=3, ngram_range=(1,3))
        model = OneVsRestClassifier(LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear", max_iter=1000))

    X_train_tfidf = tfidf.fit_transform(X_train)

    print("Final training on full training split...")
    model.fit(X_train_tfidf, y_train)

    print("Saving model, vectorizer, and binarizer...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    os.makedirs(os.path.dirname(vectorizer_path), exist_ok=True)
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(tfidf, f)
    os.makedirs(os.path.dirname(binarizer_path), exist_ok=True)
    with open(binarizer_path, 'wb') as f:
        pickle.dump(mlb, f)

    print("Training complete.")

if __name__ == '__main__':
    train_model(
        csv_path="data/all_data.csv",
        model_path="models/model.pkl",
        vectorizer_path="models/vectorizer.pkl",
        binarizer_path="models/binarizer.pkl",
        do_cv=False
    )
