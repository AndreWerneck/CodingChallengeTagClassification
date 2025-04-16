# evaluate.py
import pandas as pd
import pickle
from sklearn.metrics import classification_report, hamming_loss
from typing import Tuple, Dict, Any
from sklearn.preprocessing import MultiLabelBinarizer
from utils import preprocess_text_fields
import json
import os
from config import FOCUS_TAGS

def evaluate_model(
    test_csv_path: str,
    model_path: str,
    vectorizer_path: str,
    binarizer_path: str,
    save_reports: bool = True,
    report_dir: str = "test_reports"
) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
    print("Loading test set and model...")
    test_df = pd.read_csv(test_csv_path)

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    with open(binarizer_path, 'rb') as f:
        mlb = pickle.load(f)

    print("Preprocessing test input...")
    
    test_df["new_tags"] = test_df["new_tags"].apply(eval)
    
    X_test = vectorizer.transform(test_df["model_input"])
    y_test = mlb.transform(test_df["new_tags"])

    print("Making predictions...")
    y_pred = model.predict(X_test)

    print("\n Classification Report — All Tags:")
    full_report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    print(classification_report(y_test, y_pred, zero_division=0, target_names=mlb.classes_))
    hamming = hamming_loss(y_test, y_pred)
    print(f"\n Hamming Loss (All Tags): {hamming:.4f}")

    # Focus tag evaluation
    focus_indices = [i for i, tag in enumerate(mlb.classes_) if tag in FOCUS_TAGS]
    if not focus_indices:
        print(" No focus tags found in binarizer labels.")
        return full_report, {}, hamming

    y_test_focus = y_test[:, focus_indices]
    y_pred_focus = y_pred[:, focus_indices]

    print("\n Classification Report — Focus Tags Only:")
    focus_report = classification_report(y_test_focus, y_pred_focus, target_names=FOCUS_TAGS, zero_division=0, output_dict=True)
    hamming_focus = hamming_loss(y_test_focus, y_pred_focus)
    print(classification_report(y_test_focus, y_pred_focus, target_names=FOCUS_TAGS, zero_division=0))
    print(f"Hamming Loss: {hamming_focus:.4f}")

    if save_reports:
        print("Saving reports to disk...")
        os.makedirs(report_dir, exist_ok=True)
        with open(f"{report_dir}/full_report.json", "w") as f:
            json.dump(full_report, f, indent=2)
        with open(f"{report_dir}/focus_report.json", "w") as f:
            json.dump(focus_report, f, indent=2)
        with open(f"{report_dir}/hamming_loss.txt", "w") as f:
            f.write(f"Hamming Loss: {hamming:.4f}\n")
            f.write(f"Hamming Loss (Focus Tags): {hamming_focus:.4f}\n")        

    return full_report, focus_report, hamming, hamming_focus

if __name__ == '__main__':
    evaluate_model(
        test_csv_path="data/test_data.csv",
        model_path="models/model.pkl",
        vectorizer_path="models/vectorizer.pkl",
        binarizer_path="models/binarizer.pkl"
    )
