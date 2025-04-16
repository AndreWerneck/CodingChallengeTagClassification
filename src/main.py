# main.py
import argparse
from train import train_model
from evaluate import evaluate_model

def main():
    parser = argparse.ArgumentParser(description="Code Tag Classification CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # === Train parser ===
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--csv_path", type=str, required=True, help="Path to the whole CSV dataset")
    train_parser.add_argument("--model_path", type=str, default="models/model.pkl")
    train_parser.add_argument("--vectorizer_path", type=str, default="models/vectorizer.pkl")
    train_parser.add_argument("--binarizer_path", type=str, default="models/binarizer.pkl")
    train_parser.add_argument("--cv", action="store_true", help="Use cross-validation for hyperparameter tuning")

    # === Evaluate parser ===
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the model")
    eval_parser.add_argument("--test_csv", type=str, required=True, help="Path to test CSV dataset")
    eval_parser.add_argument("--model_path", type=str, default="models/model.pkl")
    eval_parser.add_argument("--vectorizer_path", type=str, default="models/vectorizer.pkl")
    eval_parser.add_argument("--binarizer_path", type=str, default="models/binarizer.pkl")
    eval_parser.add_argument("--save_reports", action="store_true", help="Save evaluation reports to disk")
    eval_parser.add_argument("--report_dir", type=str, default="reports", help="Directory to save reports")

    args = parser.parse_args()

    if args.command == "train":
        train_model(
            csv_path=args.csv_path,
            model_path=args.model_path,
            vectorizer_path=args.vectorizer_path,
            binarizer_path=args.binarizer_path,
            do_cv=args.cv
        )

    elif args.command == "evaluate":
        evaluate_model(
            test_csv_path=args.test_csv,
            model_path=args.model_path,
            vectorizer_path=args.vectorizer_path,
            binarizer_path=args.binarizer_path,
            save_reports=args.save_reports,
            report_dir=args.report_dir
        )

if __name__ == "__main__":
    main()
