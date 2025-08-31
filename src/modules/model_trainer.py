"""
Train and evaluate a sentiment classifier using TF-IDF + Logistic Regression.

Artifacts produced (by default):
 - models/trained_model.pkl       (scikit-learn Pipeline)
 - reports/metrics.json           (accuracy, f1, report)
 - reports/predictions.csv        (text, y_true, y_pred, y_prob)

Usage (PowerShell, one line):
python src/models/model_trainer.py --train data/processed/train.csv --test data/processed/test.csv --text_col review --label_col sentiment

Or, if you don't have saved splits, point to raw and it will split for you:
python src/models/model_trainer.py --input data/raw/IMDB_Dataset.csv --text_col review --label_col sentiment --train_frac 0.8
"""
from __future__ import annotations
import os
import json
import argparse
import logging
import pandas as pd
from typing import Tuple, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger("model_trainer")


def build_pipeline(max_features: int = 30000) -> Pipeline:
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    clf = LogisticRegression(max_iter=300, n_jobs=None)
    pipe = Pipeline([("tfidf", vec), ("clf", clf)])
    return pipe


def train_and_eval(train_df: pd.DataFrame, test_df: pd.DataFrame, text_col: str, label_col: str, out_models: str, out_reports: str) -> Tuple[Pipeline, dict, pd.DataFrame]:
    os.makedirs(out_models, exist_ok=True)
    os.makedirs(out_reports, exist_ok=True)

    pipe = build_pipeline()
    LOGGER.info("Training model on %d rows ...", len(train_df))
    pipe.fit(train_df[text_col], train_df[label_col])

    LOGGER.info("Evaluating on %d rows ...", len(test_df))
    y_true = test_df[label_col].values
    # predict_proba might not exist for some models; LR has it
    try:
        proba = pipe.predict_proba(test_df[text_col])[:, 1]
    except Exception:
        proba = None
    preds = pipe.predict(test_df[text_col])

    metrics = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "f1": float(f1_score(y_true, preds, average="weighted")),
        "report": classification_report(y_true, preds, output_dict=True),
    }

    # Save artifacts
    model_path = os.path.join(out_models, "trained_model.pkl")
    joblib.dump(pipe, model_path)
    LOGGER.info("Saved model -> %s", model_path)

    metrics_path = os.path.join(out_reports, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    LOGGER.info("Saved metrics -> %s", metrics_path)

    pred_df = test_df[[text_col, label_col]].copy()
    pred_df.rename(columns={text_col: "text", label_col: "y_true"}, inplace=True)
    pred_df["y_pred"] = preds
    if proba is not None:
        pred_df["y_prob"] = proba
    preds_path = os.path.join(out_reports, "predictions.csv")
    pred_df.to_csv(preds_path, index=False)
    LOGGER.info("Saved predictions -> %s", preds_path)

    return pipe, metrics, pred_df


def main():
    ap = argparse.ArgumentParser(description="Train TF-IDF + LogisticRegression model")
    # Paths
    ap.add_argument("--train", help="Path to train.csv (optional if --input provided)")
    ap.add_argument("--test", help="Path to test.csv (optional if --input provided)")
    ap.add_argument("--input", help="Raw dataset CSV; if provided, will split inside")
    ap.add_argument("--out_models", default="models", help="Directory to save model artifacts")
    ap.add_argument("--out_reports", default="reports", help="Directory to save reports/predictions")

    # Data columns & split
    ap.add_argument("--text_col", default="review")
    ap.add_argument("--label_col", default="sentiment")
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    # Load data
    if args.train and args.test:
        LOGGER.info("Loading provided train/test CSVs")
        train_df = pd.read_csv(args.train)
        test_df = pd.read_csv(args.test)
    elif args.input:
        from src.data.prepare_data import load_csv, prepare_and_split
        LOGGER.info("No explicit train/test provided. Splitting from raw input: %s", args.input)
        df = load_csv(args.input)
        train_df, test_df = prepare_and_split(
            df, text_col=args.text_col, label_col=args.label_col,
            train_frac=args.train_frac, seed=args.seed, save_dir=None
        )
    elif args.input:
        # lazy import only if we actually need to split from raw
        from src.data.prepare_data import load_csv, prepare_and_split
        LOGGER.info("No explicit train/test provided. Splitting from raw input: %s", args.input)
        df = load_csv(args.input)
        train_df, test_df = prepare_and_split(
            df, text_col=args.text_col, label_col=args.label_col,
            train_frac=args.train_frac, seed=args.seed, save_dir=None
        )
    else:
        raise SystemExit("Provide --train and --test, or provide --input to auto-split.")

    # Ensure label is numeric
    if train_df[args.label_col].dtype == object:
        mapping = {"positive": 1, "negative": 0, "pos": 1, "neg": 0}
        train_df[args.label_col] = train_df[args.label_col].str.lower().map(mapping).fillna(train_df[args.label_col]).astype(int)
        test_df[args.label_col] = test_df[args.label_col].str.lower().map(mapping).fillna(test_df[args.label_col]).astype(int)

    # Train & evaluate
    train_and_eval(train_df, test_df, args.text_col, args.label_col, args.out_models, args.out_reports)


if __name__ == "__main__":
    main()