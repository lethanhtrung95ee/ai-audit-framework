# src/modules/explainability.py
from __future__ import annotations
import os
import argparse
import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# SHAP (global)
import shap
# LIME (local)
from lime.lime_text import LimeTextExplainer

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger("explainability")


def ensure_out(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)


def shap_summary_for_tfidf(pipe, train_texts: pd.Series, out_path: str, max_samples: int = 1000) -> str:
    """Create a global SHAP summary for TF-IDF + linear classifier pipelines.
    Works best with LogisticRegression/LinearSVC.
    """
    vec = pipe.named_steps.get("tfidf")
    clf = pipe.named_steps.get("clf")
    sample = train_texts.sample(min(max_samples, len(train_texts)), random_state=42)
    LOGGER.info("Vectorizing %d texts for SHAP background ...", len(sample))
    X = vec.transform(sample)

    LOGGER.info("Building LinearExplainer ...")
    explainer = shap.LinearExplainer(clf, X)
    shap_values = explainer.shap_values(X)

    LOGGER.info("Rendering SHAP summary plot ...")
    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=vec.get_feature_names_out(), show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def lime_explain_text(pipe, text: str, out_path: str, class_names=("neg", "pos")) -> str:
    """Create a local LIME explanation image for a single input text."""
    if not hasattr(pipe, "predict_proba"):
        raise RuntimeError("LIME needs a classifier with predict_proba().")
    explainer = LimeTextExplainer(class_names=list(class_names))
    exp = explainer.explain_instance(text, pipe.predict_proba, num_features=10)
    fig = exp.as_pyplot_figure()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Generate SHAP & LIME explanations for a TF-IDF model")
    ap.add_argument("--model", required=True, help="Path to models/trained_model.pkl")
    ap.add_argument("--train", required=True, help="Path to train.csv for SHAP background")
    ap.add_argument("--test", required=True, help="Path to test.csv for a LIME example")
    ap.add_argument("--text_col", default="review")
    ap.add_argument("--label_col", default="sentiment")
    ap.add_argument("--out_dir", default="reports")
    args = ap.parse_args()

    ensure_out(args.out_dir)

    LOGGER.info("Loading model: %s", args.model)
    pipe = joblib.load(args.model)

    LOGGER.info("Loading train/test data ...")
    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    # SHAP (global)
    shap_path = os.path.join(args.out_dir, "shap_summary.png")
    try:
        shap_out = shap_summary_for_tfidf(pipe, train_df[args.text_col], shap_path)
        LOGGER.info("Saved SHAP summary -> %s", shap_out)
    except Exception as e:
        LOGGER.error("SHAP failed: %s", e)

    # LIME (local) on the first test example
    lime_path = os.path.join(args.out_dir, "lime_example.png")
    try:
        example_text = str(test_df[args.text_col].iloc[0])
        lime_out = lime_explain_text(pipe, example_text, lime_path)
        LOGGER.info("Saved LIME example -> %s", lime_out)
    except Exception as e:
        LOGGER.error("LIME failed: %s", e)


if __name__ == "__main__":
    main()