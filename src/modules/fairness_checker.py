# src/modules/fairness_checker.py
from __future__ import annotations
import os
import argparse
import logging
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger("fairness")


def tag_protected(df: pd.DataFrame, text_col: str, keywords: list[str]) -> pd.Series:
    """Return boolean mask for rows containing protected keywords."""
    pattern = r"|".join([rf"\b{k}\b" for k in keywords])
    return df[text_col].str.lower().str.contains(pattern, regex=True)


def group_metrics(df: pd.DataFrame, label_col: str, pred_col: str, mask: pd.Series) -> dict:
    y_true, y_pred = df[label_col].values, df[pred_col].values
    def m(idx):
        if idx.sum() == 0:
            return {"accuracy": None, "tpr": None}
        return {
            "accuracy": float(accuracy_score(y_true[idx], y_pred[idx])),
            "tpr": float(recall_score(y_true[idx], y_pred[idx], pos_label=1))
        }
    g1 = m(mask)
    g0 = m(~mask)
    return {
        "protected": g1,
        "non_protected": g0,
        "demographic_parity_diff": None if g1["accuracy"] is None or g0["accuracy"] is None else abs(g1["accuracy"] - g0["accuracy"]),
        "equalized_odds_tpr_diff": None if g1["tpr"] is None or g0["tpr"] is None else abs(g1["tpr"] - g0["tpr"])
    }


def main():
    ap = argparse.ArgumentParser(description="Fairness metrics from predictions.csv")
    ap.add_argument("--predictions", required=True, help="Path to predictions.csv")
    ap.add_argument("--out", default="reports/fairness.json")
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--label_col", default="y_true")
    ap.add_argument("--pred_col", default="y_pred")
    ap.add_argument("--keywords", nargs="*", default=["he","she","man","woman","girl","boy","male","female"])
    args = ap.parse_args()

    LOGGER.info("Loading predictions: %s", args.predictions)
    df = pd.read_csv(args.predictions)

    mask = tag_protected(df, args.text_col, args.keywords)
    metrics = group_metrics(df, args.label_col, args.pred_col, mask)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    import json
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    LOGGER.info("Saved fairness metrics -> %s", args.out)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()