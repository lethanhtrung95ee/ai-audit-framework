from __future__ import annotations
import os, argparse, logging
import pandas as pd

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.pipeline.column_mapping import ColumnMapping

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger("drift")


def _require_cols(df: pd.DataFrame, path: str, cols: list[str]):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns {missing} in {path}. Have: {list(df.columns)[:8]}")


def load_df(path: str, text_col: str, label_col: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise SystemExit(f"File not found: {path}")
    df = pd.read_csv(path)
    _require_cols(df, path, [text_col, label_col])
    out = df[[text_col, label_col]].copy()
    return out


def coerce_labels_numeric(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    if pd.api.types.is_numeric_dtype(df[label_col]):
        return df
    mapping = {"positive": 1, "negative": 0, "pos": 1, "neg": 0}
    lowered = df[label_col].astype(str).str.lower()
    try:
        df[label_col] = lowered.map(mapping).fillna(lowered).astype(int)
    except Exception:
        # Last resort: try to cast directly (e.g., '0'/'1')
        try:
            df[label_col] = df[label_col].astype(int)
        except Exception as e:
            raise SystemExit(f"Cannot coerce labels in column '{label_col}' to numeric: {e}")
    return df


def build_and_save_report(train_csv: str, test_csv: str, text_col: str, label_col: str, out_html: str):
    os.makedirs(os.path.dirname(out_html), exist_ok=True)

    ref = load_df(train_csv, text_col, label_col)
    cur = load_df(test_csv, text_col, label_col)

    ref = coerce_labels_numeric(ref, label_col)
    cur = coerce_labels_numeric(cur, label_col)

    # Column mapping tells Evidently what each column means
    column_mapping = ColumnMapping(
        target=label_col,
        text_features=[text_col],
        numerical_features=[],
        categorical_features=[],
    )

    report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
    report.run(reference_data=ref, current_data=cur, column_mapping=column_mapping)
    report.save_html(out_html)
    return out_html


def main():
    ap = argparse.ArgumentParser(description="Generate Evidently drift report for train vs test")
    ap.add_argument("--train", required=True, help="Path to train.csv")
    ap.add_argument("--test", required=True, help="Path to test.csv")
    ap.add_argument("--text_col", default="review", help="Text column name (default: review)")
    ap.add_argument("--label_col", default="sentiment", help="Label/target column name (default: sentiment)")
    ap.add_argument("--out", default="reports/drift_report.html", help="Output HTML path")
    args = ap.parse_args()

    path = build_and_save_report(args.train, args.test, args.text_col, args.label_col, args.out)
    LOGGER.info("Saved drift report -> %s", path)


if __name__ == "__main__":
    main()