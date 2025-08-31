# modules/prepare_data.py
"""
Prepare and split dataset for the audit pipeline.

Produces:
 - data/processed/processed_imdb.csv  (full cleaned dataset)
 - data/processed/train.csv
 - data/processed/test.csv
"""

from __future__ import annotations
import argparse
import os
import re
import logging
from typing import Tuple, Optional
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger("prepare_data")


URL_RE = re.compile(r"https?://\S+")
HTML_RE = re.compile(r"<.*?>")
CONTRACTIONS = {
    "can't": "can not", "won't": "will not", "n't": " not", "'re": " are",
    "'s": " is", "'d": " would", "'ll": " will", "'t": " not", "'ve": " have", "'m": " am"
}


def load_csv(path: str) -> pd.DataFrame:
    LOGGER.info("Loading CSV: %s", path)
    df = pd.read_csv(path)
    LOGGER.info("Loaded %d rows, %d columns", len(df), len(df.columns))
    return df


def clean_text(text: str,
               lowercase: bool = True,
               strip_urls: bool = True,
               strip_html: bool = True,
               decontract: bool = True,
               remove_punct: bool = False) -> str:
    if pd.isna(text):
        return ""
    s = str(text)
    if lowercase:
        s = s.lower()
    if strip_urls:
        s = URL_RE.sub(" ", s)
    if strip_html:
        s = HTML_RE.sub(" ", s)
    if decontract:
        for a, b in CONTRACTIONS.items():
            s = s.replace(a, b)
    if remove_punct:
        s = re.sub(r"[^a-zA-Z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def prepare_and_split(df: pd.DataFrame,
                      text_col: str = "text",
                      label_col: str = "label",
                      train_frac: float = 0.8,
                      seed: int = 42,
                      save_dir: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # validate columns
    missing = [c for c in (text_col, label_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataframe: {missing}")

    LOGGER.info("Cleaning text column '%s' ...", text_col)
    df = df.copy()
    df[text_col] = df[text_col].astype(str).apply(clean_text)

    # drop empty texts
    before = len(df)
    df = df[df[text_col].str.len() > 0].reset_index(drop=True)
    after = len(df)
    LOGGER.info("Dropped %d empty-text rows ( %d -> %d )", before - after, before, after)

    # normalize labels: if labels are strings like 'positive'/'negative', map to 1/0
    if df[label_col].dtype == object:
        uniq = df[label_col].str.lower().unique().tolist()
        LOGGER.info("Detected string labels: %s", uniq[:10])
        # try automatic mapping common cases
        mapping = {}
        if all(x in ["positive", "negative"] for x in [u.lower() for u in uniq]):
            mapping = {"positive": 1, "negative": 0}
        elif all(x in ["pos", "neg"] for x in [u.lower() for u in uniq]):
            mapping = {"pos": 1, "neg": 0}
        elif all(u.isdigit() for u in uniq):
            mapping = {}
        if mapping:
            df[label_col] = df[label_col].str.lower().map(mapping)
            LOGGER.info("Applied label mapping: %s", mapping)
        else:
            # try cast to int if possible
            try:
                df[label_col] = df[label_col].astype(int)
            except Exception:
                LOGGER.warning("Could not auto-convert labels to integers; leaving as-is.")

    # split
    LOGGER.info("Splitting into train/test (train_frac=%s, seed=%s)", train_frac, seed)
    train_df, test_df = train_test_split(df, train_size=train_frac, random_state=seed, stratify=df[label_col] if df[label_col].nunique() > 1 else None)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # save
    # optionally save — disabled for now
    if save_dir:
        LOGGER.info("Skipping save, returning DataFrames directly")

    return train_df, test_df


def main():
    ap = argparse.ArgumentParser(description="Prepare IMDB dataset: clean and split.")
    ap.add_argument("--input", required=True, help="Path to raw CSV (e.g., data/IMDB_Dataset.csv)")
    ap.add_argument("--out_dir", default="data/processed", help="Directory to write processed files")
    ap.add_argument("--text_col", default="text", help="Name of the text column in CSV (default: text)")
    ap.add_argument("--label_col", default="label", help="Name of the label column in CSV (default: label)")
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = load_csv(args.input)
    try:
        train_df, test_df = prepare_and_split(df, text_col=args.text_col, label_col=args.label_col,
                                              train_frac=args.train_frac, seed=args.seed, save_dir=args.out_dir)
        LOGGER.info("Done. Train rows: %d, Test rows: %d", len(train_df), len(test_df))
    except Exception as e:
        LOGGER.error("Failed to prepare dataset: %s", e)
        raise


if __name__ == "__main__":
    main()
