# src/modules/similarity_detector.py
from __future__ import annotations
import os, argparse, logging
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger("similarity_detector")

def main():
    ap = argparse.ArgumentParser(description="Detect near-duplicate or similar texts")
    ap.add_argument("--input", required=True, help="Path to CSV file (train/test)")
    ap.add_argument("--text_col", default="review")
    ap.add_argument("--sample", type=int, default=5000, help="Limit rows for faster runtime")
    ap.add_argument("--threshold", type=float, default=0.9, help="Cosine similarity cutoff for duplicates")
    ap.add_argument("--out_csv", default="reports/similarity_pairs.csv")
    args = ap.parse_args()

    LOGGER.info("Loading dataset: %s", args.input)
    df = pd.read_csv(args.input)
    texts = df[args.text_col].astype(str).tolist()
    if args.sample:
        texts = texts[:args.sample]

    LOGGER.info("Encoding %d texts with Sentence-BERT ...", len(texts))
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda" if torch.cuda.is_available() else "cpu")
    embeddings = model.encode(texts, batch_size=64, convert_to_tensor=True, show_progress_bar=True)

    LOGGER.info("Computing cosine similarities ...")
    sim_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy()

    pairs = []
    n = len(texts)
    for i in range(n):
        for j in range(i+1, n):
            if sim_matrix[i, j] >= args.threshold:
                pairs.append((i, j, sim_matrix[i, j], texts[i][:200], texts[j][:200]))

    out_df = pd.DataFrame(pairs, columns=["idx1", "idx2", "similarity", "text1", "text2"])
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)

    LOGGER.info("Saved %d near-duplicate pairs -> %s", len(out_df), args.out_csv)

if __name__ == "__main__":
    main()