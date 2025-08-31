# src/modules/bias_detector.py
from __future__ import annotations
import os, argparse, logging, json
import pandas as pd
import torch
from transformers import pipeline

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger("bias_detector")


def device_id() -> int:
    dev = 0 if torch.cuda.is_available() else -1
    if dev == 0:
        LOGGER.info("Using GPU: %s (CUDA %s, torch %s)", torch.cuda.get_device_name(0), torch.version.cuda, torch.__version__)
    else:
        LOGGER.info("CUDA not available; using CPU (torch %s)", torch.__version__)
    return dev


def load_texts(path: str, text_col: str, sample: int | None) -> list[str]:
    df = pd.read_csv(path)
    if text_col not in df.columns:
        raise SystemExit(f"Column '{text_col}' not in {path} (cols={list(df.columns)[:8]}...)")
    s = df[text_col].astype(str)
    if sample:
        s = s.head(sample)
    return s.tolist()


def score_toxicity(texts: list[str], model_name: str, device: int, batch_size: int = 16) -> list[float]:
    clf = pipeline("text-classification", model=model_name, device=device, truncation=True, top_k=None, batch_size=batch_size)
    results = clf(texts)
    scores: list[float] = []
    for r in results:
        # r is list[{'label': 'toxic'|'non-toxic'|..., 'score': float}, ...]
        toxic = 0.0
        for item in r:
            if "toxic" in item["label"].lower():
                toxic = float(item["score"]); break
        scores.append(toxic)
    return scores


def identity_bias_scores(texts: list[str], models: list[str], device: int, batch_size: int = 16) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for m in models:
        LOGGER.info("Scoring identity model: %s", m)
        clf = pipeline("text-classification", model=m, device=device, truncation=True, top_k=None, batch_size=batch_size)
        res = clf(texts)
        vals = [float(max(item, key=lambda x: x['score'])['score']) for item in res]
        out[m.split('/')[-1] + "_score"] = vals
    return out


def main():
    ap = argparse.ArgumentParser(description="Bias & toxicity scoring on text samples")
    ap.add_argument("--input", required=True, help="Path to CSV (e.g., data/processed/train.csv)")
    ap.add_argument("--text_col", default="review")
    ap.add_argument("--sample", type=int, default=1500, help="Max rows to score (cap runtime)")
    ap.add_argument("--tox_model", default="unitary/toxic-bert")
    ap.add_argument("--id_models", nargs="*", default=["Hate-speech-CNERG/dehatebert-mono-english"])
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--out_csv", default="reports/bias_toxicity_sample.csv")
    ap.add_argument("--out_json", default="reports/bias_summary.json")
    args = ap.parse_args()

    dev = device_id()
    texts = load_texts(args.input, args.text_col, args.sample)
    LOGGER.info("Loaded %d rows to score", len(texts))

    LOGGER.info("Scoring toxicity with %s ...", args.tox_model)
    tox_scores = score_toxicity(texts, args.tox_model, device=dev, batch_size=args.batch_size)

    id_out = {}
    if args.id_models:
        id_out = identity_bias_scores(texts, args.id_models, device=dev, batch_size=args.batch_size)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df = pd.DataFrame({"text": texts, "toxicity": tox_scores, **id_out})
    df.to_csv(args.out_csv, index=False)

    summary = {
        "n_scored": len(texts),
        "toxicity_mean": float(df["toxicity"].mean()),
        "toxicity_p95": float(df["toxicity"].quantile(0.95)),
        "max_identity_scores": {k: float(df[k].max()) for k in id_out.keys()}
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    LOGGER.info("Saved sample scores -> %s", args.out_csv)
    LOGGER.info("Saved summary -> %s", args.out_json)


if __name__ == "__main__":
    # Optional: suppress parallelism warning on Windows
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()