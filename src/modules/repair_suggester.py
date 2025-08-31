# src/modules/repair_suggester.py
from __future__ import annotations
import os, json, argparse, logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
LOGGER = logging.getLogger("repair")

def load_json(path: str) -> dict | None:
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def read_similarity_pairs(path: str) -> pd.DataFrame | None:
    if path and os.path.exists(path):
        return pd.read_csv(path)
    return None

def class_balance(df: pd.DataFrame, label_col: str) -> dict:
    counts = df[label_col].value_counts(dropna=False).to_dict()
    total = sum(counts.values())
    frac = {k: v/total for k,v in counts.items()}
    return {"counts": counts, "fractions": {str(k): float(v) for k,v in frac.items()}}

def suggest_actions(sim_pairs: pd.DataFrame | None,
                    bias_summary: dict | None,
                    robustness_summary: dict | None,
                    fairness_json: dict | None,
                    imbalance_info: dict) -> dict:
    suggestions = []

    # 1) DEDUP
    if sim_pairs is not None and len(sim_pairs) > 0:
        suggestions.append({
            "title": "Remove near-duplicates",
            "reason": f"Found {len(sim_pairs)} pairs above similarity threshold.",
            "action": "drop one item from each high-similarity pair to reduce leakage and redundancy."
        })

    # 2) REBALANCE
    fracs = imbalance_info.get("fractions", {})
    if fracs:
        maj_class, maj_frac = max(fracs.items(), key=lambda x: x[1])
        min_class, min_frac = min(fracs.items(), key=lambda x: x[1])
        if maj_frac >= 0.6:
            suggestions.append({
                "title": "Rebalance classes",
                "reason": f"Class distribution is skewed (majority={maj_class} ~{maj_frac:.2f}, minority={min_class} ~{min_frac:.2f}).",
                "action": "either downsample majority or upsample minority for training."
            })

    # 3) ROBUSTNESS
    if robustness_summary and "perturbations" in robustness_summary:
        worst = sorted(robustness_summary["perturbations"], key=lambda x: x["flip_rate"], reverse=True)[0]
        suggestions.append({
            "title": "Augment for robustness",
            "reason": f"Highest flip rate under '{worst['name']}' perturbation.",
            "action": f"augment training data with '{worst['name']}' style noise; consider char-level normalization & token cleanup."
        })

    # 4) BIAS / TOXICITY
    if bias_summary:
        tox_mean = bias_summary.get("toxicity_mean")
        tox_p95 = bias_summary.get("toxicity_p95")
        if tox_p95 is not None and tox_p95 > 0.7:
            suggestions.append({
                "title": "Moderate toxic content",
                "reason": f"High 95th percentile toxicity score (~{tox_p95:.2f}).",
                "action": "filter or downweight highly toxic samples; include counterfactual data if relevant."
            })

    # 5) FAIRNESS (optional)
    if fairness_json:
        eo = fairness_json.get("equalized_odds_tpr_diff")
        if eo and eo > 0.1:
            suggestions.append({
                "title": "Fairness reweighting",
                "reason": f"Equalized odds TPR difference ~{eo:.2f}.",
                "action": "reweight sensitive slices or perform post-hoc thresholding by group."
            })

    return {"suggestions": suggestions}

def auto_apply_dedupe(train_csv: str, text_col: str,
                      sim_pairs_csv: str | None,
                      threshold: float = 0.92,
                      out_csv: str = "data/processed/train_repaired.csv") -> str:
    """
    Simple, deterministic dedupe:
    - read train
    - mark all 'j' indices in high-similarity pairs for removal
    """
    df = pd.read_csv(train_csv)
    to_drop = set()

    if sim_pairs_csv and os.path.exists(sim_pairs_csv):
        pairs = pd.read_csv(sim_pairs_csv)
        # if file has a 'score' column, respect threshold when present
        if "score" in pairs.columns:
            pairs = pairs[pairs["score"] >= threshold]
        for _, r in pairs.iterrows():
            j = int(r["j"]) if "j" in r else None
            if j is not None and 0 <= j < len(df):
                to_drop.add(j)

    if to_drop:
        LOGGER.info("Dropping %d duplicate rows based on similarity pairs.", len(to_drop))
        df = df.drop(index=list(to_drop), errors="ignore").reset_index(drop=True)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    return out_csv

def auto_apply_rebalance(train_csv: str, label_col: str,
                         out_csv: str = "data/processed/train_repaired_balanced.csv",
                         target_majority_max_frac: float = 0.6) -> str:
    df = pd.read_csv(train_csv)
    info = class_balance(df, label_col)
    fracs = info["fractions"]
    if not fracs:
        df.to_csv(out_csv, index=False)
        return out_csv

    maj_class, maj_frac = max(fracs.items(), key=lambda x: x[1])
    if maj_frac <= target_majority_max_frac:
        # already balanced enough
        df.to_csv(out_csv, index=False)
        return out_csv

    # downsample majority to the target
    import math
    total = len(df)
    maj_target = int(target_majority_max_frac * total)
    df_maj = df[df[label_col] == int(maj_class)].sample(n=maj_target, random_state=42, replace=False)
    df_min = df[df[label_col] != int(maj_class)]
    out = pd.concat([df_maj, df_min], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    out.to_csv(out_csv, index=False)
    return out_csv

def main():
    ap = argparse.ArgumentParser(description="Generate repair suggestions and optionally apply simple fixes.")
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--text_col", default="review")
    ap.add_argument("--label_col", default="sentiment")

    # audit artifacts (optional)
    ap.add_argument("--similarity_pairs", default="reports/similarity_pairs.csv")   # or reports/similarity_clusters.csv/near_duplicates.json from SBERT
    ap.add_argument("--bias_summary", default="reports/bias_summary.json")
    ap.add_argument("--robustness_summary", default="reports/robustness_summary.json")
    ap.add_argument("--fairness_json", default="reports/fairness.json")

    # outputs
    ap.add_argument("--out_plan", default="reports/repair_plan.json")

    # auto-apply flags
    ap.add_argument("--apply_dedupe", action="store_true")
    ap.add_argument("--apply_rebalance", action="store_true")

    args = ap.parse_args()

    # Load main train and compute imbalance
    train_df = pd.read_csv(args.train_csv)
    imbalance = class_balance(train_df, args.label_col)

    sim_df = read_similarity_pairs(args.similarity_pairs)
    bias = load_json(args.bias_summary)
    robust = load_json(args.robustness_summary)
    fair = load_json(args.fairness_json)

    plan = suggest_actions(sim_df, bias, robust, fair, imbalance)
    os.makedirs(os.path.dirname(args.out_plan), exist_ok=True)
    with open(args.out_plan, "w", encoding="utf-8") as f:
        json.dump({"class_balance": imbalance, **plan}, f, indent=2)
    logging.info("Saved repair plan -> %s", args.out_plan)

    # auto-apply fixes if requested
    if args.apply_dedupe:
        dedup_out = auto_apply_dedupe(args.train_csv, args.text_col, args.similarity_pairs)
        logging.info("Wrote deduplicated training set -> %s", dedup_out)

    if args.apply_rebalance:
        reb_out = auto_apply_rebalance(args.train_csv if not args.apply_dedupe else "data/processed/train_repaired.csv",
                                       args.label_col)
        logging.info("Wrote rebalanced training set -> %s", reb_out)

if __name__ == "__main__":
    main()
