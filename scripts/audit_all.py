# scripts/audit_all.py
from __future__ import annotations
import argparse, subprocess, sys, os, shutil, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def run(cmd: list[str]):
    print("[RUN]", " ".join(cmd))
    rc = subprocess.call(cmd, shell=False)
    if rc != 0:
        print("[WARN] command failed with code", rc)
    return rc

def ensure_dirs():
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)
    (ROOT / "models").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)

def main():
    ap = argparse.ArgumentParser(description="End-to-end audit runner")
    ap.add_argument("--raw", default="data/raw/IMDB_Dataset.csv")
    ap.add_argument("--text_col", default="review")
    ap.add_argument("--label_col", default="sentiment")
    ap.add_argument("--train", default="data/processed/train.csv")
    ap.add_argument("--test",  default="data/processed/test.csv")
    ap.add_argument("--repaired", action="store_true", help="use repaired/balanced train if available")
    args = ap.parse_args()

    ensure_dirs()

    # 1) If no splits, prepare
    if not (ROOT / args.train).exists() or not (ROOT / args.test).exists():
        run([sys.executable, str(ROOT / "src" / "data" / "prepare_data.py"),
             "--input", args.raw, "--out_dir", "data/processed",
             "--text_col", args.text_col, "--label_col", args.label_col, "--train_frac", "0.8", "--seed", "42"])

    # choose training file
    train_path = args.train
    if args.repaired:
        repaired_bal = ROOT / "data" / "processed" / "train_repaired_balanced.csv"
        repaired = ROOT / "data" / "processed" / "train_repaired.csv"
        if repaired_bal.exists():
            train_path = str(repaired_bal)
        elif repaired.exists():
            train_path = str(repaired)
        else:
            print("[INFO] repaired flag set but no repaired file found; using", args.train)

    # 2) Train
    run([sys.executable, str(ROOT / "src" / "modules" / "model_trainer.py"),
         "--train", train_path, "--test", args.test,
         "--text_col", args.text_col, "--label_col", args.label_col])

    # 3) Explainability
    run([sys.executable, str(ROOT / "src" / "modules" / "explainability.py"),
         "--model", "models/trained_model.pkl",
         "--train", args.train, "--test", args.test,
         "--text_col", args.text_col, "--label_col", args.label_col,
         "--out_dir", "reports"])

    # 4) Bias/Toxicity (GPU-aware pipeline)
    run([sys.executable, str(ROOT / "src" / "modules" / "bias_detector.py"),
         "--input", args.test, "--text_col", args.text_col,
         "--sample", "1500", "--batch_size", "32",
         "--out_csv", "reports/bias_toxicity_sample.csv",
         "--out_json", "reports/bias_summary.json"])

    # 5) Similarity (TF-IDF fast)
    run([sys.executable, str(ROOT / "src" / "modules" / "similarity_detector.py"),
         "--input", args.train, "--text_col", args.text_col,
         "--threshold", "0.90", "--out_csv", "reports/similarity_pairs.csv"])

    # 6) Robustness (perturbations)
    run([sys.executable, str(ROOT / "src" / "modules" / "robustness_checker.py"),
         "--model", "models/trained_model.pkl",
         "--data", args.test, "--text_col", args.text_col, "--label_col", args.label_col,
         "--sample", "800",
         "--out_csv", "reports/robustness_details.csv",
         "--out_json", "reports/robustness_summary.json"])

    # 7) Repair suggestions (no auto-apply here)
    run([sys.executable, str(ROOT / "src" / "modules" / "repair_suggester.py"),
         "--train_csv", args.train, "--text_col", args.text_col, "--label_col", args.label_col,
         "--similarity_pairs", "reports/similarity_pairs.csv",
         "--bias_summary", "reports/bias_summary.json",
         "--robustness_summary", "reports/robustness_summary.json",
         "--fairness_json", "reports/fairness.json",
         "--out_plan", "reports/repair_plan.json"])

    # 8) Final Markdown summary
    run([sys.executable, str(ROOT / "src" / "modules" / "report_builder.py"),
         "--out", "reports/audit_summary.md"])

    print("\n[OK] Audit complete. See reports/audit_summary.md")

if __name__ == "__main__":
    main()
