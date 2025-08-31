# src/modules/report_builder.py
from __future__ import annotations
import os, json, argparse, logging
import pandas as pd
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
LOGGER = logging.getLogger("report_builder")

def read_json(p: str):
    f = Path(p)
    if f.exists():
        with open(f, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return None

def read_csv(p: str):
    f = Path(p)
    return pd.read_csv(f) if f.exists() else None

def fmt(v, nd=4):
    if v is None: return "—"
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return str(v)

def build_md(ctx: dict) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []
    lines.append(f"# AI Audit Summary — {ts}")
    lines.append("")
    lines.append("## Project")
    lines.append("- Tool: **AI Audit Framework** (dataset QC → model → explainability → fairness/bias → robustness → repair)")
    lines.append("- Dataset: IMDb reviews (binary sentiment)")
    lines.append("")
    # Metrics
    base_acc = ctx.get("metrics_base", {}).get("accuracy")
    base_f1  = ctx.get("metrics_base", {}).get("f1")
    rep_acc  = ctx.get("metrics_rep", {}).get("accuracy")
    rep_f1   = ctx.get("metrics_rep", {}).get("f1")

    lines.append("## Model Quality")
    lines.append("| version | accuracy | f1 |")
    lines.append("|---|---:|---:|")
    lines.append(f"| original | {fmt(base_acc, 4)} | {fmt(base_f1, 4)} |")
    lines.append(f"| repaired | {fmt(rep_acc, 4)} | {fmt(rep_f1, 4)} |")
    lines.append("")

    # Bias/Toxicity
    b_base = ctx.get("bias_base", {})
    b_rep  = ctx.get("bias_rep", {})
    lines.append("## Bias & Toxicity")
    lines.append("| version | tox_mean | tox_p95 |")
    lines.append("|---|---:|---:|")
    lines.append(f"| original | {fmt(b_base.get('toxicity_mean'), 4)} | {fmt(b_base.get('toxicity_p95'), 4)} |")
    lines.append(f"| repaired | {fmt(b_rep.get('toxicity_mean'), 4)} | {fmt(b_rep.get('toxicity_p95'), 4)} |")
    lines.append("")

    # Robustness
    r_base = ctx.get("robust_base", {}).get("perturbations", [])
    r_rep  = ctx.get("robust_rep", {}).get("perturbations", [])
    def table_robust(rows, title):
        if not rows:
            lines.append(f"_{title}: no data_")
            return
        lines.append(title)
        lines.append("| perturbation | flip_rate | avg_conf_change |")
        lines.append("|---|---:|---:|")
        for p in rows:
            lines.append(f"| {p['name']} | {fmt(p.get('flip_rate'),4)} | {fmt(p.get('avg_conf_change'),4)} |")
        lines.append("")
    lines.append("## Robustness (prediction flips under simple perturbations)")
    table_robust(r_base, "Original")
    table_robust(r_rep,  "Repaired")

    # Similarity
    sim_pairs = ctx.get("sim_pairs_n", 0)
    lines.append("## Similarity & Redundancy")
    lines.append(f"- High-similarity pairs (≥ threshold): **{sim_pairs}**")
    lines.append("")

    # Repair plan
    plan = ctx.get("repair_plan", {})
    suggestions = plan.get("suggestions", [])
    lines.append("## Repair Suggestions Applied / Proposed")
    if suggestions:
        for i, s in enumerate(suggestions, 1):
            lines.append(f"**{i}. {s.get('title','')}**")
            lines.append(f"- Reason: {s.get('reason','')}")
            lines.append(f"- Action: {s.get('action','')}")
            lines.append("")
    else:
        lines.append("_No plan or no suggestions available._\n")

    # Takeaways
    lines.append("## Key Takeaways")
    notes = []
    if rep_acc is not None and base_acc is not None:
        delta = float(rep_acc) - float(base_acc)
        notes.append(f"- Accuracy delta (repaired - original): **{fmt(delta,4)}**")
    if r_rep:
        worst_rep = max(r_rep, key=lambda x: x.get("flip_rate", 0.0))
        notes.append(f"- Worst robustness after repair: **{worst_rep['name']}** (flip={fmt(worst_rep.get('flip_rate'))})")
    if not notes:
        notes.append("- Filled baseline audit; future work: BERT model, slice-based fairness, data drift.")
    lines += notes
    lines.append("")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser(description="Build a consolidated Markdown audit report.")
    ap.add_argument("--out", default="reports/audit_summary.md")
    # base (original) artifacts — optional
    ap.add_argument("--metrics_base", default="reports/metrics.json")
    ap.add_argument("--bias_base", default="reports/bias_summary.json")
    ap.add_argument("--robust_base", default="reports/robustness_summary.json")
    # repaired artifacts
    ap.add_argument("--metrics_rep", default="reports/metrics_repaired.json")
    ap.add_argument("--bias_rep", default="reports/bias_summary_repaired.json")
    ap.add_argument("--robust_rep", default="reports/robustness_summary_repaired.json")
    # similarity & plan
    ap.add_argument("--sim_pairs", default="reports/similarity_pairs.csv")
    ap.add_argument("--repair_plan", default="reports/repair_plan.json")
    args = ap.parse_args()

    ctx = {
        "metrics_base": read_json(args.metrics_base) or {},
        "bias_base":    read_json(args.bias_base) or {},
        "robust_base":  read_json(args.robust_base) or {},
        "metrics_rep":  read_json(args.metrics_rep) or {},
        "bias_rep":     read_json(args.bias_rep) or {},
        "robust_rep":   read_json(args.robust_rep) or {},
        "repair_plan":  read_json(args.repair_plan) or {},
    }
    sim_df = read_csv(args.sim_pairs)
    ctx["sim_pairs_n"] = 0 if sim_df is None else int(len(sim_df))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    md = build_md(ctx)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md)
    LOGGER.info("Wrote report -> %s", args.out)

if __name__ == "__main__":
    main()
