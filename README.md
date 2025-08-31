# 🛡️ AI Audit Framework

*A Lightweight, End-to-End Toolkit for Auditing Text Classification Models*

---

## ✨ Overview

This framework provides a **pre-flight audit pipeline** for text classification models.
It detects **bias, robustness gaps, data leakage, redundancy, and fairness issues** before and after training, then produces a **single-page audit report** suitable for technical and non-technical reviewers.

Designed to be:

* **Practical** – built on PyTorch, HuggingFace, scikit-learn
* **Lightweight** – runs on CPU or GPU with simple commands
* **Transparent** – outputs clear artifacts and summaries for governance
* **EB2-ready** – produces `audit_summary.md` as a concise audit record

---

## 🔍 Key Features

* **Dataset Health Checks**

  * Train/test split reproducibility
  * Similarity & duplicate detection (TF-IDF cosine)
  * Imbalance snapshot

* **Model Training**

  * Fast baseline: TF-IDF + Logistic Regression
  * Saved artifacts (`models/trained_model.pkl`, metrics, predictions)

* **Explainability**

  * SHAP global feature importances
  * LIME local interpretability

* **Bias & Fairness**

  * Identity & toxicity checks (HuggingFace models)
  * JSON/CSV summaries for review

* **Robustness**

  * Perturbation testing (typos, casing, whitespace, punctuation)
  * Flip-rate and confidence drop metrics

* **Repair Suggestions**

  * JSON plan with deduplication, rebalancing, and bias mitigation proposals

* **Unified Reporting**

  * Generates `reports/audit_summary.md` for decision-makers

---

## ⚡ Quickstart

### 1. Setup

```bash
# Clone & enter project
git clone <repo-url>
cd ai-audit-framework

# Create virtual environment (Python 3.12 recommended)
python -m venv .venv
.venv\Scripts\activate   # (Windows)
# source .venv/bin/activate  # (Linux/Mac)

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Pipeline

```bash
# Train/Test split
python src/modules/data_splitter.py --input data/raw/reviews.csv --text_col review --label_col sentiment

# Train baseline model
python src/modules/model_trainer.py --train data/processed/train.csv --test data/processed/test.csv --text_col review --label_col sentiment

# Run bias check
python src/modules/bias_detector.py --input data/processed/test.csv --text_col review

# Run robustness check
python src/modules/robustness_checker.py --model models/trained_model.pkl --data data/processed/test.csv --text_col review --label_col sentiment

# Generate audit report
python src/modules/report_builder.py --out reports/audit_summary.md
```

---

## 📂 Project Structure

```
├── data/
│   ├── raw/               # Original datasets
│   ├── processed/         # Train/test splits
├── models/                # Saved trained models
├── reports/               # Bias, robustness, repair, and audit summaries
├── src/
│   ├── modules/           # Core audit modules
│   └── utils/             # Shared utilities
└── requirements.txt       # Python dependencies
```

---

## 📑 Output Artifacts

* **Model** → `models/trained_model.pkl`
* **Metrics** → `reports/metrics.json`
* **Bias Summary** → `reports/bias_summary.json`
* **Robustness Summary** → `reports/robustness_summary.json`
* **Repair Plan** → `reports/repair_plan.json`
* **Final Audit** → `reports/audit_summary.md`

---

## ✅ Example Use Case

* Train a sentiment analysis model on reviews
* Detect toxic/identity bias
* Test robustness against typos and noise
* Suggest repair strategies (e.g., rebalance, deduplication)
* Export a one-page `audit_summary.md` for governance

---

## 🤝 Contributing

Pull requests and issue reports are welcome! This framework is designed to grow with community feedback.

---

## 📜 License

MIT License – free to use, modify, and distribute.
