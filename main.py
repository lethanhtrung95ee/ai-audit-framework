import pandas as pd
from src.modules.schema_checker import check_schema_issues
from src.modules.duplicate_checker import find_duplicates
from src.modules.imbalance_checker import check_class_imbalance
from src.modules.bias_detector import detect_toxicity
from src.modules.similarity_checker import check_similarity
from src.modules.report_generator import save_report
from src.modules.fairness_checker import calculate_fairness_metrics
import numpy as np

def run_audit(csv_path: str, text_column: str, label_column: str, protected_attr: str = None):
    df = pd.read_csv(csv_path)

    schema = check_schema_issues(df)
    duplicates = find_duplicates(df, text_column)
    imbalance = check_class_imbalance(df, label_column)
    toxicity = detect_toxicity(df[text_column].dropna().sample(5).tolist())
    similarity = check_similarity(df[text_column].dropna().sample(5).tolist())

    final_report = {
        "schema_issues": schema,
        "duplicates": duplicates,
        "class_imbalance": imbalance,
        "toxicity_analysis": toxicity,
        "text_similarity": similarity
    }

    if protected_attr and protected_attr in df.columns:
        # For now, use dummy predictions (all ones)
        predictions = np.ones(len(df))
        fairness_metrics = calculate_fairness_metrics(
            df=df,
            label_col=label_column,
            protected_attr=protected_attr,
            privileged_groups=[{protected_attr: 1}],
            unprivileged_groups=[{protected_attr: 0}],
            predictions=predictions
        )
        final_report["fairness_metrics"] = fairness_metrics

    save_report(final_report)
    print("✅ Audit completed. Report saved.")


if __name__ == "__main__":
    run_audit(
        "data/processed/processed_imdb.csv",
        text_column="text",
        label_column="label",
        protected_attr="gender"
    )
