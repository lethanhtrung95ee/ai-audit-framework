import pandas as pd

def check_class_imbalance(df: pd.DataFrame, label_column: str):
    distribution = df[label_column].value_counts().to_dict()
    return {
        "label_distribution": distribution,
        "most_common": max(distribution, key=distribution.get),
        "least_common": min(distribution, key=distribution.get)
    }
