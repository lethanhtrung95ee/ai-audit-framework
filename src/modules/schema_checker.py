import pandas as pd

def check_schema_issues(df: pd.DataFrame):
    report = {
        "missing_values": df.isnull().sum().to_dict(),
        "column_types": df.dtypes.astype(str).to_dict(),
        "empty_strings": (df == '').sum().to_dict()
    }
    return report
