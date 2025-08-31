import pandas as pd
from fuzzywuzzy import fuzz

def find_duplicates(df: pd.DataFrame, text_column: str):
    duplicates = df[df.duplicated(subset=[text_column])]
    return {
        "exact_duplicates_count": len(duplicates),
        "duplicate_rows": duplicates.head(5).to_dict(orient='records')
    }
