import pandas as pd
from sklearn.model_selection import train_test_split

# Load your dataset
df = pd.read_csv("../../data/raw/IMDB_Dataset.csv")

# Split dataset: 80% train, 20% test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["sentiment"])

# Save to CSV
train_df.to_csv("../../data/processed/train.csv", index=False)
test_df.to_csv("../../data/processed/test.csv", index=False)
