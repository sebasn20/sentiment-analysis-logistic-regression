from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data" / "airline_splits"

splits = {
    "train": pd.read_csv(DATA / "train.csv"),
    "validation": pd.read_csv(DATA / "validation.csv"),
    "test": pd.read_csv(DATA / "test.csv"),
}

for name, df in splits.items():
    print("=" * 60)
    print(name.upper(), df.shape)
    print("=" * 60)
    print(df.head(3).to_string())
    print("\nMissing values:\n", df.isna().sum())
    print("\nLabel counts:\n", df["airline_sentiment"].value_counts())
    print("\nLabel proportions:\n",
          df["airline_sentiment"].value_counts(normalize=True).round(3))
    print("\nAvg tweet length (chars):", round(df["text"].str.len().mean(), 1))
    print()

# Leakage check
train_t, val_t, test_t = (set(splits[k]["text"]) for k in splits)
print("Overlap train/val:", len(train_t & val_t))
print("Overlap train/test:", len(train_t & test_t))
print("Overlap val/test:", len(val_t & test_t))