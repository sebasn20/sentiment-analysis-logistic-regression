from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data" / "stock_tweets"

stock = pd.read_csv(DATA / "stock_tweets.csv")
print("Shape:", stock.shape)
print("Columns:", list(stock.columns))
print(stock.head(5).to_string())
print("\nMissing values:\n", stock.isna().sum())
print("\nDuplicate tweets:", stock.duplicated(subset=stock.columns[1]).sum())
print("\nTweets per stock:\n", stock.iloc[:, 2].value_counts().head(10))