import pandas as pd
from sklearn.model_selection import validation_curve

from preprocess import clean_text

# load the 3 files
train = pd.read_csv("../data/airline_splits/train.csv")
validation = pd.read_csv("../data/airline_splits/validation.csv")
test = pd.read_csv("../data/airline_splits/test.csv")

# clean all tweets and save it to a new column
train["clean_text"] = train["text"].apply(clean_text)
validation["clean_text"] = validation["text"].apply(clean_text)
test["clean_text"] = test["text"].apply(clean_text)

# save cleaned files
train.to_csv("../data/airline_clean/train.csv", index=False)
validation.to_csv("../data/airline_clean/validation.csv", index=False)
test.to_csv("../data/airline_clean/test.csv", index=False)

print("Rows:", len(train), len(validation), len(test))
print("Empty after cleaning:", (train["clean_text"] == "").sum())
print()
print(train[["text", "clean_text", "airline_sentiment"]].head(5).to_string())