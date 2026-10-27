import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# load clean files
train = pd.read_csv("../data/airline_clean/train.csv")
test = pd.read_csv("../data/airline_clean/test.csv")

# same final model from before
vectorizer = TfidfVectorizer()
x_train = vectorizer.fit_transform(train["clean_text"])
x_test = vectorizer.transform(test["clean_text"])

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(x_train, train["airline_sentiment"])

# words model relies on mose for each class
words = vectorizer.get_feature_names_out()
for i, label in enumerate(model.classes_):
    top = model.coef_[i].argsort()[-15:][::-1]
    print(label.upper(), "->", [words[i] for i in top])
    print()

# mistakes
test["prediction"] = model.predict(x_test)
wrong = test[test["prediction"] != test["airline_sentiment"]]
print("Wrong predictions:", len(wrong), "out of", len(test))
print()
print("Which mix ups happen the msot (rows = true, columns = predicted):")
print(pd.crosstab(wrong["airline_sentiment"], wrong["prediction"]))
print()
print("15 random mistakes:")
sample = wrong[["text", "airline_sentiment", "prediction"]].sample(15, random_state=1)
print(sample.to_string())