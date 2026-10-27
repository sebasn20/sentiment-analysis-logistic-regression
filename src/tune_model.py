import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score

# load clean files
train = pd.read_csv("../data/airline_clean/train.csv")
validation = pd.read_csv("../data/airline_clean/validation.csv")

y_train = train["airline_sentiment"]
y_validation = validation["airline_sentiment"]

def try_settings(name, ngram_range=(1,1), min_df=1, C=1, class_weight=None):
    # tweets into numbers
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df)
    x_train = vectorizer.fit_transform(train["clean_text"])
    x_validation = vectorizer.transform(validation["clean_text"])

    # train model
    model = LogisticRegression(max_iter=1000, C=C, class_weight=class_weight)
    model.fit(x_train, y_train)

    # score on validation set
    prediction = model.predict(x_validation)
    accuracy = round(accuracy_score(y_validation, prediction), 3)
    macro_f1 = round(f1_score(y_validation, prediction, average="macro"), 3)

    print(f"{name:30} accuracy: {accuracy} macro F1: {macro_f1}")

try_settings("baseline")
try_settings("balanced weights", class_weight="balanced")
try_settings("word pairs (bigrams)", ngram_range=(1,2))
try_settings("ignore rare words", min_df=2)
try_settings("C = 0.5", C=0.5)
try_settings("C = 5", C=5)
try_settings("C = 10", C=10)
try_settings("balances + C=5", C=5, class_weight="balanced")
try_settings("balanced + min_df=2", min_df=2, class_weight="balanced")
try_settings("balanced + min_df=2 + C=5", min_df=2, C=5, class_weight="balanced")
try_settings("balanced + min_df=2 + C=2", min_df=2, C=2, class_weight="balanced")