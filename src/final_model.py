import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay)

# load clean files
train = pd.read_csv("../data/airline_clean/train.csv")
validation = pd.read_csv("../data/airline_clean/validation.csv")
test = pd.read_csv("../data/airline_clean/test.csv")

# tweets into numbers
vectorizer = TfidfVectorizer()
x_train = vectorizer.fit_transform(train["clean_text"])
x_validation = vectorizer.transform(validation["clean_text"])
x_test = vectorizer.transform(test["clean_text"])

y_train = train["airline_sentiment"]
y_validation = validation["airline_sentiment"]
y_test = test["airline_sentiment"]

# final model
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(x_train, y_train)

# validation report
print("VALIDATION RESULTS")
validation_prediction = model.predict(x_validation)
print(classification_report(y_validation, validation_prediction))

# test results
print("TEST RESULTS")
test_prediction = model.predict(x_test)
print("Accuracy:", round(accuracy_score(y_test, test_prediction), 3))
print("Macro F1:", round(f1_score(y_test, test_prediction, average="macro"), 3))
print()
print(classification_report(y_test, test_prediction))

# confusion matrix pic
labels = ["negative", "neutral", "positive"]
ConfusionMatrixDisplay.from_predictions(y_test, test_prediction, labels=labels, cmap="Blues")
plt.title("Airline Sentiment: Logistic Regression (Test Set)")
plt.savefig("../results/airline_sentiment.png", dpi=150, bbox_inches="tight")
plt.show()