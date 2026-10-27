import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report

# load clean files
train = pd.read_csv("../data/airline_clean/train.csv")
validation = pd.read_csv("../data/airline_clean/validation.csv")

# turn tweets into numbers
vectorizer = TfidfVectorizer()
x_train = vectorizer.fit_transform(train["clean_text"])
x_validation = vectorizer.transform(validation["clean_text"])

# labels
y_train = train["airline_sentiment"]
y_validation = validation["airline_sentiment"]
print("Number of words (features): ", x_train.shape[1])

# train model
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# predict validation set
prediction = model.predict(x_validation)

print("\nAccuracy:", round(accuracy_score(y_validation, prediction), 3))
print("Macro F1:", round(f1_score(y_validation, prediction, average="macro"), 3))
print()
print(classification_report(y_validation, prediction))