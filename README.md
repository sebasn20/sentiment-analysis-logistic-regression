# Sentiment Analysis on Social Media: Logistic Regression

CS 3793 (Artificial Intelligence) group research project. This repo contains
the Logistic Regression component, which classifies social media posts as
positive, negative, or neutral using TF-IDF features. It is compared against
Naive Bayes and LSTM models built by my teammates.

## Datasets
- [Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
- [Stock Tweets for Sentiment Analysis and Prediction](https://www.kaggle.com/datasets/equinxx/stock-tweets-for-sentiment-analysis-and-prediction)

All models share the same train/validation/test splits so results are
directly comparable. Data files are not included in this repo.

## Method
Text cleaning, then TF-IDF vectorization, then scikit-learn Logistic Regression,
evaluated with accuracy, precision, recall, and F1-score.

## Setup
Create a virtual environment, then run: pip install -r requirements.txt

## Results (Twitter US Airline Sentiment, test set)

| Model | Accuracy | Macro F1 |
|---|---|---|
| Logistic Regression + TF-IDF (baseline) | 0.794* | 0.716* |
| Logistic Regression + TF-IDF (balanced class weights) | 0.776 | 0.732 |

*Validation set. Settings were tuned on the validation set; the test set was
used once for the final score.

Balanced class weights raised recall on the minority classes
(neutral 0.48 to 0.68, positive 0.61 to 0.76 on validation) at a small cost to
negative recall. Most remaining errors involve the neutral class, including
complaints phrased as questions and sarcasm.

## Status
Airline dataset complete. Stock Tweets in progress.