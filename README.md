# Sentiment Analysis on Social Media: Logistic Regression

CS 3793 (Artificial Intelligence) group research project. This repo contains
the Logistic Regression component, which classifies social media posts as
positive, negative, or neutral using TF-IDF features. It is compared against
Naive Bayes and LSTM models built by my teammates.

## Datasets
- [Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
- [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)
- [Stock Tweets for Sentiment Analysis and Prediction](https://www.kaggle.com/datasets/equinxx/stock-tweets-for-sentiment-analysis-and-prediction)

All models share the same train/validation/test splits so results are
directly comparable. Data files are not included in this repo.

## Method
Text cleaning, then TF-IDF vectorization, then scikit-learn Logistic Regression,
evaluated with accuracy, precision, recall, and F1-score.

## Setup
Create a virtual environment, then run: pip install -r requirements.txt

## Status
Work in progress.