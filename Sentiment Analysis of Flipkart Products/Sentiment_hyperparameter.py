import pandas as pd
import numpy as np
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec

import optuna
import mlflow
import mlflow.sklearn
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
import time
import joblib
import os

#from transformers import BertTokenizer, BertModel
#import torch

import warnings
warnings.filterwarnings("ignore")

'''nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')'''

df=pd.read_csv("data.csv")
#df.head()

df=df[['Review text','Ratings']]
#df.head()

df.dropna( inplace=True)

#Creating Sentiment Labels
""" 1-->positive
    0-->negative"""
def label_sentiment(rating):
    return 1 if rating >= 4 else 0

df['sentiment'] = df['Ratings'].apply(label_sentiment)

#Data Cleaning
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)  # remove URLs
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # remove special characters
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text if word not in stop_words]
    return " ".join(text)

# Apply cleaning
df['Cleaned_Review'] = df['Review text'].apply(clean_text)
y = df['sentiment']
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df['Cleaned_Review'],
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

bow = CountVectorizer(max_features=5000)

X_train_bow = bow.fit_transform(X_train_text)
X_test_bow = bow.transform(X_test_text) 

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))

X_train_tfidf = tfidf.fit_transform(X_train_text)
X_test_tfidf = tfidf.transform(X_test_text)

tokenized_train = X_train_text.apply(lambda x: x.split())
tokenized_test = X_test_text.apply(lambda x: x.split())

w2v_model = Word2Vec(
    sentences=tokenized_train,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4
)

def avg_word2vec(tokens, model, size):
    vectors = [model.wv[w] for w in tokens if w in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(size)

X_train_w2v = np.array([
    avg_word2vec(tokens, w2v_model, 100)
    for tokens in tokenized_train
])

X_test_w2v = np.array([
    avg_word2vec(tokens, w2v_model, 100)
    for tokens in tokenized_test
])

def tune_model(trial, X, y, model_type):

    if model_type == "logreg":
        C = trial.suggest_float("C", 1e-4, 10, log=True)
        model = LogisticRegression(C=C, max_iter=2000)

    elif model_type == "svm":
        C = trial.suggest_float("C", 1e-4, 10, log=True)
        loss = trial.suggest_categorical("loss", ["hinge", "squared_hinge"])
        model = LinearSVC(C=C, loss=loss, max_iter=5000)

    elif model_type == "rf":
        n_estimators = trial.suggest_int("n_estimators", 100, 300)
        max_depth = trial.suggest_int("max_depth", 5, 30)
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )

    score = cross_val_score(model, X, y, cv=3, scoring="f1").mean()
    return score

def run_experiment(X_train, X_test, y_train, y_test, representation_name):

    for model_type in ["logreg", "svm", "rf"]:

        # Hyperparameter tuning
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: tune_model(trial, X_train, y_train, model_type),
                       n_trials=15)

        best_params = study.best_params
        best_f1_cv = study.best_value

        # Build model
        if model_type == "logreg":
            model = LogisticRegression(**best_params, max_iter=2000)

        elif model_type == "svm":
            model = LinearSVC(**best_params, max_iter=5000)

        elif model_type == "rf":
            model = RandomForestClassifier(**best_params)

        # ⏱ Train Time
        start_train = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_train

        # ⏱ Test Time
        start_test = time.time()
        y_pred = model.predict(X_test)
        test_time = time.time() - start_test

        # Metrics
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred)

        # 💾 Model Size
        model_filename = f"{representation_name}_{model_type}.pkl"
        joblib.dump(model, model_filename)
        model_size = os.path.getsize(model_filename) / (1024 * 1024)  # MB

        # MLflow Logging
        with mlflow.start_run(run_name=f"{representation_name}_{model_type}"):

            mlflow.log_param("representation", representation_name)
            mlflow.log_param("model_type", model_type)

            for param, value in best_params.items():
                mlflow.log_param(param, value)

            mlflow.log_metric("cv_f1_score", best_f1_cv)
            mlflow.log_metric("train_accuracy", train_acc)
            mlflow.log_metric("test_accuracy", test_acc)
            mlflow.log_metric("test_f1_score", test_f1)
            mlflow.log_metric("train_time_seconds", train_time)
            mlflow.log_metric("test_time_seconds", test_time)
            mlflow.log_metric("model_size_mb", model_size)

            mlflow.sklearn.log_model(model, "model")

        print(f"\n{representation_name} + {model_type}")
        print("CV F1:", best_f1_cv)
        print("Train Acc:", train_acc)
        print("Test Acc:", test_acc)
        print("Test F1:", test_f1)
        print("Train Time:", train_time)
        print("Test Time:", test_time)
        print("Model Size (MB):", model_size)

run_experiment(X_train_bow, X_test_bow, y_train, y_test, "BoW")
run_experiment(X_train_tfidf, X_test_tfidf, y_train, y_test, "TFIDF")
run_experiment(X_train_w2v, X_test_w2v, y_train, y_test, "Word2Vec")

