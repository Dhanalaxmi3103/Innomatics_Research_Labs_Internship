## Sentiment Analysis of Real-time Flipkart Product Reviews

In this project, I built an end-to-end Sentiment Analysis system to classify customer reviews as Positive or Negative and understand customer dissatisfaction patterns.

## 1. Data Loading and Initial Analysis

First, I loaded the dataset using pandas.


  --Checked the dataset shape

  --Verified missing values

  --Analyzed class distribution (positive vs negative)

  --Explored frequently occurring words

Observed patterns in negative reviews to understand customer pain points

This helped me understand the data quality and sentiment distribution.

## 2. Data Cleaning and Preprocessing

The raw review text contained noise such as:

  --Special characters

  --URLs

  --Numbers

  --Stopwords

So I performed the following preprocessing steps:

 -- Converted text to lowercase

  --Removed URLs

  --Removed special characters and numbers

  --Removed stopwords using NLTK

  --Applied lemmatization

  --Tokenized the text

This normalization improved model learning and reduced noise.

## 3. Text Representations (Feature Engineering)

Since ML models cannot understand raw text, I experimented with multiple text embedding techniques:

  ✅ 1. Bag of Words (BoW)

   Used CountVectorizer

   Converted text into word frequency vectors

   Simple and effective baseline approach

  ✅ 2. TF-IDF (Term Frequency – Inverse Document Frequency)

   Used max_features = 5000

   Used ngram_range = (1,2)

   Reduced impact of frequent words

   Captured important unigrams and bigrams

TF-IDF performed better than BoW in most cases.

  ✅ 3. Word2Vec

   Used Word2Vec to generate word embeddings

   Converted each review into a vector by averaging word vectors

   Captured semantic meaning of words

   More advanced representation compared to BoW and TF-IDF

This helped capture contextual similarity between words.

## 4. Train-Test Split

I split the dataset into:

  80% Training data

  20% Testing data

I used stratified sampling to maintain class balance.

## 5. Model Training

I trained three machine learning models on each representation:

  ### 🔹 Logistic Regression

   Strong baseline model

   Works well with linear text features

  ### 🔹 Support Vector Machine (SVM)

   Effective in high-dimensional spaces

   Performed well with TF-IDF

  ### 🔹 Random Forest

   Ensemble-based model

  Captured non-linear relationships

## 7. Model Evaluation

I evaluated all models using:

  Train Accuracy

  Test Accuracy

  F1-score (primary metric)


F1-score was chosen because it balances Precision and Recall, which is important for sentiment classification.

After comparison, the best performing combination was:

👉 TF-IDF + Logistic Regression

## 8. Model Serialization

After selecting the best model, I saved:

  TF-IDF Vectorizer → vectorizer.pkl

  Trained Model → best_model.pkl

This allowed me to deploy the model without retraining.

## 9. Deployment using Streamlit

Finally, I developed a Streamlit web application.

Workflow:

User enters review
  → Text cleaning

  → Vectorization

  → Model prediction
  
  → Sentiment output with confidence score

This made the system interactive and user-friendly.

## Project Structure

Sentiment Analysis of Flipkart Product/

│

├── data.csv

│

├── sentiment.ipynb                 # Initial experimentation & EDA

│

├── best_model.pkl                  # Saved best performing model

├── vectorizer.pkl                  # Saved TF-IDF vectorizer

│

├── sent_app.py                     # Streamlit deployment app

│

├── mlruns/                         # MLflow experiment tracking folder

│

├── requirements.txt

└── README.md


## Skills Demonstrated

  NLP preprocessing

  Feature engineering

  Multiple text embeddings

  Model comparison

  Performance evaluation

  Model deployment (Streamlit)




