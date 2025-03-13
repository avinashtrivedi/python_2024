"""
This script iterates over required models, trains them, and saves the models for use in prediction.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
import pandas as pd
import re
from urllib.parse import urlparse, unquote
import requests
from bs4 import BeautifulSoup
import time
import pickle
import url_classification_utility  # Custom utility module
from gensim.models import Word2Vec




# Load the dataset
df = pd.read_csv('url_classification_data.csv')

# Initialize the models from the utility module
models = url_classification_utility.initiate_models()

models

# +
# Create and train the Word2Vec model
# sentences = df['URL'].apply(lambda x: url_classification_utility.url_to_words(x).split()).tolist()
# w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)
# -

with open('word2vec_model.pkl', 'rb') as file: 
    w2v_model = pickle.load(file) 

# +
# Save the Word2Vec model for future use
# url_classification_utility.save_word2vec_model(w2v_model)
# -

# Iterate over each model and train it
# for model_name, model in models.items():
#     try:
#         url_classification_utility.train_models(df, model_name, model)
#         print(f'[success] trained {model_name}')
#     except Exception as e:
#         print(f'[error] failed training for {model_name}')
#         print(e)
#         print('##################################')


# Iterate over each model and train it
for model_name, model in models.items():
    if model_name in ['dt_tuned', 'lr_tuned', 'rf_tuned', 'knn_tuned', 'svm_tuned']:
        print("Model Name: ", model_name)
        df, vectorizer = url_classification_utility.preprocess_data(df, use_word2vec=True, w2v_model=w2v_model)
    else:
        df, vectorizer = url_classification_utility.preprocess_data(df)

    # Handle advanced models separately
    if model_name in ['gpt', 'bert']:
        continue  # GPT and BERT will be handled differently in the prediction phase

    url_classification_utility.train_models(df, model_name, model)

    # Save the tuned models separately
    with open(f'url_classification_{model_name}.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)

print('[success] All models have been trained and saved.')

pd

df = pd.read_csv('a.csv')

from sklearn.ensemble import RandomForestClassifier

df.columns = df.columns.astype(str)

X = df.drop(['Unnamed: 0','Classification', 'URL', 'University', 'URL_words'], axis=1)
y = df['Classification']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier()

clf.fit(X_train,y_train)


