"""
This script contains the required functions for:
    1. Pre-processing data
    2. Feature engineering
    3. Model training
    4. Prediction
"""


import re
import pickle
import warnings
import requests
import pandas as pd
from urllib.parse import urlparse, unquote
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score, cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, accuracy_score
# from selenium import webdriver
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
# from selenium.webdriver.common.by import By
# from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
# from selenium.webdriver.common.action_chains import ActionChains
import os
import pandas as pd
import shutil
import time
# from selenium.webdriver.common.by import By
# from urllib.parse import urlparse
# from selenium.webdriver.common.proxy import Proxy, ProxyType
import pandas as pd
# import url_classification_utility
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from gensim.models import Word2Vec
import numpy as np
# from transformers import pipeline
# import add_bert_mode
# import torch
# from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
# import add_gpt_mode

warnings.filterwarnings("ignore")

os.environ['OMP_NUM_THREADS'] = '1'

# Ignore warnings
warnings.filterwarnings("ignore")

print("Hellooooo")

def url_to_words(url):
    """
    Converts a URL into a string of words extracted from its components.
    
    Parameters:
    url (str): The URL to be processed.
    
    Returns:
    str: A string of words extracted from the URL.
    """
    parsed_url = urlparse(unquote(url))
    words = []

    # Extract and clean domain components
    words.extend(parsed_url.netloc.split('.'))
    words = [''.join(c for c in word if c.isalnum()) for word in words]

    # Extract and clean path components
    path_components = parsed_url.path.strip('/').split('/')
    for component in path_components:
        words.extend(re.findall(r'[^\W_]+', component))

    # Join and return the words
    words = ' '.join([word.lower() for word in words if word])
    return words

def save_word2vec_model(w2v_model, filepath='word2vec_model.pkl'):
    with open(filepath, 'wb') as file:
        pickle.dump(w2v_model, file)

def load_word2vec_model(filepath='word2vec_model.pkl'):
    with open(filepath, 'rb') as file:
        w2v_model = pickle.load(file)
    return w2v_model

def preprocess_data(df, vectorizer=None, use_word2vec=False, w2v_model=None):
    # Basic feature extraction
    df['url_length'] = df['URL'].str.len()
    df['path_depth'] = df['URL'].apply(lambda x: len(urlparse(x).path.strip('/').split('/')))

    # Keyword presence as binary features
    keyword_cols = ['calendar', 'academic', 'erasmus']
    for keyword in keyword_cols:
        df[f'contains_{keyword}'] = df['URL'].str.lower().str.contains(keyword).astype(int)

    # Extract words from URL
    df['URL_words'] = df['URL'].apply(lambda x: url_to_words(x))

    if use_word2vec:
        if w2v_model is None:
            raise ValueError("Word2Vec model must be provided when use_word2vec is True")
        w2v_df = word_to_vec_transform(df, w2v_model=w2v_model)
        df = pd.concat([df, w2v_df], axis=1)
        vectorizer = None  # No need for a TF-IDF vectorizer if Word2Vec is used
    else:
        if vectorizer is None:
            vectorizer = TfidfVectorizer(stop_words='english', min_df=1, max_df=0.95)
            tfidf_matrix = vectorizer.fit_transform(df['URL_words'] + ' ' + df['University'].str.lower())
        else:
            tfidf_matrix = vectorizer.transform(df['URL_words'] + ' ' + df['University'].str.lower())

        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
        df = pd.concat([df, tfidf_df], axis=1)

    return df, vectorizer

def word_to_vec_transform(df, w2v_model=None, vector_size=100):
    if w2v_model is None:
        sentences = df['URL_words'].apply(lambda x: x.split()).tolist()
        w2v_model = Word2Vec(sentences, vector_size=vector_size, window=5, min_count=1)
    word_vectors = w2v_model.wv.key_to_index #w2v_model.wv
#     print(word_vectors)

    def vectorize_sentence(sentence):
        words = sentence.split()
        word_vecs = [word_vectors[word] for word in words if word in word_vectors]
        if len(word_vecs) > 0:
            return np.mean(word_vecs, axis=0)
        else:
            return np.zeros(vector_size)
    
    df_w2v = df['URL_words'].apply(lambda x: vectorize_sentence(x))
    return pd.DataFrame(df_w2v.tolist())

def tune_hyperparameters(model, X_train, y_train, param_grid):
    grid_search = GridSearchCV(model, param_grid, cv=10, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


# Function to load and use advanced models
def use_advanced_model(text, model_type='gpt'):
    if model_type == 'gpt':
        model = pipeline('text-generation', model='gpt-2')
    elif model_type == 'bert':
        model = pipeline('feature-extraction', model='bert-base-uncased')

    return model(text)


def initiate_models():
    # Define models with default parameters
    rf = RandomForestClassifier()
    lr = LogisticRegression()
    dt = DecisionTreeClassifier()
    knn = KNeighborsClassifier()
    svm = SVC()

    # Define hyperparameter grids for tuning
    param_grid_rf = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20],
    }

    param_grid_lr = {
        'C': [0.1, 1, 10],
    }

    param_grid_dt = {
        'max_depth': [5, 10],
        'min_samples_split': [2, 5],
    }

    param_grid_knn = {
        'n_neighbors': [5, 7],
        'weights': ['uniform', 'distance'],
    }

    param_grid_svm = {
        'C': [0.1, 1, 10],
        'degree': [3, 4]
    }

    # Tune each model using GridSearchCV
    tuned_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=10, n_jobs=-1)
    tuned_lr = GridSearchCV(estimator=lr, param_grid=param_grid_lr, cv=10, n_jobs=-1, scoring='accuracy')
    tuned_dt = GridSearchCV(estimator=dt, param_grid=param_grid_dt, cv=10, n_jobs=-1, scoring='accuracy')
    tuned_knn = GridSearchCV(estimator=knn, param_grid=param_grid_knn, cv=10, n_jobs=-1, scoring='accuracy')
    tuned_svm = GridSearchCV(estimator=svm, param_grid=param_grid_svm, cv=10, n_jobs=-1, scoring='accuracy')


    # Create a dictionary to store all models
    models = {
        'rf_tuned': tuned_rf,
        'dt_tuned': tuned_dt,
        'lr_tuned': tuned_lr,
        'knn_tuned': tuned_knn,
        'svm_tuned': tuned_svm,
        'gpt': None,  # Placeholder for GPT
        'bert': None   # Placeholder for BERT
    }

    return models

def train_models(df, model_name, model, use_word2vec=False):
#     df = df.reset_index(drop=True,inplace=True)
    df.to_csv('a.csv')
    from sklearn.tree import DecisionTreeClassifier
    # Preprocess the data
    if use_word2vec:
        df, _ = preprocess_data(df, use_word2vec=True)
    else:
        df, vectorizer = preprocess_data(df)

    print(df['Classification'].unique())
    
    df['Classification'] = df['Classification'].replace(['Erasmus/Exchange','Academic Calendar','Others'],[0,1,2])
    
    df.to_csv('a.csv')
    df = df.reset_index(drop=True)
    
    print(list(df))
    # Separate features (X) and target labels (y)
    X = df.drop(['Classification', 'URL', 'University', 'URL_words'], axis=1)
    y = df['Classification']

    X.to_csv('a1.csv')
    
    # Ensure all column names are strings
    X.columns = X.columns.astype(str)
    # Splitting the data into 80% training and 20% testing with stratified sampling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if isinstance(model, LogisticRegression):
        model.max_iter = 500  # Increase the number of iterations

    print(X_train.shape)
    print(y_train.shape)
    model = DecisionTreeClassifier()
    model.fit(X_train,y_train)
    best_model = model.best_estimator_
    # Perform cross-validation to evaluate the model
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    scoring = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score, average='macro'),
        'recall': make_scorer(recall_score, average='macro'),
        'f1': make_scorer(f1_score, average='macro')
    }
    
    print('*******')
    print(type(X_train),type(y_train))
    print('*******')
    cv_scores = cross_validate(best_model, X_train.values, y_train.values, cv=kf, scoring=scoring)

    # Print cross-validation results
    print(f'Mean accuracy score: {cv_scores["test_accuracy"].mean()}')
    print(f'Mean precision score: {cv_scores["test_precision"].mean()}')
    print(f'Mean recall score: {cv_scores["test_recall"].mean()}')
    print(f'Mean f1 score: {cv_scores["test_f1"].mean()}')

    # Fit the model on the entire dataset
#     model.fit(X, y)
    
    # Save the trained model to a pickle file
    with open(f'url_classification_{model_name}.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)

    # Save the TF-IDF vectorizer if used
    if not use_word2vec:
        with open('url_classification_tfidf_vec.pkl', 'wb') as vectorizer_file:
            pickle.dump(vectorizer, vectorizer_file)


def use_advanced_model(text, model_type='gpt'):
    if model_type == 'gpt':
        model = pipeline('text-generation', model='gpt-2')
    elif model_type == 'bert':
        model = pipeline('feature-extraction', model='bert-base-uncased')

    return model(text)


def predict_class_and_prob(model, vectorizer, university_name, url, use_word2vec=False, w2v_model=None):
    # Prepare the input data as a DataFrame
    if isinstance(university_name, list) and isinstance(url, list): 
        df_input = pd.DataFrame(columns=['University', 'URL'])
        df_input['University'] = university_name
        df_input['URL'] = url
    else:
        df_input = pd.DataFrame([[university_name, url]], columns=['University', 'URL'])

    # Preprocess the data
    df_input, _ = preprocess_data(df_input, vectorizer=vectorizer, use_word2vec=use_word2vec, w2v_model=w2v_model)

    # Drop non-feature columns
    X_input = df_input.drop(['University', 'URL', 'URL_words'], axis=1)

    # Ensure all column names are strings
    X_input.columns = X_input.columns.astype(str)

    # Handle missing features: Add missing columns with default value 0
    missing_features = set(model.best_estimator_.feature_names_in_) - set(X_input.columns)
    for feature in missing_features:
        X_input[feature] = 0

    # Reorder columns to match the order used during training
    X_input = X_input[model.best_estimator_.feature_names_in_]

    # Predict the class and its probability
    class_pred = model.predict(X_input)
    prob_pred = model.predict_proba(X_input)

    # Structure the output
    prob_preds_dict = {}
    for i, class_label in enumerate(model.classes_):
        prob_preds_dict[class_label] = prob_pred[:, i]
    prob_preds_dict['url'] = url
    
    return prob_preds_dict


def predict_class_and_prob_bert(model, tokenizer, university_name, url):
    # Prepare the input data as a DataFrame
    if isinstance(university_name, list) and isinstance(url, list): 
        df_input = pd.DataFrame(columns=['University', 'URL'])
        df_input['University'] = university_name
        df_input['URL'] = url
    else:
        df_input = pd.DataFrame([[university_name, url]], columns=['University', 'URL'])

    # Preprocess the data for BERT
    df_input = add_bert_mode.preprocess_data(df_input)

    # Extract the text input for BERT
    text_input = df_input['text_for_bert'].tolist()

    # Tokenize the input
    inputs = tokenizer(text_input, padding=True, truncation=True, return_tensors="pt", max_length=512)

    # Make predictions with BERT
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        class_pred = torch.argmax(probabilities, dim=1).tolist()

    # Structure the output
    prob_preds_dict = {}
    for i, class_label in enumerate(model.config.id2label.values()):
        prob_preds_dict[class_label] = probabilities[:, i].tolist()
    prob_preds_dict['url'] = url
    
    return prob_preds_dict


def load_model(model_name):
    with open(f'url_classification_{model_name}.pkl', 'rb') as file: 
        model = pickle.load(file)
    with open('word2vec_model.pkl', 'rb') as file: 
        vectorizer = pickle.load(file) 
    return model, vectorizer

def load_model_bert():

    # # Load the trained model and tokenizer
    model = BertForSequenceClassification.from_pretrained('./BERT')
    tokenizer = BertTokenizer.from_pretrained('./BERT')

    return model, tokenizer

def load_model_GPT():

    # # Load the trained model and tokenizer
    model = GPT2ForSequenceClassification.from_pretrained('./GPT')
    tokenizer = GPT2Tokenizer.from_pretrained('./GPT')

    return model, tokenizer


def scrape_google_search(university_name, search_type, model, vectorizer, uni_url,model_name = ''):
      
    erasmus_query_list = [
        f'Erasmus program {university_name}',
        f'{university_name} Erasmus exchange program',
        f'{university_name} Erasmus student mobility',
        f'Erasmus information {university_name}',
        f'{university_name} Erasmus office'
    ]

    calendar_query_list = [
        f'{university_name} academic calendar',
    ]

    if search_type == 'Erasmus/Exchange':
        query_list = erasmus_query_list
    else:
        query_list = calendar_query_list

    base_url = 'https://www.google.com/search'
    results = []
    start = 0
    pages = 1

    for query in query_list:
        for _ in range(pages):
            params = {
                'q': query,
                'start': start
            }
            response = requests.get(base_url, params=params, headers={'User-Agent': 'Mozilla/5.0'})
            soup = BeautifulSoup(response.text, 'html.parser')

            search_results = soup.find_all('a')
            for result in search_results:
                href = result.get('href')
                if href and '/url?q=' in href:
                    link = href.split('/url?q=')[1].split('&')[0]
                    # Filter out unwanted links based on specific keywords and types
                    if (
                        "google.com" not in link and
                        "youtube.com" not in link and
                        "/search" not in link and
                        "/imgres?" not in link and
                        "/shopping?" not in link and
                        uni_url in link
                    ):
                        results.append(unquote(link))
            start += 10
        
    records = []
    for url in results:
        if model_name == 'BERT' or model_name == 'GPT':
            records.append(predict_class_and_prob_bert(model, vectorizer, university_name= university_name, url = url))
        else:
            records.append(predict_class_and_prob(model, vectorizer=None, use_word2vec=True, w2v_model=vectorizer, university_name= university_name, url = url))
    result_df = pd.DataFrame(records)

# Debugging: Print out the columns and first few rows of result_df
    print("Columns in result_df:", result_df.columns)
    print(result_df.head())

    if search_type == 'Erasmus/Exchange':
        try:
            result_df = result_df[['url', 'Erasmus/Exchange']]
        except KeyError as e:
            print(f"Error: {e}. Available columns: {result_df.columns}")
            return
        result_df.columns = ['url', 'score']
    elif search_type == 'Academic Calendar':
        try:
            result_df = result_df[['url', 'Academic Calendar']]
        except KeyError as e:
            print(f"Error: {e}. Available columns: {result_df.columns}")
            return
        result_df.columns = ['url', 'score']

    result_df.score = result_df.score.apply(lambda x: x[0])
    result_df = result_df[result_df['score'] > 0.6].drop_duplicates(['url'])
    print(' '.join([f'<a href={i}>{i}</a><br>' for i in result_df.sort_values(by='score', ascending=False)[:10]['url'].to_list()]))

def scrape_website(university_name, search_type, model, ml_method, vectorizer, uni_url,model_name = ''):
    
    
    def get_prediction(url, ml_method):
        if model_name == 'BERT' or model_name == 'GPT':
            proba = predict_class_and_prob_bert(model, vectorizer, university_name, url)
        else: 
            proba = predict_class_and_prob(model, university_name= university_name, url= url,vectorizer=None,use_word2vec=True, w2v_model=vectorizer)
        thresh = 0.6
        if ml_method == "Random Forest":
            if search_type=='Erasmus/Exchange':
                thresh = 0.9
            else:
                thresh = 0.9
        if ml_method == "K-Nearest Neighbours":
            if search_type=='Erasmus/Exchange':
                thresh = 0.9
            else:
                thresh = 0.9
        if ml_method == "Support Vector Machine":
            if search_type=='Erasmus/Exchange':
                thresh = 0.85
            else:
                thresh = 0.9
            
        if proba[search_type][0] > thresh:
            return True, proba[search_type]
        else:
            return False, proba[search_type]
    
    options = Options()
    options.page_load_strategy = 'eager'
    driver=webdriver.Chrome(options=options)
    
    base_url = uni_url
    visited_urls = set()
    base_domain = urlparse(base_url).netloc
    url_queue = [base_url]
    driver.get(base_url)

    stop_flag = False
    result_df = pd.DataFrame(columns=['preference', 'url', 'score'])
    
    response = requests.get(base_url+'sitemap.xml', headers={'User-Agent': 'Mozilla/5.0'})
    url_pattern = r'https?://[^\s<]+'

    # Find all matches
    sitemap_urls = [i for i in re.findall(url_pattern, response.text) if base_url in i]
    if sitemap_urls:
        print("sitemapppppppppppppppppp")

        if model_name == 'BERT' or model_name == 'GPT':
            sitemap_df = pd.DataFrame(predict_class_and_prob_bert(model,vectorizer, university_name=[university_name]*len(sitemap_urls), url=sitemap_urls))
        else: 
            sitemap_df = pd.DataFrame(predict_class_and_prob(model,vectorizer=None,use_word2vec=True, w2v_model=vectorizer, university_name=[university_name]*len(sitemap_urls), url=sitemap_urls))


        print("sitemapppppppppppppppppp423421413")
        if search_type == 'Erasmus/Exchange':
            sitemap_df = sitemap_df[['url', 'Erasmus/Exchange']]
            sitemap_df.columns = ['url', 'score']
        elif search_type == 'Academic Calendar':
            sitemap_df = sitemap_df[['url', 'Academic Calendar']]
            sitemap_df.columns = ['url', 'score']

        #sitemap_df = sitemap_df[sitemap_df['score']>0.6].drop_duplicates(['url'])
        if search_type=='Erasmus/Exchange':
            print(' '.join([f'<a href={i}>{i}</a><br>' for i in sitemap_df.sort_values(by='score', ascending=False)['url'].to_list() if any([j in i.lower() for j in ['erasmus', 'exchange']])][:10]))
        else:
            print(' '.join([f'<a href={i}>{i}</a><br>' for i in sitemap_df.sort_values(by='score', ascending=False)['url'].to_list() if any([j in i.lower() for j in ['academic', 'calendar']])][:10]))
    
    while url_queue:
        current_url = url_queue.pop(0)
    
        if current_url in visited_urls:
            continue
        visited_urls.add(current_url)

        try:
            driver.get(current_url)
            links = driver.find_elements(By.TAG_NAME, 'a')
            for link in links:
                href = link.get_attribute('href')
                if href and urlparse(href).netloc == base_domain:
                    if href not in visited_urls and href not in url_queue and not any(ext in href for ext in ['jpg', 'png', 'pdf', 'ics', '#', '?']):
                        i,j = get_prediction(href, ml_method)
                        result_df = pd.concat([result_df, pd.DataFrame([{'preference': 'b', 'url': href, 'score': j}])])
                        url_queue.append(href)
                        if len(visited_urls)>=10:
                            stop_flag=True
                            break
        except Exception as e:
            print(f"Error processing URL: {href}, Error: {e}")

        if stop_flag:
            break
       
    result_df.score = result_df.score.apply(lambda x: x[0])
    result_df = result_df.drop_duplicates(['url'])
    print(' '.join([f'<a href={i}>{i}</a><br>' for i in result_df.sort_values(by='score', ascending=False)[:10]['url'].to_list()]))
