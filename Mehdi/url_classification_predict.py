# +
import warnings
warnings.filterwarnings('ignore')
import sys
# import url_classification_utility
import pandas as pd
import pickle
import os

def load_model(model_name):
    with open(f'url_classification_{model_name}.pkl', 'rb') as file: 
        model = pickle.load(file)
    with open('word2vec_model.pkl', 'rb') as file: 
        vectorizer = pickle.load(file) 
    return model, vectorizer


# -

university_name = sys.argv[1]
search_type = sys.argv[2]
scraping_method = sys.argv[3]
ml_method = sys.argv[4]
uni_url = sys.argv[5]

# +
# w2v_model = url_classification_utility.load_word2vec_model()
# -

if ml_method == 'Random Forest':
    model, vectorizer = load_model('rf_tuned')
    best_params = model.get_params()
    d ={k:best_params[k] for k in best_params if k in ('n_estimators','max_depth')}
    print('---Random Forest---')
    print(d)
if ml_method == "K-Nearest Neighbours":
    model, vectorizer =load_model('knn_tuned') 
    best_params = model.get_params()
    d ={k:best_params[k] for k in best_params if k in ('n_neighbors','weights')}
    print('---K-Nearest Neighbours---')
    print(d)
if ml_method == "Logistic Regression":
    model, vectorizer = load_model('lr_tuned') 
    best_params = model.get_params()
    d ={k:best_params[k] for k in best_params if k in ('C')}
    print('---Logistic Regression---')
    print(d)
if ml_method == "Support Vector Machine":
    model, vectorizer = load_model('svm_tuned') 
    best_params = model.get_params()
    d ={k:best_params[k] for k in best_params if k in ('degree','C')}
    print('---Support Vector Machine---')
    print(d)
if ml_method == "Decision Tree":
    model, vectorizer = load_model('dt_tuned') 
    best_params = model.get_params()
    d ={k:best_params[k] for k in best_params if k in ('max_depth','min_samples_split')}
    print('---Decision Tree---')
    print(d)
if ml_method == "BERT":
    model, vectorizer = load_model_bert()

if ml_method == "GPT":
    model, vectorizer = url_classification_utility.load_model_GPT()


if ml_method == "BERT":
    if scraping_method=='Google Search Scraping':
        url_classification_utility.scrape_google_search(university_name, search_type, model, vectorizer, uni_url,model_name = "BERT")
    else:
        url_classification_utility.scrape_website(university_name, search_type, model, ml_method, vectorizer, uni_url,model_name = "BERT")
elif ml_method == "GPT":
    if scraping_method=='Google Search Scraping':
        url_classification_utility.scrape_google_search(university_name, search_type, model, vectorizer, uni_url,model_name = "GPT")
    else:
        url_classification_utility.scrape_website(university_name, search_type, model, ml_method, vectorizer, uni_url,model_name = "GPT")
else:
    if scraping_method=='Google Search Scraping':
        url_classification_utility.scrape_google_search(university_name, search_type, model, w2v_model, uni_url,model_name = "")
    else:
        url_classification_utility.scrape_website(university_name, search_type, model, ml_method, w2v_model, uni_url,model_name = "")
