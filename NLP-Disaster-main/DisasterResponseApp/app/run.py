# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 09:16:38 2021

@author: Victor Ponce-Lopez @ UCL Energy Institute

==============================================================================
Visualisation Web Interface for Disaster Response Classification for Filtering
==============================================================================
"""

import json, joblib, plotly
try:
    import _pickle as pickle
except:
    import pickle
import numpy as np
import pandas as pd


from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar, Pie

from class_def import Lemmer
from deepclass_def import ModelData

#import torch

# load data
df_ = pd.read_csv('../data/MDRM-lite.csv', low_memory=False)
df = df_.drop('related',axis=1)
df2 = pd.read_csv('../data/Stevens2016_Met_CEH_Dataset_Public_Version.csv', header=1, encoding = 'latin-1', low_memory=False)
df2 = df2.loc[~df2['Description'].isnull()]
df2['label'] = np.nan
df2['label'].loc[~df2['2'].isnull() | ~df2['3'].isnull()] = 1
df2.drop(['1','2','3'], axis = 1, inplace = True)
df2['label'].iloc[[i for i,e in enumerate(pd.to_numeric(df2['label']).isnull()) if e]] = 0
df2['label'] = pd.to_numeric(df2['label']).astype(int)

   
# load models
model_multi = joblib.load("../model/DS_model_SVC.pkl")
deep_models, tokenizer = ModelData.getDeepModels()

app = Flask(__name__)

# index webpage displays visuals and receives user input text for model_multi
@app.route('/')
@app.route('/index')
def index():
        
    # assign data needed for visualisations and create graphs 
    ## Disaster category
    label_sums = pd.Series({'Disaster': df_.iloc[:, 4].sum(), 'Non Disaster': len(df_)-df_.iloc[:, 4].sum()})
    label_names = list(label_sums.index)
    graphs = [{'data': [Bar(x=label_names, y=label_sums)],
               'layout': {'title': {'text':'Distribution of Disaster Category'},
                          'yaxis': {'title': "Count"},'xaxis': {'title': "Category"}}}]
    ## genre groups
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    graphs.append({'data': [Bar(x=genre_names, y=genre_counts)],
                   'layout': {'title': 'Distribution of Message Genres',
                              'yaxis': {'title': "Count"},'xaxis': {'title': "Genre"}}})
    ## and all categories
    label_sums = df.iloc[:, 4:].sum()
    label_names = list(label_sums.index)
    graphs.append({'data': [Pie(labels=label_names, values=label_sums)],
                   'layout': {'title': {'text':'Distribution of Disaster Categories','y': 0.95},
                              'yaxis': {'title': "Count"},'xaxis': {'title': "Category"}}})                            
    ## humanitarian standard and medical categories
    label_sums = pd.Series({'Humanitarian Standards': df.iloc[:, [6, 8, 20, 26, 28]].sum().sum(), 
                  'Medical': df.iloc[:, [7, 12, 13, 14]].sum().sum()})
    label_names = list(label_sums.index)
    graphs.append({'data': [Pie(labels=label_names, values=label_sums)],
                   'layout': {'title': {'text':'Distribution of Humanitarian Standards and Medical Categories'},
                              'yaxis': {'title': "Count"},'xaxis': {'title': "Category"}}})
    ## Severity category
    label_sums = pd.Series({'Severe': df2.iloc[:, 22].sum(), 'Mild': len(df2)-df2.iloc[:, 22].sum()})
    label_names = list(label_sums.index)
    graphs.append({'data': [Bar(x=label_names, y=label_sums)],
               'layout': {'title': {'text':'Distribution of Severity Category from the "UnifiedCEHMET" dataset'},
                          'yaxis': {'title': "Count"},'xaxis': {'title': "Category"}}})
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays deep_models results
@app.route('/multi')
def goMulti():
    # save user input in query
    query = request.args.get('query', '') 

    # use model_multi to predict classification for query
    classification_labels = model_multi.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels[1:]))

    # This will render the go.html Please see that file. 
    return render_template(
        'multi.html',
        query=query,
        classification_result=classification_results
    )

# web page that handles user query and displays model_multi results
@app.route('/deep')
def goDeep():
    # save user input in query
    query = request.args.get('query', '') 

    test_encodings = tokenizer([query], truncation=True, padding=True)
    test_dataset = ModelData(test_encodings)
        
    # use deep_models to predict classifications for query
    classification_scores, classification_labels = [],[]
    for trainer in deep_models:
        trainer.model.eval()        
        preds = trainer.predict(test_dataset)[0]
        softmax_score = (np.exp(preds[0])/np.sum(np.exp(preds[0])))
        score = np.argmax(softmax_score[0])
        
        ## Verify calculation using Pytorch tensors in local GPU device (requires torch import)
        # pt_inputs = {k: torch.tensor(v).to(torch.device("cuda")) for k, v in test_encodings.items()}
        # with torch.no_grad():
        #     output = trainer.model(**pt_inputs)
        # softmax_score = torch.nn.functional.softmax(output.logits[0],dim=-1)
        # score = torch.argmax(softmax_score).cpu().numpy()
        # softmax_score = softmax_score.cpu().numpy()
        # print(softmax_score, score)
        
        classification_scores.append(softmax_score[1])
        classification_labels.append(score.item())
    
    categories = ['Disaster', 'Humanitarian Standards', 'Medical', 'Severe']
    classification_results = dict(zip(categories, classification_scores))

    # This will render the go.html Please see that file. 
    return render_template(
        'deep.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(debug=True)

if __name__ == '__main__':
    main()