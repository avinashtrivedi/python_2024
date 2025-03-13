# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 17:16:38 2020

@author: Victor Ponce-Lopez @ UCL Energy Institute

===================================================================
Utils for text data analysis and visualisation
===================================================================
"""


try:
    import _pickle as pickle
except:
    import pickle
import pandas as pd
import numpy as np
import nltk
nltk.download('words')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from bs4 import BeautifulSoup
import re, random
import preprocessor as p
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from itertools import islice


###############################################################################
# Get info about data filenames, headers to drop and rename
def getDataInfo(dataset):    
    if dataset == 'socialmedia_disaster':
        filenames = ['socialmedia-disaster-tweets-DFE.csv']
        headerDropper = ['_golden', '_unit_id', '_unit_state','_trusted_judgments','_last_judgment_at','choose_one:confidence','choose_one_gold','keyword','location','tweetid','userid']
        renamedColumns = {'choose_one': 'labels', 'text': 'message'}
    elif dataset == 'multilingual_disaster':
        filenames = ['disaster_response_messages_training.csv',
                 'disaster_response_messages_validation.csv',
                 'disaster_response_messages_test.csv',]
        headerDropper = ['ï»¿id', 'split', 'original','genre','PII','request','offer','aid_related',
                       'medical_help','medical_products','search_and_rescue','security','military',
                       'child_alone','water','food','shelter','clothing','money','missing_people',
                       'refugees','death','other_aid','infrastructure_related','transport',
                       'buildings','electricity','tools','hospitals','shops','aid_centers',
                       'other_infrastructure','weather_related','storm','fire',
                       'earthquake','cold','other_weather','direct_report','weather_related',
                       'floods']
        renamedColumns = {'related': 'label'}    
    elif dataset == "UnifiedMETCEHFloodsUK":
        filenames = ['Stevens2016_Met_CEH_Dataset_Public_Version.csv']
        headerDropper = ['CID','ID','Year','Month','Day','Place(s) Affected','Source',
                         'Flood_Type','Flood','Dataset','England','Anglian','EA_Wales',
                         'Midlands','North-East','North-West','NI','Scotland',
                         'South-East','South-West','Unspecified']
        renamedColumns = {'Description': 'message'}
    return filenames, headerDropper, renamedColumns


###############################################################################
# Prepare data and splits for the given dataset
def prepareData(path, dataset, pos_ratio, targets):
    
    filenames, headerDropper, renamedColumns = getDataInfo(dataset)
    
    data = pd.DataFrame()
    trainData, valData, testData = list(), list(), list()
    for filename in filenames:
        print('Reading file '+filename+' ...')
        if dataset in ['UnifiedMETCEHFloodsUK']:
            tweets = pd.read_csv(path+filename, header=1, encoding = 'latin-1')
        else:
            tweets = pd.read_csv(path+filename, encoding = 'latin-1')
            
        if dataset == 'multilingual_disaster' and 'disaster' not in targets:
            tweets = tweets.loc[tweets['related'] == 1]
            idx = pd.Int64Index([])
            for target in targets:
                idx = idx.union(tweets.loc[tweets[target] == 1].index)
            tweets['related'] = 0
            tweets['related'].loc[idx] = 1
            
        tweets.drop(headerDropper, axis = 1, inplace = True)
        tweets.rename(columns = renamedColumns, inplace = True)
        
        if dataset == 'socialmedia_disaster':
            tweets['label'] = tweets['labels'].map({'Relevant': 1, 'Not Relevant': 0})
            tweets.drop(['labels'], axis = 1, inplace = True)
        elif dataset == 'UnifiedMETCEHFloodsUK':
            tweets = tweets.loc[~tweets['message'].isnull()]
            tweets['label'] = np.nan
            tweets['label'].loc[~tweets['2'].isnull() + ~tweets['3'].isnull()] = 1
            tweets.drop(['1','2','3'], axis = 1, inplace = True)
            
        # Correction of missing or nan labels
        if 'label' in tweets.columns:
            idx_miss = [i for i, e in enumerate(tweets['label'].to_list()) if e>1 or e<0]
            tweets['label'].iloc[idx_miss] = 0
            idx_nan = [i for i,e in enumerate(pd.to_numeric(tweets['label']).isnull()) if e]
            tweets['label'].iloc[idx_nan] = 0
            tweets['label'] = pd.to_numeric(tweets['label']).astype(int)
            
        if pos_ratio > 0 and dataset in ['multilingual_disaster','UnifiedMETCEHFloodsUK'] and any([True for x in ['floods','severe'] if x in targets]):
            # Delete random training data with negative samples for unbalanced datasets - positive ratio < pos_ratio (default: 0.45)
            print("Deleting random negative samples at positive ratio :", pos_ratio)
            neg_delete = len(tweets[tweets['label']==0]) - int(len(tweets[tweets['label']==1]) / pos_ratio)
            if np.sign(neg_delete) > 0:
                neg_randIdx = random.sample(tweets[tweets['label']==0].index.to_list(), neg_delete)
                tweets = tweets.drop(index=neg_randIdx)
                tweets.reset_index(inplace = True)
                tweets.drop(['index'], axis = 1, inplace = True)
        else: 
            pos_ratio = 0
        
        # Cleaning the data
        words = set(nltk.corpus.words.words())
        for index, row in tweets.iterrows():
            tweets.at[index, 'original'] = tweets.at[index, 'message']
            tweets.at[index, 'message'] = p.clean(str(row['message']))
            tweets.at[index, 'message'] = BeautifulSoup(row['message'], 'html.parser')
            tweets.at[index, 'message'] = re.sub(r'@[A-Za-z0-9]+','',str(row['message']))
            #tweets.at[index, 'message'] = re.sub('https?://[A-Za-z0-9./]+','',str(row['message']))
            tweets.at[index, 'message'] = re.sub(r'^https?:\/\/.*[\r\n]*','',str(row['message']), flags=re.MULTILINE)
            tweets.at[index, 'message'] = re.sub("[^a-zA-Z]", " ", str(row['message']))
        
        if dataset == 'multilingual_disaster':
            if filename == filenames[0]:
                trainData = tweets
            elif filename == filenames[1]:
                valData = tweets
            elif filename == filenames[2]:
                testData = tweets
        else:
            data = data.append(tweets)
            data.reset_index(inplace = True)
            data.drop(['index'], axis = 1, inplace = True)
        del tweets
        
    ###########################################################################
    # Split into training, and test if required by the dataset
    if dataset in ['socialmedia_disaster', 'UnifiedMETCEHFloodsUK']:
        print("Creating random Training (60%) and Test (40%) sets ...")
        trainIndex, testIndex = list(), list()
        for i in range(data.shape[0]):
            if np.random.uniform(0, 1) < 0.6:
                trainIndex += [i]
            else:
                testIndex += [i]                
        valIndex = [i for i in trainIndex if np.random.uniform(0, 1) < 0.4]
        valData = data.loc[valIndex]
        trainData = data.loc[[x for x in trainIndex if x not in valIndex]]
        testData = data.loc[testIndex]
        valData.reset_index(inplace = True)
        valData.drop(['index'], axis = 1, inplace = True)
        trainData.reset_index(inplace = True)
        trainData.drop(['index'], axis = 1, inplace = True)
        testData.reset_index(inplace = True)
        testData.drop(['index'], axis = 1, inplace = True)
        testData.head()
        
    return trainData, valData, testData, data, pos_ratio




###############################################################################
# Show tweet data
def showWords(trainData, valData, testData, dataset, data):
    TotalTweets = pd.concat([trainData,valData,testData]) if dataset == 'multilingual_disaster' else data 
    undersiredWords = ['co','https t','D D','D RT','t co','https co','amp','don','re','s','u','m','gt gt','t','C','will','idioti','still']
    # Show WordCloud
    label_words = ' '.join(list(TotalTweets[TotalTweets['label'] == 1]['message']))
    label_words = WordCloud(width = 512,height = 512).generate(label_words)
    plt.figure(figsize = (10, 8), facecolor = 'k')
    plt.imshow(label_words); plt.axis('off'); plt.tight_layout(pad = 0)
    plt.show(); plt.close()
    # Show Histogram
    sortedWords = list(islice(label_words.words_.items(), 15))
    x, y = [i[0] for i in sortedWords if i[0] not in undersiredWords], [i[1] for i in sortedWords if i[0] not in undersiredWords]
    plt.bar(x, np.array(y)*100); plt.ylabel('Ocurrences (%)'); plt.xticks(x, rotation=80)
    plt.show(); plt.close()
    
    normal_words = ' '.join(list(TotalTweets[TotalTweets['label'] == 0]['message']))
    
    # Show WordCloud
    normal_wc = WordCloud(width = 512,height = 512).generate(normal_words)
    plt.figure(figsize = (10, 8), facecolor = 'k')
    plt.imshow(normal_wc); plt.axis('off'); plt.tight_layout(pad = 0)
    plt.show(); plt.close()
    # Show Histogram
    sortedWords = list(islice(normal_wc.words_.items(), 15))
    x, y = [i[0] for i in sortedWords if i[0] not in undersiredWords], [i[1] for i in sortedWords if i[0] not in undersiredWords]
    plt.bar(x, np.array(y)*100); plt.ylabel('Ocurrences (%)'); plt.xticks(x, rotation=80)
    plt.show(); plt.close()



###############################################################################
# Save Indices Prediction Results 
def saveResults(idx_target, idx_sent, dataset):
    results = {'idx_bow': idx_target[0], 'idx_tfidf': idx_target[1], 
               'idx_distBert': idx_target[2], 'idx_pos_sent': idx_sent[0],
               'idxFail': idx_sent[1]}
    with open('results_'+dataset+'.pkl','wb') as outfile:
        print("Saving indices of prediction results for "+dataset+" ...")
        pickle.dump(results, outfile)
    print("Done!")
