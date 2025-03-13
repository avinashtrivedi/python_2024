# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 12:34:34 2020

@author: Victor Ponce-Lopez @ UCL Energy Institute

===================================================================
Filtering, Text and Sentiment Analysis, and Topic Classification
===================================================================
 """
if __name__ == '__main__':
    
    from utils import prepareData, showWords, saveResults
    from binary import classifCat, classifySentiment
    from multiclass import LearnInferTopics    
    #%matplotlib inline

    # Set parameters and import data
    path = 'data/'
    datasets = ['socialmedia_disaster','multilingual_disaster','UnifiedMETCEHFloodsUK']
    dataset = datasets[1]
    bMethods = ['bow', 'tfidf', 'distBert', 'all']; bMethod = bMethods[2]
    mMethods = ['lda-bow', 'lda-tfidf', 'all']; mMethod = mMethods[2]
    targets = ['severe']  #['aid_related','medical_products','other_aid','hospitals','aid_centers']  #['water','food','shelter','medical_help']  #['disaster']  #['search_and_rescue','missing_people']  #['floods']
    save, show, retrain, pos_ratio = False, True, False, 0  #(default for floods: 0.45)
    
    trainData, valData, testData, data, pos_ratio = prepareData(path, dataset, pos_ratio, targets)
    
    ###########################################################################
    # Show tweet data
    if show: showWords(trainData, valData, testData, dataset, data)
    
    # Assign default models depending on the target categories

    # use model of target category trained on specific dataset
    if targets == ['disaster']: 
        modeldata = dataset[:5]+'disaster'
    elif targets == ['aid_related','medical_products','other_aid','hospitals','aid_centers']: modeldata = dataset[:5]+'medical'
    elif targets == ['water','food','shelter','medical_help']: modeldata = dataset[:5]+'humanstand'
    elif targets == ['severe']: modeldata = dataset[:5]+'severe'
    elif targets == ['floods']: modeldata = dataset[:5]+'floods'
        
    
    ###########################################################################
    # Classify Target Category and get indexes
    idx_bow, idx_tfidf, idx_distBert = classifCat(trainData, valData, testData, 
                                                  retrain, dataset, modeldata, 
                                                  save, pos_ratio, bMethod, show)
    ###########################################################################
    
