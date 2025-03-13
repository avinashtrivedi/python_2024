# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 17:16:38 2021

@author: Victor Ponce-Lopez @ UCL Energy Institute

==============================================================================
Class def to load multiclass model
==============================================================================
"""
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

class Lemmer:
    def __init__(self, tokenize):
        self.tokenize = tokenize
    
    # Tokenizing lowercase   
    def tokenize(text):
    
        tokens = word_tokenize(text)
        lemmatizer = WordNetLemmatizer()
    
        clean_tokens = []
        for tok in tokens:
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_tok)    
        return clean_tokens
    
