# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 17:34:18 2021

@author: Victor Ponce-Lopez @ UCL Energy Institute

==============================================================================
Class def to load a Simple dataset
==============================================================================
"""

from transformers import DistilBertForSequenceClassification, Trainer, DistilBertTokenizerFast

class ModelData:
    def __init__(self, tokenized_texts):
        self.tokenized_texts = tokenized_texts
    
    def __len__(self):
        return len(self.tokenized_texts["input_ids"])
    
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.tokenized_texts.items()}
    
    def getDeepModels():
        trainers = []
        model_paths = ['distilBert_multidisaster','distilBert_multihumanstand',
                       'distilBert_multimedical','distilBert_Unifisevere']
        
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        
        for model_path in model_paths:    
            try: 
                print("Loading Fine-Tuned BERT pretrained model "+model_path+" ...")
                if model_path == 'distilBert_multidisaster': model_path += '/checkpoint-31600'
                elif model_path == 'distilBert_multihumanstand': model_path += '/checkpoint-2100'
                elif model_path == 'distilBert_multimedical': model_path += '/checkpoint-1900'
                elif model_path == 'distilBert_Unifisevere': model_path += '/checkpoint-1'            
                model = DistilBertForSequenceClassification.from_pretrained('../../'+model_path, return_dict=True)
                trainers.append(Trainer(model=model))
            except:
                print("The model {} could not be loaded. Please ensure paths are correct".format(model_path))
            
        return trainers, tokenizer