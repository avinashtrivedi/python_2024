# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 21:00:28 2020

@author: Victor Ponce-Lopez @ UCL Energy Institute

===================================================================
Binary classification methods
===================================================================
"""

try:
    import _pickle as pickle
except:
    import pickle
from DisasterClassifier import DisasterClassifier
import numpy as np    
from utils import showWords
from transformers import pipeline
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments, DistilBertTokenizerFast
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch, random
import matplotlib.pyplot as plt
    

# Torch Dataset Format
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
        
###############################################################################
# classify target category using TF-IDF and BoW models
def classifCat(trainData, valData, testData, retrain, dataset, modeldata, save, 
               pos_ratio, method, showRes):
    
    idxClass_bow, idxClass_tfidf, idxClass_distBert = [], [], []
    
    # BoW
    if method == 'bow' or method == 'all':
        # Train BoW with Training data only
        try: 
            with open('model_BoW_'+str(modeldata)+'_'+str(pos_ratio)+'.pkl','rb') as infile:
                print("Loading BoW "+modeldata+" model for "+dataset+" ...")
                model_bow = pickle.load(infile)
        except:
            print("Training BoW "+modeldata+" model for "+dataset+" ...")
            model_bow = DisasterClassifier(trainData, 'bow')
            model_bow.train()
            if save:
                with open('model_BoW_'+str(modeldata)+'_'+str(pos_ratio)+'.pkl','wb') as outfile:
                    pickle.dump(model_bow, outfile)
        if dataset == 'multilingual_disaster':
            print("Predicting BoW labels on Validation data ...")
            preds_bow = model_bow.predict(valData['message'])
            print("Calculating results on Validation data ...")
            print(calcMetrics(valData['label'].values, preds_bow))
        print("Predicting BoW labels on Test data ...")
        preds_bow = model_bow.predict(testData['message'])
        if dataset in ['socialmedia_disaster','multilingual_disaster','UnifiedMETCEHFloodsUK']:
            print("Calculating results on Test data ...")
            print(calcMetrics(testData['label'].values, preds_bow))
        
        # Train BoW with all learning data 
        if dataset == 'multilingual_disaster':
            try: 
                with open('model_BoW_'+str(modeldata)+'_full_'+str(pos_ratio)+'.pkl','rb') as infile:
                    print("Loading full BoW "+modeldata+" model for "+dataset+" ...")
                    model_bow = pickle.load(infile)
            except:
                print("Training full BoW "+modeldata+" model for "+dataset+" ...")
                model_bow = DisasterClassifier(trainData.append(valData, ignore_index=True), 'bow')
                model_bow.train()
                if save:
                    with open('model_BoW_'+str(modeldata)+'_full_'+str(pos_ratio)+'.pkl','wb') as outfile:
                        pickle.dump(model_bow, outfile)
            print("Predicting BoW labels on Test data ...")
            preds_bow = model_bow.predict(testData['message'])
            print("Calculating BoW results on Test data ...")
            print(calcMetrics(testData['label'].values, preds_bow))
        
        idxClass_bow = [i for i, e in enumerate(list(preds_bow.values())) if e == 1]
        if showRes:
            print("Examples of 5 out of {} messages detected:".format(len(idxClass_bow)))
            print(testData.loc[random.sample(idxClass_bow,5)]['original'].values)
            showWords(False, False, False, dataset, testData.loc[idxClass_bow], True)
        #######################################################################
        print("\n")
    
    # TF-IDF
    if method == 'tfidf' or method == 'all': 
        # Train TF-IDF with Training data only
        try: 
            with open('model_TF-IDF_'+str(modeldata)+'_'+str(pos_ratio)+'.pkl','rb') as infile:
                print("Loading TF-IDF "+modeldata+" model for "+dataset+" ...")
                model_tfidf = pickle.load(infile)
        except:
            print("Training TF-IDF "+modeldata+" model for "+dataset+" ...")
            model_tfidf = DisasterClassifier(trainData, 'tf-idf')
            model_tfidf.train()
            if save:
                with open('model_TF-IDF_'+str(modeldata)+'_'+str(pos_ratio)+'.pkl','wb') as outfile:
                    pickle.dump(model_tfidf, outfile)
                    
        if dataset == 'multilingual_disaster':
            print("Prediting TF-IDF labels on Validation data ...")
            preds_tf_idf = model_tfidf.predict(valData['message'])
            print("Calculating results on Validation data ...")
            print(calcMetrics(valData['label'].values, preds_tf_idf))
        print("Prediting TF-IDF labels on Test data ...")
        preds_tf_idf = model_tfidf.predict(testData['message'])
        if dataset in ['socialmedia_disaster','multilingual_disaster','UnifiedMETCEHFloodsUK']:
            print("Calculating results on Test data ...")
            print(calcMetrics(testData['label'].values, preds_tf_idf))
        
        # Train TF-IDF with all learning data 
        if dataset == 'multilingual_disaster':
            try: 
                with open('model_TF-IDF_'+str(modeldata)+'_full_'+str(pos_ratio)+'.pkl','rb') as infile:
                    print("Loading full TF-IDF "+modeldata+" model for "+dataset+" ...")
                    model_tfidf = pickle.load(infile)
            except:
                print("Training full TF-IDF "+modeldata+" model for "+dataset+" ...")
                model_tfidf = DisasterClassifier(trainData.append(valData, ignore_index=True), 'tf-idf')
                model_tfidf.train()
                if save:
                    with open('model_TF-IDF_'+str(modeldata)+'_full_'+str(pos_ratio)+'.pkl','wb') as outfile:
                        pickle.dump(model_tfidf, outfile)
            print("Prediting TF-IDF labels on Test data ...")
            preds_tf_idf = model_tfidf.predict(testData['message'])
            print("Calculating results on Test data ...")
            print(calcMetrics(testData['label'].values, preds_tf_idf))
        idxClass_tfidf = [i for i, e in enumerate(list(preds_tf_idf.values())) if e == 1]
        if showRes:
            print("Examples of 5 out of {} messages detected:".format(len(idxClass_tfidf)))
            print(testData.loc[random.sample(idxClass_tfidf,5)]['original'].values)
            showWords(False, False, False, dataset, testData.loc[idxClass_tfidf], True)
        #######################################################################
        print("\n")
    
    # Fine-tuned DistilBERT 
    torch.random.manual_seed(4)        # Set a manual seed for our experiments
    if method == 'distBert' or method == 'all':
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        train_dataset, val_dataset = [], []
        if len(trainData) > 0:
            train_encodings = tokenizer(trainData['message'].to_list(), truncation=True, padding=True)
            train_dataset = Dataset(train_encodings, trainData['label'].to_list())
            if len(valData) > 0: 
                val_encodings = tokenizer(valData['message'].to_list(), truncation=True, padding=True)
                val_dataset = Dataset(val_encodings, valData['label'].to_list())
        test_encodings = tokenizer(testData['message'].to_list(), truncation=True, padding=True)
        testLabels = np.zeros(shape=(len(testData)), dtype=int).tolist() if 'label' not in testData.columns else testData['label'].to_list()
        test_dataset = Dataset(test_encodings, testLabels)
        
        training_args = TrainingArguments(
            output_dir='./distilBert_'+str(modeldata),   # output directory
            overwrite_output_dir=True,       # Overwrite checkpoint
            num_train_epochs=10,              # total number of training epochs
            per_device_train_batch_size=4,   # batch size per device during training
            per_device_eval_batch_size=8,    # batch size for evaluation
            warmup_steps=500,                # number of warmup steps for learning rate scheduler
            weight_decay=0.01,               # strength of weight decay
            logging_dir='./logs',            # directory for storing logs
            logging_steps=100,               # Number of update steps between two logs
            evaluation_strategy = "steps",   # Monitor both the training and evaluation losses
            save_steps=10000,                # Number of updates steps before two checkpoint saves
        )         
        try: 
            print("Loading Fine Tuned BERT "+modeldata+" pretrained model for "+dataset+" ...")
            trainer = DistilBertForSequenceClassification.from_pretrained(training_args.output_dir+'/checkpoint-1')
            model = DistilBertForSequenceClassification.from_pretrained(training_args.output_dir+'/checkpoint-1', return_dict=True)
        except:
            model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", return_dict=True)
            retrain = True
           
        trainer = Trainer(
            model=model,                         # the instantiated 🤗 Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=val_dataset              # evaluation dataset
        )
        if retrain:
            print("Fine Tunning BERT model for "+dataset+" ...")
            trainer.train()
            if save:
                trainer.save_model(training_args.output_dir)     
        if showRes and len(valData) > 0: print(trainer.evaluate())
        print("Prediting BERT labels on Test data ...")
        
        preds_distBert = trainer.predict(test_dataset)
        if len(train_dataset) > 0:
            print("Calculating results on Test data ...")
            print(compute_metrics(preds_distBert))
        idxClass_distBert = [i for i, e in enumerate(preds_distBert.predictions.argmax(-1).tolist()) if e == 1]
        if showRes:
            print("Examples of 5 out of {} messages detected:".format(len(idxClass_distBert)))
            print(testData.loc[random.sample(idxClass_distBert,5)]['original'].values)
            showWords(False, False, False, dataset, testData.loc[idxClass_distBert], True)
            
        #######################################################################
    
    return idxClass_bow, idxClass_tfidf, idxClass_distBert

###############################################################################
# Compute Metrics BERT
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': acc,
    }

###############################################################################
# Calculate Metrics
def calcMetrics(labels, predictions):
    true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
    for i in range(len(labels)):
        true_pos += int(labels[i] == 1 and predictions[i] == 1)
        true_neg += int(labels[i] == 0 and predictions[i] == 0)
        false_pos += int(labels[i] == 0 and predictions[i] == 1)
        false_neg += int(labels[i] == 1 and predictions[i] == 0)
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    try:
        Fscore = 2 * precision * recall / (precision + recall)
    except:
        Fscore = np.nan
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
    return {
        'Precision': precision,
        'Recall': recall,
        'F-score': Fscore,
        'Accuracy': accuracy,
    }