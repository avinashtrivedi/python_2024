# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 12:44:05 2020

@author: Victor Ponce-Lopez @ UCL Energy Institute

===================================================================
Disaster Classifier
Initially forked from:
    https://github.com/pointoflight/tweet_classification
===================================================================
"""

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from math import log

def process_message(message, lower_case = True, stem = True, stop_words = True, gram = 2):
    if lower_case:
        message = message.lower()
    words = word_tokenize(message)
    words = [w for w in words if len(w) > 2]
    if gram > 1:
        w = []
        for i in range(len(words) - gram + 1):
            w += [' '.join(words[i:i + gram])]
        return w
    if stop_words:
        sw = stopwords.words('english')
        words = [word for word in words if word not in sw]
    if stem:
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]   
    return words

class DisasterClassifier(object):
    def __init__(self, trainData, method = 'tf-idf'):
        self.tweets, self.labels = trainData['message'], trainData['label']
        self.method = method

    def train(self):
        self.calc_TF_and_IDF()
        if self.method == 'tf-idf':
            self.calc_TF_IDF()
        else:
            self.calc_prob()

    def calc_prob(self):
        self.prob_disaster = dict()
        self.prob_normal = dict()
        for word in self.tf_disaster:
            self.prob_disaster[word] = (self.tf_disaster[word] + 1) / (self.disaster_words + len(list(self.tf_disaster.keys())))
        for word in self.tf_normal:
            self.prob_normal[word] = (self.tf_normal[word] + 1) / (self.normal_words + len(list(self.tf_normal.keys())))
        self.prob_disaster_mail, self.prob_normal_mail = self.disaster_tweets / self.total_tweets, self.normal_tweets / self.total_tweets 

    
    # TF-IDF
    def calc_TF_and_IDF(self):
        noOfMessages = self.tweets.shape[0]
        self.disaster_tweets, self.normal_tweets = self.labels.value_counts()[1], self.labels.value_counts()[0]
        self.total_tweets = self.disaster_tweets + self.normal_tweets
        self.disaster_words = 0
        self.normal_words = 0
        self.tf_disaster = dict()
        self.tf_normal = dict()
        self.idf_disaster = dict()
        self.idf_normal = dict()
        for i in range(noOfMessages):
            message_processed = process_message(self.tweets[self.tweets.index[i]])
            count = list() #To keep track of whether the word has ocured in the message or not.
                           #For IDF
            for word in message_processed:
                if self.labels[self.labels.index[i]]:
                    self.tf_disaster[word] = self.tf_disaster.get(word, 0) + 1
                    self.disaster_words += 1
                else:
                    self.tf_normal[word] = self.tf_normal.get(word, 0) + 1
                    self.normal_words += 1
                if word not in count:
                    count += [word]
            for word in count:
                if self.labels[self.labels.index[i]]:
                    self.idf_disaster[word] = self.idf_disaster.get(word, 0) + 1
                else:
                    self.idf_normal[word] = self.idf_normal.get(word, 0) + 1

    def calc_TF_IDF(self):
        self.prob_disaster = dict()
        self.prob_normal = dict()
        self.sum_tf_idf_disaster = 0
        self.sum_tf_idf_normal = 0
        for word in self.tf_disaster:
            self.prob_disaster[word] = (self.tf_disaster[word]) * log((self.disaster_tweets + self.normal_tweets)                                                           / (self.idf_disaster[word] + self.idf_normal.get(word, 0)))
            self.sum_tf_idf_disaster += self.prob_disaster[word]
        for word in self.tf_disaster:
            self.prob_disaster[word] = (self.prob_disaster[word] + 1) / (self.sum_tf_idf_disaster + len(list(self.prob_disaster.keys())))
            
        for word in self.tf_normal:
            self.prob_normal[word] = (self.tf_normal[word]) * log((self.disaster_tweets + self.normal_tweets)                                                           / (self.idf_disaster.get(word, 0) + self.idf_normal[word]))
            self.sum_tf_idf_normal += self.prob_normal[word]
        for word in self.tf_normal:
            self.prob_normal[word] = (self.prob_normal[word] + 1) / (self.sum_tf_idf_normal + len(list(self.prob_normal.keys())))
            
    
        self.prob_disaster_mail, self.prob_normal_mail = self.disaster_tweets / self.total_tweets, self.normal_tweets / self.total_tweets 
                    
    def classify(self, processed_message):
        pdisaster, pnormal = 0, 0
        for word in processed_message:                
            if word in self.prob_disaster:
                pdisaster += log(self.prob_disaster[word])
            else:
                if self.method == 'tf-idf':
                    pdisaster -= log(self.sum_tf_idf_disaster + len(list(self.prob_disaster.keys())))
                else:
                    pdisaster -= log(self.disaster_words + len(list(self.prob_disaster.keys())))
            if word in self.prob_normal:
                pnormal += log(self.prob_normal[word])
            else:
                if self.method == 'tf-idf':
                    pnormal -= log(self.sum_tf_idf_normal + len(list(self.prob_normal.keys()))) 
                else:
                    pnormal -= log(self.normal_words + len(list(self.prob_normal.keys())))
            pdisaster += log(self.prob_disaster_mail)
            pnormal += log(self.prob_normal_mail)
        return pdisaster >= pnormal
    
    def predict(self, testData):
        result = dict()
        for (i, message) in enumerate(testData):
            processed_message = process_message(message)
            result[i] = int(self.classify(processed_message))
        return result