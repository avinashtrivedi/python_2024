# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 20:58:06 2020

@author: Victor Ponce-Lopez @ UCL Energy Institute

===================================================================
Multi-classification methods
===================================================================
"""

try:
    import _pickle as pickle
except:
    import pickle
import gensim
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem import PorterStemmer
from gensim import corpora, models


###############################################################################
# lemmatize and stem preprocessing steps
def lemmatize_stemming(text):
    ps = PorterStemmer()
    return ps.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


###############################################################################
# Infer topics
def LearnInferTopics(text, test_messages, save, dataset, n_topics, method, show):

    # preview example after preprocessing
    if show:    
        doc_sample = text.loc[210]['message']
        print('Sample from the original document: ')
        words = []
        for word in doc_sample.split(' '):
            words.append(word)
        print(words)
        print('\ntokenized and lemmatized document: ')
        print(preprocess(doc_sample))
    
    text.rename(columns = {'message' : 'headline_text'}, inplace = True)
    processed_docs = text['headline_text'].map(preprocess)
    
    # Create a dictionary from 'processed_docs' containing the number of times 
    # a word appears in the training set
    print("\nCreating dictionary from document ...")
    dictionary = gensim.corpora.Dictionary(processed_docs)
    print("Examples of words in the dictionary:")
    count = 0
    for k, v in dictionary.iteritems():
        print(k, v)
        count += 1
        if count > 10:
            break
    print('\n')
        
    # Filter out tokens that appear in less than no_below documents, more than 
    # no_above documents, and keep only the first keep_n documents
    print("Filtering dictionary contents ...")
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
    
    # Create a BoW dictrionary and preview
    print("\nCreating BoW dictionary ...")
    bow_corpus = []
    for doc in processed_docs:
        bow_corpus.append(dictionary.doc2bow(doc))
    # show example
    if show:
        print("BoW Tuple Example:")
        print(bow_corpus[210])
        for i in range(len(bow_corpus[210])):
            print("\tWord {} (\"{}\") appears {} time.".format(bow_corpus[210][i][0], 
                                                       dictionary[bow_corpus[210][i][0]], 
                                                       bow_corpus[210][i][1]))
        print('\n')

    # 1) RUN Latent Dirichlet allocation model using BoW and/or TF-IDF
    # 2) Explore the words occuring in that topic and its relative weight
    if method == 'lda-bow' or method == 'all':
        try: 
            with open('model_lda-bow_'+str(dataset)+'.pkl','rb') as infile:
                print("Loading LDA BoW model for "+dataset+" ...")
                lda_model = pickle.load(infile)
        except:
            print("Training LDA BoW model for "+dataset+" ...")
            lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=n_topics, id2word=dictionary, passes=2, workers=6)
        if save:
            with open('model_lda-bow_'+str(dataset)+'.pkl','wb') as outfile:
                pickle.dump(lda_model, outfile)
        # visualise topics and scores
        for idx, topic in lda_model.print_topics(-1):
            print('Topic: {} Words: {}'.format(idx, topic))
        if show:
            for index, score in sorted(lda_model[bow_corpus[210]], key=lambda tup: -1*tup[1]):
                print("\nScore: {}\t \nTopic: {} Word: {}".format(score, index, lda_model.print_topic(index, 10)))   
        print('\n')
    
    if method == 'lda-tfidf' or method == 'all':
        try: 
            with open('model_lda-tfidf_'+str(dataset)+'.pkl','rb') as infile:
                print("Loading LDA TF-IDF model for "+dataset+" ...")
                lda_model_tfidf = pickle.load(infile)
        except:
            tfidf = models.TfidfModel(bow_corpus)
            corpus_tfidf = tfidf[bow_corpus]
            from pprint import pprint
            for doc in corpus_tfidf:
                pprint(doc)
                break
            print("Training LDA TF-IDF model for "+dataset+" ...")
            lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=n_topics, id2word=dictionary, passes=2, workers=4)
        if save:
            with open('model_lda-tfidf_'+str(dataset)+'.pkl','wb') as outfile:
                pickle.dump(lda_model_tfidf, outfile)
        # visualise topics and scores
        for idx, topic in lda_model_tfidf.print_topics(-1):
            print('Topic: {} Word: {}'.format(idx, topic))
        if show:
            for index, score in sorted(lda_model_tfidf[bow_corpus[210]], key=lambda tup: -1*tup[1]):
                print("\nScore: {}\t \nTopic: {} Word: {}".format(score, index, lda_model_tfidf.print_topic(index, 10)))
        print('\n')
            
    # Test on unseen document
    if len(test_messages) > 0:
        print('\nTesting model on the unseen message: \n\t"{}" \n... \n'.format(test_messages))
        bow_vector_te = dictionary.doc2bow(preprocess(test_messages))
        # Clear non-indexed dictionary elements
        for v in bow_vector_te:
            try:
                lda_model[[v]]; lda_model_tfidf[[v]]; 
            except:
                bow_vector_te.remove(v)
        # Test on unseen document
        if method == 'lda-bow' or method == 'all':
            print("Topics inferred from LDA BoW model:")
            for index, score in sorted(lda_model[bow_vector_te], key=lambda tup: -1*tup[1]):
                print("Score: {}\t Topic: {} Word: {}".format(score, index, lda_model.print_topic(index, 5)))
        if method == 'lda-tfidf' or method == 'all':
            print("\nTopics inferred from LDA TF-IDF model:")
            for index, score in sorted(lda_model_tfidf[bow_vector_te], key=lambda tup: -1*tup[1]):
                print("Score: {}\t Topic: {} Word: {}".format(score, index, lda_model_tfidf.print_topic(index, 5)))