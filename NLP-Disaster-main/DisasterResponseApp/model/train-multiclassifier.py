# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import joblib
import string 
import sys 

import nltk
nltk.download(['punkt', 'wordnet','stopwords'])

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from scipy.sparse import csr_matrix
from sklearn import svm

from class_def import Lemmer

import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import validation_curve

# Loading data lite from file path
def load_data(database_filepath, noDisaster):
    """
	Load database and get dataset
	Args: 
		database_filepath (str): file path of sqlite database
	Return:
		X (pandas dataframe): Features
		y (pandas dataframe): Targets/ Labels
        categories (list): List of categorical columns
        :param databse_filepath:
    """
    engine = create_engine('sqlite:///../'+database_filepath)
    df = pd.read_sql_table('DS_messages', engine)
    engine.dispose()
    
    if noDisaster: df = df.drop('related',axis=1)
    
    X = df['message']
    y = df[df.columns[4:]]
    categories = y.columns.tolist()

    return X, y, categories

# Loading original data from file path
def load_OrigData(noDisaster=True):
    """
	Load database and get dataset
	Args: 
		database_filepath (str): file path of sqlite database
	Return:
		X_train (pandas dataframe): train Features
        X_test (pandas dataframe): test Features
		y_train (pandas dataframe): Targets/ Labels for training
        y_test (pandas dataframe): Targets/ Labels for testing
        categories (list): List of categorical columns
        :param databse_filepath:
    """
    path = '../../data/'
    filenames = ['disaster_response_messages_training.csv',
                 'disaster_response_messages_validation.csv',
                 'disaster_response_messages_test.csv',]
    for filename in filenames:
        df = pd.read_csv(path+filename, encoding = 'latin-1')
        df.drop(['ï»¿id', 'split', 'original', 'genre', 'PII', 'offer', 'child_alone'], 
                axis = 1, inplace = True)
        if noDisaster: df = df.drop('related', axis=1)
        if filename == 'disaster_response_messages_training.csv':
            X_train = df['message']
            y_train = df[df.columns[1:]]
            categories = y_train.columns.tolist()
        elif filename == 'disaster_response_messages_validation.csv':
            X_val = df['message']
            y_val = df[df.columns[1:]]
        elif filename == 'disaster_response_messages_test.csv':
            X_test = df['message']
            y_test = df[df.columns[1:]]

    return X_train, y_train, X_val, y_val, X_test, y_test, categories 

# Tokenizing
# def tokenize(text):
#     # normalize text and remove punctuation
#     text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
#     # tokenize text
#     tokens = word_tokenize(text)
#     stop_words = stopwords.words("english")
#     words = [w for w in tokens if w not in stop_words]
    
#     # Reduce words to their stems
#     stemmer = PorterStemmer()
#     stemmed = [stemmer.stem(w) for w in words]
    
#     # Reduce words to their root form
#     lemmatizer = WordNetLemmatizer()
#     lemmed = [lemmatizer.lemmatize(w) for w in stemmed]
    
#     return lemmed
  
# Building model
def build_model():
    """Returns the GridSearchCV model
    Args:
        None
    Returns:
        cv: Grid search model object
    """
    #clf = svm.SVC() #MultinomialNB() #BernoulliNB()

    # The pipeline has tfidf, dimensionality reduction, and classifier
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=Lemmer.tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(svm.SVC(gamma=10)))  #MultinomialNB() # #BernoulliNB()
    ])

    # Parameters for GridSearchCV
    param_grid = {
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__ngram_range': ((1, 1), (1,2)),
        'vect__max_features': (None, 5000,10000),
        'tfidf__use_idf': (True, False)
    }

    cv = GridSearchCV(pipeline, param_grid)

    return cv

def evaluate_model(model, X_test, y_test, categories):
    """Prints multi-output classification results
    Args:
        model (pandas dataframe): the scikit-learn fitted model
        X_text (pandas dataframe): The X test set
        y_test (pandas dataframe): the y test classifications
        category_names (list): the category names
    Returns:
        None
    """

    # Generate predictions
    y_pred = model.predict(X_test)

    # Print out the full classification report
    print(classification_report(y_test, y_pred, target_names=categories))

# Save model 
def save_model(model, model_filepath):
    """
    Dumps the model to given path 
    Args: 
        model: the fitted model
        model_filepath (str): filepath to save model
    Return:
        None
	"""
    joblib.dump(model, model_filepath)
    
def plot_validation_curve(estimator, X, y, title, axes=None, cv=None, n_jobs=None):    
    if axes is None:
        _, axes = plt.subplots(1, 1, figsize=(20, 5))
        
    param_range = np.logspace(0.5, 1.5, 5)
    train_scores, test_scores = validation_curve(
        estimator,
        X,
        y,
        param_name='clf__estimator__gamma',
        cv=cv,
        n_jobs=n_jobs,
        param_range=param_range,
        #scoring="accuracy",
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.title(title)
    plt.xlabel(r"$\gamma$")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(
        param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw
    )
    plt.fill_between(
        param_range,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.2,
        color="darkorange",
        lw=lw,
    )
    plt.semilogx(
        param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw
    )
    plt.fill_between(
        param_range,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.2,
        color="navy",
        lw=lw,
    )
    plt.legend(loc="best")
    
    return plt
    
def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, "o-")
    axes[2].fill_between(
        fit_times_mean,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt
  
def custom_cv(trainIndices, valIndices):
    yield trainIndices, valIndices
    
  
def main():
    if len(sys.argv) == 1:
        # database_filepath, model_filepath = sys.argv[1:]
        database_filepath = 'data/MDRM-lite.db'
        model_filepath = 'DS_model_SVC.pkl'
        #model_filepath = 'Kaggle_SVCLITE05.pkl'
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        
        X, y, categories = load_data(database_filepath, noDisaster=False)
        # cv = ShuffleSplit(n_splits=5, test_size=0.33, random_state=42)
        cv = joblib.load('datasplits.pkl')        
        for tr, te in cv.split(X): print("TRAIN:", tr, "TEST:", te)
        X_train = X[tr]; y_train = y.loc[tr]; X_test = X[te]; y_test = y.loc[te]
        #joblib.dump(cv, 'datasplits.pkl')
        
        #X_train, y_train, X_val, y_val, X_test, y_test, categories = load_OrigData(noDisaster=True)
        # #idx = np.random.choice(len(X_train), size=int(0.75*len(X_train)))
        # #joblib.dump(idx, 'datasplitsOrigLITE075.pkl')
        #idx = joblib.load('datasplitsOrigLITE05.pkl')        
        #X_train = X_train[idx]; y_train = y_train.loc[idx]
        #X = pd.concat([X_train,X_val], sort=False); X.reset_index(drop=True, inplace=True)
        #y = pd.concat([y_train,y_val], sort=False); y.reset_index(drop=True, inplace=True)
        #valIndices =  y.index[len(y_train):].values.astype(int)
        #cv = custom_cv(valIndices, valIndices)
        
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        if 'related' not in categories: model_filepath = model_filepath[:-4]+'_noRelatedClass.pkl'

        #print('Building model...')
        #model = build_model()
        
        model = joblib.load(model_filepath)

        #print('Training model {} ...'.format(model_filepath))
        #model.fit(X_train, y_train)
        # model.fit(X_train.to_numpy(), csr_matrix(y_train).todense())  # for CategoricalNB

        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, categories)

        #print('Saving model...\n    MODEL: {}'.format(model_filepath))
        #save_model(model, model_filepath)
        #print('Trained model saved!')       
        
        
        # plot validation curves
        # fig, axes = plt.subplots(1, 1, figsize=(10, 15))
        # title = r"Validation Curve with (SVC)"
        # # train_scores, test_scores = validation_curve(
        # #     model.estimator,
        # #     X,
        # #     y,
        # #     param_name='clf__estimator__gamma',
        # #     cv=cv,
        # #     n_jobs=1,
        # #     param_range=1,
        # #     scoring="accuracy",
        # # )
        # plot_validation_curve(
        #     model.estimator, title, X, y, axes=axes, cv=cv, n_jobs=1
        # )
        # plt.show()
        
        # plot learning curves
        # plt.close()
        #fig, axes = plt.subplots(3, 1, figsize=(10, 15))
        #title = r"Learning Curves (SVM, RBF kernel)"
        #plot_learning_curve(
        #    model.estimator, title, X, y, axes=axes[:], ylim=(0, 1.01), cv=cv, n_jobs=1
        #)
        #plt.show()
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
