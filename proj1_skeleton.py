# -*- coding: utf-8 -*-
"""
EECS 445 - Introduction to Machine Learning
Winter 2015
Project 1 - Everyone's a Critic
"""

import sys
import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
import sklearn
from sklearn import metrics
from string import punctuation
import string
import random

def read_vector_file(fname):
    """
      HELPER FUNCTION
      Reads and returns a vector from the label file specified by fname.
      Input: 
        fname- string specifying a filename 
      Returns an (n,) array where n is the number of lines in the file.
    """
    return np.genfromtxt(fname)
    
def write_label_answer(vec, outfile):
    """
      HELPER FUNCTION
      Writes your label vector to the given file.
      The vector must be of shape (70, ) or (70, 1),
      Input:
          vec- (70,) or (70,1) array containing labels  {1,-1}, 
          outfile- string with filename to write to
    """
    
    if(vec.shape[0] != 70):
        print("Error - output vector should have 70 rows.")
        print("Aborting write.")
        return

    for v in vec:
        if((v != -1.0) and (v != 1.0)):
            print("Invalid value in input vector.")
            print("Aborting write.")
            return
        
    np.savetxt(outfile, vec)    

def extract_dictionary(infile):
    """
      Given a filename, reads the text file and builds a dictionary of unique 
      words/punctuations. 
      Input: 
        infile- string specificying a file name
      Returns: a dictionary containg d unique keys, (key, value) pairs are
      (word, index)
    """
    word_list = {}

    with open(infile, 'rU') as f:
        text = f.read().lower().strip()
        for c in text:
            if c in punctuation:
                word_list[c] = word_list.get(c,0) + 1
        for k in word_list.keys():
            text = text.replace(k,' ')
        tokens = text.split()
        for t in tokens:
            word_list[t] = word_list.get(t,0) + 1
    return word_list

def extract_feature_vectors(infile, word_list):
    """
      Produces a bag-of-words representation for each line of a text file 
      specified by the filename infile based on the dictionary word_list.
      Input:
          infile- string specificying file name 
          word_list- dictionary with (word,index) as (key,value) pairs
      Returns: feature_matrix (n,d) array , where the text file has n non-blank
      lines, and the dicionary has length d
    """
    num_lines = sum(1 for line in open(infile,'rU'))
    feature_matrix = np.zeros([num_lines, len(word_list)])
    
    with open(infile,'rU') as f:
        for i, l in enumerate(f):
            l = l.lower()
            l_list = l.translate(string.maketrans(punctuation,' '*len(punctuation))).strip().split()
            for j, k in enumerate(word_list.keys()):
                if k in punctuation:
                    feature_matrix[i,j] = 1 if k in l else 0
                else:
                    feature_matrix[i,j] = 1 if k in l_list else 0
                    
    return feature_matrix

def performance(y_true, y_pred, metric="accuracy"):
    """
        Calculates the performance metric based on the agreement between the 
        true labels and the predicted labels
        Input:
          y_true- (n,) array containing known labels
          y_pred- (n,) array containing predicted labels
          metric- string option used to select the performance measure
        Returns: the performance as a np.float64
    """
    if metric == "f1-score":
        score = metrics.f1_score(y_true, y_pred)
    elif metric == "auroc":
        score = metrics.roc_auc_score(y_true, y_pred)
    elif metric == "precision":
        score = metrics.precision_score(y_true, y_pred)
    elif metric == "sensitivity":
        cf_matrix = metrics.confusion_matrix(y_true, y_pred)
        score = np.trace(cf_matrix) / float(cf_matrix.sum())
    elif metric == "specificity":
        cf_matrix = metrics.confusion_matrix(y_true, y_pred)
        n = cf_matrix.shape[0]
        score = (n-2) / float(n-1) + float(np.trace(cf_matrix)) / (cf_matrix.sum() * (n-1))
    else:
        # accuracy
        score = metrics.accuracy_score(y_true, y_pred)

    performance=np.float64(score)

    return performance
    
def cv_performance(clf, X, y, k=5, metric="accuracy"):
    """
        Splits the data, X and y, into k-folds and runs k-fold crossvalidation: 
        training a classifier on K-1 folds and testing on the remaining fold. 
        Calculates the k-fold crossvalidation performance metric for classifier
        clf by averaging the performance across all test folds. 
        Input:
          clf- an instance of SVC()
          X- (n,d) array of feature vectors, where n is the number of examples
             and d is the number of features
          y- (n,) array of binary labels {1,-1}
          k- int specifying the number of folds (default=5)
          metric- string specifying the performance metric (default='accuracy',
                   other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                   and 'specificity')
        Returns: average 'test' performance across the k folds as np.float64
    """
    skf = sklearn.cross_validation.StratifiedKFold(y,n_folds=k)
    metric_values = []
    for train_index, test_index in skf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        
        metric_values.append(performance(y_test,pred,metric=metric))
    
    avg_performance=np.float64(metric_values).mean()
    print avg_performance
    return avg_performance
    
def select_param_rbf(X, y, k=5, metric="accuracy"):
    """
        Sweeps different settings for the hyperparameters of an RBF-kernel SVM,
        calculating the k-fold CV performance for each setting on X, y.         
        Input:
          X- (n,d) array of feature vectors, where n is the number of examples
             and d is the number of features
          y- (n,) array of binary labels {1,-1}
          k- int specifying the number of folds (default=5)
          metric- string specifying the performance metric (default='accuracy',
                   other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                   and 'specificity')
        Returns the parameter value(s) for an RBF-kernel SVM, that 'maximize'
        the average 5-fold CV performance.
    """   
    best_gamma = 0
    best_C = 0
    
    best_perf = 0
    for c in [1e-3,1e-2,1e-1,1e0,1e1,1e2][::-1]:
        for g in [1e-3,1e-2,1e-1,1e0,1e1,1e2]:
            clf = SVC(kernel='rbf', C=c, gamma=g)
            performance = cv_performance(clf,X,y,k=k,metric=metric)
            #print performance, c, g, metric
            if performance > best_perf: best_gamma, best_C, best_perf = g, c, performance

    return best_gamma, best_C

def select_param_linear(X, y, k=5, metric="accuracy"):
    """
        Sweeps different settings for the hyperparameter of a linear-kernel SVM,
        calculating the k-fold CV performance for each setting on X, y.         
        Input:
          X- (n,d) array of feature vectors, where n is the number of examples
             and d is the number of features
          y- (n,) array of binary labels {1,-1}
          k- int specifying the number of folds (default=5)
          metric- string specifying the performance metric (default='accuracy',
                   other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                   and 'specificity')
        Returns the parameter value for linear-kernel SVM, that 'maximizes' the
        average 5-fold CV performance.
    """
    best_C=0
    best_perf = 0
    for c in [1e-3,1e-2,1e-1,1e0,1e1,1e2]:
        clf = SVC(kernel='linear', C=c)
        performance = cv_performance(clf,X,y,k=k,metric=metric)
        if performance > best_perf: best_C, best_perf = c, performance

    return best_C
 
def performance_CI(clf, X, y, metric="accuracy"):
    """
        Estimates the performance of clf on X,y and the corresponding 95%CI
        (lower and upper bounds) 
        Input:
          clf-an instance of SVC() that has already been fit to data
          X- (n,d) array of feature vectors, where n is the number of examples
             and d is the number of features
          y- (n,) array of binary labels {1,-1}
          metric- string specifying the performance metric (default='accuracy',
                   other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                   and 'specificity')
        Returns:
            a tuple containing the performance of clf on X,y and the corresponding
            confidence interval (all three values as np.float64's)
    """
    n = X.shape[0]
    # bootstrap parameter
    B = 1000
    p_l = 25
    p_r = 975
    scores = []

    for i in range(B):
        # select n samples
        samples = [random.randint(0,n-1) for j in range(n)]
        sample_pred = clf.predict(X[samples])
        y_sample = y[samples]
        score = performance(y_sample, sample_pred, metric=metric)
        scores.append(score)

    scores.sort()

    perf=np.float64(np.mean(scores))
    lower=np.float64(scores[p_l])
    upper=np.float64(scores[p_r])

    return perf, lower, upper  
 
def main():
    """**TO COMPLETE**"""
    # Read the tweets and its labels   
    dictionary = extract_dictionary('tweets.txt')
    X = extract_feature_vectors('tweets.txt', dictionary)
    y = read_vector_file('labels.txt')
    
    # Split the data into training and testing set (train: 1-560, test: 561-630)
    X_train = X[:560,:]
    y_train = y[:560]
    X_test = X[560:,:]
    y_test = y[560:]

    # Select Hyperparameters for Linear-Kernel SVM
    # options = ['accuracy','f1-score', 'auroc', 'precision', 'sensitivity', 'specificity']
    # for opt in options:
    #     print select_param_linear(X_train,y_train,k=5,metric=opt)

    # Select Hyperparameters for RBF-Kernel SVM
    # options = ['accuracy','f1-score', 'auroc', 'precision', 'sensitivity', 'specificity']
    # for opt in options:
    #     print "method: %s-----------------------------------" % opt
    #     print select_param_rbf(X_train,y_train,k=5,metric=opt)

    C = 100; gamma = 0.01 # to be deleted~!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Train Linear-Kernel with selected C
    clf_linear = SVC(kernel='linear', C=C)
    clf_linear.fit(X_train, y_train)
    # Train RBF-Kernel with selected C and gamma
    clf_rbf = SVC(kernel='rbf', C=C, gamma=gamma)
    clf_rbf.fit(X_train, y_train)

    # Apply Classifiers to Test Set and Measure Performance (with CI)
    perf, lower, upper = performance_CI(clf_rbf, X_test, y_test, metric="accuracy")
    print perf, lower, upper
    # Appy your best classifier to the held_out_tweets.txt

main()
