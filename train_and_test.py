#!/usr/bin/env python

###########################################################################
# train_and_test.py: Training & validation for sklearn mini-project
# Author: Chris Hodapp (hodapp87@gmail.com)
# Date: 2017-11-03
###########################################################################

import data_preprocessing

import pandas as pd
import sklearn.metrics
import sklearn.neighbors
import sklearn.svm

def train_and_validate(algorithm):
    """This function is mandated by the requirements.  Argument
    'algorithm' takes values of 'naive_bayes', 'decision_tree', 'knn',
    and 'svm', and the function will train the respective classifier
    on training data, predict on the testing data, and prints that
    classifier's testing accuracy.
    """
    # I'm going to assume we should use the specified classifier by
    # itself, and using it as the base classifier in an ensemble is
    # cheating.
    #
    # Below maps algorithm name to a function producing a model
    # from training data:
    algos = {
        "naive_bayes": train_naive_bayes,
        "decision_tree": train_decision_tree,
        "knn": train_knn,
        "svm": train_svm,
    }
    # Load data and train respective model:
    train_X, train_y, test_X, test_y = data_preprocessing.get_processed_data()
    train_fn = algos[algorithm]
    model = train_fn(train_X, train_y)
    # Predict from testing data and get accuracy:
    predict_y = model.predict(test_X)
    accuracy = sklearn.metrics.accuracy_score(test_y, predict_y)
    print(accuracy)

def train_naive_bayes(train_X, train_y):
    pass

def train_decision_tree(train_X, train_y):
    pass

def train_knn(train_X, train_y):
    knn = sklearn.neighbors.KNeighborsClassifier(5, n_jobs=-1)
    knn.fit(train_X, train_y)
    return knn

def train_svm(train_X, train_y):
    svm = sklearn.svm.SVC(random_state=123456)
    svm.fit(train_X, train_y)
    return svm

# N.B.:
# The labels are very lopsided (it's 0 around 25% of the time)
# 'education' appears redundant with 'education_num'
# 'relationship' is likely redundant with 'marital_status' (certain ones at least)
# 'race' might have some redundancy with 'native_country'

if __name__ == '__main__':
    for algo in ("naive_bayes", "decision_tree", "svm", "knn"):
        print(algo + ": ")
        try:
            train_and_validate(algo)
        except Exception as e:
            print(e)
