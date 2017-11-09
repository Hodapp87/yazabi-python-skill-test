#!/usr/bin/env python

###########################################################################
# train_and_test.py: Training & validation for sklearn mini-project
# Author: Chris Hodapp (hodapp87@gmail.com)
# Date: 2017-11-03
###########################################################################

import data_preprocessing

import pandas as pd
import numpy as np
import sklearn.decomposition
import sklearn.ensemble
import sklearn.metrics
import sklearn.feature_selection
import sklearn.model_selection
import sklearn.naive_bayes
import sklearn.neighbors
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.svm
import sklearn.tree

def train_and_validate(algorithm):
    """This function is mandated by the requirements.  Argument
    'algorithm' takes values of 'naive_bayes', 'decision_tree', 'knn',
    and 'svm', and the function will train the respective classifier
    on training data, predict on the testing data, and print that
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
    # Get testing accuracy:
    predict_y = model.predict(test_X)
    test_acc = sklearn.metrics.accuracy_score(test_y, predict_y)
    print(test_acc)

def train_naive_bayes(train_X, train_y):
    # GaussianNB (which I'm assuming is what 'naive_bayes' meant) has
    # seemingly no hyperparameters to speak of, so feature
    # selection/transformation is where I started.  The below features
    # were found via a couple runs of forward selection:
    columns = ['net_capital', 'education_Prof-school',
               'education_Doctorate', 'occupation_Transport-moving',
               'education_Masters', 'marital_status_Never-married',
               'education_Bachelors', 'relationship_Not-in-family',
               'occupation_Exec-managerial']
    # Turn columns to indices, as FunctionTransformer seems to receive
    # normal NumPy arrays (not dataframes):
    idxs = [train_X.columns.get_loc(c) for c in columns]
    pipeline = sklearn.pipeline.make_pipeline(
        sklearn.preprocessing.FunctionTransformer(lambda x: x[:, idxs]),
        sklearn.naive_bayes.GaussianNB(),
    )
    pipeline.fit(train_X, train_y)
    return pipeline

def train_decision_tree(train_X, train_y):
    # Hyperparameters found with GridSearchCV:
    dt = sklearn.tree.DecisionTreeClassifier(
        min_samples_split = 2,
        min_samples_leaf = 4,
        max_depth = 10,
    )
    dt.fit(train_X, train_y)
    return dt

def train_knn(train_X, train_y):
    # Features selected with forward selection:
    columns = ['education_num', 'marital_status_Married-civ-spouse',
               'net_capital']
    # Same arrangement as train_naive_bayes:
    idxs = [train_X.columns.get_loc(c) for c in columns]
    pipeline = sklearn.pipeline.make_pipeline(
        sklearn.preprocessing.FunctionTransformer(lambda x: x[:, idxs]),
        sklearn.neighbors.KNeighborsClassifier(11, weights="distance", n_jobs=-1),
    )
    pipeline.fit(train_X, train_y)
    return pipeline

def train_svm(train_X, train_y):
    # Hyperparameters found with GridSearchCV:
    svm = sklearn.svm.SVC(kernel="rbf", C=40.0, gamma=1/100.0, random_state=123456)
    svm.fit(train_X, train_y)
    return svm

if __name__ == '__main__':
    for algo in ("naive_bayes", "decision_tree", "knn", "svm"):
        print(algo + ": ")
        train_and_validate(algo)
