#!/usr/bin/env python

###########################################################################
# train_and_test.py: Training & validation for sklearn mini-project
# Author: Chris Hodapp (hodapp87@gmail.com)
# Date: 2017-11-03
###########################################################################

import data_preprocessing

import pandas as pd
import numpy as np
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
    # GaussianNB (which I'm assuming is what 'naive_bayes' meant) has
    # seemingly no hyperparameters to speak of, so feature
    # selection/transformation is the place to begin with tuning.
    #
    columns = ['net_capital', 'education_Prof-school',
               'education_Doctorate', 'occupation_Transport-moving',
               'education_Masters', 'marital_status_Never-married',
               'education_Bachelors', 'relationship_Not-in-family',
               'occupation_Exec-managerial']
    #columns = ['education_num', 'marital_status_Married-civ-spouse', 'net_capital']
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
    columns = ['education_num', 'marital_status_Married-civ-spouse', 'net_capital']
    idxs = [train_X.columns.get_loc(c) for c in columns]
    pipeline = sklearn.pipeline.make_pipeline(
        sklearn.preprocessing.FunctionTransformer(lambda x: x[:, idxs]),
        sklearn.tree.DecisionTreeClassifier(),
    )
    pipeline.fit(train_X, train_y)
    return pipeline

def train_knn(train_X, train_y):
    columns = ['education_num', 'marital_status_Married-civ-spouse', 'net_capital']
    idxs = [train_X.columns.get_loc(c) for c in columns]
    pipeline = sklearn.pipeline.make_pipeline(
        sklearn.preprocessing.FunctionTransformer(lambda x: x[:, idxs]),
        sklearn.neighbors.KNeighborsClassifier(5, weights="distance", n_jobs=-1),
    )
    pipeline.fit(train_X, train_y)
    return pipeline

def train_svm(train_X, train_y):
    # Rank 1 features of RFECV:
    columns = [ 'age', 'native_country_Germany',
                'native_country_France', 'native_country_England',
                'native_country_Dominican-Republic', 'native_country_Cuba',
                'native_country_Columbia', 'native_country_China',
                'native_country_Canada', 'native_country_Cambodia',
                'race_White', 'race_Other', 'race_Black',
                'race_Amer-Indian-Eskimo', 'relationship_Wife',
                'relationship_Unmarried', 'relationship_Own-child', 'male',
                'relationship_Not-in-family', 'relationship_Husband',
                'native_country_Greece', 'native_country_Guatemala',
                'native_country_Honduras', 'native_country_Hong',
                'native_country_Yugoslavia', 'native_country_Vietnam',
                'native_country_United-States', 'native_country_Thailand',
                'native_country_Taiwan', 'native_country_South',
                'native_country_Scotland', 'native_country_Portugal',
                'native_country_Poland', 'occupation_Transport-moving',
                'native_country_Philippines',
                'native_country_Outlying-US(Guam-USVI-etc)',
                'native_country_Nicaragua', 'native_country_Mexico',
                'native_country_Laos', 'native_country_Japan',
                'native_country_Jamaica', 'native_country_Italy',
                'native_country_Ireland', 'native_country_Hungary',
                'native_country_Peru', 'occupation_Tech-support',
                'relationship_Other-relative', 'occupation_Protective-serv',
                'education_Preschool', 'occupation_Sales',
                'education_Assoc-acdm', 'education_9th', 'education_7th-8th',
                'education_5th-6th', 'education_1st-4th',
                'workclass_Without-pay', 'workclass_State-gov',
                'workclass_Self-emp-not-inc', 'workclass_Private',
                'workclass_Local-gov', 'workclass_Federal-gov',
                'hours_per_week', 'education_num', 'education_Prof-school',
                'marital_status_Divorced', 'net_capital',
                'occupation_Priv-house-serv',
                'marital_status_Married-civ-spouse', 'occupation_Other',
                'marital_status_Married-spouse-absent',
                'marital_status_Never-married', 'occupation_Prof-specialty',
                'occupation_Exec-managerial', 'marital_status_Separated',
                'marital_status_Married-AF-spouse',
                'occupation_Machine-op-inspct', 'occupation_Armed-Forces',
                'occupation_Handlers-cleaners', 'occupation_Farming-fishing',
                'occupation_Other-service', 'marital_status_Widowed' ]
    idxs = [train_X.columns.get_loc(c) for c in columns]
    pipeline = sklearn.pipeline.make_pipeline(
        sklearn.preprocessing.FunctionTransformer(lambda x: x[:, idxs]),
        sklearn.svm.SVC(kernel="rbf", random_state=123456),
    )
    pipeline.fit(train_X, train_y)
    return pipeline

"""
print("-"*70)
print("KNN:")
print("-"*70)
for i in range(30):
    print("First {0} features:".format(i + 10))
    knn = sklearn.neighbors.KNeighborsClassifier(n_jobs=-1)
    params = {'n_neighbors': range(1, 13)}
    clf = sklearn.model_selection.GridSearchCV(knn, params)
    clf.fit(train_X[list(features.Feature[:(i+10)])], train_y)
    #print(clf.cv_results_)
    print("Best score: {0}".format(clf.best_score_))
    print("Best params: {0}".format(clf.best_params_))
"""

if __name__ == '__main__':
    for algo in ("naive_bayes", "decision_tree", "svm", "knn"):
        print(algo + ": ")
        train_and_validate(algo)
