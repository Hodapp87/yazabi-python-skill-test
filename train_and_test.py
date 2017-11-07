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
    # The features below are the top 18 features that
    # RandomForestClassifier returns in order of importance.  The
    # number was selected via cross-validation on the training data.
    columns = ('marital_status_Married-civ-spouse',
               'net_capital',
               'relationship_Husband',
               'age',
               'marital_status_Never-married',
               'education_num')
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
    dt = sklearn.tree.DecisionTreeClassifier()
    dt.fit(train_X, train_y)
    return dt

def train_knn(train_X, train_y):
    knn = sklearn.neighbors.KNeighborsClassifier(10, n_jobs=-1)
    knn.fit(train_X, train_y)
    return knn

def train_svm(train_X, train_y):
    columns = ('marital_status_Married-civ-spouse',
               'net_capital',
               'relationship_Husband',
               'age',
               'marital_status_Never-married',
               'education_num')
    """columns = ('marital_status_Married-civ-spouse',
               'relationship_Husband', 'net_capital',
               'marital_status_Never-married', 'education_num',
               'age', 'relationship_Own-child', 'hours_per_week',
               'relationship_Not-in-family',
               'occupation_Exec-managerial', 'male',
               'occupation_Other-service', 'education_Bachelors',
               'education_Masters', 'relationship_Wife',
               'occupation_Prof-specialty', 'education_Prof-school')"""
    idxs = [train_X.columns.get_loc(c) for c in columns]
    pipeline = sklearn.pipeline.make_pipeline(
        sklearn.preprocessing.FunctionTransformer(lambda x: x[:, idxs]),
        sklearn.svm.SVC(random_state=123456),
    )
    pipeline.fit(train_X, train_y)
    return pipeline

"""
# Feature importance tuning:
rf = sklearn.ensemble.RandomForestClassifier(
    n_estimators=100, criterion="entropy", max_depth=4, n_jobs=-1,
    random_state=12348)
rf = rf.fit(train_X, train_y)
features = pd.DataFrame(
    {"Feature": list(train_X.columns),
     "Importance": rf.feature_importances_})
features.sort_values("Importance", inplace=True, ascending=False)
#features

nb = sklearn.naive_bayes.GaussianNB()
rfe = sklearn.feature_selection.RFECV(nb)
rfe.fit(train_X, train_y)
"""

"""
knn = sklearn.neighbors.KNeighborsClassifier(10, n_jobs=-1)
scores = sklearn.model_selection.cross_val_score(
    knn,
    train_X[list(features.Feature[:15])],
    train_y,
    cv=sklearn.model_selection.StratifiedShuffleSplit())
print(scores)

nb = sklearn.naive_bayes.GaussianNB()
scores = sklearn.model_selection.cross_val_score(
    nb,
    train_X[list(features.Feature[:18])],
    train_y,
    cv=sklearn.model_selection.StratifiedShuffleSplit())
print(scores.mean(), scores.std())

# KNN tuning:
train_X, train_y, test_X, test_y = data_preprocessing.get_processed_data()

print("-"*70)
print("SVM:")
print("-"*70)
n_features = train_X.shape[1]
params = {
    'gamma': np.linspace(n_features / 3, n_features * 3, 20),
    'C': np.linspace(0.1, 1.0, 20),
}
svm = sklearn.svm.SVC(random_state=123456)
clf.fit(train_X, train_y)
#print(clf.cv_results_)
print("Best score: {0}".format(clf.best_score_))
print("Best params: {0}".format(clf.best_params_))
"""

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

# N.B.:
# The labels are very lopsided (it's 0 around 25% of the time)
# 'education' appears redundant with 'education_num'
# 'relationship' is likely redundant with 'marital_status' (certain ones at least)
# 'race' might have some redundancy with 'native_country'
# See if PCA can improve anything here.
# Perhaps use sklearn.feature_selection.RFE with GaussianNB or KNN.

if __name__ == '__main__':
    for algo in ("naive_bayes", "decision_tree", "svm"): # , "knn"):
        print(algo + ": ")
        train_and_validate(algo)
