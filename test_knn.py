#!/usr/bin/env python

import data_preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.ensemble
import sklearn.metrics
import sklearn.feature_selection
import sklearn.manifold
import sklearn.model_selection
import sklearn.naive_bayes
import sklearn.neighbors
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.svm
import sklearn.tree

def forward_selection(X, y, estimator, features=None, max_iters=None):
    if features is None:
        features = list(X.columns)
    else:
        features = list(features)
    if max_iters is None:
        max_iters = X.shape[1]
    best_features = []
    incr_accuracy = []
    base_accuracy = 0
    for i in range(max_iters):
        print("Iteration {0}/{1}:".format(i+1, max_iters))
        best_accuracy = 0
        best_feature = None
        for feature in features:
            scores = sklearn.model_selection.cross_val_score(
                estimator,
                X[best_features + [feature]],
                y,
                cv=sklearn.model_selection.StratifiedShuffleSplit())
            acc = scores.mean()
            if (acc > best_accuracy):
                best_feature = feature
                best_accuracy = acc
        if best_accuracy < base_accuracy:
            print("No features improved; quitting")
            break
        print("Feature \"{1}\" raises accuracy from {2} to {3}".format(i, best_feature, base_accuracy, best_accuracy))
        base_accuracy = best_accuracy
        best_features.append(best_feature)
        incr_accuracy.append(best_accuracy)
        print("{0} features: {1}".format(len(best_features), best_features))
        features.remove(best_feature)
    df = pd.DataFrame.from_dict(
        {"Feature": best_features, 
         "Accuracy": incr_accuracy,
        })
    return df

train_X, train_y, test_X, test_y = data_preprocessing.get_processed_data()

print("-"*70)
print("kNN:")
print("-"*70)
knn = sklearn.neighbors.KNeighborsClassifier(n_jobs=-1)
params = {'n_neighbors': range(1, 17), 'weights': ['distance', 'uniform']}
features = ['education_num', 'marital_status_Married-civ-spouse', 'net_capital']
clf = sklearn.model_selection.GridSearchCV(knn, params)
clf.fit(train_X[features], train_y)
print(clf.cv_results_)
print("Best score: {0}".format(clf.best_score_))
print("Best params: {0}".format(clf.best_params_))


#features = forward_selection(train_X, train_y, sklearn.neighbors.KNeighborsClassifier(6, n_jobs=-1))
#print(features)
