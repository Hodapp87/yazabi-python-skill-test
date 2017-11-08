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

train_X, train_y, test_X, test_y = data_preprocessing.get_processed_data()

"""
#svm = sklearn.svm.SVC(kernel="linear", random_state=123456)
svm = sklearn.svm.LinearSVC(random_state=123456)
selector = sklearn.feature_selection.RFECV(svm, step=1,
    cv=sklearn.model_selection.StratifiedShuffleSplit())
selector = selector.fit(train_X, train_y)
features = pd.DataFrame(
    {"Feature": list(train_X.columns),
     "RFE rank": selector.ranking_,
    })
features.sort_values("RFE rank", inplace=True, ascending=True)
features.to_csv("rfe_linearsvc.csv", index=False)
print(features)
"""

print("-"*70)
print("SVM:")
print("-"*70)
svm = sklearn.svm.SVC(kernel="rbf", random_state=123456)
params = {'C': np.linspace(0.4, 2, 5)}
clf = sklearn.model_selection.GridSearchCV(knn, params)
clf.fit(train_X[list(features.Feature[:(i+10)])], train_y)
print(clf.cv_results_)
print("Best score: {0}".format(clf.best_score_))
print("Best params: {0}".format(clf.best_params_))
