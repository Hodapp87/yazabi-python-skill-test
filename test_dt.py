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
print("Decision Tree:")
print("-"*70)
dt = sklearn.tree.DecisionTreeClassifier()
params = {
    'max_depth': (2,3,4,5,6,7,8,9,10,12,14,16,18,20,23,26,30,35,40,None),
    'min_samples_split': (2, 4, 8, 16),
    'min_impurity_decrease': (0.0, 0.05, 0.1),
}
clf = sklearn.model_selection.GridSearchCV(dt, params, verbose=100)
clf.fit(train_X, train_y)
print(clf.cv_results_)
print("Best score: {0}".format(clf.best_score_))
print("Best params: {0}".format(clf.best_params_))
