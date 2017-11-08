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

print("-"*70)
print("Linear SVM:")
print("-"*70)
svm = sklearn.svm.LinearSVC(random_state=123456, dual=False)
params = {
    'C': np.linspace(0.1, 10.0, 20),
    'penalty': ('l1', 'l2'),
    #'loss': ('hinge', 'squared_hinge'),
}
clf = sklearn.model_selection.GridSearchCV(svm, params, verbose=2, cv=sklearn.model_selection.StratifiedShuffleSplit(), n_jobs=-1)
clf.fit(train_X, train_y)
print(clf.cv_results_)
print("Best score: {0}".format(clf.best_score_))
print("Best params: {0}".format(clf.best_params_))
