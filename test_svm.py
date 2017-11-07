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
